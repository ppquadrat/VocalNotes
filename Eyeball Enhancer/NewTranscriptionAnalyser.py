"""
   Author: John M. McBride
   Date: 2020.04.22
"""


import argparse
import os
from pathlib import Path
import pickle
import sys
import time

import tkinter as tk
from tkinter import Tk, Canvas, Frame, Menu
from tkinter import StringVar, IntVar, DoubleVar, BooleanVar
from tkinter import messagebox as msg
from tkinter import Button, Radiobutton, Checkbutton, Label, Entry
from tkinter import NORMAL, DISABLED, RAISED, END

import librosa
import matplotlib as mpl
from matplotlib import cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import MouseEvent
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import simpleaudio as sa

import transcription_evaluation as TE


PATH_BASE = Path("/home/jmcbride/projects/BirdSongSpeech/Polina_transcriptions/JapanGroup")


#----------------------------------------------------------
# Parse_arguments

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("songname", default="", type=str,
                        help="Choose a songname")
    parser.add_argument("transcriber_A", default=False, type=str,
                        help="Enter a two-letter code for transcriber A")
    parser.add_argument("transcriber_B", default=False, type=str,
                        help="Enter a two-letter code for transcriber B")
    return parser.parse_args()


#----------------------------------------------------------
# Load / check paths

# This function finds any file name that starts with the song name,
# so that it can recognize different file types.
# This may break if there is more than just audio in this folder.
def get_audio_path(songname):
    path_audio = sorted(PATH_BASE.joinpath("Audio").glob(f"{songname}*"))
    return path_audio[0]


def get_paths(songname, tA, tB):
    path_pitches = PATH_BASE.joinpath("Transcription", f"{songname}", f"{tA}", f"{songname}_f0.csv")
    path_audio = get_audio_path(songname)
    path_transA = PATH_BASE.joinpath("Transcription", f"{songname}", f"{tA}", f"{songname}_note.csv")
    path_transB = PATH_BASE.joinpath("Transcription", f"{songname}", f"{tB}", f"{songname}_note.csv")
    return path_pitches, path_audio, path_transA, path_transB


def check_files_exist(songname, tA, tB):
    exists = True
    for path in get_paths(songname, tA, tB):
        if not path.exists():
            print(f"Required path does not exist:\n{path}\n")
            exists = False
    return exists
    

#----------------------------------------------------------
# Load data

def load_audio(path, allow_stereo=False):
    try:
        ext = path.suffix
    except:
        ext = Path(path).suffix

    try:
        wav, fr = sf.read(path)
    except:
        raise Exception(f"File extension '{ext}' not recognized\n{path}")

    if allow_stereo:
        return fr, wav

    if len(wav.shape)>1:
        return fr, wav.mean(axis=1)
    else:
        return fr, wav


def load_transcriptions(f):
    on, freq, dur = np.loadtxt(f, delimiter=',').T
    off = on + dur 
    return {'on':on, 'off':off, 'freq':freq, 'dur':dur}


#----------------------------------------------------------
# Audio tools

def wav2int(wav, c=20000):
    return (wav * c / np.max(wav)).astype(np.int16)


def clip_audio(fr, wav, beg, end, e1=0.5, e2=4., repeat=0):
    ibeg = int(beg * fr)
    iend = int(end * fr)
    wav = wav[ibeg:iend]

    envelope = np.ones(len(wav), float)
    imid = int(len(wav)/2)
    envelope[:imid] = np.linspace(0, 1, imid)**e1
    envelope[imid:] = 1 - np.linspace(0, 1, len(wav)-imid)**e2
    clipped_audio = wav * envelope

    if repeat:
        clipped_audio = np.concatenate([clipped_audio] * repeat, axis=0)

    return clipped_audio


def synth_tone(freq, N, fr, ramp=10, level=60, maxLevel=110):
    amp = 10**((level - maxLevel) / 20)
    tone = amp * np.sin(2*np.pi * freq * np.arange(N) / fr)
    iramp = int(ramp * 0.001 * fr)
    tone[:iramp] = tone[:iramp] * (1 - np.cos(np.pi * 2 * np.linspace(0, 1, iramp)))
    tone[-iramp:] = tone[-iramp:] * (1 - np.cos(np.pi * 2 * np.linspace(1, 0, iramp)))
    return tone


#----------------------------------------------------------

class TransAnalyse(Frame):

    #------------------------------------------------------
    def __init__(self, root, songname, tA, tB):
        Frame.__init__(self, root)
        self.master = root
        self.grid_rows = 0
        self.grid_columns = 0

        self.songname = songname
        self.transcriber_A = tA
        self.transcriber_B = tB


        # Title and size of the window
        self.master.title('Transcription Analyser')


        ### Initialise algorithm parameters
        self.initialise_parameters()
        self.load_paths()
        self.load_data()

        ### Create figure
        self.init_figure()

        ### Add parameter/control widgets
        self.playback_widget()
        self.run_save_reset_widget()


        ### Add figure to Frame
        self.graph.get_tk_widget().grid(row=0, column=3, rowspan=self.grid_rows+2, sticky='nsew')
        self.toolbar.grid(row=self.grid_rows+2, column=3)

        ### Adjust grid parameters
        self.master.grid_columnconfigure(3, weight=1)
        self.master.grid_rowconfigure(self.grid_rows+1, weight=1)


        ### Define the default file options for opening files:
        self.file_opt = {}
        self.file_opt['defaultextension'] = '.wav'
        self.file_opt['filetypes'] = [('audio files', '.wav .mp3'), ('previous analyses', '.pickle')]
        self.file_opt['initialdir'] = '.'
 
        # the window to disable when the file open/save window opens on top of it:
        self.file_opt['parent'] = root
        self.input_filename = 'None chosen'
        self.input_path = None

        self.plot_data()


    def load_paths(self):
        path_list = get_paths(self.songname, self.transcriber_A, self.transcriber_B)
        self.path_pitches = path_list[0]
        self.path_audio   = path_list[1]
        self.path_transA  = path_list[2]
        self.path_transB  = path_list[3]


    def load_data(self):
        self.load_pitches()
        self.TranscriptionA = load_transcriptions(self.path_transA)
        self.TranscriptionB = load_transcriptions(self.path_transB)
        self.load_audiofile()
        self.note_groups = TE.get_note_groups(self.TranscriptionA, self.TranscriptionB)
        self.get_minmax_overlap()

    def load_pitches(self):
        tm, freq = np.loadtxt(self.path_pitches, delimiter=',').T
        self.pitches = {}
        self.pitches['time'] = tm[freq > 0]
        self.pitches['freq'] = freq[freq > 0]


    def get_minmax_overlap(self):
        self.min_overlap = 1.0
        self.max_overlap = 0.0
        for k, v in self.note_groups.items():
            for paired_notes in v:
                for note_list in paired_notes:
                    for note in note_list:
                        self.min_overlap = min(self.min_overlap, note[1])
                        self.max_overlap = max(self.max_overlap, note[1])


    def initialise_parameters(self):

        ### Playback Parameters
        self.play_start  = StringVar(self.master, "0:00")
        self.play_end    = StringVar(self.master, "0:00")
        self.play_speed = DoubleVar(self.master, 1)
        self.is_playing  = False
        self.play_obj    = None
        self.audio_track = None
        self.audio_track_notesA = None
        self.audio_track_notesB = None

        self.play_audio = BooleanVar(self.master, True)
        self.play_notesA = BooleanVar(self.master, False)
        self.play_notesB = BooleanVar(self.master, False)
        self.play_on_loop = BooleanVar(self.master, False)

        ### Plotting Parameters
        self.is_bare = True
        self.color_choice = IntVar(self.master, 0)


    def playback_widget(self):
        Label(self.master, text="Playback options"+' '*40).grid(row=self.grid_rows, column=0, columnspan=3, sticky=tk.W)
        self.grid_rows += 1

        Button(self.master, text='Set time by window', command=self.set_window_time, relief=RAISED,
               bd=4, padx=10, pady=5, font=('Helvetica', 16)).grid(row=self.grid_rows, column=2, rowspan=3)

        Label(self.master, text="Start time").grid(row=self.grid_rows, column=0, sticky=tk.E)
        Entry(self.master, textvariable=self.play_start).grid(row=self.grid_rows, column=1)
        self.grid_rows += 1

        Label(self.master, text="End time").grid(row=self.grid_rows, column=0, sticky=tk.E)
        Entry(self.master, textvariable=self.play_end).grid(row=self.grid_rows, column=1)
        self.grid_rows += 1

        Button(self.master, text='Play', command=self.play_wav, relief=RAISED,
               bd=4, padx=10, pady=5, font=('Helvetica', 16)).grid(row=self.grid_rows, column=2, rowspan=3)
 
        Label(self.master, text="Speed").grid(row=self.grid_rows, column=0, sticky=tk.E)
        Entry(self.master, textvariable=self.play_speed).grid(row=self.grid_rows, column=1)
        self.grid_rows += 1


        Checkbutton(self.master, text='Loop', variable=self.play_on_loop).\
                    grid(row=self.grid_rows, column=0)
        self.grid_rows += 1

        Checkbutton(self.master, text='Audio', variable=self.play_audio).\
                    grid(row=self.grid_rows, column=0)
        self.grid_rows += 1

        Checkbutton(self.master, text='Notes A', variable=self.play_notesA).\
                    grid(row=self.grid_rows, column=0)
        Checkbutton(self.master, text='Notes B', variable=self.play_notesB).\
                    grid(row=self.grid_rows, column=1)
        self.grid_rows += 1

    def run_save_reset_widget(self):
        self.grid_rows += 3
        Label(self.master, text="Plotting options").grid(row=self.grid_rows, column=0, sticky=tk.W)
        self.grid_rows += 2
        Label(self.master, text="Color notes by...").grid(row=self.grid_rows, column=0, sticky=tk.W)
        self.grid_rows += 1

        Radiobutton(self.master, text='none', variable=self.color_choice, value=0).\
                    grid(row=self.grid_rows, column=0)
        Radiobutton(self.master, text='pitch', variable=self.color_choice, value=1).\
                    grid(row=self.grid_rows, column=1)
        self.grid_rows += 1

        Radiobutton(self.master, text='onset difference', variable=self.color_choice, value=2).\
                    grid(row=self.grid_rows, column=0)
        Radiobutton(self.master, text='note matching', variable=self.color_choice, value=3).\
                    grid(row=self.grid_rows, column=1)
        self.grid_rows += 1

        Radiobutton(self.master, text='note duration overlap', variable=self.color_choice, value=4).\
                    grid(row=self.grid_rows, column=0)
        self.grid_rows += 1

        Button(self.master, text='Update colours', command=self.replot_data, relief=RAISED,
               bd=4, padx=10, pady=5, font=('Helvetica', 16)).grid(row=self.grid_rows, column=2)
 
 
#       Button(self.master, text='Save', command=self.save_data, relief=RAISED,
#              bd=4, padx=10, pady=5, font=('Helvetica', 16)).grid(row=self.grid_rows, column=1)
 
        self.grid_rows += 2

        Button(self.master, text='Reset all', command=self.reset_parameters, relief=RAISED,
               bd=4, padx=10, pady=5, font=('Helvetica', 16)).grid(row=self.grid_rows, column=2)
 
 
        

    #------------------------------------------------------
    def quit(self):
        '''
        Quit the program
        '''
        sys.exit(0)

    #------------------------------------------------------
    def init_figure(self):
        self.fig = Figure()
        self.ax = [plt.subplot2grid((2,1), (0,0), fig=self.fig)]
        self.ax.append(plt.subplot2grid((2,1), (1,0), fig=self.fig, sharex=self.ax[0]))

        for i in range(len(self.ax)):
            self.ax[i].set_ylabel('Frequency (Hz)')

        self.ax[-1].set_xlabel('Time (sec)')

        self.graph = FigureCanvasTkAgg(self.fig, self.master)
        self.graph.draw()

        self.toolbar = NavigationToolbar2Tk(self.fig.canvas, self.master)
        self.toolbar.pack_forget()
        self.toolbar.update()


    def plot_data(self):
        for a in self.ax:
            a.plot(self.pitches['time'], self.pitches['freq'], 'ok', ms=1)

        self.plot_note_groups()


    def replot_data(self):
        self.reset_figure()
        self.plot_data()
            

    def plot_note_groups(self):
        data = [self.TranscriptionA, self.TranscriptionB]
        for (i, j) in self.note_groups['no_match']:
            on, off, freq = [data[i][x][j] for x in ['on', 'off', 'freq']]
            self.ax[i].plot([on], [freq], 'o', fillstyle='none', ms=6, mew=2, mec='k')
            self.ax[i].plot([on, off], [freq]*2, '-k')
        
        col = sns.color_palette()
        cmap = mpl.cm.hsv
        cmap = mpl.cm.YlOrRd
        ci = 0

        if self.color_choice.get() == 1:
            vmin = np.log2(self.pitches['freq'].min())
            vmax = np.log2(self.pitches['freq'].max())

        if self.color_choice.get() == 2:
            onset_diff = np.abs(np.diff(np.meshgrid(self.TranscriptionA['on'], self.TranscriptionB['on']), axis=0)[0])
            vmin = np.min(onset_diff)
            vmax = min([np.min(onset_diff, axis=0).max(), np.min(onset_diff, axis=1).max()])

        if self.color_choice.get() == 4:
            vmin = self.min_overlap
            vmax = self.max_overlap

        for k in ['one2one', 'many2one', 'many2many']:
            for notes in self.note_groups[k]:
                for i, p in zip([0,1], ['-', '-']):
                    for (j, ovl, on) in notes[i]:
                        on, off, freq = [data[i][x][j] for x in ['on', 'off', 'freq']]
                        if self.color_choice.get() == 0:
                            c = col[i]

                        elif self.color_choice.get() == 1:
                            c = cmap((np.log2(freq) - vmin) / (vmax - vmin))

                        elif self.color_choice.get() == 2:
                            val = np.min(onset_diff, axis=i)[j]
                            c = cmap((val - vmin) / (vmax - vmin))

                        elif self.color_choice.get() == 3:
                            c = col[ci%10]

                        elif self.color_choice.get() == 4:
                            c = cmap((ovl - vmin) / (vmax - vmin))

                        self.ax[i].plot([on], [freq], 'o', fillstyle='none', ms=8, mew=3, mec=c)
                        self.ax[i].plot([on, off], [freq]*2, p, c=c, lw=3)
                ci += 1

        if self.color_choice.get() in [1, 2, 4]:
            norm = Normalize(vmin, vmax)
            self.cbar = [self.fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=self.ax[i]) for i in [0, 1]]


    def reset_figure(self, j=-1):
        if j >= 0:
            self.ax[j].clear()
            self.ax[j].set_ylabel('Frequency (Hz)')
            try:
                self.cbar[j].clear()
            except:
                pass
        else:
            for i in range(len(self.ax)):
                self.ax[i].clear()
                self.ax[i].set_ylabel('Frequency (Hz)')
                try:
                    self.cbar[i].clear()
                except:
                    pass
        self.ax[-1].set_xlabel('Time (sec)')



    def load_audiofile(self):
        wav, fr = librosa.load(self.path_audio)
        self.wav = wav
        self.fr = fr
        self.play_end.set(self.get_recording_length())
        self.audio_track_notesA = self.prepare_note_track(*[self.TranscriptionA[x] for x in ['on', 'off', 'freq']])
        self.audio_track_notesB = self.prepare_note_track(*[self.TranscriptionB[x] for x in ['on', 'off', 'freq']])


    def get_recording_length(self):
        length = self.wav.size / self.fr
        mins = int(length / 60)
        sec  = int(length % 60) + 1
        return f"{mins}:{sec:02d}"


    def convert_string_to_seconds(self, time_string):
        mins, sec = [int(x) for x in time_string.split(':')]
        return (mins*60 + sec)


    def convert_seconds_to_string(self, time_sec):
        mins = int(time_sec // 60)
        sec = int(time_sec % 60)
        return f"{mins}:{sec}"


    def set_window_time(self):
        start, end = self.ax[0].get_xlim()
        self.play_start.set(self.convert_seconds_to_string(start))
        self.play_end.set(self.convert_seconds_to_string(end))

 
    def prepare_note_track(self, on, off, freq):
        audio = np.zeros(self.wav.size)
        xtime = np.arange(self.wav.size) / self.fr
        for i in range(len(on)):
            idx = (xtime >= on[i]) & (xtime <= off[i])
            audio[idx] = synth_tone(freq[i], np.sum(idx), self.fr)
        return audio


    def prepare_audio(self):
        track_created = False
        start = self.convert_string_to_seconds(self.play_start.get())
        end   = self.convert_string_to_seconds(self.play_end.get())
        speed = self.play_speed.get()
        if self.play_audio.get():
            if self.play_on_loop.get():
                self.audio_track = librosa.effects.time_stretch(clip_audio(self.fr, self.wav, start, end, repeat=5), speed)
            else:
                self.audio_track = librosa.effects.time_stretch(clip_audio(self.fr, self.wav, start, end), speed)
            self.audio_track = wav2int(self.audio_track)
            track_created = True

        if self.play_notesA.get():
            notesA = librosa.effects.time_stretch(clip_audio(self.fr, self.audio_track_notesA, start, end), speed)
            if track_created:
                self.audio_track = self.audio_track + notesA
            else:
                self.audio_track = notesA
                track_created = True

        if self.play_notesB.get():
            notesB = librosa.effects.time_stretch(clip_audio(self.fr, self.audio_track_notesA, start, end), speed)
            if track_created:
                self.audio_track = self.audio_track + notesB
            else:
                self.audio_track = notesB


    def play_wav(self):
        if not self.is_playing:
            self.is_playing = True
            self.prepare_audio()
            self.play_obj = sa.play_buffer(self.audio_track, 1, 2, self.fr)
        else:
            self.is_playing = False
            self.play_obj.stop()
            print("Stopped playing audio")


    def reset_parameters(self):

        ### Playback Parameters
        self.play_start.set("0:00")
        self.play_end.set(self.get_recording_length())
        self.audio_track = None





if __name__ == '__main__':

    args = parse_args()

    if not check_files_exist(args.songname, args.transcriber_A, args.transcriber_B):
        sys.exit()

    root = Tk()
    app = TransAnalyse(root, args.songname, args.transcriber_A, args.transcriber_B)
    app.mainloop()

