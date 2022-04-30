from collections import Counter, defaultdict
from pathlib import Path
import pickle

from Bio import pairwise2
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import simpleaudio


PATH_BASE = Path("/home/jmcbride/projects/BirdSongSpeech/Polina_transcriptions")


#------------------------------------------------------------#
#-- Input / Output


def load_transcriptions(f):
    on, freq, dur = np.loadtxt(f, delimiter=',').T
    off = on + dur 
    return {'on':on, 'off':off, 'freq':freq, 'dur':dur}

def load_pitches(f):
    try:
        return np.loadtxt(PATH_BASE.joinpath("Pitches", f"{f}.txt"))
    except:
        return []

def load_all_data():
    files = sorted(PATH_BASE.joinpath("Notes").glob("*csv"))
    stem = ['_'.join(f.stem.split('_')[1:-1]) for f in files]
    prsn = [f.stem.split('_')[-1].split('.')[0] for f in files]
    df = pd.DataFrame(data={'path':files, 'song':stem, 'transcriber':prsn})
    key = {s:i for i, s in enumerate(df.song.unique())}
    df['songID'] = df.song.apply(lambda x: key.get(x))
    df = df.sort_values(by='songID').reset_index(drop=True)
    data = [load_transcriptions(f) for f in df.path]
    pitches = {ID: load_pitches(f) for ID, f in zip(df.songID, df.song)}
    return df, data, pitches


#------------------------------------------------------------#
#-- Alignment


def find_note_match(on1, off1, on2, off2):
    overlap = [calc_note_overlap(on1, off1, on2[i], off2[i]) for i in range(len(on2))]
    i, o = np.argmax(overlap), np.max(overlap)
    if o > 0:
        return i, o
    else:
        return -1, 0


def calc_note_overlap(on1, off1, on2, off2):
    return (min(off1, off2) - max(on1, on2)) / (off1 - on1)


def match_notes(trans1, trans2, plot=False):
    on1, off1, freq1, dur1 = [trans1[x] for x in ['on', 'off', 'freq', 'dur']]
    on2, off2, freq2, dur2 = [trans2[x] for x in ['on', 'off', 'freq', 'dur']]

    idx1, over1 = np.array([find_note_match(on1[i], off1[i], on2, off2) for i in range(len(on1))]).T
    idx2, over2 = np.array([find_note_match(on2[i], off2[i], on1, off1) for i in range(len(on2))]).T

    if plot:
        fig, ax = plt.subplots(2,1,sharex=True,sharey=True)
        plot_matches(ax[0], on1, freq1, off1, on2, freq2, off2, idx1, idx2)
        plot_matches(ax[1], on2, freq2, off2, on1, freq1, off1, idx2, idx1)

    return idx1.astype(int), over1, idx2.astype(int), over2


def match_indices(i1, i2, idx1, idx2):
    old_i1, old_i2 = set(), set()
    while (i1 != old_i1) | (i2 != old_i2):
        old_i1, old_i2 = i1.copy(), i2.copy()
        i1 = [i for j in i2 for i in np.where(idx1==j)[0]]
        i2 = [j for i in i1 for j in np.where(idx2==i)[0]]
    return sorted(i1), sorted(i2)


def merge_transcriptions(idx1, idx2, olp1, olp2, on1, on2):
    used1, used2 = set(), set()
    note_groups = {x:[] for x in ['no_match', 'one2one', 'many2one', 'many2many']}
    no_match = {'idx1:':[], 'idx2':[]}
    one2one = {'idx1:':[], 'idx2':[]}
    many2one = {'idx1:':[], 'idx2':[]}
    many2many = {'idx1:':[], 'idx2':[]}
    for i, j in enumerate(idx1):
        if i in used1 or j in used2:
            continue
        if j == -1:
            note_groups['no_match'].append((0, i))
            used1.add(i)
        else:
            i1, i2 = match_indices(set([i]), set([j]), idx1, idx2)
            if (len(i1) == 1) & (len(i2) == 1):
                # Save as ( [(note 1 index, note 1 overlap)], [(note 2 index, note 2 overlap)] )
                note_groups['one2one'].append(([(ii, olp1[ii], on1[ii]) for ii in i1], [(jj, olp2[jj], on2[jj]) for jj in i2]))
            elif (len(i1) > 1) & (len(i2) > 1):
                note_groups['many2many'].append(([(ii, olp1[ii], on1[ii]) for ii in i1], [(jj, olp2[jj], on2[jj]) for jj in i2]))
            else:
                note_groups['many2one'].append(([(ii, olp1[ii], on1[ii]) for ii in i1], [(jj, olp2[jj], on2[jj]) for jj in i2]))
            used1 = used1.union(i1)
            used2 = used1.union(i2)

    for i, j in enumerate(idx2):
        if i in used2 or j in used1:
            continue
        if j == -1:
            note_groups['no_match'].append((1, i))
            used1.add(i)
        else:
            i1, i2 = match_indices(set([i]), set([j]), idx1, idx2)
            if (len(i1) == 1) & (len(i2) == 1):
                # Save as ( [(note 1 index, note 1 overlap)], [(note 2 index, note 2 overlap)] )
                note_groups['one2one'].append(([(ii, olp1[ii], on1[ii]) for ii in i1], [(jj, olp2[jj], on2[jj]) for jj in i2]))
            if (len(i1) > 1) & (len(i2) > 1):
                note_groups['many2many'].append(([(ii, olp1[ii], on1[ii]) for ii in i1], [(jj, olp2[jj], on2[jj]) for jj in i2]))
            else:
                note_groups['many2one'].append(([(ii, olp1[ii], on1[ii]) for ii in i1], [(jj, olp2[jj], on2[jj]) for jj in i2]))
            used1 = used1.union(i1)
            used2 = used1.union(i2)

    return note_groups


def get_note_groups(trans1, trans2):
    on1, off1, freq1, dur1 = [trans1[x] for x in ['on', 'off', 'freq', 'dur']]
    on2, off2, freq2, dur2 = [trans2[x] for x in ['on', 'off', 'freq', 'dur']]
    idx1, olp1, idx2, olp2 = match_notes(trans1, trans2)
    note_groups = merge_transcriptions(idx1, idx2, olp1, olp2, on1, on2)
    return note_groups


#------------------------------------------------------------#
#-- Plotting


def plot_points(ax, i, j, c, on1, off1, seq1, on2, off2, seq2):
    ax.plot(on1[i], seq1[i], 'o', mec='k', fillstyle='none', mew=2, ms=8)
    ax.plot([on1[i], off1[i]], [seq1[i]]*2, '-', lw=2, c=c)
    if j >= 0:
        ax.plot([on1[i], on2[j]], [seq1[i], seq2[j]], '--', lw=2, c=c)


def plot_matches(ax, on1, seq1, off1, on2, seq2, off2, idx1, idx2):
    seq2 = seq2 - 10
    for i in range(len(on1)):
        j = int(idx1[i])
        plot_points(ax, i, j, 'k', on1, off1, seq1, on2, off2, seq2)

    for i in range(len(on2)):
        j = int(idx2[i])
        j = -1
        plot_points(ax, i, j, 'r', on2, off2, seq2, on1, off1, seq1)


def plot_note(ax, on, off, freq):
    ax.plot([on], [freq], 'o', fillstyle='none', ms=6, mew=2, mec='k')
    ax.plot([on, off], [freq]*2, '-k')


def plot_note_groups(data, note_groups):
    fig, ax = plt.subplots()
    for (i, j) in note_groups['no_match']:
        on, off, freq = [data[i][x][j] for x in ['on', 'off', 'freq']]
        ax.plot([on], [freq], 'o', fillstyle='none', ms=6, mew=2, mec='k')
        ax.plot([on, off], [freq]*2, '-k')

    
    col = sns.color_palette()
    ci = 0
    offset = [0, 10]
    for k in ['one2one', 'many2one', 'many2many']:
        for notes in note_groups[k]:
            for i, p in zip([0,1], ['-', ':']):
                for (j,o) in notes[i]:
                    on, off, freq = [data[i][x][j] for x in ['on', 'off', 'freq']]
                    c = col[ci%10]
                    ax.plot([on], [freq - offset[i]], 'o', fillstyle='none', ms=6, mew=2, mec=c)
                    ax.plot([on, off], [freq - offset[i]]*2, p, c=c, lw=3)
            ci += 1


#------------------------------------------------------------#
#-- Overall Analyses


### Questions:
### What are the ioi, dur, and ioi_ratio distributions?
### What are the durations / ioi distributions for matched vs non-matched notes?
### Distribution of frequency differences in many2one matches;
###     are there clear differences between onset disagreements, and ornamentation disagreements?
def disagreement_stats():
    df, trans = load_all_data()
    dur_stats = defaultdict(list)
    ovl_stats = []
    onset_diff = []
    offset_diff = []
    mean_freq_diff = []
    max_freq_diff = []
    match_type = []
    for song_id in range(6):
#       i, j = df.loc[(df.songID==song_id)&(df.transcriber!='AUTO')].index
        i, j = df.loc[(df.songID==song_id)&(df.transcriber!='OV')].index
        on1, off1, freq1, dur1 = [trans[i][x] for x in ['on', 'off', 'freq', 'dur']]
        on2, off2, freq2, dur2 = [trans[j][x] for x in ['on', 'off', 'freq', 'dur']]
        idx1, olp1, idx2, olp2 = match_notes(trans[i], trans[j])
        note_groups = merge_transcriptions(idx1, idx2, olp1, olp2)

        for k, ng in note_groups.items():
            match_type.extend([k] * len(ng))
            if k == 'no_match':
                dur_stats[k].extend([[dur1, dur2][ii][jj] for (ii, jj) in ng])
            else:
                for notes in ng:
                    dur_stats[k].extend([dur1[ii] for (ii, ol) in notes[0]])
                    dur_stats[k].extend([dur2[ii] for (ii, ol) in notes[1]])

            # Get the overlap fractions for all "one2one" pairs of notes.
            # A more useful metric would be the onset and offset differences,
            # measured in seconds
            if k == 'one2one':
                for notes in ng:
                    ovl_stats.extend([ol for (ii, ol) in notes[0]])
                    ovl_stats.extend([ol for (ii, ol) in notes[1]])

                    onset_diff.append(abs(on1[notes[0][0][0]] - on2[notes[1][0][0]]))
                    offset_diff.append(abs(off1[notes[0][0][0]] - off2[notes[1][0][0]]))

            if k == 'many2one':
                for notes in ng:
                    if not len(notes[0]):
                        continue
                    meanfd = np.mean([abs(np.log2(freq1[ii] / freq2[jj])*1200) for (ii, ol) in notes[0] for (jj, ol) in notes[1]])
                    maxfd = np.max([abs(np.log2(freq1[ii] / freq2[jj])*1200) for (ii, ol) in notes[0] for (jj, ol) in notes[1]])
#                   f1 = [freq1[ii] for (ii, ol) in notes[0]]
#                   f2 = [freq2[ii] for (ii, ol) in notes[1]]
                    mean_freq_diff.append(meanfd)
                    max_freq_diff.append(maxfd)
    onset_diff = np.array(onset_diff)
    offset_diff = np.array(offset_diff)
    meanfd = np.array(mean_freq_diff)
    maxfd = np.array(max_freq_diff)

    fig, ax = plt.subplots(2,3)

    sns.countplot(match_type, ax=ax[0,0])
    for k in ['no_match', 'one2one', 'many2one']:
        sns.distplot(np.log10(dur_stats[k]), ax=ax[1,0], label=k)


    sns.distplot(meanfd[np.isfinite(meanfd)], bins = np.arange(0, 220, 10), ax=ax[0,1])
    sns.distplot(maxfd[np.isfinite(maxfd)], bins = np.arange(0, 720, 10), ax=ax[1,1])


    sns.distplot(ovl_stats, bins=np.arange(0, 1.05, 0.05), kde=False, norm_hist=True, ax=ax[0,2])
    sns.distplot(np.log10(onset_diff[onset_diff>0]), ax=ax[1,2], label='onset')
    sns.distplot(np.log10(offset_diff[offset_diff>0]), ax=ax[1,2], label='offset')

    xlbl = ['Note matches',
            'mean interval (cents) between notes in many-to-one group',
            'Overlap fraction (one-to-one)',
            r'$\log_{10}$ duration (log-seconds)',
            'max interval (cents) between notes in many-to-one group',
            r'$\log_{10}$ time difference (log-seconds; one-to-one)']
    for i, a in enumerate(ax.ravel()):
        a.set_xlabel(xlbl[i])
        a.set_ylabel('Density')
    ax[0,0].set_ylabel("Count")

    ax[1,0].legend(loc='best', frameon=False)
    ax[1,2].legend(loc='best', frameon=False)
    
#   return dur_stats, np.array(ovl_stats), np.array(mean_freq_diff)


def get_ioi_ratio_from_on(on):
    ioi = np.diff(on)
    return np.array([ioi[i]/ioi[i:i+2].sum() for i in range(ioi.size-1)])


def get_ints_from_freq(freq):
    return np.array([np.log2(freq[i+1] / freq[i]) * 1200 for i in range(freq.size-1)])


def transcription_stats():
    df, trans = load_all_data()
    idx_list = [df.loc[df.transcriber!='AUTO'].index,
                df.loc[df.transcriber=='PP'].index, 
                df.loc[df.transcriber=='OV'].index] 

    fig, ax = plt.subplots(2,2)
    ax = ax.T.reshape(ax.size)

    for idx, t_id in zip(idx_list, ['Both', 'PP', 'OV']):
        data = [trans[i] for i in idx]
        ioi = np.array([i for d in data for i in np.diff(d['on'])])
        ioi_ratio = np.array([i for d in data for i in get_ioi_ratio_from_on(d['on'])])
        freq = np.array([i for d in data for i in d['freq']])
        ints = np.array([i for d in data for i in get_ints_from_freq(d['freq'])])

        X = [np.log10(ioi[ioi>0]), ioi_ratio, freq[freq<1000], ints[np.abs(ints)<2400]]
        lbls = [r'$\log_{10}$IOI', 'IOI ratio', 'freq (Hz)', 'ints (cents)']
        for a, x, l in zip(ax, X, lbls):
            if l == 'ints (cents)':
                sns.distplot(x, ax=a, label=t_id, bins=np.arange(-1250, 1260, 20))
            else:
                sns.distplot(x, ax=a, label=t_id)
            a.set_xlabel(l)
            a.set_ylabel('Density')
            a.legend(loc='best', frameon=False)

        


if __name__ == "__main__":

#   usage_example1()
#   usage_example2()
#   usage_example3()

    transcription_stats()
    disagreement_stats()







