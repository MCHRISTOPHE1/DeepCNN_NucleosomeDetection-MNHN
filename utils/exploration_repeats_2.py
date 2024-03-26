from sys import path

path.insert(1, "/home/maxime/data/utils")
import multiprocessing
from enum import IntEnum

import Levenshtein
import logomaker as lm
import matplotlib as mpl
import matplotlib.pyplot as plt
import met_brewer
import myfunc as mf
import numpy as np
import pandas as pd
import pyBigWig as pbw
import seaborn as sns
from matplotlib.colors import LogNorm
from preprocess import default
from scipy.stats import linregress
from sklearn.decomposition import PCA

bb = pbw.open("/home/maxime/data/sequences/mm10/JASPAR2024.bb")

def parse_bb(bbEntry, thresh=400):
    pos = list(bbEntry[:2])
    string = bbEntry[-1].split("\t")
    score = int(string[1])
    if score<=thresh:
        return False
    else:
        return pos + [score] + [string[-1]]

def smith_waterman(seq1, seq2):
    # Assigning the constants for the scores
    class Score(IntEnum):
        MATCH = 1
        MISMATCH = -1
        GAP = -10

    # Assigning the constant values for the traceback
    class Trace(IntEnum):
        STOP = 0
        LEFT = 1 
        UP = 2
        DIAGONAL = 3

    # Generating the empty matrices for storing scores and tracing
    row = len(seq1) + 1
    col = len(seq2) + 1
    matrix = np.zeros(shape=(row, col), dtype=np.int32)  
    tracing_matrix = np.zeros(shape=(row, col), dtype=np.int32)  
    
    # Initialising the variables to find the highest scoring cell
    max_score = -1
    max_index = (-1, -1)
    
    # Calculating the scores for all cells in the matrix
    for i in range(1, row):
        for j in range(1, col):
            # Calculating the diagonal score (match score)
            match_value = Score.MATCH if seq1[i - 1] == seq2[j - 1] else Score.MISMATCH
            diagonal_score = matrix[i - 1, j - 1] + match_value
            
            # Calculating the vertical gap score
            vertical_score = matrix[i - 1, j] + Score.GAP
            
            # Calculating the horizontal gap score
            horizontal_score = matrix[i, j - 1] + Score.GAP
            
            # Taking the highest score 
            matrix[i, j] = max(0, diagonal_score, vertical_score, horizontal_score)
            
            # Tracking where the cell's value is coming from    
            if matrix[i, j] == 0: 
                tracing_matrix[i, j] = Trace.STOP
                
            elif matrix[i, j] == horizontal_score: 
                tracing_matrix[i, j] = Trace.LEFT
                
            elif matrix[i, j] == vertical_score: 
                tracing_matrix[i, j] = Trace.UP
                
            elif matrix[i, j] == diagonal_score: 
                tracing_matrix[i, j] = Trace.DIAGONAL 
                
            # Tracking the cell with the maximum score
            if matrix[i, j] >= max_score:
                max_index = (i,j)
                max_score = matrix[i, j]
    
    # Initialising the variables for tracing
    aligned_seq1 = ""
    aligned_seq2 = ""   
    current_aligned_seq1 = ""   
    current_aligned_seq2 = ""  
    (max_i, max_j) = max_index
    aligned_seq1="-"*(len(seq1)-(max_i))
    aligned_seq2="-"*(len(seq2)-(max_j))
    # Tracing and computing the pathway with the local alignment
    while tracing_matrix[max_i, max_j] != Trace.STOP:
        if tracing_matrix[max_i, max_j] == Trace.DIAGONAL:
            current_aligned_seq1 = seq1[max_i - 1]
            current_aligned_seq2 = seq2[max_j - 1]
            max_i = max_i - 1
            max_j = max_j - 1
            
        elif tracing_matrix[max_i, max_j] == Trace.UP:
            current_aligned_seq1 = seq1[max_i - 1]
            current_aligned_seq2 = '-'
            max_i = max_i - 1    
            
        elif tracing_matrix[max_i, max_j] == Trace.LEFT:
            current_aligned_seq1 = '-'
            current_aligned_seq2 = seq2[max_j - 1]
            max_j = max_j - 1
            
        aligned_seq1 = aligned_seq1 + current_aligned_seq1
        aligned_seq2 = aligned_seq2 + current_aligned_seq2
    
    # Reversing the order of the sequences
    aligned_seq1 = aligned_seq1[::-1]
    aligned_seq2 = aligned_seq2[::-1]
    
    return aligned_seq1, aligned_seq2, max_score

def localign(ref, sequence):
    """
    sequences must be 'ATGC' one-hot encoded
    """
    #offsets = [0]
    #ref = "".join(mf.BASES[np.argmax(ref, axis =1)])

    aseq1, aseq2, score = smith_waterman(ref, sequence)
    off1 = len(ref)-len(aseq1)-1
    off2 = len(sequence)-len(aseq2)-1

    return [off2-off1, score]

def H(x):
    a = np.log2(x)
    a[np.isnan(a)]=0
    a[np.isinf(a)]=0
    return -np.sum(x*a, axis = 1)

def meta_idx(motif, chr, size, rm):
    rttmp = rm[(rm.repName==motif) & (rm.genoName==chr)].copy()
    try:
        assert len(rttmp)>0
    except AssertionError:
        return None
    strand = rttmp.strand.values=="-"
    offset = rttmp[["repStart", "repLeft"]].values
    offset = offset[np.arange(len(offset)), np.array(strand, dtype=int)]
    offset[strand]*=-1
    pos = rttmp[["genoStart","genoEnd"]].values
    pos = pos[np.arange(pos.shape[0]), np.array(strand, dtype=int)]
    pos -= offset
    x = mf.create_ranges(pos-size, pos+size).reshape((-1, 2*size))
    x[strand,:] = x[strand,::-1]
    return x, strand, np.abs(offset), (rttmp.genoEnd-rttmp.genoStart).values

def period(cor):
    lag, cor = mf.autocorr(cor)
    st=int(np.where(lag==0)[0])
    lag = lag[st:]
    cor=cor[st:]
    return lag[np.argmin(cor)+np.argmax(cor[np.argmin(cor):])]

def module(a, b):
    return np.sqrt(a**2+b**2)

#Parameters
size = 5000
plotsize = 3000
size_for_alignement = 75
interest_size = 8 #half of muatzone of interest
s = 30
n_components =  50 #PCA components

bw = pbw.open("/home/maxime/data/mnase/labels/sparse_A_16.bw")
bb = pbw.open("/home/maxime/data/sequences/mm10/JASPAR2024.bb")
FILEPATH = "/home/maxime/data/mnase/train/full_genome_rmdense/full_genome_rmdense_1"

#Repeat masker
rm = pd.read_csv("/home/maxime/data/sequences/mm10/repeat_masker", sep = "\t")
rp =  pd.read_csv("/home/maxime/data/mnase/train/full_genome_rmdense/full_genome_rmdense_1/exploration_lax_gap20_thresh5/extracted_repeatsrepeats.csv", sep ="\t", index_col=0)
per = rp.repName.value_counts()/rm["repName"].value_counts()
cou =rp.repName.value_counts()
rphist = per[np.isnan(per)==False]
rphist = rphist.to_frame().reset_index()
rpcou = cou[np.isnan(cou)==False]
rpcou = rpcou.to_frame().reset_index()
rphist.columns = ["repName", "perc"]
rpcou.columns = ["repName", "cou"]
rphist = rphist.merge(rpcou)
rmnames = rm[["repClass", "repFamily", "repName"]].drop_duplicates()
rmnames = rmnames.merge(rphist, on="repName", how="right")
selectrp = rmnames[(rmnames.perc>=0.01) & (rmnames.cou>=100)].sort_values("cou",ascending=False).drop_duplicates("repName", keep="last")
del rp, per, cou, rphist, rpcou, rmnames



#For all interesting tfbs
for classmot, familymot, mot  in selectrp[["repClass", "repFamily", "repName"]].values:
    try:
        #Create figures frame
        mosaic = [["NPR_autocorr",  "NPR_nuc",      "tmpmut",   "nNPR_nuc",     "nNPR_autocorr"],
            ["EMPTY",         "NPR_tmplab",        "tmpmut",   "nNPR_tmplab",  "EMPTY"],
            ["NPR_seq",         "NPR_seq",   "tmpmut","nNPR_seq",        "nNPR_seq"   ],
            # ["EMPTY", "Enrichement", "Enrichement", "Enrichement", "EMPTY"],
            ["EMPTY", "Mutasome_order", "Alignement", "levenshtein", "EMPTY"],
            ["individuals", "individuals", "factors", "factors", "correlations"]]
        
        fig  = plt.figure(layout="constrained", figsize=(30, 30))
        axs = fig.subplot_mosaic(
        mosaic, empty_sentinel="EMPTY", gridspec_kw={"width_ratios":[1,2,2,2,1],
                                                    "height_ratios":[2,3,2,6,6]})

        max_length = -1
        tmpmut, tmpseq, tmplab, tmpmask, tmppred = [], [], [], [], []
        dfdata = []

        print(classmot, familymot, mot)
        allpos = {}
        #For all chromosomes
        for i in range(1, 20):
            print(">", end = "")
            try:
                pos, strand, rep_offset, rep_length = meta_idx(mot, f"chr{i}", size, rm)
                tmp_length = np.min((np.max(rep_length), size))
                max_length = np.max((max_length, tmp_length))
                allpos[i] = pos[:,size]
            except TypeError:
                continue
            
            chromosome = f"chr{i}"
            try:
                mut = mf.zscore(np.mean(mf.loadnp(f"{FILEPATH}/{i}_mutasome.npz"), axis = 1))
                lab = default(bw.values(chromosome, 0, bw.chroms()[chromosome]))
                pred = mf.loadnp(f"{FILEPATH}/{i}_mutasomePREDICTION.npz")
                seq = mf.loadnp(f"/home/maxime/data/sequences/mm10/one_hot/{chromosome}.npz")
            except FileNotFoundError:
                print("mutasome or sequence file is incorrect")
                continue

            tmpmut.append(mut[pos])
            tmplab.append(lab[pos])
            tmppred.append(pred[pos])
            
            tseq = seq[pos,:]
            tseq[strand,...] = tseq[strand,:,::-1]
            tmpseq.append(tseq)

            tmp = np.arange(size, 2*size*len(rep_length)+1, 2*size) + np.clip(rep_offset,0,size-1)
            a = np.zeros(size*2*len(rep_length))
            maskidx = np.clip(mf.create_ranges(tmp, tmp+np.clip(rep_length,0, size-1)),0,a.size-1)
            a[maskidx] = 1
            tmpmask.append(a)

            print(f"chr{i} ", end="")
        print()
        tmpmask = np.vstack([x.reshape((-1, 2*size)) for x in tmpmask])
        tmpmut = np.vstack(tmpmut)
        tmplab = np.vstack(tmplab)
        tmppred = np.vstack(tmppred)
        tmpseq = np.vstack(tmpseq).reshape(-1, 2*size, 4)

        print("find peak")
        selidx = np.argmax(
            np.convolve(
                np.mean(tmpmut[:,size:2*size-interest_size], axis = 0),
                mf.gauss(interest_size//4),
                mode = "same")
            ) + size

        selmask = np.sum(tmpmask[:,selidx-interest_size:selidx+interest_size], axis = 1)==2*interest_size

        tmpmut = tmpmut[selmask, ...]
        tmppred = tmppred[selmask, ...]
        tmpseq = tmpseq[selmask, ...]
        tmplab = tmplab[selmask, ...]
        tmpmask = tmpmask[selmask, ...]


        interest = np.sum(tmpmut[:,selidx-interest_size:selidx+interest_size], axis =1)

        # Number of sequences in the cluster
        #TODO limit to half the number of sequences (avoid crossing low and high cluster)
        nclust = np.max(
            (
                np.sum(interest>=100),
                np.sum(interest>(np.max(interest)*0.25))
                )
            )


        print(f"N = {nclust}")
        nidx = np.argsort(interest)
        plotrange=slice(plotsize//2, (3*plotsize//2)+1)

        
        #  Local alignment
        ref_sequence = "".join(mf.BASES[np.argmax(tmpseq[nidx[-1], selidx-interest_size:selidx+interest_size,:], axis=1)])
        print(f"Align seq :{ref_sequence}... ", end="")
        # Multiprocess
        with multiprocessing.Pool(processes=24) as pool:
            alignment = pool.starmap(localign, 
                                [(ref_sequence,
                                "".join(mf.BASES[np.argmax(seq, axis=1)])) 
                                for seq in tmpseq[nidx[:][::-1], selidx-size_for_alignement:selidx+size_for_alignement,:]])


        alignment = np.array(alignment).reshape((-1, 2))
        offsets = alignment[:, 0] - (size_for_alignement-interest_size*2) 
        scores = alignment[:, 1]
        del alignment
        print("Aligned!")
        
        
        offidx = mf.create_ranges(np.repeat(size-plotsize, len(nidx)),
                                np.repeat(size+plotsize, len(nidx))).reshape((len(nidx), -1)) +offsets.reshape((-1, 1))


        tmplab = tmplab[nidx[::-1].reshape((-1, 1)), offidx][::-1,...]
        tmppred = tmppred[nidx[::-1].reshape((-1, 1)), offidx][::-1,...]
        tmpseq = tmpseq[nidx[::-1].reshape((-1, 1)), offidx, :][::-1,...]

        selidx = selidx-size+plotsize
        rtmpmut = tmpmut[nidx[::-1].reshape((-1, 1)), offidx][::-1, ...]
        
        selidx  = np.argmax(np.mean(rtmpmut, axis=0)) #This is where you look, once align : Where the mutasome is at its most
        mutsum = np.sum(rtmpmut[::-1,selidx-interest_size:selidx+interest_size], axis=1)
        mutorder = np.argsort(mutsum)


        cmap = mpl.colors.ListedColormap(colors = ["green", "blue", "yellow", "red"])
        hmap = sns.heatmap(np.argmax(tmpseq[:,selidx-s:selidx+s,:][mutorder, ...], axis = 2),
                            cmap=cmap, ax=axs["Alignement"],
                            xticklabels=np.arange(selidx-s,selidx+s,)-plotsize)
        colorbar = hmap.collections[0].colorbar
        colorbar.set_ticks(np.linspace(0.4, 2.6, 4))
        colorbar.set_ticklabels(['A', 'C', 'G', 'T'])

        axs["Mutasome_order"].plot(
            np.clip(mutsum[mutorder],1e-5,None),
            range(len(rtmpmut)-1, -1, -1)
                                    )

        # axs["Mutasome_order"].twiny()
        # axs["Mutasome_order"].plot(scores,range(len(tmpmut)-1, -1, -1))

        axs["Mutasome_order"].set_title("Mutasome sum", fontsize=20)
        axs["Mutasome_order"].set_xscale("log")
        axs["Alignement"].set_title("Aligned motif", fontsize=20)


        

        # GRAPHICAL
        print("Draw signals...", end = " ")

        hmaptmp = rtmpmut.copy()
        hmaptmp[tmpmask[nidx[::-1].reshape((-1, 1)), offidx]==0] = -2
        cmap = sns.color_palette("YlOrRd", as_cmap=True).copy()
        cmap.set_under("gray")
        sns.heatmap(hmaptmp[::-1,np.min(np.where(hmaptmp!=-2)[1]) : np.max(np.where(hmaptmp!=-2)[1])],
                    vmax=5,
                    vmin=-1,
                    cmap=cmap,
                    ax=axs["tmpmut"],
                    cbar=False)


        # NPR
        axs["NPR_nuc"].plot(
                            np.mean(tmplab[-nclust:,plotrange], axis = 0),
                            label = "Experimental",
                            alpha = 0.6,
                            color = "b")
        
        axs["NPR_nuc"].plot(
                            np.mean(tmppred[-nclust:,plotrange], axis = 0),
                            label = "Prediction",
                            alpha = 0.6,
                            color = "r")
        axs["NPR_nuc"].legend(fontsize = 15)
        
        sns.heatmap(tmplab[-nclust:,plotrange][::-1],
                    ax=axs["NPR_tmplab"],
                    cbar=False,
                    cmap="YlOrRd")
        lag, cor = mf.autocorr(np.mean(tmplab[-nclust:,plotrange], axis = 0))
        axs["NPR_autocorr"].plot(lag[plotsize:], cor[plotsize:])
        axs["NPR_autocorr"].annotate(f"Base period: {period(cor)}b", (20, 0.5), fontsize=15)



        a = np.mean(tmpseq[-nclust:,selidx-s:selidx+s,:], axis = 0)
        if len(a)>0:
            lm.Logo(pd.DataFrame(a*(2-H(a).reshape((-1, 1))), columns=mf.BASES), ax=axs["NPR_seq"])



        interange = slice(np.min(np.argwhere(H(a)<1)+selidx-s),
                        np.max(np.argwhere(H(a)<1))+1+selidx-s)
        levseqs = mf.BASES[np.argmax(tmpseq[:,interange,:][mutorder, ...], axis = 2)]
        mutsum = np.sum(rtmpmut[::-1,interange], axis=1)[mutorder[::-1]]
        tmp = [0]
        for levs in levseqs[1:]:
            tmp.append(
            Levenshtein.distance("".join(levseqs[0]), "".join(levs)))
        
        sns.kdeplot(data=pd.DataFrame({"mutsum":mutsum,
                                        "Levenshtein": tmp}), x="mutsum", y="Levenshtein", fill=True, ax=axs["levenshtein"])
        #axs["levenshtein"].hist2d(mutsum[mutorder[::-1]], tmp,bins=(100,int(np.max(tmp))), cmap="YlOrRd", norm=LogNorm())
        xfit = np.arange(0,np.max(mutsum))
        slope, intercept, r_value, p_value, std_err = linregress(mutsum, tmp)
        axs["levenshtein"].plot(xfit*slope+intercept, c="r", linestyle="--")
        axs["levenshtein"].set_xscale("log")

        #nNPR
        axs["nNPR_nuc"].plot(np.mean(tmplab[:nclust,plotrange], axis = 0),
                            label = "Experimental",
                            alpha = 0.6,
                            color = "b")
        
        axs["nNPR_nuc"].plot(np.mean(tmppred[:nclust,plotrange], axis = 0),
                            label = "Prediction",
                            alpha = 0.6,
                            color = "r")
            
        
        lag, cor = mf.autocorr(np.mean(tmplab[:nclust,plotrange], axis = 0))
        sns.heatmap(tmplab[:nclust,plotrange], ax=axs["nNPR_tmplab"], cbar=False, cmap="YlOrRd")
        axs["nNPR_autocorr"].plot(lag[plotsize:], cor[plotsize:])
        axs["nNPR_autocorr"].annotate(f"Base period: {period(cor)}b", (20, 0.5), fontsize=15)

        b = np.mean(tmpseq[:nclust,selidx-s:selidx+s,:], axis = 0)
        if len(b)>0:
            lm.Logo(pd.DataFrame(b*(2-H(b).reshape((-1, 1))), columns=mf.BASES), ax=axs["nNPR_seq"])

        # if len(a)>0 and len(b)>0:
        #     lm.Logo(pd.DataFrame(a-b, columns=mf.BASES), ax=axs["Enrichement"])

        for tmpax in ("nNPR_nuc","NPR_nuc", "NPR_tmplab", "nNPR_tmplab"):
            axs[tmpax].set_xticks(range(0, plotsize, plotsize//10),
                            np.arange(-plotsize//2, plotsize//2, plotsize//10)+selidx-plotsize)
            #axs[tmpax].axvline((selidx), c="r")

        # for tmpax in ("NPR_seq", "nNPR_seq", "Enrichement"):
        #     axs[tmpax].set_xticks(range(0, 2*s, 10),
        #                     np.arange(-s, s, 10)+selidx-plotsize)


        axs["NPR_tmplab"].sharex(axs["NPR_nuc"])
        axs["nNPR_tmplab"].sharex(axs["nNPR_nuc"])

        axs["nNPR_nuc"].sharey(axs["NPR_nuc"])
        axs["nNPR_seq"].sharey(axs["NPR_seq"])
        axs["tmpmut"].axhline(nclust, c="black")
        axs["tmpmut"].annotate("NPR flagged", (0, nclust), (15,nclust-1), fontsize=15)
        axs["tmpmut"].axhline(len(tmpmut)-nclust, c="black")
        axs["tmpmut"].annotate("NPR lost", (0, len(tmpmut)-nclust), (15,len(tmpmut)-nclust+3), fontsize=15)

        axs["Mutasome_order"].sharey(axs["Alignement"])

        #Titles
        fig.suptitle(mot + f"{classmot} {familymot} {mot} N={len(tmpmut)}", fontsize=40)
        axs["nNPR_autocorr"].set_title("metaplot\nautocorrelation", fontsize=25)
        axs["nNPR_nuc"].set_title("NPR lost nucleosomes position", fontsize=25)

        axs["NPR_autocorr"].set_title("metaplot\nautocorrelation", fontsize=25)
        axs["NPR_nuc"].set_title("NPR flagged nucleosomes position", fontsize=25)


        axs["tmpmut"].set_title("Mutasome", fontsize=25)
        # axs["Enrichement"].set_title("Nucleotide enrichment in NPRs")

        axs["levenshtein"].set_title(f"a = {slope:.2f}, R = {r_value:.2f} , p = {p_value:.4e}", fontsize=20)
        axs["levenshtein"].set_xlabel("Mutascore")
        axs["levenshtein"].set_ylabel("Levenshtein")

        # PCA CALCULATION
        # "individuals", "factors", "correlations"
        print("Perform PCA...", end = " ")
        for metachrom, metapos in allpos.items():
            mut = mf.zscore(
                    np.mean(
                            mf.loadnp(f"/home/maxime/data/mnase/train/full_genome_rmdense/full_genome_rmdense_1/{metachrom}_mutasome.npz"),
                            axis = 1))
            
            for i, p in enumerate(metapos):
                p = p + selidx-plotsize
                dat = bb.entries(f"chr{metachrom}", p-25,p+25)
                df = pd.DataFrame(dat)
                try:
                    df[["ID", "jascore", "strand", "name"]] = df[2].str.split("\t", expand=True)
                    df["rep"] = chromosome + ":"+ str(p)
                    df["mutscore"] = np.mean(mut[p-25:p+25])
                    dfdata.append(df)
                except (ValueError, KeyError) as e:
                    continue
        
        dfdata = pd.concat(dfdata)
        dfdata["jascore"] = dfdata["jascore"].astype(float)
        
        #filter prediction quality
        dfdata = dfdata[dfdata.jascore>=400]
        dfdata = dfdata.sort_values('rep')


        # cross = pd.crosstab( dfdata["rep"], dfdata["name"])
        vcross = pd.crosstab( dfdata["rep"], dfdata["name"], values=dfdata["jascore"], aggfunc="max")
        vcross = vcross.fillna(0)
        # vcross[vcross<400]=0
        vcross = vcross.sort_values("rep")
        navcross = vcross
        navcross[vcross==0] = pd.NA
        norm_vcross=(navcross-navcross.min(skipna=True))/(navcross.max(skipna=True)-navcross.min(skipna=True))
        norm_vcross[pd.isna(norm_vcross)] = 0

        # PCA AND FIGURE
        # "individuals", "factors", "correlations"
        pca = PCA(n_components=n_components)

        pcadata = pca.fit_transform(norm_vcross)
        #     n_components = len(pca.explained_variance_)


        mutval = dfdata.drop_duplicates("rep").sort_values("rep").mutscore.values.reshape((-1, 1))
        stck = np.hstack([mutval, pcadata])
        ticklabs = ["Mutasome"] + [f"PC{i+1}" for i in range(n_components)]

        pcmut_corr = np.corrcoef(stck.T)
        sns.heatmap(pcmut_corr[0,1:].reshape((-1, 1)),
                    cmap = "seismic",
                    vmin = -1,
                    vmax = 1,
                    annot = True,
                    ax = axs["correlations"],
                    fmt =".2f",
                    xticklabels=[ticklabs[0]],
                    yticklabels=ticklabs[1:])

        pc1, pc2  = np.argsort(np.abs(pcmut_corr[0]))[-3:-1]
        variance_percentage = 100 * pca.explained_variance_ratio_

        indiv_scat = axs["individuals"].scatter(
                                                pcadata[:,pc1-1],
                                                pcadata[:, pc2-1],
                                                alpha=0.5, 
                                                c = dfdata.drop_duplicates("rep").sort_values("rep").mutscore.values,
                                                cmap = "YlOrRd",
                                                vmin = 0,
                                                vmax = 5 )


        modules = [module(pca.components_[pc1-1, i],  
                        pca.components_[pc2-1, i] )
                for i in range(0, pca.components_.shape[1])]


        idx = np.argsort(modules)[-5:] #Number of TFBS
        colors = met_brewer.met_brew(name="Austria", n=len(idx), brew_type='continuous')
        arrows, legend = [], []
        for n, i in enumerate(idx):

                axs["factors"].arrow(0,
                        0,  # Start the arrow at the origin
                        pca.components_[pc1-1, i],  
                        pca.components_[pc2-1, i], 
                        head_width=0.01,
                        head_length=0.01)

                if vcross.columns.values[i][:2] == "PC":
                        c = "r"
                
                else:
                        c = "k"
                
                
                arrows.append(
                        axs["factors"].arrow(0,
                                0,  # Start the arrow at the origin
                                pca.components_[pc1-1, i],  
                                pca.components_[pc2-1, i],  
                                head_width=0.03,
                                head_length=0.03, color = colors[n])
                )
                
                legend.append(vcross.columns.values[i])
                # ax.text(pca.components_[pc1-1, i] + 0.01,
                #         pca.components_[pc2-1, i] + 0.01,
                #         vcross.columns.values[i], color = colors[n])

        an = np.linspace(0, 2 * np.pi, 100)

        axs["correlations"].set_title("Correlations with Mutasome", fontsize = 20)
        axs["individuals"].set_title("Repeats projection", fontsize = 20)

        # plt.xlim(0, 1)
        # plt.ylim(-0.75, 0.75)

        #Circle
        axs["factors"].plot(np.cos(an), np.sin(an), alpha = 0.8)  # Add a unit circle for scale
        axs["factors"].axis('equal')
        axs["factors"].set_title(f'Variable factor map  ({np.sum(variance_percentage):.2f}%)', fontsize = 20)
        axs["factors"].arrow(0,0,pcmut_corr[0][pc1],
                pcmut_corr[0][pc2],
                color = "r",
                head_width=0.01,
                head_length=0.01, alpha = 0.7)
        axs["factors"].text(pcmut_corr[0][pc1]+0.01, pcmut_corr[0][pc2]+0.01, "Mutasome", color = "r")
        axs["factors"].axvline(0, linestyle = "--", alpha = 0.8)
        axs["factors"].axhline(0, linestyle = "--", alpha = 0.8)
        axs["factors"].set_xlabel(f"PC{pc1} ({variance_percentage[pc1-1]:.3f}%)", fontsize = 15)
        axs["factors"].set_ylabel(f"PC{pc2} ({variance_percentage[pc2-1]:.3f}%)", fontsize = 15)
        legend = axs["factors"].legend(arrows, legend, loc = "lower left", ncol = 1, fontsize=14, framealpha = 1)
        legend.get_frame().set_edgecolor('k')
        axs["factors"].spines[["bottom", "top", "left", "right"]].set_visible(False)
        axs["factors"].set_xticks([])
        axs["factors"].set_yticks([])
        # axs["factors"].set_ylim(-0.95)



        plt.savefig(f"/home/maxime/repeats_figures/{classmot}_{familymot}_{mot}_PCA.png")
        print("SAVED!")
        
    except ValueError:
        print(f"{classmot}_{familymot}_{mot}... ERROR")
        continue
