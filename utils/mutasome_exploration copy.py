import numpy as np
import pandas as pd
import myfunc as mf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--offset",
                    help = 'start of the mutasome (to align with sequence)',
                    default = 3_000_000, type = int, required = False)
parser.add_argument("--thresh",
                    help = "#Mutasome threshold to select peaks",
                    default = 10, type = float, required = False)
parser.add_argument("--gap",
                    help = "#Max peak distance to NPR",
                    default = 20, type = int, required = False)
parser.add_argument("--size",
                    help = "#half size of sequence output",
                    default = 10, type = int,required = False)
parser.add_argument("--minpeak",
                    help = "minimum number of peaks to form a NPR",
                    type = int, default = 2, required = False)

parser.add_argument("--output", 
                    help = "output directory path",
                    type = str, required = True)
parser.add_argument("--mutasome",
                    help = "mutasome path",
                    type = str,required = True)
parser.add_argument("--repeat",
                    help = "repeat masker formatted file path",
                    type = str,   required = True)

args = parser.parse_args()

OFFSET = args.offset #start of the mutasome (to align with sequence)
THRESH = args.thresh #Mutasome threshold to select peaks
STD_GAP = args.gap #Max peak distance to Nucleosome Positionning Region
SEQ_SIZE = args.size #half size of sequence output
MIN_PEAK = args.minpeak #minimum number of peaks to form a NPR

output_dir = args.output
file = args.mutasome
repeat_masker = args.repeat

cpt_t, cpt_r = 0, 0
for chr in range(1,20):
    print("start chr {}".format(chr))
    try:
        #Load and clean mutasome
        mut = np.load(file.format(chr))
        mut[np.abs(np.isfinite(mut) - 1)] = 0
        mut[np.isinf(mut)] = 0
        mut[np.isnan(mut)] = 0

        nanval = np.round(100 * np.sum(mut == 0) / len(mut), 2)
        print("{} % value are nan" .format(nanval))

        #Mean and zscore mutasome
        mut = np.mean(mut, axis = 1)
        mut = mf.zscore(mut)

        #Sequence as string of nucleotide
        seq = "".join(
            mf.BASES[
                np.argmax(
                    np.load(
                        "/home/maxime/data/sequences/mm10/one_hot/chr{}.npz".format(chr)
                    )["arr_0"][OFFSET:]
                ,axis = 1)
                    ]
                )

        rm = pd.read_csv("/home/maxime/data/sequences/mm10/repeat_masker", sep = "\t")
        rm = rm[rm["genoName"]=="chr{}".format(chr)]
    except FileNotFoundError:
        print("{} not found")
        continue
    
    #Get peaks cluster
    inter_pos = mf.consecutive(np.where(mut > THRESH)[0], STD_GAP)
    f_inter_pos = np.array([(x[-1]+x[0])//2 for x in inter_pos if len(x)>=MIN_PEAK])

    #Separate repeats 
    st = np.array(rm["genoStart"]).reshape((-1, 1)) - OFFSET
    end = np.array(rm["genoEnd"]).reshape((-1, 1)) - OFFSET
    masked_idx = set(np.concatenate([np.arange(x,y) for x,y in zip(st, end)]))

    rep_clusters = set(f_inter_pos) & masked_idx
    non_rep_clusters = set(f_inter_pos) - masked_idx
    assert len(rep_clusters) + len(non_rep_clusters) == len(f_inter_pos)
    del masked_idx, st, end



    with open(output_dir + "/repeats.fasta", "a") as f:
        for i, p in enumerate(rep_clusters):
            f.write(">chr{}_{}_{}\n".format(chr, p+OFFSET, np.round(mut[p], 2)))
            f.write(seq[p-SEQ_SIZE:p+SEQ_SIZE])
            f.write("\n")
            cpt_r += 1

    with open(output_dir + "/out_of_repeats.fasta", "a") as f:
        for i, p in enumerate(non_rep_clusters):
            f.write(">chr{}_{}_{}\n".format(chr, p+OFFSET, np.round(mut[p], 2)))
            f.write(seq[p-SEQ_SIZE:p+SEQ_SIZE])
            f.write("\n")
            cpt_t += 1

    print("chr{} - {} repeats - {} non repeats".format(chr, len(rep_clusters), len(non_rep_clusters)))
    print(" {} total repeats {} total non repeat {} total".format(cpt_r, cpt_t, cpt_r + cpt_t))