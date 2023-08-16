import numpy as np
import pandas as pd
import myfunc as mf
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--offset",
                    help = 'start of the mutasome (to align with sequence)',
                    default = 0, type = int, required = False)

parser.add_argument("--thresh",
                    help = "#Mutasome threshold to select peaks",
                    default = 10, type = float, required = False)

parser.add_argument("--gap",
                    help = "#Max peak distance to NPR",
                    default = 20, type = int, required = False)

parser.add_argument("--size",
                    help = "#half size of sequence output",
                    default = 10, type = int,required = False)

parser.add_argument("--chr", 
                    help="chromosomes to use", required=True, type=int, nargs="+")

parser.add_argument("--minpeak",
                    help = "minimum number of peaks to form a NPR",
                    type = int, default = 2, required = False)

parser.add_argument("--output", 
                    help = "output directory path",
                    type = str, required = True)

parser.add_argument("--mutasome",
                    help = "mutasome path",
                    type = str,required = True, nargs="+")

parser.add_argument("--repeat",
                    help = "repeat masker formatted file path",
                    type = str,   required = True)
args = parser.parse_args()

OFFSET = args.offset #start of the mutasome (to align with sequence)
THRESH = args.thresh #Mutasome threshold to select peaks
STD_GAP = args.gap #Max peak distance to Nucleosome Positionning Region
SEQ_SIZE = args.size #half size of sequence output
MIN_PEAK = args.minpeak #minimum number of peaks to form a NPR
CHROM = args.chr
output_dir = args.output
file = args.mutasome
repeat_masker = args.repeat

cpt_t, cpt_r = 0, 0
os.mkdir(output_dir)
with open(output_dir + "/readme", "a") as f:
    print(f"Chromosomes: {CHROM}", file = f)
    print(f"Mutasome threshold: {THRESH}", file = f)
    print(f"Maximum gap: {STD_GAP}", file = f)
    print(f"Min number of peaks: {MIN_PEAK}", file = f)
    print(f"Sequence size: {SEQ_SIZE}", file = f)
    for chr in CHROM:
        print("start chr {}".format(chr))
        try:
            #Load and clean mutasome
            mut = np.mean([mf.loadnp(f.format(chr)) for f in file], axis = 0)
            mut[np.abs(np.isfinite(mut) - 1)] = 0
            mut[np.isinf(mut)] = 0
            mut[np.isnan(mut)] = 0

            nanval = np.round(100 * np.sum(mut == 0) / len(mut), 2)
            print("{} % value are nan" .format(nanval))

            #Mean and zscore mutasome
            mut = np.mean(mut, axis = 1)
            mut = mf.zscore(mut)
            print(f"mutasome quartiles : {np.quantile(mut, [0.25, 0.5, 0.75, 0.99])}")
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
        
        if SEQ_SIZE<0:
            f_inter_pos = np.array([x[0]
                                    for x in inter_pos if len(x)>=MIN_PEAK])
            lenpos = np.array([np.min((np.max((x[-1]-x[0], .20)), -SEQ_SIZE))
                               for x in inter_pos if len(x)>=MIN_PEAK])
        else:
            f_inter_pos = np.array([(x[-1]+x[0])//2 for x in inter_pos if len(x)>=MIN_PEAK])

        #Separate repeats 
        st = np.array(rm["genoStart"]).reshape((-1, 1)) - OFFSET
        end = np.array(rm["genoEnd"]).reshape((-1, 1)) - OFFSET
        masked_idx = set(list(mf.create_ranges(st.ravel(), end.ravel())))

        rep_clusters = set(f_inter_pos) & masked_idx
        non_rep_clusters = set(f_inter_pos) - masked_idx
        
        assert len(rep_clusters) + len(non_rep_clusters) == len(f_inter_pos)
        del masked_idx, st, end

        if SEQ_SIZE<0:
            rep_length = np.array(lenpos[np.argwhere(np.isin(f_inter_pos, list(rep_clusters)))], dtype=np.int).ravel()
            non_rep_length = np.array(lenpos[np.argwhere(np.isin(f_inter_pos, list(non_rep_clusters)))], dtype=np.int).ravel()
            #Corriger pour avoir les séquences entières.
            with open(output_dir + "/repeats.fasta", "a") as ap:

                for i, p in enumerate(rep_clusters):
                    ap.write(">chr{}_{}_{}_{}\n".format(chr, 
                                                        p+OFFSET,
                                                        p+OFFSET+rep_length[i],
                                                        np.round(mut[p], 2)))
                    ap.write(seq[p:p+rep_length[i]])
                    ap.write("\n")
                    cpt_r += 1

            with open(output_dir + "/out_of_repeats.fasta", "a") as ap:
                for i, p in enumerate(non_rep_clusters):
                    ap.write(">chr{}_{}_{}_{}\n".format(chr, 
                                                        p+OFFSET,
                                                        p+OFFSET+non_rep_length[i],
                                                        np.round(mut[p], 2)))
                    ap.write(seq[p:p+non_rep_length[i]])
                    ap.write("\n")
                    cpt_t += 1

        else:       
            with open(output_dir + "/repeats.fasta", "a") as ap:

                for i, p in enumerate(rep_clusters):
                    ap.write(">chr{}_{}_{}\n".format(chr, 
                                                     p+OFFSET, 
                                                     np.round(np.sum(mut[p-SEQ_SIZE:p+SEQ_SIZE]), 2)))
                    ap.write(seq[p-SEQ_SIZE:p+SEQ_SIZE])
                    ap.write("\n")
                    cpt_r += 1

            with open(output_dir + "/out_of_repeats.fasta", "a") as ap:
                for i, p in enumerate(non_rep_clusters):
                    ap.write(">chr{}_{}_{}\n".format(chr, 
                                                     p+OFFSET, 
                                                     np.round(np.sum(mut[p-SEQ_SIZE:p+SEQ_SIZE]), 2)))
                    ap.write(seq[p-SEQ_SIZE:p+SEQ_SIZE])
                    ap.write("\n")
                    cpt_t += 1

        print("chr{} - {} repeats - {} non repeats".format(chr, len(rep_clusters), len(non_rep_clusters)), file=f)
        print(" {} total repeats {} total non repeat {} total".format(cpt_r, cpt_t, cpt_r + cpt_t), file=f)

        print("chr{} - {} repeats - {} non repeats".format(chr, len(rep_clusters), len(non_rep_clusters)))
        print(" {} total repeats {} total non repeat {} total".format(cpt_r, cpt_t, cpt_r + cpt_t))