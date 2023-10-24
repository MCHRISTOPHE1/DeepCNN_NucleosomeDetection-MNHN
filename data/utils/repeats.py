import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from sys import path

path.insert(1, "/home/maxime/data/utils")
import myfunc as mf

parser = argparse.ArgumentParser()
parser.add_argument("--file",   help = 'repeats fasta path', type = str, required = True)
parser.add_argument("--output", help="Output path for the csv", type = str, required= True )
args = parser.parse_args()
file =  args.file
# chr, pos, score = [], [], []

with open(file, "r") as f:
    l = np.array(f.readlines()[::2])
l = np.array(np.char.split(l, "_"))
l = np.stack(l, axis = 0)
try:
    df = pd.DataFrame(l, columns=["chr", "pos", "score"])
except ValueError:
    df = pd.DataFrame(l, columns=["chr", "pos", "posplength", "score"])
    
df["chr"] = df["chr"].str.lstrip(">")
df["score"] = df["score"].str.rstrip("\n")
df = df.astype({'pos':'int64'})

rm = pd.read_csv("/home/maxime/data/sequences/mm10/repeat_masker", sep = "\t")
rm.genoName = [int(x) if x.isnumeric() else 0 for x in [x[0].lstrip("chr") for x in np.array(rm.genoName.str.split("_"))] ]
rm = rm[rm.genoName!=0]
rm = rm.astype({'genoStart':'int64', 'genoEnd':'int64'})

idx = []
d = []
for chr in range(1, 20):
    rmtmp = rm[rm.genoName==chr]
    dftmp = df[df.chr ==f"chr{chr}"]
    r = mf.create_ranges(np.array(rmtmp.genoStart), np.array(rmtmp.genoEnd))
    rlen = np.array(rmtmp.genoEnd-rmtmp.genoStart)
    
    rcol = []
    for k, reps in enumerate(rlen):
        rcol.extend([k]*reps)

    sparse_rep = sparse.csc_matrix((np.ones(len(rcol)), (r, rcol)),
                        shape=(np.max(rmtmp.genoEnd), len(rcol)), dtype=np.int8)
    sparse_df = sparse.csr_matrix((np.ones(len(dftmp)), (np.arange(len(dftmp)), dftmp.pos.values)),
                            shape= (len(dftmp), np.max(rmtmp.genoEnd)), dtype=np.int8)
    prod = sparse_df.dot(sparse_rep)
    
    idx_tmp = sparse.find(prod>0)
    idx.append([np.array(dftmp.iloc[idx_tmp[0]].index),
                np.array(rmtmp.iloc[idx_tmp[1]].index)])
    
    d.append(pd.concat([dftmp.iloc[idx_tmp[0]].reset_index(drop=True), rmtmp.iloc[idx_tmp[1]].reset_index(drop=True)], axis = 1))



merg = pd.concat(d, ignore_index=True)
merg.to_csv(args.output + "repeats.csv", sep = "\t")