import numpy as np
import matplotlib.pyplot as plt
import seaborn
import pandas
import argparse
from sys import path

path.insert(1, "/home/maxime/data/utils")

import myfunc as mf

parser = argparse.ArgumentParser()
parser.add_argument("--file",     help = 'repeats fasta path', type = str, required = True)
parser.add_argument("--output", help="Output path for the csv", type = str, required= True )
args = parser.parse_args()
file =  args.file
chr, pos, score = [], [], []
with open(file, "r") as f:
    l = np.array(f.readlines()[::2])


l = np.array(np.char.split(l, "_"))
l = np.stack(l, axis = 0)

import pandas as pd

df = pd.DataFrame(l, columns=["chr", "pos", "score"])

df["chr"] = df["chr"].str.lstrip(">")
df["score"] = df["score"].str.rstrip("\n")
df = df.astype({'pos':'int64'})

rm = pd.read_csv("/home/maxime/data/sequences/mm10/repeat_masker", "\t")

def search(x):
    mask = np.logical_and(rm.genoName == x["chr"],np.logical_and(rm.genoStart<=x['pos'], rm.genoEnd>=x['pos']))
    return rm.loc[mask].repName.tolist()[0]

df["repName"] = df.apply(search, axis=1)
df.to_csv("/home/maxime/genomewide_streme_mA/repeats.csv", sep = "\t")