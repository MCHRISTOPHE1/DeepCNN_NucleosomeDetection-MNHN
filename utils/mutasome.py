# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 10:46:38 2022

@author: maxime christophe
"""


import argparse

import numpy as np
import time
import tensorflow as tf
import pandas as pd
from os import path

try:
    import myfunc as mf
    from myfunc import sliding_window_view
    from losses import mae_cor, correlate
except ModuleNotFoundError:
    import sys
    sys.path.insert(1, "/home/maxime/data/utils")
    import myfunc as mf
    from myfunc import sliding_window_view
    from losses import mae_cor, correlate






"""parser = argparse.ArgumentParser()
parser.add_argument("model", help="Keras model to predict data", type=tf.keras.Model )
parser.add_argument("sequence")
"""

def mutasome(model, sequence, labels, positions, step, winsize = 2001,offset=1000, multi:bool=False, output_file: str = "mutasome", save: bool = True, result: bool = True, batch:int = 4096, reverse = False):
    # Faire varier le nombre de steps
    """
    Compute mutasome around position (corresponding to size b around the position)
    -> to compute full sequence position
    positions = np.arange(size,len(sequence),2*size)

    :param model: keras model loaded
    :param sequence: one hot encoded sequence
    :param labels: 1000bp offset sequence labels (pos 1000 in sequence match with pos 0 in label)
    :param positions: array of positions
    :param int step: size to compute around the pos
    :param str output_file: path to output the file
    :param bool save: Save result to file if True
    :param bool result: return result if True
    :param int offset: locus of the first prediction
    """

    assert len(step)>=1, "step est une liste de positions contenant les tetes du mutasome (entre 0 et 2000)"
    nb_heads = len(step)

    model = model
    seq = sequence
    if multi:
        lab = labels.reshape(-1,nb_heads)
    else:
        lab = labels

    mut_batch = []
    lab_batch = []

    Fresult = []
    cpt = 0
    t0 = time.time()


   
    for pos in positions:  # For each position
        t1 = time.time()


        if multi:
            s = seq[pos-(winsize//2):pos+(winsize//2) + 1].copy()
            if reverse:
                s = s[::-1, [3,2,1,0]]       
           
        else:
            # Get a window of 4002b around the pos
            s = seq[pos-winsize + 1 :pos + winsize].copy()
            if reverse:
                s = s[::-1, [3,2,1,0]]     

        lmut = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        lmut.pop(np.argmax(s[winsize//2]))  # Discard the wt nucleotide

        for mut in lmut:  # For each mutation
            # Get all the windows with the mutation, considering the step applied
            s[winsize//2] = mut  # Mutate
            if multi:
                mut_batch.append(s.copy())
            

            else:
                s_tmp = sliding_window_view(s, (2001, 4))[step, ...]
                mut_batch.append(s_tmp)
        

        if multi:
            lab_batch.append(lab[pos-offset])
        else:
            l = lab[pos-(winsize//2) -offset:pos+(winsize//2)+1-offset].ravel()  # get corresponding labels
            lab_batch.append(l[step])
        
        if len(mut_batch) >= batch or pos == positions[-1]:
            #print("Batch : {}".format(time.time() - t1))
            #cpt += len(lab_batch)/nb_windows
            t1 = time.time()
            # Make the prediction
            res = model.predict(np.array(mut_batch).reshape((-1, winsize, 4)), verbose = 0)
            res = res.reshape((-1, 3, nb_heads))

            lab_batch = np.array(lab_batch).reshape((-1, 1, nb_heads))

            sumtmp = np.abs(res-lab_batch)/nb_heads

            lab_batch = lab_batch.reshape(-1, nb_heads)
            sumtmp[:, 0, :] = sumtmp[:, 0, :] + 1 - mf.vcorrcoef(res[:, 0, :], lab_batch).reshape((-1, 1))
            sumtmp[:, 1, :] = sumtmp[:, 1, :] + 1 - mf.vcorrcoef(res[:, 0, :], lab_batch).reshape((-1, 1))
            sumtmp[:, 2, :] = sumtmp[:, 2, :] + 1 - mf.vcorrcoef(res[:, 0, :], lab_batch).reshape((-1, 1))
            sumtmp = np.sum(sumtmp, axis = 2).ravel()
            # print(sumtmp)
            lab_batch = []
            mut_batch = []

            Fresult.extend(sumtmp)
            """print("{}%, {},  {}s.".format(np.round(cpt/len(positions), 2),
                  np.round(res_mut, 2), np.round(time.time()-t1, 2)))"""


    if save:
        try:

            np.save(output_file, np.array(Fresult).reshape((-1, 3)))
            print("Saved - {}s".format(time.time()-t0))
            
        except FileNotFoundError:
            name = output_file.split("/")
            np.save(name[-1], np.array(Fresult).reshape((-1, 3)))


    if result:
        return np.array(Fresult).reshape((-1, 3))


########################################################################################################################
#
# Experimental mutasome
#
########################################################################################################################
def exp_mutasome(model, sequence, labels, positions, step: int, size: int, output_file: str = "mutasome", save: bool = True, result: bool = True):
    """
    Compute mutasome around position (corresponding to size b around the position)
    -> to compute full sequence position
    positions = np.arange(size,len(sequence),2*size)

    :param model: keras model loaded
    :param sequence: one hot encoded sequence
    :param labels: 1000bp offset sequence labels (pos 1000 in sequence match with pos 0 in label)
    :param positions: array of positions
    :param int step: size to compute around the pos
    :param str output_file: path to output the file
    :param bool save: Save result to file if True
    :param bool result: return result if True

    """

    model = model
    seq = sequence
    lab = labels
    step = step

    Fresult = []
    cpt = 0
    t0 = time.time()

    for pos in positions:  # For each position
        cpt += 1
        Fresult.append([])
        t1 = time.time()

        for i in range(-size, size + 1):  # for size around position
            # Get a window of 4002b around the pos
            s = seq[pos-2000+i:pos+2001+i].copy()

            res_mut = 0

            for mut in lmut:  # For each mutation
                s[(len(s)//2)+1] = mut  # Mutate
                # Get all the windows with the mutation, considering the step applied
                s_tmp = sliding_window_view(s, (2001, 4))
                mid_stmp = len(s_tmp)//2
                s_tmp = s_tmp[[mid_stmp-step, mid_stmp, mid_stmp+step], ...]
                l = lab[pos-2000+i:pos+1+i:step].ravel()
                # get corresponding labels
                l = l[[mid_stmp-step, mid_stmp, mid_stmp+step]]
                # Make the prediction
                res = model.predict(s_tmp.reshape(-1, 2001, 4)).ravel()
                res_mut += np.sum(np.abs(res-l)) + 1 - \
                    np.corrcoef(res, l)[0][1]  # Calculate the score

            Fresult[-1].append(res_mut)
        print("{}/{}, {},  {}s.".format(cpt, len(positions),
              np.round(res_mut, 2), np.round(time.time()-t1, 2)))

    if save:
        np.save(output_file, np.array(Fresult))

    if result:
        return Fresult

##########################################################################################
#
# Fast mutasome
#
##########################################################################################
def fast_mutasome(model, sequence, labels, positions, output_file: str = "mutasome", save: bool = True, result: bool = True):
    """
    Compute mutasome around position (corresponding to size b around the position)
    -> to compute full sequence position
    positions = np.arange(size,len(sequence),2*size)

    :param model: keras model loaded
    :param sequence: one hot encoded sequence
    :param labels: 1000bp offset sequence labels (pos 1000 in sequence match with pos 0 in label)
    :param positions: array of positions
    :param int step: size to compute around the pos
    :param str output_file: path to output the file
    :param bool save: Save result to file if True
    :param bool result: return result if True
    """

    model = model
    seq = sequence
    lab = labels

    Fresult = []
    cpt = 0
    t0 = time.time()
    mut_batch = []
    lab_batch = []
    t0 = time.time()
    for p, pos in enumerate(positions):  # For each position

        t1 = time.time()

        # Get a window of 4002b around the pos
        s = seq[pos-1000:pos+1001].copy()
        lmut = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        lmut.pop(np.argmax(s[(len(s)//2)+1]))  # Check the central base

        for mut in lmut:  # For each mutation
            s[(len(s)//2)+1] = mut  # Mutate
            mut_batch.append(s)

            l = lab[pos-1000]  # get corresponding labels
            lab_batch.append(l)

        if len(lab_batch) >= 4096 or pos == positions[-1]:
            cpt += len(lab_batch)
            mut_batch = np.array(mut_batch).reshape((-1, 2001, 4))
            lab_batch = np.array(lab_batch).ravel()
            res = model.predict(mut_batch).ravel()  # Make the prediction
            res_mut = np.abs(res-lab_batch)  # Calculate the score

            res_mut = np.sum(res_mut.reshape((-1, 3)), axis=1)
            mut_batch = []
            lab_batch = []
            Fresult.extend(res_mut)
            print("{}%,  step: {}s. Total: {}s.".format(np.round(100*cpt/len(positions), 2),
                                                        np.round(
                                                            time.time()-t1, 2),
                                                        np.round(time.time()-t0), 2))

    if save:

        np.save(output_file, np.array(Fresult))
        print("Saved - {}s".format(time.time()-t0))

    if result:
        return Fresult


# ctcf = pd.read_csv("/home/maxime/data/sequences/CTCF/LociSets/tfsites/CTCF/bookmarked-CTCF.bed", "\t", header=None)
# ctcf6p = np.array(ctcf[ctcf[0]=="chr6"][1])
# ctcf6m = np.array(ctcf[ctcf[0]=="chr6"][2])
# sense = np.array(ctcf[ctcf[0]=="chr6"][5])
# ctcf6 = [ctcf6p[i] if x == "+" else ctcf6m[i] for i,x in enumerate(sense)]

# """cpg = pd.read_csv("/home/maxime/data/sequences/mm10/CpG", sep = "\t")
# cpg = cpg[cpg["chrom"]=="chr6"]
# cpg = np.array(cpg["chromStart"])"""

# seq = np.load("/home/maxime/data/sequences/mm10/one_hot/chr6.npz")["arr_0"]


# """mutasome(model=tf.keras.models.load_model("/home/maxime/data/chemical/train/mutasome1/best.h5",
#                                         custom_objects={"mae_cor": mae_cor, "correlate": correlate}),
#         sequence=seq,
#         labels=np.load("/home/maxime/data/chemical/predictions/6__mutasome1.npy").ravel(),
#         positions=np.arange(ct-5000, ct+5001), output_file="/home/maxime/data/chemical/mutasome/chr6_full/test_fullhead",
#         save=True, result=False, step=[500,750,1000,1250,1500])"""



# mutasome(model=tf.keras.models.load_model("/home/maxime/data/chemical/train/6_12_2022___12_10_16/best.h5",
#                                         custom_objects={"mae_cor": mae_cor, "correlate": correlate}),
#         sequence=seq,
#         labels=np.load("/home/maxime/data/chemical/predictions/RAWRESULT_6__6_12_2022___12_10_16.npy"),
#         positions=np.array([np.arange(x-1000,x+1001) for x in ctcf6[:20]]).ravel(), output_file="/home/maxime/data/chemical/mutasome/chr6_full/multi_test_fullhead",
#         save=True, result=False, step=[500,750,1000,1250,1500], multi=True)

# for chr in range(1, 20):
#     if not path.exists("/home/maxime/data/chemical/mutasome/mut_chr{}_3M_minus3M.npy".format(chr)):
#         try:
#             print(chr)
#             seq = np.load("/home/maxime/data/sequences/mm10/one_hot/chr{}.npz".format(chr))["arr_0"]
#             mutasome(model=tf.keras.models.load_model("/home/maxime/data/chemical/train/6_12_2022___12_10_16/best.h5",
#                                                     custom_objects={"mae_cor": mae_cor, "correlate": correlate}),
#                     sequence=seq,
#                     labels=np.load("/home/maxime/data/chemical/predictions/RAWRESULT_{}__6_12_2022___12_10_16.npy".format(chr)),
#                     positions=np.arange(3_000_000, len(seq)-3_000_000), output_file="/home/maxime/data/chemical/mutasome/mut_chr{}_3M_minus3M".format(chr),
#                     multi= True, offset=1000, save=True, result=True, step=[500,750,1000,1250,1500])
#         except FileNotFoundError:
#             continue

chr=6
seq = np.load("/home/maxime/data/sequences/mm10/one_hot/chr6.npz")["arr_0"]
lab = np.load("/home/maxime/data/mnase/predictions/RAWRESULT_6__mA_1.npy".format(chr))

print("lab :", lab.shape[0]//5)
print("seq :", seq.shape)

mutasome(model=tf.keras.models.load_model("/home/maxime/data/mnase/train/mA_1/best.h5",
                                                    custom_objects={"mae_cor": mae_cor, "correlate": correlate}),
                    sequence=seq,
                    labels=lab,
                    positions=np.arange(5_000_000,6_000_000), output_file="/home/maxime/data/mnase/mutasome/mut_mA_chr{}_5M_6M_reversed".format(chr),
                    multi= True, offset=1000, save=True, result=True, step=[500,750,1000,1250,1500], reverse = True)

