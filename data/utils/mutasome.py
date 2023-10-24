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
import sys

try:
    import myfunc as mf
    from myfunc import sliding_window_view, loadbar
    from losses import mae_cor, correlate
except ModuleNotFoundError:
    import sys
    sys.path.insert(1, "/home/maxime/data/utils")
    import myfunc as mf
    from myfunc import sliding_window_view, loadbar
    from losses import mae_cor, correlate


physical_devices = tf.config.list_physical_devices('GPU') 
for gpu_instance in physical_devices: 
    tf.config.experimental.set_memory_growth(gpu_instance, True)
strategy = tf.distribute.MirroredStrategy()



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

    Fresult = [0]*3*(positions[0]-1)
    cpt = 0
    t0 = time.time()


   
    for pos in positions:  # For each position
        t1 = time.time()


        if multi:
            s = seq[pos-(winsize//2):pos+(winsize//2) + 1].copy()
            if reverse:
                s = s[::-1, ::-1]       
           
        else:
            # Get a window of 4002b around the pos
            s = seq[pos-winsize + 1 :pos + winsize].copy()
            if reverse:
                s = s[::-1, ::-1]     

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
            cpt+=1
            sys.stdout.write(f"\r{cpt}/{np.ceil(3*len(positions)/batch)}")
            sys.stdout.flush()
            #print("Batch : {}".format(time.time() - t1))
            #cpt += len(lab_batch)/nb_windows
            t1 = time.time()
            # Make the prediction
            res = model.predict(np.array(mut_batch).reshape((-1, winsize, 4)), verbose = 0)
            res = res.reshape((-1, 3, nb_heads))
            # print("pred")
            # print(res.shape)

            #MAE
            lab_batch = np.array(lab_batch).reshape((-1, 1, nb_heads))
            sumtmp = np.abs(res-lab_batch)
            lab_batch = lab_batch.reshape(-1, nb_heads)
            # print("#############################################  mae")
            # print(sumtmp)

            #Pearson's correlation
            sumtmp[:, 0, :] = sumtmp[:, 0, :] + 1 - mf.vcorrcoef(res[:, 0, :], lab_batch).reshape((-1, 1))
            sumtmp[:, 1, :] = sumtmp[:, 1, :] + 1 - mf.vcorrcoef(res[:, 1, :], lab_batch).reshape((-1, 1))
            sumtmp[:, 2, :] = sumtmp[:, 2, :] + 1 - mf.vcorrcoef(res[:, 2, :], lab_batch).reshape((-1, 1))
            sumtmp = sumtmp/nb_heads
            # print("############################### PCC")
            # print(sumtmp.shape)
            # print(sumtmp)
            sumtmp = np.sum(sumtmp, axis = 2).ravel()
            # print("RES")
            # print(sumtmp.shape)
            # print(sumtmp)
            lab_batch = []
            mut_batch = []

            Fresult.extend(sumtmp)
            """print("{}%, {},  {}s.".format(np.round(cpt/len(positions), 2),
                  np.round(res_mut, 2), np.round(time.time()-t1, 2)))"""

    Fresult.extend([0]*(3*(len(seq)-positions[-1])))
    if save:
        try:

            np.savez_compressed(output_file, np.array(Fresult).reshape((-1, 3)))
            print("Saved - {}s".format(time.time()-t0))
            
        except FileNotFoundError:
            name = output_file.split("/")
            np.savez_compressed(name[-1], np.array(Fresult).reshape((-1, 3)))


    if result:
        return np.array(Fresult).reshape((-1, 3))



def mse_mutasome(model, sequence, labels, positions, step, winsize = 2001,offset=1000, multi:bool=False, output_file: str = "mutasome", save: bool = True, result: bool = True, batch:int = 8*16384, reverse = False):
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
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    model = model
    seq = sequence
    if multi:
        lab = labels.reshape(-1,nb_heads)
    else:
        lab = labels

    mut_batch = []
    lab_batch = []

    Fresult = [0]*3*(positions[0]-1)
    cpt = 0
    t0 = time.time()
   
    for pos in positions:  # For each position
        t1 = time.time()


        if multi:
            s = seq[pos-(winsize//2):pos+(winsize//2) + 1].copy()
            if reverse:
                s = s[::-1, ::-1]       
           
        else:
            # Get a window of 4002b around the pos
            s = seq[pos-winsize + 1 :pos + winsize].copy()
            if reverse:
                s = s[::-1, ::-1]     

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
            cpt+=1
            loadbar(cpt, np.ceil(3*len(positions)/batch), t0)
    #         x, n = cpt, np.ceil(3*len(positions)/batch)
    #         sys.stdout.flush()
    #         sys.stdout.write(f"|{'='*(int(30*x/(n-1))-1)}>{'.'*int(30*(1-(x/(n-1))))}|\
    # {x+1}/{n} {time.time()-t0:.2f}s")
            
            #print("Batch : {}".format(time.time() - t1))
            #cpt += len(lab_batch)/nb_windows
            t1 = time.time()
            # Make the prediction

            tfdata = tf.data.Dataset.from_tensor_slices(np.array(mut_batch).reshape((-1, winsize, 4)))
            tfdata = tfdata.batch(len(mut_batch))
            tfdata = tfdata.with_options(options)
            res = model.predict(tfdata, verbose = 0)
            res = res.reshape((-1, 3, nb_heads))
            # print("pred")
            # print(res.shape)

            #MSE
            lab_batch = np.array(lab_batch).reshape((-1, 1, nb_heads))
            sumtmp = (res-lab_batch)**2
            sumtmp = np.mean(sumtmp, axis = 2).ravel()

            # print("RES")
            # print(sumtmp.shape)
            # print(sumtmp)
            lab_batch = []
            mut_batch = []

            Fresult.extend(sumtmp)
            
            """print("{}%, {},  {}s.".format(np.round(cpt/len(positions), 2),
                  np.round(res_mut, 2), np.round(time.time()-t1, 2)))"""

    Fresult.extend([0]*(3*(len(seq)-positions[-1])))
    if save:
        try:

            np.savez_compressed(output_file, np.array(Fresult).reshape((-1, 3)))
            print("Saved - {}s".format(time.time()-t0))
            
        except FileNotFoundError:
            name = output_file.split("/")
            np.savez_compressed(name[-1], np.array(Fresult).reshape((-1, 3)))


    if result:
        return np.array(Fresult).reshape((-1, 3))

def neutral_mutasome(model, sequence, labels, positions, step, winsize = 2001,offset=1000, multi:bool=False, output_file: str = "mutasome", save: bool = True, result: bool = True, batch:int = 4096, reverse = False):
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

    Fresult = [0]*3*(positions[0]-1)
    cpt = 0
    t0 = time.time()


   
    for pos in positions:  # For each position
        t1 = time.time()


        if multi:
            s = seq[pos-(winsize//2):pos+(winsize//2) + 1].copy()
            if reverse:
                s = s[::-1, ::-1]       
           
        else:
            # Get a window of 4002b around the pos
            s = seq[pos-winsize + 1 :pos + winsize].copy()
            if reverse:
                s = s[::-1, ::-1]     

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
            cpt+=1
            sys.stdout.write(f"\r{cpt}/{np.ceil(3*len(positions)/batch)}")
            sys.stdout.flush()
            #print("Batch : {}".format(time.time() - t1))
            #cpt += len(lab_batch)/nb_windows
            t1 = time.time()
            # Make the prediction
            res = model.predict(np.array(mut_batch).reshape((-1, winsize, 4)), verbose = 0)
            res = res.reshape((-1, 3, nb_heads))
            # print("pred")
            # print(res.shape)

            #MSE
            lab_batch = np.array(lab_batch).reshape((-1, 1, nb_heads))
            sumtmp = res-lab_batch
            sumtmp = np.mean(sumtmp, axis = 2).ravel()

            # print("RES")
            # print(sumtmp.shape)
            # print(sumtmp)
            lab_batch = []
            mut_batch = []

            Fresult.extend(sumtmp)
            """print("{}%, {},  {}s.".format(np.round(cpt/len(positions), 2),
                  np.round(res_mut, 2), np.round(time.time()-t1, 2)))"""

    Fresult.extend([0]*(3*(len(seq)-positions[-1])))
    if save:
        try:

            np.savez_compressed(output_file, np.array(Fresult).reshape((-1, 3)))
            print("Saved - {}s".format(time.time()-t0))
            
        except FileNotFoundError:
            name = output_file.split("/")
            np.savez_compressed(name[-1], np.array(Fresult).reshape((-1, 3)))


    if result:
        return np.array(Fresult).reshape((-1, 3))


def mutgen(model, sequence, labels, positions, step, winsize = 2001,offset=1000, multi:bool=False, batch:int = 8*16384, reverse = False):

    assert len(step)>=1, "step est une liste de positions contenant les tetes du mutasome (entre 0 et 2000)"
    nb_heads = len(step)

    model = model
    seq = sequence.astype("float32")
    if multi:
        lab = labels.reshape(-1,nb_heads).astype("float32")
    else:
        lab = labels

    mut_batch = []
    lab_batch = []

    Fresult = [0]*3*(positions[0]-1)

    for pos in positions:  # For each position
        if multi:
            s = seq[pos-(winsize//2):pos+(winsize//2) + 1].copy()
            if reverse:
                s = s[::-1, ::-1]       
        else:
            # Get a window of 4002b around the pos
            s = seq[pos-winsize + 1 :pos + winsize].copy()
            if reverse:
                s = s[::-1, ::-1]     

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
            yield tf.reshape(tf.convert_to_tensor(mut_batch),(-1, winsize, 4)), tf.reshape(tf.convert_to_tensor(lab_batch), (-1, 1, nb_heads))

            lab_batch = []
            mut_batch = []

def mutcalc(mut_batch,lab_batch, nb_heads, Fresult):
    res = model.predict(mut_batch, verbose = 0)
    res = res.reshape((-1, 3, nb_heads))

    #MSE
    sumtmp = (res-lab_batch)**2
    sumtmp = tf.reduce_mean(sumtmp, axis = 2)

    Fresult.extend(sumtmp.numpy().ravel())

# file = "chemicalF"
# with strategy.scope():
#     model = tf.keras.models.load_model(f"/home/maxime/data/chemical/train/{file}/best.h5", custom_objects={"mae_cor": mae_cor, "correlate": correlate})
    
# for chr in range(1, 20):
#     Fresult = []
#     seq = np.load(f"/home/maxime/data/sequences/mm10/one_hot/chr{chr}.npz")["arr_0"]
#     lab = np.load(f"/home/maxime/data/chemical/predictions/RAWRESULT_{chr}__{file}.npz")["arr_0"]
#     batch = 4098
#     print("Initialize generator")
#     #gen = mutgen(model = model, sequence=seq, labels=lab, positions=np.arange(3_000_000, len(seq)-3_000_000), step=[500, 750, 1000, 1250, 1500], multi=True)
#     print("Convert to tf.Dataset")
#     dataset = tf.data.Dataset.from_generator(lambda: mutgen(model = model, sequence=seq, labels=lab,batch=batch, positions=np.arange(3_000_000, len(seq)-3_000_000), step=[500, 750, 1000, 1250, 1500], multi=True),
#                                                 output_signature=(
#                                                     tf.TensorSpec(shape=(batch, 2001, 4), dtype=tf.float32),
#                                                     tf.TensorSpec(shape=(batch//3, 1,5), dtype=tf.float32)
#                                                 ))#.prefetch(tf.data.AUTOTUNE)
#     options = tf.data.Options()
#     #options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
#     options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
#     dataset = dataset.with_options(options)
#     cpt = 0
#     for data in dataset:
#         mutcalc(data[0], data[1], 5, Fresult)
#         cpt+=1





file = "mut_highcorr_0"
#                     multi= True, offset=1000, save=True, result=True, step=[500,750,1000,1250,1500], reverse = True)
with strategy.scope():
    model = tf.keras.models.load_model(f"/home/maxime/data/mnase/train/highcorr_10813239/best.h5", custom_objects={"mae_cor": mae_cor, "correlate": correlate})
    for chr in range(12, 20):
        seq = np.load(f"/home/maxime/data/sequences/mm10/one_hot/chr{chr}.npz")["arr_0"]
        lab = np.load(f"/home/maxime/data/mnase/predictions/RAWRESULT_{chr}__highcorr_10813239.npz")["arr_0"]
        mse_mutasome(model = model, sequence=seq, labels=lab,
                positions=np.arange(3_000_000, len(seq)-3_000_000), output_file=f"/home/maxime/data/mnase/mutasome/{chr}_{file}",
                save=True, result=False, step=[500, 750, 1000, 1250, 1500], multi=True, batch=4096)
        






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

# chr=6
# seq = np.load("/home/maxime/data/sequences/mm10/one_hot/chr6.npz")["arr_0"]
# lab = np.load("/home/maxime/data/mnase/predictions/RAWRESULT_6__mA_1.npy".format(chr))

# print("lab :", lab.shape[0]//5)
# print("seq :", seq.shape)

# mutasome(model=tf.keras.models.load_model("/home/maxime/data/mnase/train/mA_1/best.h5",
#                                                     custom_objects={"mae_cor": mae_cor, "correlate": correlate}),
#                     sequence=seq,
#                     labels=lab,
#                     positions=np.arange(5_000_000,6_000_000), output_file="/home/maxime/data/mnase/mutasome/mut_mA_chr{}_5M_6M_reversed".format(chr),
