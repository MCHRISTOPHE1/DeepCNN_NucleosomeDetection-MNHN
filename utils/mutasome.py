# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 10:46:38 2022

@author: maxime christophe
"""


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="Keras model to predict data", type=str, required=True)
parser.add_argument("--chr", help = 'chr to predict (unique)', required = True)
parser.add_argument("--seq", help = "path to seq", default = "/home/maxime/data/sequences/mm10/one_hot/", required = False)
parser.add_argument("--output_dir", help = "output directory file")
parser.add_argument("--output_file", help = "structure for file name")
parser.add_argument("--pos", help="start position", default = 3_000_000, type=int)
parser.add_argument("--size", help = "size of the mutasome", default=None, type=int)
parser.add_argument("--gpu", help = "gpu index to use", required = True)

args = parser.parse_args()
import os
os.environ["CUDA_VISIBLE_DEVICES"]=f"{args.gpu}"
from os import path

import numpy as np
import time
import tensorflow as tf
import pandas as pd
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
    
def mutasome_opt(seq, model, winsize, batch, nb_heads, start=None, end=None):
    """_summary_

    Args:
        seq (np.array): one hot encoded sequence
        model (tf.keras.Model): loaded model
        winsize (int): window size for prediction
        batch (int): batch size > winsize
        nb_heads (_type_): number of model output
    """
    
    if start is None: start = winsize//2
    if end is None: end = len(seq)-batch-winsize//2
    results = [0]*start*3
    wt = [0]*start
    for pos in range(start, end, batch):
        #get sequeces
        sliding = mf.sliding_window_view(seq[pos:pos+(batch)+winsize-winsize%2], (winsize, 4)).reshape((-1, winsize, 4))

        #mutate
        reps = np.repeat(sliding, 4, axis=0)
        reps[:, winsize//2,:] = np.tile(np.eye(4).ravel(), len(sliding)).reshape((-1, 4))

        #predic
        preds = model.predict(reps)
        preds_reshape = preds.reshape((-1, 4, nb_heads))

        #separate WT
        lab = preds_reshape[seq[pos:pos+len(preds_reshape)]==1].reshape((-1, 1, nb_heads))
        res = preds_reshape[seq[pos:pos+len(preds_reshape)]==0].reshape((-1, 3, nb_heads))

        #Calculate score
        sumtmp = (res-lab)**2
        results.extend(np.mean(sumtmp, axis=2).ravel())
        wt.extend(lab[:,:,(nb_heads//2)])
    
    results.extend([0]*(len(results)//3-len(seq)))
    return np.array(results).reshape((-1, 3))
    
def mse_mutasome_opt(model, sequence, pos = None,size = None,winsize = 2001,
                    output_file: str = "mutasome", save: bool = True, result: bool = True,
                    batch:int = 8192, reverse = False, write_batch = 5_000_000):
    
    # Faire varier le nombre de steps
    """
    Compute mutasome around position (corresponding to size b around the position)
    -> to compute full sequence position
    positions = np.arange(size,len(sequence),2*size)

    :param model: keras model loaded
    :param sequence: one hot encoded sequence
    :param positions: first position
    
    :param int size: length of desired mutasome
    :param str output_file: path to output the file
    :param bool save: Save result to file if True
    :param bool result: return result if True
    :param int offset: locus of the first prediction
    """

    
        
    nb_heads = 5
    # options = tf.data.Options()
    # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    seq = sequence
    
    if pos is None:
        pos = 0
        
    if size is None:
        size = len(seq)-pos-winsize
        
    t0 = time.time()

    mutation = np.tile(np.eye(4), (batch//4, 1)).reshape((batch//4,4, 4))
    #mutate sequence
    MUTASOME_SCORE = []
    PREDICTION = []
    
    try:
        os.remove(output_file + "tmp")
        
    except FileNotFoundError:
        pass
    
    try:
        os.remove(output_file + "PREDICTIONtmp")
    except FileNotFoundError:
        pass
    
    with open(output_file + "tmp", "a") as mutscor_file, open(output_file + "PREDICTIONtmp", "a") as pred_file:
        
        for cpt, p in enumerate(range(pos-winsize//2, pos+size, batch//4)):
            try:
                seqs = mf.sliding_window_view(seq[p:p+(batch//4)+winsize-winsize%2],  (winsize, 4))#.reshape((batch//4), winsize, 4)
                repseq = np.repeat(seqs, 4, axis = 1)
                #np.expand_dims(seqs, axis=1)
                repseq[:, :, winsize//2,:] = mutation[:len(repseq)]
                repseq = repseq.reshape((-1, winsize, 4))

                #separate indices of lab and pred
                idxlab = np.argmax(seqs[:,:,winsize//2], axis=-1).ravel()+np.arange(0,batch,4)
                idxpred = np.ones(batch, dtype=np.bool)
                idxpred[idxlab] = 0

                #Predict
                rawpred = model.predict_on_batch(repseq)
                res = rawpred[idxpred].reshape((batch//4, 3, nb_heads))
                lab_batch = rawpred[idxlab].reshape((batch//4, 1, nb_heads))
                
                #MSE
                sumtmp = (res-lab_batch)**2
                sumtmp = np.mean(sumtmp, axis = 2).ravel()
                MUTASOME_SCORE.extend(list(sumtmp))
                PREDICTION.extend(list(lab_batch[:, :, 2]))
                loadbar(cpt, np.round(size/(batch//4)), t0)
                
                if cpt%10000==0 and cpt>0:
                    np.savetxt(mutscor_file, np.array(MUTASOME_SCORE).reshape((-1, 3)))
                    np.savetxt(pred_file, np.array(PREDICTION ))
                    MUTASOME_SCORE = []
                    PREDICTION = []
            except ValueError:
                break
                
        np.savetxt(mutscor_file, np.array(MUTASOME_SCORE, dtype=np.float32).reshape((-1, 3)))
        np.savetxt(pred_file, np.array(PREDICTION, dtype=np.float32 ))
    
    
    if save:
        MUTASOME_SCORE = np.zeros((len(seq), 3), dtype=np.float32)
        muttmp = np.loadtxt(output_file + "tmp")
        MUTASOME_SCORE[pos:pos+len(muttmp),:] = muttmp
                                   
        np.savez_compressed(output_file, np.array(MUTASOME_SCORE).reshape((-1, 3)))
        os.remove(output_file + "tmp")
        
        PREDICTION = np.zeros(len(seq))
        predtmp = np.loadtxt(output_file + "PREDICTIONtmp")
        PREDICTION[pos:pos+len(predtmp)] = predtmp
        np.savez_compressed(output_file + "PREDICTION", 
                            np.array(PREDICTION).ravel())
        
        os.remove(output_file + "PREDICTIONtmp")
        print("Saved - {}s".format(time.time()-t0))
            


    if result:
        return (np.array(PREDICTION, dtype=np.float16).ravel(), 
                np.array(MUTASOME_SCORE, dtype=np.float16).reshape((-1, 3)))


#                     multi= True, offset=1000, save=True, result=True, step=[500,750,1000,1250,1500], reverse = True)
# with strategy.scope():

model = tf.keras.models.load_model(f"{args.model}/best.h5", custom_objects={"mae_cor": mae_cor, "correlate": correlate})

seq = np.load(f"{args.seq}/chr{args.chr}.npz")["arr_0"]
mse_mutasome_opt(model = model, sequence=seq,
        output_file=f"{args.output_dir}/{args.chr}_{args.output_file}", save=True,
        result=False, batch=4096, pos = args.pos, size=args.size)