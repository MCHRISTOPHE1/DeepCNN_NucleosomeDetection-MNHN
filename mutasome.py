# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 10:46:38 2022

@author: maxime christophe
"""

import numpy as np
import argparse
from myfunc import sliding_window_view
import time
import tensorflow as tf
from losses import mae_cor, correlate
import pandas as pd


parser = argparse.ArgumentParser()


def mutasome(model, sequence, labels, positions, step, size: int, output_file: str = "mutasome", save: bool = True, result: bool = True):
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
    """

    assert len(step)>=1, "step est une liste de positions contenant les tetes du mutasome (entre 0 et 2000)"

    model = model
    seq = sequence
    lab = labels
    mut_batch = []
    lab_batch = []

    Fresult = []
    cpt = 0
    t0 = time.time()

    nb_windows = len(step)

    for pos in positions:  # For each position
        t1 = time.time()

        # Get a window of 4002b around the pos
        s = seq[pos-2000:pos+2001].copy()
        lmut = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        lmut.pop(np.argmax(s[(len(s)//2)+1]))  # Check the central base

        for mut in lmut:  # For each mutation
            s[(len(s)//2)+1] = mut  # Mutate
            # Get all the windows with the mutation, considering the step applied
            s_tmp = sliding_window_view(s, (2001, 4))[step, ...]
            mut_batch.append(s_tmp)

        l = lab[pos-2000:pos+1].ravel()  # get corresponding labels
        lab_batch.append(l[step])

        if len(mut_batch) >= 4096 or pos == positions[-1]:
            #cpt += len(lab_batch)/nb_windows

            # Make the prediction
            res = model.predict(
                np.array(mut_batch).reshape(-1, 2001, 4)).ravel()
            res = res.reshape((-1, 3, nb_windows))

            lab_batch = np.array(lab_batch).reshape((-1, 1, nb_windows))

            sumtmp = np.sum(
                np.sum(np.abs(res-lab_batch), axis=1), axis=1).ravel()

            lab_batch = lab_batch.reshape(-1, nb_windows)
            res_mut = sumtmp + 3 - np.diag(np.corrcoef(res[:, 0, :], lab_batch, rowvar=True), len(lab_batch)) \
                - np.diag(np.corrcoef(res[:, 1, :], lab_batch, rowvar=True), len(lab_batch)) \
                - np.diag(np.corrcoef(res[:, 2, :], lab_batch,
                          rowvar=True), len(lab_batch))  # Calculate the score
            mut_batch = []
            lab_batch = []

            Fresult.extend(res_mut.ravel())
            """print("{}%, {},  {}s.".format(np.round(cpt/len(positions), 2),
                  np.round(res_mut, 2), np.round(time.time()-t1, 2)))"""

    if save:
        print("Saved - {}s".format(time.time()-t0))
        np.save(output_file, np.array(Fresult))

    if result:
        return Fresult

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

