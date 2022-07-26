# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 10:46:38 2022

@author: maxime christophe
"""

import numpy as np
import argparse
from myfunc import sliding_window_view
import time 



parser = argparse.ArgumentParser()

def mutasome(model, sequence, labels, positions, step:int, size:int,output_file:str = "mutasome", save:bool = True, result:bool = True ):
    """
    Compute mutasome around position (corresponding to size b around the position)
    -> to compute full sequence position
    positions = np.arange(size,len(sequence),2*size)

    :param model: keras model loaded
    :param sequence: one hot encoded sequence
    :param labels: 1000bp offset sequence labels (pos 1000 in sequence match with pos 0 in label)
    :param positions: array of positions
    :param int step: size to compute around the pos
    :output_file: path to output the file
    :param bool save: Save result to file if True
    :param bool result: return result if True

    """


    model = model
    seq = sequence
    lab = labels
    step = step

    Fresult = []
    cpt =0
    t0 =time.time()

    for pos in positions:                                                           #For each position
        cpt+=1
        print("{}/{}".format(cpt,len(positions)))
        Fresult.append([])
        t1 = time.time()

        for i in range(-size,size + 1):                                             #for size around position
            s = seq[pos-2000+i:pos+2001+i].copy()                                       #Get a window of 4002b around the pos
            lmut = [[1,0,0,0],[0,1,0,0], [0,0,1,0], [0,0,0,1]]
            lmut.pop(np.argmax(s[(len(s)//2)+1]))                                       #Check the central base
            res_mut = 0

            for mut in lmut:                                                        #For each mutation
                s[(len(s)//2)+1] = mut                                                  #Mutate
                s_tmp = sliding_window_view(s, (2001,4))[::step,...]                    #Get all the windows with the mutation, considering the step applied
                l = lab[pos-2000+i:pos+1+i:step].ravel()                                #get corresponding labels
                res = model.predict(s_tmp.reshape(-1,2001,4)).ravel()                   #Make the prediction
                res_mut += np.sum(np.abs(res-l)) + 1 - np.corrcoef(res, l)[0][1]        #Calculate the score
            #print(res_mut)

            Fresult[-1].append(res_mut)
        print(time.time()-t1)

    if save:
        np.save("output_file", np.array(Fresult))
    
    if result:
        return Fresult
