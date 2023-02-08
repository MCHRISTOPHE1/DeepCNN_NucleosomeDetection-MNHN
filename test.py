import argparse
from generator import DNAmulti, DNAmulti5H
from losses import mae_cor, correlate
import numpy as np
import tensorflow as tf
import sklearn
from myfunc import loadnp

import argparse
import os
import datetime
day = datetime.datetime.now()


parser = argparse.ArgumentParser()
parser.add_argument("data",     help = 'Kind of experimental signal to use, choose between {microc, mnase, chemical}',  type=str)
parser.add_argument("model",    help = "name of the model directory",                                                   type = str)
parser.add_argument("--batch",  help = "batch size",                                                                    type = int,     default=8192,   required=False)
parser.add_argument("--winsize",help = "Window size to use",                                                            type = int,     default=2001,   required=False)
parser.add_argument("--multi",  help = "1 if multiple output, 2 if save both result and processed result",              type = int,    default = 0,required = True)

parser.add_argument("--test_chr",       help = "nums of chromosomes for the training",                                  type=int,                       required=True)
parser.add_argument("--reverse",        help = "reverse complement",                                                    type=int,       default = 0, required = False)
parser.add_argument("--test_frac",      help = "fraction of each chromosome to use for training must be in [0-1]",      type = float,   default = 1,    required=False)

parser.add_argument("--comments", help = "comment to add in the readme", default = "", required=False)

args = parser.parse_args()
print(args)




data = args.data
global_path = "/home/maxime/data/{}/".format(data)
model_str = global_path +"train/"+ args.model +"/best.h5"
print(model_str)
print("-----------------------------------------------------------------------------")
model = tf.keras.models.load_model(model_str, custom_objects={'correlate': correlate, 'mae_cor': mae_cor})

batch = args.batch
winsize = args.winsize
chr = args.test_chr
multi = args.multi


seq = ["/home/maxime/data/sequences/mm10/one_hot/chr{}.npz".format(chr)]
lseq = len(loadnp(seq[0]))
if args.reverse == 2:
        fname = "{}__reversed".format(chr)
else:
        fname = "{}__".format(chr)
#try:
if multi>0:
        print("generator multi")
        test_gen = DNAmulti5H(seq=seq, reverse=args.reverse, apply_weights=0, sample=False, zeros=1, N=True, frac = 1)
        test_idx = test_gen.generate_split_indexes()


        test = test_gen.generate_images(test_idx, is_training=False, batch_size=batch)
        model_result = model.predict(test, verbose=1, steps=len(test_idx)//batch)

        model_result = model_result.ravel()
        if multi>1:
                np.savez_compressed(global_path+"predictions/RAWRESULT_{}".format(fname) + args.model, model_result)
                print("Raw pred saved at {}".format(global_path+"predictions/RAWRESULT_{}".format(fname) + args.model))
        
        pred = np.zeros(lseq)
        sum_idx = np.array([[i-500, i-250, i, i+250, i+500] for i in test_idx[:len(model_result)//5]]).ravel()
        uni, cou = np.unique(sum_idx, return_counts=True)
        
        np.add.at(pred, sum_idx, model_result)
        pred[uni] /= cou
else:
        print("generator")
        test_gen = DNAmulti(seq=seq, reverse=args.reverse, apply_weights=0, sample=False, zeros=1, N=True, frac = 1)
        test_idx = test_gen.generate_split_indexes()
        test = test_gen.generate_images(test_idx, is_training=False, batch_size=batch)
        model_result = model.predict(test, verbose=1, steps=len(test_idx)//batch).ravel()
        pred = np.zeros(lseq)
        pred[np.array(test_idx)] = model_result
np.savez_compressed(global_path + "predictions/{}".format(fname) + args.model, pred)
print("saved at {}".format(global_path + "predictions/{}".format(fname) + args.model))



""" uncomment to debug properly
except (IndexError, ValueError, KeyError, FileNotFoundError) as e:
        print("Something wrong!! pls check file")
        np.save(global_path+"predictions/RAWRESULT_{}__".format(chr) + args.model, model_result)
"""

