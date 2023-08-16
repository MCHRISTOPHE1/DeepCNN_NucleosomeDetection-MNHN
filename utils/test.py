import argparse
from generator import DNAmulti
from generator_opt import DNAmulti5H
from losses import mae_cor, correlate
import numpy as np
import tensorflow as tf
import sklearn
from myfunc import loadnp

physical_devices = tf.config.list_physical_devices('GPU') 
for gpu_instance in physical_devices: 
    tf.config.experimental.set_memory_growth(gpu_instance, True)
strategy = tf.distribute.MirroredStrategy()

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
parser.add_argument("--headsteps",      help = "Positions of heads",                                          type = int,     default=[-500, -250, 0, 250, 500], required=False, nargs="+")

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
with strategy.scope(): 
        model = tf.keras.models.load_model(model_str, custom_objects={'correlate': correlate, 'mae_cor': mae_cor})

batch = args.batch
winsize = args.winsize
chr = args.test_chr
multi = args.multi
headsteps = args.headsteps


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


        test = test_gen.generate_images(test_idx, is_training=False, batch_size=batch, step=headsteps)
        model_result = model.predict(test, verbose=1, steps=len(test_idx)//batch)
        model_result = model_result.ravel()
        if multi>1:
                np.savez_compressed(global_path+"predictions/RAWRESULT_{}".format(fname) + args.model, model_result)
                print("Raw pred saved at {}".format(global_path+"predictions/RAWRESULT_{}".format(fname) + args.model))
        
        model_result = model_result.reshape((-1, 5))
        pred = np.zeros((lseq, 5))
        fpos, step = 1000+headsteps[0], headsteps[-1]-headsteps[-2] #/!\ Uniquement si le pas est régulier /!\
        for i, a in enumerate(np.hsplit(model_result, len(headsteps))):
                pred[fpos+(step*i):len(a)+fpos+(step*i), i]=a.ravel()

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

