print("START-------------------------------------------------")
from secrets import choice
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
for gpu_instance in physical_devices: 
    tf.config.experimental.set_memory_growth(gpu_instance, True)
strategy = tf.distribute.MirroredStrategy()

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.mixed_precision import LossScaleOptimizer
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
opt = LossScaleOptimizer(opt)

from modeles import create_model
from generator import DNAmulti
from losses import correlate, mae_cor

import numpy as np
import pandas as pd

import argparse
import os
import datetime
day = datetime.datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument("data", help='Kind of experimental signal to use, choose between {microc, mnase, chemical}', type=str)
parser.add_argument("--batch", help="batch size", type = int, default=2048, required=False)
parser.add_argument("--winsize", help="Window size to use", type = int, default=2001, required=False)
parser.add_argument("--epochs", help="Number of epochs to train",type = int, default=3, required=False)
parser.add_argument("--model", help="name of the model (see modeles.py)",type = str, default = "MNase_simple", required=False)

parser.add_argument("--train_chr", help = "nums of chromosomes for the training",type=int, default=[2,3,4,5], nargs="+", required=False)
parser.add_argument("--train_frac", help = "fraction of each chromosome to use for training must be in [0-1]", type = float, default = 0.25, required=False)
parser.add_argument("--train_zeros", help = "Let position with no signal if true", type = bool, default=False, required=False)
parser.add_argument("--train_sample", help = "sample data if True", type = bool, default = True, required=False)
parser.add_argument("--train_reverse", help= "0: no reverse complement, 1: reverse complement only, 2: both strands", type = int, default=0, required=False, choices=[0,1,2])
parser.add_argument("--train_weights", help="0: No weights, 1: array of sample weights, 2: dict of sample weights, 3: undersampling", type = int, default=1, required=False, choices=[0,1,2,3])

parser.add_argument("--val_chr", help = "nums of chromosomes for the validation",type = int, default=[7], nargs="+", required=False)
parser.add_argument("--val_frac", help = "fraction of each chromosome to use for training must be in [0-1]", type = float, default = 1, required=False)
parser.add_argument("--val_zeros", help = "Let position with no signal if true", type = bool, default=False, required=False)
parser.add_argument("--val_sample", help = "sample data if True", type = bool, default = True, required=False)
parser.add_argument("--val_reverse", help= "0: no reverse complement, 1: reverse complement only, 2: both strands", type = int, default=0, required=False, choices=[0,1,2])
parser.add_argument("--val_weights", help="0: No weights, 1: array of sample weights, 2: dict of sample weights, 3: undersampling", type = int, default=0, required=False, choices=[0,1,2,3])

parser.add_argument("--comments", help = "comment to add in the readme", default = "", required=False)

args = parser.parse_args()
print(args)

# Params
data = args.data
global_path = "/home/maxime/data/{}/train/".format(data) + "_".join(np.array([day.day, day.month, day.year,"_", day.hour,day.minute, day.second], dtype="str"))
model_name = args.model

batch = args.batch
winsize = args.winsize
epochs = args.epochs

train_chr = args.train_chr
train_frac = args.train_frac 
train_sample = args.train_sample
train_zeros = args.train_zeros
train_reverse = args.train_reverse
train_weights = args.train_weights


val_chr = args.val_chr
val_frac = args.val_frac
val_sample = args.val_sample
val_zeros = args.val_zeros
val_reverse = args.val_reverse
val_weights = args.val_weights

comments = args.comments
print(train_chr, val_chr)
if data == "microc":
    lab = "/home/maxime/data/microc/Nsignal/Fsignal_{}.npy"
elif data == "mnase":
    lab = "/home/maxime/data/mnase/Nsignal/Fsignal_{}.npy"
elif data == "chemical":
    lab = "/home/maxime/data/chemical/labels/Fsignal_{}.npy"
else:
    lab = data

if data == "chemical":
    seq = "/home/maxime/data/sequences/mm9/one_hot/chr{}.npy"
else:
    seq = "/home/maxime/data/sequences/mm10/onehot_{}.npz"


savefile = "registered as: {}\n \
            comments: {}\n \
            data: {}\n \
            model: {}\n \
            winsize: {}\n \
            batch_size: {}\n \
            epochs: {}\n \
            train \n \
            -----------------\
            train chr: {}\n \
            train frac: {}\n \
            train sample: {}\n \
            train reverse: {}\n \
            train weights: {}\n\n \
            Validation \n \
            -----------------\
            Validation chr: {}\n \
            Validation frac: {}\n \
            Validation sample: {}\n \
            Validation reverse: {}\n \
            Validation weights: {}\n \
            ".format(global_path, comments,data, model_name, winsize, batch, epochs,
             train_chr, train_frac, train_sample, train_reverse, train_weights,
             val_chr, val_frac, val_sample, val_reverse, val_weights)


assert data in ["mnase", "microc", "chemical"], "{} not in [mnase, microc, chemical]".format(data)
os.mkdir(global_path)




loss = mae_cor
metrics = [mae_cor, "mae", correlate]
with strategy.scope():
    model = create_model(model_name, optimizer=opt, input_shape=(winsize,4), loss = loss, metrics=metrics)
    model.summary()
    mcp_save = ModelCheckpoint( "best.h5", save_best_only=True, monitor='val_loss', mode='min')



train_gen = DNAmulti(seq=[seq.format(x) for x in train_chr],
                    lab=[lab.format(x) for x in train_chr],
                    zeros=train_zeros, winsize=winsize, frac=train_frac, sample=train_sample, apply_weights=train_weights, reverse=train_reverse)
train_idx = train_gen.generate_split_indexes()
train = train_gen.generate_images(train_idx, is_training = True, batch_size = batch)
spe = len(train_idx)//batch


val_gen = DNAmulti(seq=[seq.format(x) for x in val_chr],
                        lab=[lab.format(x) for x in val_chr],
                        zeros=val_zeros, winsize=winsize, frac=val_frac, sample=val_sample, apply_weights=val_weights, reverse=val_reverse)
val_idx = val_gen.generate_split_indexes()
val = train_gen.generate_images(val_idx, is_training = True, batch_size = batch)
vspe = len(val_idx)//batch

history = model.fit(train, batch_size=batch, steps_per_epoch=spe,
                    validation_data = val, validation_steps = vspe, validation_batch_size = batch,
                    epochs=epochs, verbose=1, callbacks=[mcp_save])

model.save(global_path +"/model")
hist_df = pd.DataFrame(history.history) 

with open(global_path+"/history", mode='w') as f:
    hist_df.to_csv(f)

with open(global_path+"/readme", mode = "w") as f:
    f.write(savefile)
