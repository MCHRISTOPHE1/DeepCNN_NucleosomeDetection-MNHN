print("START-------------------------------------------------")
import tensorflow as tf
tfver = tf.__version__
physical_devices = tf.config.list_physical_devices('GPU') 
for gpu_instance in physical_devices: 
    tf.config.experimental.set_memory_growth(gpu_instance, True)
strategy = tf.distribute.MirroredStrategy()

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.mixed_precision import LossScaleOptimizer


from modeles import create_model
from generator import DNAmulti
from generator_opt import DNAmulti5H, KDNAmulti
from losses import correlate, mae_cor, max_absolute_error

import numpy as np
import pandas as pd
import myfunc as mf

import argparse
import os
import datetime
day = datetime.datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument("data",             help = 'Experimental signal, choose in {microc, mnase, chemical}',    type = str                                                                    )
parser.add_argument("--supdata",        help = "mode {multimappers, uniquereads}, type {A,M},      {gauss15}",type = str,     default = "",                             nargs = "+"         )
parser.add_argument("--batch",          help = "batch size",                                                  type = int,     default = 4096,           required=False                      )
parser.add_argument("--winsize",        help = "Window size to use",                                          type = int,     default = 2001,           required=False                      )
parser.add_argument("--epochs",         help = "Number of epochs to train",                                   type = int,     default = 3,              required=False                      )
parser.add_argument("--model",          help = "name of the model (see modeles.py)",                          type = str,     default = "MNase_simple", required=False                      )
parser.add_argument("--multi",          help = "if True : multiple output (multiple heads) ",                 type = bool,    default = True,           required = False                   )
parser.add_argument("--headsteps",      help = "Positions of heads",                                          type = int,     default=[-500, -250, 0, 250, 500], required=True, nargs="+")
parser.add_argument("--lr",             help = "set initial value of learning rate",                          type = float,   default = 0.001,          required=False)
parser.add_argument("--lr_decay",       help = "set decay mode [constant, exponential, linear], rate, steps ",  type=str,       default=["constant", 1, 1], required=False, nargs="+")

parser.add_argument("--train_chr",      help = "nums of chromosomes for the training",                        type = int,     default = [2,3,4,5],      required=False, nargs="+"           )
parser.add_argument("--train_frac",     help = "fraction of each chromosome  must be in [0-1]",               type = float,   default = 0.25,           required=False                      )
parser.add_argument("--train_zeros",    help = "Let position with no signal if true",                         type = bool,    default = False,          required=False                      )
parser.add_argument("--train_sample",   help = "sample data if True",                                         type = bool,    default = True,           required=False                      )
parser.add_argument("--train_reverse",  help = "0: no revcomp., 1: revcomp. only, 2: both ",                  type = int,     default = 0,              required=False                      )
parser.add_argument("--train_weights",  help = " sklearn",                                                    type = bool,    default = True,           required=False                      )
parser.add_argument("--train_N",        help = "0: remove N, 1: keep N",                                      type = int,     default = 0,              required=False, choices=[0,1]       )
parser.add_argument("--train_trunc",    help = "truncate the start of chr",                                   type = int,     default = 3_000_000)
parser.add_argument("--val_chr",        help = "nums of chromosomes for the validation",                      type = int,     default = [7],            required=False, nargs="+"           )
parser.add_argument("--val_frac",       help = "fraction of each chromosome  must be in [0-1]",               type = float,   default = 1,              required=False                      )
parser.add_argument("--val_zeros",      help = "Let position with no signal if true",                         type = bool,    default = False,          required=False                      )
parser.add_argument("--val_sample",     help = "sample data if True",                                         type = bool,    default = True,           required=False                      )
parser.add_argument("--val_reverse",    help = "0: no revcomp., 1: revcomp.  only, 2: both",                  type = int,     default = 0,              required=False                      )
parser.add_argument("--val_weights",    help = "0: No weights, 1: array , 2: dict, 3: undersampling",         type = int,     default = 0,              required=False                      )
parser.add_argument("--val_N",          help = "0: remove N, 1: keep N",                                      type = int,     default = 0,              required=False, choices=[0,1]       )

parser.add_argument("--comments",       help = "comment to add in the readme", default = "", required=False)

args = parser.parse_args()
print(args)

# Params
data = args.data
global_path = "/home/maxime/data/{}/train/".format(data) + "_".join(np.array([day.day, day.month, day.year,"_", day.hour,day.minute, day.second], dtype="str"))
model_name = args.model

batch = args.batch
winsize = args.winsize
epochs = args.epochs
multi = args.multi
headsteps = args.headsteps
learning_rate = args.lr
learning_parameters = args.lr_decay
learning_decay = learning_parameters[0]
if len(learning_parameters)>1:
    decay_rate = float(learning_parameters[1])
    decay_steps = float(learning_parameters[2])

train_chr = args.train_chr
train_frac = args.train_frac 
train_sample = args.train_sample
train_zeros = args.train_zeros
train_reverse = args.train_reverse
train_weights = args.train_weights
train_N = args.train_N

val_chr = args.val_chr
val_frac = args.val_frac
val_sample = args.val_sample
val_zeros = args.val_zeros
val_reverse = args.val_reverse
val_weights = args.val_weights
val_N = args.val_N

print(train_chr, val_chr)
print(train_frac, val_frac)
comments = "{} \n".format(tfver)
comments += args.comments

if data == "microc":
    lab = "/home/maxime/data/microc/MicroCsignal3std/Fsignal_{}.npy"
elif data == "mnase":
    lab = "/home/maxime/data/mnase/labels/IP_data/MNase_bed/{}/{}/{}".format(*args.supdata)
    lab = lab + "/chr{}.npz"
elif data =="test":
    lab = "/home/maxime/data/mnase/labels/multimappers/A/chr6.npy"
elif data == "chemical":
    lab = "/home/maxime/data/chemical/labels/smooth_mm10/{}.npz"
elif data == "methylation":
    lab = "/home/maxime/data/methylation/labels/chr{}.npz"
else:
    lab = data


seq = "/home/maxime/data/sequences/mm10/one_hot/chr{}.npz"




assert data in ["mnase", "microc", "chemical", "methylation", "test"], "{} not in [mnase, microc, chemical]".format(data)

if data != "test":
    os.mkdir(global_path)

opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
if int(tfver.split(".")[1])<11:
    opt = LossScaleOptimizer(opt)
else:
    losstr = "\n !! LossScaleOptimizer is not used in {}".format(tfver)
    comments += losstr

loss = mae_cor
metrics = [mae_cor, "mae", correlate]

if learning_decay == "constant":
    def scheduler(epoch, lr):
        return lr
    
elif learning_decay == "exponential":
    def scheduler(epoch, lr):
        return lr*(decay_rate**(epoch%decay_steps))
    
elif learning_decay == "linear":
    def scheduler(epoch, lr):
        return  lr/(decay_rate*(epoch%decay_steps))
    
else:
    raise AttributeError("Unknown type of learning decay (scheduler)")

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

with strategy.scope():
    if model_name == "MNase_simple" and multi ==True : model_name = "Chemical_5H"
    model = create_model(model_name, optimizer=opt, input_shape=(winsize,4), loss = loss, metrics=metrics)
    print("Model used : {}".format(model_name))
    model.summary()
    mcp_save = ModelCheckpoint( global_path + "/best.h5", save_best_only=True, monitor='val_loss', mode='min')
    print("tf version : {}".format(tfver))

if multi==True:
    # print("multi is used, the following parameters are not considered : apply_weights (default 1), reverse (default 0)")
    train_gen = DNAmulti5H(seq      = [seq.format(x) for x in train_chr],
                    lab             = [lab.format(x) for x in train_chr],
                    zeros           = train_zeros,
                    winsize         = winsize,
                    frac            = train_frac,
                    sample          = train_sample,
                    apply_weights   = train_weights,
                    reverse         = train_reverse,
                    N               = train_N)
    train_idx = train_gen.generate_split_indexes()
    train = train_gen.generate_images(train_idx, is_training = True, batch_size = batch, step= headsteps)
    spe = len(train_idx)//batch

    if data != "test":
        val_gen = DNAmulti5H(seq        = [seq.format(x) for x in val_chr],
                        lab             = [lab.format(x) for x in val_chr],
                        zeros           = val_zeros,
                        winsize         = winsize,
                        frac            = val_frac,
                        sample          = val_sample,
                        apply_weights   = val_weights,
                        reverse         = val_reverse)
        val_idx = val_gen.generate_split_indexes()
        val = val_gen.generate_images(val_idx, is_training = True, batch_size = batch, step= headsteps)
        vspe = len(val_idx)//batch

    # train = KDNAmulti5H(seq=[seq.format(x) for x in train_chr],
    #                 lab=[lab.format(x) for x in train_chr],
    #                 zeros=train_zeros, winsize=winsize, frac=train_frac,
    #                 sample=train_sample, apply_weights=train_weights,
    #                  reverse=train_reverse, N=train_N, batch_size=4096)

else:
    train_gen = DNAmulti(seq=[seq.format(x) for x in train_chr],
                        lab=[lab.format(x) for x in train_chr],
                        zeros=train_zeros, winsize=winsize, frac=train_frac, sample=train_sample, apply_weights=train_weights, reverse=train_reverse, N=train_N)
    train_idx = train_gen.generate_split_indexes()
    train = train_gen.generate_images(train_idx, is_training = True, batch_size = batch)
    spe = len(train_idx)//batch

    if data !="test":
        val_gen = DNAmulti(seq=[seq.format(x) for x in val_chr],
                        lab=[lab.format(x) for x in val_chr],
                        zeros=val_zeros, winsize=winsize, frac=val_frac, sample=val_sample, apply_weights=val_weights, reverse=val_reverse)
        val_idx = val_gen.generate_split_indexes()
        val = val_gen.generate_images(val_idx, is_training = True, batch_size = batch)
        vspe = len(val_idx)//batch


# print("number of train seq: ", len(train_idx))
# print("number of val seq: ", len(val_idx) )

try:
    if data == "test":
        history = model.fit(train, batch_size=batch, steps_per_epoch=spe,
                        epochs=epochs, verbose=1)

    else:
        print("START TRAINING")
        history = model.fit(train, batch_size=batch, steps_per_epoch=spe,
                            validation_data = val, validation_steps = vspe, validation_batch_size = batch,
                            epochs=epochs, verbose=1, callbacks=[mcp_save])#, lr_scheduler])

        # history = model.fit(train, epochs=epochs, steps_per_epoch=len(train), verbose=1)

        model.save(global_path +"/model")
        hist_df = pd.DataFrame(history.history)
        with open(global_path+"/history", mode='w') as f:
            hist_df.to_csv(f)
except KeyboardInterrupt:
    pass


if args.supdata=="":
    savefile = "registered as: {}\n \
                comments: {}\n \
                data: {}\n \
                model: {}\n \
                winsize: {}\n \
                batch_size: {}\n \
                epochs: {}\n \
                lr : {} {} \n \
                train \n \
                -----------------\
                #sequences: {}\n \
                train chr: {}\n \
                train frac: {}\n \
                train sample: {}\n \
                train reverse: {}\n \
                train weights: {}\n \
                train zeros: {}\n \
                train N: {}\n\n \
                Validation \n \
                -----------------\
                #sequences: {}\n \
                Validation chr: {}\n \
                Validation frac: {}\n \
                Validation sample: {}\n \
                Validation reverse: {}\n \
                Validation weights: {}\n \
                Validation zeros: {}\n \
                Validation N: {}\n\n \
                ".format(global_path, comments,data, model_name, winsize, batch, epochs, learning_rate, learning_parameters,
                len(train_idx), train_chr, train_frac, train_sample, train_reverse, train_weights, train_zeros, train_N,
                len(val_idx), val_chr, val_frac, val_sample, val_reverse, val_weights, val_zeros, val_N)
                
else:
    savefile = "registered as: {}\n \
                comments: {}\n \
                data: {} {} {} {}\n \
                model: {}\n \
                heads : {}\n \
                winsize: {}\n \
                batch_size: {}\n \
                epochs: {}\n \
                lr : {} {} \n \
                train \n \
                -----------------\
                #sequences: {}\n \
                train chr: {}\n \
                train frac: {}\n \
                train sample: {}\n \
                train reverse: {}\n \
                train weights: {}\n \
                train zeros: {}\n \
                train N: {}\n\n \
                Validation \n \
                -----------------\
                #sequences: {}\n \
                Validation chr: {}\n \
                Validation frac: {}\n \
                Validation sample: {}\n \
                Validation reverse: {}\n \
                Validation weights: {}\n \
                ".format(global_path, comments,data,*args.supdata, model_name,headsteps, winsize, batch, epochs, learning_rate, learning_parameters,
                len(train_idx), train_chr, train_frac, train_sample, train_reverse, train_weights, train_zeros, train_N,
                len(val_idx), val_chr, val_frac, val_sample, val_reverse, val_weights, val_zeros, val_N)

if data != "test":
    with open(global_path+"/readme.txt", mode = "w") as f:
        f.write(savefile)