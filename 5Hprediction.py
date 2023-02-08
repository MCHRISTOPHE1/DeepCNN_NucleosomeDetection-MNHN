print("START-------------------------------------------------")
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
from generator import DNAmulti5H
from losses import correlate, mae_cor

import numpy as np
import pandas as pd


sequence = "/home/maxime/data/sequences/mm10/chr6.npy"
label = "/home/maxime/data/mnase/labels/IP_data/MNase_bed/multimappers/A/counts_npy/chr6.npy"
winsize=2001
loss = mae_cor
metrics = ["mae", correlate]
batch = 2048

with strategy.scope():
    model = create_model("CNN_simple5H", optimizer=opt, input_shape=(winsize,4), loss = loss, metrics=metrics)
    model.summary()



train_gen = DNAmulti5H(seq=[sequence],
                    lab=[label], winsize=winsize, frac=1)
train_idx = train_gen.generate_split_indexes()
train = train_gen.generate_images(train_idx, is_training = True, batch_size = batch)
spe = len(train_idx)//batch

model.fit(train, batch_size=batch, steps_per_epoch=spe, epochs = 3)