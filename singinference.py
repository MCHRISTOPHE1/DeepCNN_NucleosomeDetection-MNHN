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
from generator import Signalgenerator
from losses import correlate, mae_cor

import numpy as np
import pandas as pd

slab = ["/home/maxime/data/mnase/labels/uniquereads/A/1/chr1.npy",
        "/home/maxime/data/mnase/labels/uniquereads/A/1/chr2.npy",
        "/home/maxime/data/mnase/labels/uniquereads/A/1/chr3.npy"]


sseq = ["/home/maxime/data/sequences/mm10/one_hot/onehot_chr1.npz",
        "/home/maxime/data/sequences/mm10/one_hot/onehot_chr2.npz",
        "/home/maxime/data/sequences/mm10/one_hot/onehot_chr3.npz"]

winsize=2001

loss = mae_cor
metrics = ["mse", "mae", correlate]
batch = 1024
with strategy.scope():
    model = create_model("signal", optimizer=opt, input_shape=(winsize,4), loss = loss, metrics=metrics)
    model.summary()


train_gen = Signalgenerator(sseq, slab, apply_weights=0, frac=0.1)
train_idx = train_gen.generate_split_indexes()
train = train_gen.generate_images(train_idx, is_training = True, batch_size = batch)
spe = len(train_idx)//batch

history = model.fit(train, batch_size=batch, steps_per_epoch=spe, epochs=3, verbose=1)
model.save("/home/maxime/data/mnase/train/singalInference/fulltrain_123_maecor")
