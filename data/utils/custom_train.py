print("START-------------------------------------------------")
import tensorflow as tf
tfver = tf.__version__
physical_devices = tf.config.list_physical_devices('GPU') 
for gpu_instance in physical_devices: 
    tf.config.experimental.set_memory_growth(gpu_instance, True)
if len(physical_devices)>1:
    strategy = tf.distribute.MirroredStrategy()

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.mixed_precision import LossScaleOptimizer


from modeles import create_model
from generator_opt import KDNAmulti
from losses import correlate, mae_cor, max_absolute_error, maex2_cor, rmse_cor, rmsle, stretch_cor, mae_cor_ratio

import numpy as np
import pandas as pd
import myfunc as mf

import os
import datetime
day = datetime.datetime.now()

def scheduler(epoch, lr):
        return lr*(0.75**(epoch))
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)





seq = "/home/maxime/data/sequences/mm10/one_hot/chr{}.npz"#"/home/maxime/data/sequences/mm10/masked_one_hot/chr{}.npz"#"/home/maxime/data/sequences/mm10/one_hot/chr{}.npz"
lab = "/home/maxime/data/mnase/labels/IP_data/MNase_bed/multimappers/A/gauss15/chr{}.npz"#"/home/maxime/data/mnase/labels/IP_data/MNase_bed/uniquereads/A/gauss15_maskedlabels/chr{}.npz"#"/home/maxime/data/chemical/labels/smooth_mm10/{}.npz"#"/home/maxime/data/mnase/labels/IP_data/MNase_bed/multimappers/A/counts_norm/chr{}.npz"#"/home/maxime/data/mnase/labels/IP_data/MNase_bed/multimappers/A/counts_npyOK/chr{}.npz"#"/home/maxime/data/mnase/labels/IP_data/MNase_bed/multimappers/A/ppv2/chr{}.npz"#"/home/maxime/data/mnase/labels/IP_data/MNase_bed/uniquereads/A/gauss15_060623/chr{}.npz"#"/home/maxime/data/mnase/labels/IP_data/MNase_bed/multimappers/A/gauss15_q995_025/chr{}.npz"#"/home/maxime/data/chemical/labels/smooth_mm10/{}.npz"
vlab = "/home/maxime/data/mnase/labels/IP_data/MNase_bed/multimappers/A/gauss15/chr{}.npz"#"/home/maxime/data/mnase/labels/IP_data/MNase_bed/uniquereads/A/gauss15_060623/chr{}.npz"
#"/home/maxime/data/mnase/labels/IP_data/MNase_bed/multimappers/A/gauss15/chr{}.npz"
fract = 100_000_000
fracv = 20_000_000


train_chr = (2, 10, 16)#,5,10,11,12,13,14,15,16,17,18,19)
val_chr = (11,12)
epochs = 1
batch_size = 2024

gpath = "/home/maxime/data/paper"
dirname = "test_filters"

winsize = 2001
model_name = "mnase_mod5H"
steps = [-500, -250, 0, 250, 500] if model_name !="encode" else list(range(-1000, 1000, 1))
#steps = range(-500, 500, 2)
loss = mae_cor



opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
if int(tfver.split(".")[1])<11:
    opt = LossScaleOptimizer(opt)
metrics = ["mae", correlate]


if len(physical_devices)>1:
    with strategy.scope():
        model = create_model(model_name,
                            optimizer=opt, 
                            input_shape=(winsize,4),
                            loss = loss, metrics=metrics)
else:
    model = create_model(model_name,
                    optimizer=opt,
                    input_shape=(winsize,4),
                    loss = loss, metrics=metrics)
    
print(model.summary())
os.mkdir(f"{gpath}/{dirname}")
with open(f"{gpath}/{dirname}/readme", mode='w') as f:
    f.write(f"{lab}\n \
            vlab : {vlab}\n \
            model : {model_name}\n \
            loss : {loss}\n \
            train number sequence per epoch : {fract}\n \
            val number sequence per epoch : {fracv}\n \
            train_chr {train_chr}\n \
            val_chr {val_chr}\n \
            epochs {epochs}\n \
            batch size {batch_size}\n \
            reverse mode (train only) : ""\n \
            winsize = {winsize}\n \
            lr_scheduler : reduce lr on plateau 0.1n \
            index_selection : none\n \
            generator : \n \
            self.lab[np.convolve(n_pos,np.ones(self.winsize), mode = \"same\")!=0] = 0 \n \
            binidx = np.convolve(self.lab, np.ones(1000), \"same\")\n \
            self.index = np.where((self.lab!=0) & (binidx>=500))[0]")
        
    
print("tf version : {}".format(tfver))

val_gen = KDNAmulti(seq         = [seq.format(x) for x in val_chr],
                lab             = [vlab.format(x) for x in val_chr],
                winsize         = winsize,
                frac            = fracv,
                headsteps       = steps,
                weights         = False,
                batch_size      = batch_size,
                index_selection = "validation")


train_gen = KDNAmulti(seq       = [seq.format(x) for x in train_chr],
                lab             = [lab.format(x) for x in train_chr],
                winsize         = winsize,
                frac            = fract,
                reverse         = "",
                headsteps       = steps,
                weights         = True,
                batch_size      = batch_size,
                index_selection = "")

mcp_save = ModelCheckpoint( f"{gpath}/{dirname}/best.h5",
                           save_best_only=True, monitor='val_loss', mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=3, min_lr=0.00001)
earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=7, restore_best_weights=True)

history = model.fit(train_gen, epochs=epochs,validation_data=val_gen, verbose = 1,
                    callbacks=[mcp_save, earlystop, reduce_lr])

model.save(f"{gpath}/{dirname}/model")
hist_df = pd.DataFrame(history.history)

with open(f"{gpath}/{dirname}/history", mode='w') as f:
    hist_df.to_csv(f)
