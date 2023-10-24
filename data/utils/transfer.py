
import tensorflow as tf
from generator_opt import DNAmulti5H
import pandas as pd



physical_devices = tf.config.list_physical_devices('GPU') 
for gpu_instance in physical_devices: 
    tf.config.experimental.set_memory_growth(gpu_instance, True)
strategy = tf.distribute.MirroredStrategy()

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.mixed_precision import LossScaleOptimizer



opt = tf.keras.optimizers.Adam(learning_rate=0.001)
opt = LossScaleOptimizer(opt)

from modeles import create_model
from generator_opt import DNAmulti5H
from losses import correlate, mae_cor, max_absolute_error

with strategy.scope():
    base_model = tf.keras.models.load_model("/home/maxime/data/mnase/train/mA_1/model", custom_objects={"mae_cor":mae_cor, "correlate":correlate})
    submodel = tf.keras.Sequential()
    for l in base_model.layers:
         submodel.add(l)
    submodel.trainable=False

    input = tf.keras.Input((2001,4))
    x = submodel(input, training=False)
    # x = tf.keras.layers.Dense(8)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    output = tf.keras.layers.Dense(5)(x)
    model = tf.keras.Model(input, output)

    model.compile(optimizer = opt, loss = mae_cor, metrics = [mae_cor, "mae", correlate])
    model.summary()


gen = DNAmulti5H(["/home/maxime/data/sequences/mm10/one_hot/chr2.npz"],
                 lab=["/home/maxime/data/mnase/labels/IP_data/MNase_bed/multimappers/A/gauss15/chr2.npy"],
                  winsize = 2001, frac=0.3, zeros=False, sample = True, apply_weights= True, reverse=3, N=0, truncate = 3_000_000)

idx = gen.generate_split_indexes()
gen_train = gen.generate_images(idx, batch_size=4096, is_training=True)


genv = DNAmulti5H(["/home/maxime/data/sequences/mm10/one_hot/chr7.npz"],
                 lab=["/home/maxime/data/mnase/labels/IP_data/MNase_bed/multimappers/A/gauss15/chr7.npy"],
                  winsize = 2001, frac=0.1, zeros=False, sample = True, apply_weights= False, reverse=3, N=0, truncate = 3_000_000)

vidx = genv.generate_split_indexes()
gen_val= gen.generate_images(vidx, batch_size=4096, is_training=True)


history = model.fit(gen_train, batch_size=4096, steps_per_epoch=len(idx)//4096,
                         validation_data = gen_val, validation_steps = len(vidx)//4096, validation_batch_size = 4096,
                            epochs=10, verbose=1)

history.history()
hist_df = pd.DataFrame(history.history)
with open("/home/maxime/test_transfer.csv", mode='w') as f:
        hist_df.to_csv(f)

