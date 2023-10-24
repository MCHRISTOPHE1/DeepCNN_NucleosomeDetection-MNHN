print("START-------------------------------------------------")
import tensorflow as tf
tfver = tf.__version__
physical_devices = tf.config.list_physical_devices('GPU') 
for gpu_instance in physical_devices: 
    tf.config.experimental.set_memory_growth(gpu_instance, True)
strategy = tf.distribute.MirroredStrategy()


from losses import correlate, mae_cor

import numpy as np
import myfunc as mf
from scipy.sparse import csr_matrix

import argparse
import datetime
import time
import sys

day = datetime.datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument("seq", help = 'path to one-hot dna sequence', type =str)
parser.add_argument("model",help = "path to trained model (mae_cor loss)")
parser.add_argument("output", help = "path for output file",)
parser.add_argument("--step", help = "distance between windows", default=500, type = int, required =False)
parser.add_argument("--start", help = "start of saliceny def = 3M", default=3_000_000, type = int, required=False)
parser.add_argument("--winsize", help = "inputsize of the model", default = 2001, type=int, required=False)
parser.add_argument("--batch", help = "number of sequence per batch", default = 2048, type = int, required = False)
args = parser.parse_args()

seq = mf.loadnp(args.seq).astype("float32")
model = tf.keras.models.load_model(args.model, custom_objects={"mae_cor":mae_cor, "correlate":correlate})
matsize = len(seq)
erange = (args.batch*args.step)+args.winsize-args.winsize%2
mat = csr_matrix((1, matsize), dtype="float32")
end = len(seq)-args.winsize
t0=time.time()

col = np.tile(
        np.arange(args.winsize),
        args.batch).reshape((args.batch, args.winsize)) + \
    np.arange(0, args.batch*args.step, args.step).reshape((args.batch, 1))+args.start

for pos in range(args.start, end, args.batch*args.step):
    mf.loadbar(pos, end, t0)
    seqs = mf.sliding_window_view(
            seq[pos:pos+erange], (args.winsize, 4)
            ).reshape((-1,args.winsize,4))[::500,...]
    seqs = tf.convert_to_tensor(seqs)
    
    with tf.GradientTape() as tape:
        tape.watch(seqs)
        predictions = model(seqs, training=False)
        ggrad = tape.gradient(predictions, seqs).numpy()
    ggrad -= np.mean(ggrad, keepdims=True, axis=-1)
    ggrad = np.sum(np.abs(ggrad), axis=-1)

    
    #print(ggrad.shape, col.shape, col[:2, :])
    try:
        tmp = csr_matrix((ggrad.ravel(), 
                        (np.zeros(ggrad.size), col.ravel())),
                            shape=(1, matsize), dtype="float32")
        mat += tmp
        col += args.step*args.batch 
    except ValueError:
        continue
np.savez_compressed(args.output, mat.tocsc().toarray().ravel())
print(f"saved at {args.output}")

# import tensorflow as tf
# from tensorflow.distribute import MirroredStrategy
# from tensorflow.data.experimental import AUTOTUNE
# from scipy.sparse import csr_matrix
# import numpy as np
# import argparse
# import datetime
# import time

# import myfunc as mf
# from losses import correlate, mae_cor

# day = datetime.datetime.now()

# parser = argparse.ArgumentParser()
# parser.add_argument("seq", help='path to one-hot dna sequence', type=str)
# parser.add_argument("model", help="path to trained model (mae_cor loss)")
# parser.add_argument("output", help="path for output file")
# parser.add_argument("--step", help="distance between windows", default=500, type = int, required =False)
# parser.add_argument("--start", help="start of saliceny def = 3M", default=3_000_000, type=int, required=False)
# parser.add_argument("--winsize", help="input size of the model", default=2001, type=int, required=False)
# parser.add_argument("--batch", help="number of sequences per batch", default=2048, type=int, required=False)
# args = parser.parse_args()

# tf.config.set_visible_devices([], 'GPU')  # Désactiver l'utilisation du GPU pour les prétraitements

# strategy = MirroredStrategy()

# def preprocess_seq(seq):
#     seqs = tf.convert_to_tensor(
#         mf.sliding_window_view(
#             seq, (args.winsize, 4)
#         ).reshape((-1, args.winsize, 4)).astype("float32")
#     )
#     return seqs

# def compute_gradients(seqs):
#     with tf.GradientTape() as tape:
#         tape.watch(seqs)
#         predictions = model(seqs, training=False)
#         ggrad = tape.gradient(predictions, seqs).numpy()
#     return ggrad

# def process_batch(seqs):
#     ggrad = strategy.run(compute_gradients, args=(seqs,))
#     ggrad = tf.concat(ggrad.values, axis=0)
#     return ggrad

# seq = mf.loadnp(args.seq)
# model = tf.keras.models.load_model(args.model, custom_objects={"mae_cor": mae_cor, "correlate": correlate})
# matsize = len(seq)
# erange = args.batch + args.winsize + args.winsize % 2
# mat = csr_matrix((1, matsize), dtype="float32")
# end = len(seq) - args.winsize
# t0 = time.time()

# dataset = tf.data.Dataset.from_tensor_slices(seq)
# preprocessed_dataset = dataset.map(preprocess_seq, num_parallel_calls=AUTOTUNE)

# for pos in range(args.start, end, args.batch):
#     mf.loadbar(pos, end, t0)

#     batch_seqs = preprocessed_dataset.skip(pos).take(args.batch)

#     ggrad = strategy.run(process_batch, args=(batch_seqs,))
#     ggrad = tf.concat(ggrad.values, axis=0)
    
#     ggrad -= tf.reduce_mean(ggrad, keepdims=True, axis=-1)
#     ggrad = tf.reduce_sum(tf.abs(ggrad), axis=-1)

#     col = tf.tile(
#         tf.range(ggrad.shape[1]),
#         [ggrad.shape[0]]
#     ) + tf.expand_dims(tf.range(ggrad.shape[0]), axis=1) + pos

#     tmp = csr_matrix(
#         (ggrad.numpy().ravel(), (np.zeros(ggrad.size), col.numpy().ravel())),
#         shape=(1, matsize), dtype="float32"
#     )
#     mat += tmp

# np.savez_compressed(args.output, mat.tocsc().toarray().ravel())
# print(f"saved at {args.output}")
