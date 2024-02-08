import tensorflow as tf
import tensorflow.keras.backend as K
from sys import path
path.insert(1, "/home/maxime/data/utils")
from losses import mae_cor, correlate


class RevCompConv1D(tf.keras.layers.Layer):
    def __init__(self) -> None:
        super().__init__()
        pass

    def call(self, x, mask = None):
        rev_comp_W = K.concatenate([self.W, self.W[::-1,:,::-1,::-1]],axis=-1)
        if (self.bias):
            rev_comp_b = K.concatenate([self.b, self.b[::-1]], axis=-1)
        x = K.expand_dims(x, 2)  # add a dummy dimension
        output = K.conv2d(x, rev_comp_W, strides=self.subsample,
                            border_mode=self.border_mode,
                            dim_ordering='tf')
        output = K.squeeze(output, 2)  # remove the dummy dimension
        if self.bias:
            output += K.reshape(rev_comp_b, (1, 1, 2*self.nb_filter))
        output = self.activation(output)
        return output

model = tf.keras.models.Sequential()
model.add(RevCompConv1D(input_shape=(2001,4),
                                                   nb_filter=32,
                                                   filter_length=3))
model.add(rkeras.layers.normalization.RevCompConv1DBatchNorm())
model.add(rkeras.layers.core.Activation("relu"))
model.add(rkeras.layers.convolutional.RevCompConv1D(nb_filter=32,
                                                   filter_length=11))
model.add(rkeras.layers.normalization.RevCompConv1DBatchNorm())
model.add(rkeras.layers.core.Activation("relu"))
model.add(rkeras.layers.convolutional.RevCompConv1D(nb_filter=32,
                                                   filter_length=21))
model.add(rkeras.layers.normalization.RevCompConv1DBatchNorm())
model.add(rkeras.layers.core.Activation("relu"))
model.add(rkeras.layers.pooling.MaxPooling1D(pool_length=10))
model.add(rkeras.layers.convolutional.WeightedSum1D(symmetric=False,
                                                   input_is_revcomp_conv=True,
                                                   bias=False,
                                                   init="fanintimesfanouttimestwo"))
model.add(rkeras.layers.core.DenseAfterRevcompWeightedSum(output_dim=10))
model.add(rkeras.layers.core.Activation("relu"))
model.add(rkeras.layers.core.Dense(output_dim=10))
model.add(rkeras.layers.core.Activation("sigmoid"))
model.compile(optimizer="adam", loss=mae_cor)