import tensorflow as tf
from losses import mae_cor, correlate
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization
from tensorflow.keras.layers import MultiHeadAttention, TimeDistributed, Conv1D
from tensorflow.keras.layers import Flatten, GlobalAveragePooling1D, concatenate
from tensorflow.keras.models import Model




def create_model(name:str = "CNN_simple", filters= 16,activation = "sigmoid",  input_shape = (2001,4), loss = mae_cor, metrics = ["mae", correlate], optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)):
    """
    return compiled model

    :param str name: name of the architecture
    :param input_shape: input shape as iterable
    :param loss: loss as string (keras function) or object (custom function)
    :param metrics: iterable of string (keras function) and/or objects (custom function)
    :param optimizer: keras optimizer
    """


    if name == "CNN_simple5H":

        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv1D(32, kernel_size = 3,activation ='relu', input_shape=input_shape))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())
        #model.add(tf.keras.layers.Dropout(0.1))

        model.add(tf.keras.layers.Conv1D(32, kernel_size=10,activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())
        #model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Conv1D(32, kernel_size=20,activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())
        #model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(8, activation="relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(5, activation="sigmoid"))
        
        model.compile(optimizer=optimizer,
                    loss=loss, metrics=metrics)
        return model
    
    elif name == "mnase_mod5H":

        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv1D(32, kernel_size = 6,activation ='elu', input_shape=input_shape))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1D(64, kernel_size=11,activation='elu'))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1D(128, kernel_size=21,activation='elu'))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(16, activation="elu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(5, activation="sigmoid"))
        
        model.compile(optimizer=optimizer,
                    loss=loss, metrics=metrics)
        return model


    elif name == "encode":
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv1D(128, kernel_size = 5,padding = "same", activation ='elu', input_shape=input_shape))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1D(64, kernel_size = 5,padding = "same", activation ='elu', input_shape=input_shape))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1D(64, kernel_size=5,padding="same", activation='elu'))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())

        # model.add(tf.keras.layers.Conv1D(128, kernel_size=21, padding = "same", activation='elu'))
        # model.add(tf.keras.layers.MaxPooling1D(2))
        # model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Dense())

        # model.add(tf.keras.layers.Conv1DTranspose(128, kernel_size=250, strides=2, padding='same', activation='elu'))
        # model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1DTranspose(64, kernel_size=5, strides=2, padding='same', activation='elu'))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1DTranspose(64, kernel_size=5, strides=2, padding='same', activation='elu'))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1DTranspose(128, kernel_size=5, strides=2, padding='same', activation='elu'))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1D(1, kernel_size=1, padding='same', activation='sigmoid'))
       
        
        model.compile(optimizer=optimizer,
                    loss=loss, metrics=metrics)

        return model
    

    elif name == "Chemical_5H":
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv1D(64, kernel_size = 5,activation ='relu', input_shape=input_shape))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1D(64, kernel_size=11,activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1D(64, kernel_size=21,activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(8, activation="relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(5, activation="sigmoid"))
        
        model.compile(optimizer=optimizer,
                    loss=loss, metrics=metrics)
        return model
    
    elif name == "lstm":
        model = tf.keras.Sequential()
        model.add(layers.LSTM(128, input_shape=input_shape, return_sequences=True))
        # Dropout layer to prevent overfitting
        model.add(layers.Dropout(0.2))
        # Dense layer with sigmoid activation for binary classification
        model.add(layers.Dense(5, activation='sigmoid'))
        model.compile(optimizer=optimizer,
                    loss=loss, metrics=metrics)
        return model

    elif name=="exptest":

        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.LSTM(256, return_sequences=True, input_shape=input_shape))
        # model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.LSTM(128))
        # model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(5, activation="sigmoid"))
        
        model.compile(optimizer=optimizer,
                    loss=loss, metrics=metrics)
        return model

    else:
        print("no model found")
        return None

