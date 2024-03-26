import tensorflow as tf
from losses import correlate, mae_cor, wmae_cor
from tensorflow.keras import layers
from tensorflow.keras.layers import (Conv1D, Dense, Dropout, Flatten,
                                     GlobalAveragePooling1D, Input,
                                     LayerNormalization, MultiHeadAttention,
                                     TimeDistributed, concatenate)
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
    
    elif name == "wCNN_simple5H":
        inp = Input((2001, 4), name="sequence")
        weights = Input((5,), name = "weights")
        true = Input((5,), name = "labels")
        
        x = tf.keras.layers.Conv1D(128, kernel_size = 5,activation ='relu')(inp)
        x = tf.keras.layers.MaxPooling1D(5)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        x = tf.keras.layers.Conv1D(64, kernel_size = 11,activation ='relu')(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        x = tf.keras.layers.Conv1D(32, kernel_size = 21,activation ='relu')(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        out = tf.keras.layers.Dense(5, activation="sigmoid")(x)
        
        model = Model([inp, true, weights], out)
        model.add_loss(wmae_cor(true, out, weights))
        model.compile(optimizer=optimizer,
                    loss=None, metrics=metrics)
        
        return model
    
    elif name == "deep_maxPool2":
        
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv1D(256, kernel_size = 5,activation ='relu', input_shape=input_shape))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())
        
        model.add(tf.keras.layers.Conv1D(128, kernel_size = 7,activation ='relu'))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1D(64, kernel_size=11,activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1D(32, kernel_size=21,activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(2))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(16, activation="relu"))
        model.add(tf.keras.layers.Dense(5, activation="sigmoid"))
        
        model.compile(optimizer=optimizer,
                    loss=loss, metrics=metrics)
        
        return model
    
    elif name == "mnase_mod5H":

        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv1D(128, kernel_size = 5,activation ='relu', input_shape=input_shape))
        model.add(tf.keras.layers.MaxPooling1D(5))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1D(64, kernel_size=11,activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1D(32, kernel_size=21,activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1D(1, kernel_size=1, padding="same", activation='relu'))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Dense(128, activation="relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(64, activation="relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(5, activation="sigmoid"))
        
        model.compile(optimizer=optimizer,
                    loss=loss, metrics=metrics)
        return model

    elif name == "CNN_standard5H":

        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv1D(128, kernel_size = 5,activation ='relu', input_shape=input_shape))
        model.add(tf.keras.layers.MaxPooling1D(5)) # 5 ?
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1D(64, kernel_size=11,activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1D(32, kernel_size=21,activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(2))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(32, activation="relu"))
        model.add(tf.keras.layers.Dense(5, activation="sigmoid"))
        
        model.compile(optimizer=optimizer,
                    loss=loss, weighted_metrics=metrics)
        return model
    
    
    elif name == "CNN_motifs":

        input = tf.keras.layers.Input(shape = input_shape)

        mers5 = tf.keras.layers.Conv1D(32, kernel_size = 5,activation ='relu', padding='same')(input)
        mers9 = tf.keras.layers.Conv1D(32, kernel_size = 9,activation ='relu', padding='same')(input)
        mers11 = tf.keras.layers.Conv1D(32, kernel_size = 15,activation ='relu', padding='same')(input)
        
        x = tf.keras.layers.Concatenate()([mers5, mers9, mers11])
        x = tf.keras.layers.Conv1D(64, kernel_size = 5,activation ='relu', dilation_rate=2, padding='same')(x)
        x = tf.keras.layers.MaxPooling1D(5)(x)
        x = tf.keras.layers.Conv1D(64, kernel_size = 5,activation ='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling1D(5)(x)
        x = tf.keras.layers.Conv1D(1, kernel_size = 5,activation ='relu', padding='same')(x)
        
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        output = tf.keras.layers.Dense(5, activation="sigmoid")(x)
        model = tf.keras.Model(inputs = input, outputs = output)
        
        model.compile(optimizer=optimizer,
                    loss=loss, metrics=metrics)
        return model
    
    elif name == "CNN_directmotifs":
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv1D(256, kernel_size = 11,
                                         activation ='relu',
                                         input_shape=input_shape,
                                         kernel_constraint=tf.keras.constraints.NonNeg(),
                                         padding="same",
                                         kernel_regularizer="l1"))
        
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        
        model.compile(optimizer=optimizer,
                    loss=loss, metrics=metrics)
        return model
        
    elif name == "experimental":
        l1_strength = 0.005

        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv1D(128, kernel_size = 5,activation="exponential", input_shape=input_shape))
        model.add(tf.keras.layers.MaxPooling1D(5)) #5 ?
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1D(64, kernel_size=11,activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1D(32, kernel_size=21,activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(2))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l1(l1_strength)))  # L1 regularization here
        model.add(tf.keras.layers.Dense(5, activation="sigmoid"))  # L1 regularization here
                
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

        model.add(tf.keras.layers.Conv1D(32, kernel_size = 5,activation ='relu', input_shape=input_shape))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1D(64, kernel_size=11,activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1D(128, kernel_size=21,activation='relu'))
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

