import tensorflow as tf
from losses import mae_cor, correlate



def create_model(name:str = "CNN_simple", filters= 16,activation = "sigmoid",  input_shape = (2001,4), loss = mae_cor, metrics = ["mae", correlate], optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)):
    """
    return compiled model

    :param str name: name of the architecture
    :param input_shape: input shape as iterable
    :param loss: loss as string (keras function) or object (custom function)
    :param metrics: iterable of string (keras function) and/or objects (custom function)
    :param optimizer: keras optimizer
    """
    if name == "CNN_simple":
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv1D(filters, kernel_size = 3,activation ='relu', input_shape=input_shape))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Conv1D(filters, kernel_size=10,activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Conv1D(filters, kernel_size=20,activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1, activation=activation))

        model.compile(optimizer=optimizer,
                    loss=loss, metrics=metrics)
        return model

    elif name == "CNN_simple5H":

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

    elif name == "Chemical":
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv1D(64, kernel_size = 3,activation ='relu', input_shape=input_shape))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1D(64, kernel_size=10,activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1D(64, kernel_size=20,activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(16, activation="relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        
        model.compile(optimizer=optimizer,
                    loss=metrics, metrics=metrics)
        return model

    elif name == "dilated":

        """

        inputs = tf.keras.Input(input_shape)
        conv1 = tf.keras.layers.Conv1D(32, kernel_size = 5, activation="relu", padding = "same")(inputs)
        convd1 = tf.keras.layers.Conv1D(32, kernel_size =5, activation = "relu", dilation_rate = 20, padding = "same")(inputs)
        concat = tf.keras.layers.Concatenate()([conv1, convd1])
        concat = tf.keras.layers.Conv1D(32, kernel_size = 5, activation = "relu")(concat)
        concat = tf.keras.layers.MaxPooling1D(2)(concat)
        concat = dense = tf.keras.layers.BatchNormalization()(concat)
        concat = tf.keras.layers.Conv1D(32, kernel_size = 5, activation = "relu")(concat)
        concat = tf.keras.layers.MaxPooling1D(2)(concat)
        concat = dense = tf.keras.layers.BatchNormalization()(concat)
        concat = tf.keras.layers.Conv1D(32, kernel_size = 5, activation = "relu")(concat)
        concat = tf.keras.layers.MaxPooling1D(2)(concat)
        concat = dense = tf.keras.layers.BatchNormalization()(concat)
        concat = tf.keras.layers.Conv1D(32, kernel_size = 5, activation = "relu")(concat)
        concat = tf.keras.layers.MaxPooling1D(2)(concat)
        concat = dense = tf.keras.layers.BatchNormalization()(concat)
        concat = tf.keras.layers.Conv1D(32, kernel_size = 5, activation = "relu")(concat)
        concat = tf.keras.layers.MaxPooling1D(2)(concat)
        concat = dense = tf.keras.layers.BatchNormalization()(concat)

        flat = tf.keras.layers.Flatten()(concat)
        dense =  tf.keras.layers.Dense(8, activation="relu")(flat)
        dense = tf.keras.layers.BatchNormalization()(dense)
        outputs = tf.keras.layers.Dense(1, activation="relu")(dense)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        """
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv1D(32, kernel_size = 3,activation ='relu', input_shape=input_shape))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Conv1D(32, kernel_size=10,activation='relu', dilation_rate=2))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Conv1D(32, kernel_size=20,activation='relu', dilation_rate=2))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1, activation=activation))

        model.compile(optimizer=optimizer, loss=metrics, metrics=metrics)
        return model
    
    elif name == "MNase_simple":
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
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        
        model.compile(optimizer=optimizer,
                    loss=loss, metrics=metrics)
        return model

    elif name == "bases_inference":
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
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        
        model.compile(optimizer=optimizer,
                    loss=metrics, metrics=metrics)
        return model

    elif name == "Quick_train":
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv1D(16, kernel_size = 3,activation ='relu', input_shape=input_shape))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())
        #model.add(tf.keras.layers.Dropout(0.1))

        model.add(tf.keras.layers.Conv1D(16, kernel_size=10,activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())
        #model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Conv1D(16, kernel_size=20,activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())
        #model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(16, activation="relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        
        model.compile(optimizer=optimizer,
                    loss=metrics, metrics=metrics)
        return model

    elif name == "test":
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv1D(32, kernel_size = 25,activation ='relu', input_shape=input_shape))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1D(32, kernel_size=3,dilataion_rate=2,activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1D(32, kernel_size=3,dilataion_rate=4,activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
    
        model.add(tf.keras.layers.Conv1D(32, kernel_size=3,dilataion_rate=8,activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1D(32, kernel_size=3,dilataion_rate=16,activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1D(32, kernel_size=3,dilataion_rate=32,activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        
        model.compile(optimizer=optimizer,
                    loss=metrics, metrics=metrics)
        return model

    elif name == "chem_simple":
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv1D(64, kernel_size = 3, input_shape=input_shape))
        model.add(tf.keras.layers.Conv1D(64, kernel_size = 3,activation ='relu'))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())
        #model.add(tf.keras.layers.Dropout(0.1))

        model.add(tf.keras.layers.Conv1D(64, kernel_size=10))
        model.add(tf.keras.layers.Conv1D(64, kernel_size=10, activation = "relu"))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())
        #model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Conv1D(64, kernel_size=20))
        model.add(tf.keras.layers.Conv1D(64, kernel_size=20,activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())
        #model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(8, activation="relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        
        model.compile(optimizer=optimizer,
                    loss=metrics, metrics=metrics)
        return model
    
    elif name =="CNN_deep":
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv1D(32, kernel_size = 3,activation ='relu', input_shape=input_shape))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Conv1D(32, kernel_size=11,activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Conv1D(32, kernel_size=21,activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        model.compile(optimizer=optimizer,
                    loss=loss, metrics=metrics)
        return model

    elif name == "signal":
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv1D(32, kernel_size = 5,activation ='relu', input_shape=input_shape))
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
        model.add(tf.keras.layers.Dense(1000, activation="sigmoid"))
        
        model.compile(optimizer=optimizer,
                    loss=loss, metrics=metrics)
        return model

    else:
        raise IndexError("Unknown model")