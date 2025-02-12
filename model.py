import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model


def createModelSuit():
    model = Sequential([
        layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (35,25,1)),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(32, (3,3), activation = 'relu'),
        layers.MaxPooling2D((2,2)),
        
        layers.Flatten(),
        layers.Dense(64, activation = 'relu'),
        layers.Dense(4, activation = 'softmax')
    ])
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.summary()
    return model

def createModelNumber():
    #modelv2.keras
    model = Sequential()

    model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (35,25,1)))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(32, (3,3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dense(13, activation = 'softmax'))


    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    #model.summary()
    return model

def loadModel(modelName):
    reconstructed_model = load_model(modelName)
    return reconstructed_model