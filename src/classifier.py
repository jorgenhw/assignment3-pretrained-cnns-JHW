##################################
############ IMPORTS #############
##################################

# tensorflow
import tensorflow as tf

# VGG16 model
from tensorflow.keras.applications.vgg16 import (VGG16)

# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     BatchNormalization)

# generic model object
from tensorflow.keras.models import Model

# optimizers
from tensorflow.keras.optimizers import SGD


##################################
############ FUNCTIONS ###########
##################################

# load the model without classifier layers
def load_model(model = VGG16, 
               include_top=False, # false = no classifier
               pooling = 'avg', # Pooling method
               input_shape=(32, 32, 3)): # input shape of the images
    model = model(include_top=include_top, 
              pooling=pooling, 
              input_shape=input_shape)
    # Mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    return model

# adding new classifier layers to the model
def add_classifier_layers(model, nodes1 = 256, nodes2 = 128, classes = 15):
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output) # 
    bn = BatchNormalization()(flat1)
    class1 = Dense(nodes1, # number of nodes
                   activation='relu')(bn)
    class2 = Dense(nodes2, # number of nodes
                   activation='relu')(class1)
    output = Dense(classes, # number of classes
                   activation='softmax')(class2)
    
    # defining the new and modified model
    model = Model(inputs=model.inputs, 
                  outputs=output)
    return model


# Function to compile the model using keras
def compile_model(model, learning_rate=0.01):
    # compile model
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=10000,
        decay_rate=0.9)
    sgd = SGD(learning_rate=lr_schedule) 

    model.compile(optimizer=sgd,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model

# Function to train the model (fits the model on batches with real-time data augmentation)
def train_model(model, train_gen, val_gen, epochs=5):
    H = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=epochs)
    return H, epochs # returns the history of the model (H=history)

