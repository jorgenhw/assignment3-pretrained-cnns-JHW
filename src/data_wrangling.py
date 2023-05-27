#########################
######## IMPORTS ########
#########################

import json
import pandas as pd
import os

# Tensorflow imports
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#########################
####### FUNCTIONS #######
#########################

## Function to convert JSON data to a dataframe
def make_dataframe_from_json(json_path):
    # Load JSON data into a list of dictionaries
    data = []
    with open(json_path) as f:
        for line in f:
            data.append(json.loads(line))

    # Convert list of dictionaries to a dataframe
    return pd.DataFrame(data)

## Function to convert image paths to absolute paths
def convert_image_path(image_path):
    base_dir = os.path.join("..","431824")
    print(os.path.join(base_dir, image_path))
    return os.path.join(base_dir, image_path)


## Function defining the image data generator
def define_data_generator(horizontal_flip=True):
    train_datagen = ImageDataGenerator(horizontal_flip=horizontal_flip)
    val_datagen = ImageDataGenerator(horizontal_flip=horizontal_flip)
    test_datagen = ImageDataGenerator(horizontal_flip=horizontal_flip)
    return train_datagen, val_datagen, test_datagen

## Function to create the data generators
def create_data_generators(train_df, val_df, test_df, # the json datasets
                           train_datagen, val_datagen, test_datagen, # the data generators
                           img_size = (224, 224), batch_size = 128, # image size and batch size (can be changed using argparse)
                           seed = 42): # seed for reproducibility
    train_gen = train_datagen.flow_from_dataframe(
        train_df, # json filen
        x_col = 'image_path', # path to images
        y_col = 'class_label', # class labels
        target_size=img_size, # image size
        batch_size=batch_size, # batch size
        class_mode='categorical', # class mode: categorical since we have more than 2 classes
        seed = seed # seed for reproducibility
    )

    val_gen = val_datagen.flow_from_dataframe(
        val_df,
        x_col = 'image_path',
        y_col = 'class_label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        seed = seed
    )

    test_gen = test_datagen.flow_from_dataframe(
        test_df,
        x_col = 'image_path',
        y_col = 'class_label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        seed = seed
    )
    return train_gen, val_gen, test_gen