####### IMPORTS #######
# Built-in imports
import src.data_wrangling as dw
import src.classifier as classifier
import src.evaluation as eva

# other imports
import argparse # For parsing arguments
import os # For defining paths

######## MAIN ########

def main(args):
    ## DATA WRANGLING ##

    # Load JSON data into dataframes
    train_df = dw.make_dataframe_from_json(os.path.join("data", "train_data.json"))
    val_df = dw.make_dataframe_from_json(os.path.join("data", "val_data.json"))
    test_df = dw.make_dataframe_from_json(os.path.join("data", "test_data.json"))

    # Add absolute path to image path
    test_df['image_path'] = test_df['image_path'].apply(dw.convert_image_path)
    val_df['image_path'] = val_df['image_path'].apply(dw.convert_image_path)
    train_df['image_path'] = train_df['image_path'].apply(dw.convert_image_path)

    # Defining the data generators
    train_datagen, val_datagen, test_datagen = dw.define_data_generator(horizontal_flip=args.horizontal_flip)

    # Creating the data generators
    train_gen, val_gen, test_gen = dw.create_data_generators(train_df, val_df, test_df, train_datagen, val_datagen, test_datagen, 
                                                             img_size=args.img_size, batch_size=args.batch_size, seed=args.seed)

    ## MODEL PREPERATION ##

    ## Load model
    model = classifier.load_model(model=args.model,include_top=args.include_top, pooling=args.pooling, input_shape=args.input_shape)

    # adding new classifier layers to the model
    model = classifier.add_classifier_layers(model,
                                             nodes1=args.nodes1,nodes2=args.nodes2,classes=args.classes)

    # Compile the model using keras
    model = classifier.compile_model(model,
                                     learning_rate=args.learning_rate)

    ## MODEL TRAINING ##

    # Initiate training of model
    H, epochs = classifier.train_model(model, train_gen, val_gen, epochs=args.epochs)

    ## MODEL EVALUATION ##

    # Saving training and validation plots to drive
    eva.plot_history(H, epochs)

    # Get predictions from the test data
    predictions = eva.get_predictions(model, test_gen)

    # Save classification report to txt file
    eva.save_classification_report(test_gen, predictions, test_df)

# Parse arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script trains a model to classify images of different clothing items. Type --help to see all the possible arguments.')
    
    # Arguments for data wrangling
    parser.add_argument('--horizontal_flip', default=True, type=bool, help='Whether to use horizontal flip augmentation')
    parser.add_argument('--img_size', default=(224, 224), type=tuple, help='Image size of input images')
    parser.add_argument('--seed', default=42, type=int, help='seed for reproducibility')
    
    # Arguments for defining the model
    parser.add_argument('--model', default='VGG16', type=str, help='Model to use for the classifier')
    parser.add_argument('--include_top', default=False, type=bool, help='Whether to include the top layers of the model')
    parser.add_argument('--pooling', default='avg', type=str, help='Pooling method to use for the model')
    parser.add_argument('--input_shape', default=(32, 32, 3), type=tuple, help='Image input shape to use for the model')
    
    # Arguments for changing the classifier layer of the model
    parser.add_argument('--nodes1', default=256, type=int, help='Number of nodes in the first classifier layer')
    parser.add_argument('--nodes2', default=128, type=int, help='Number of nodes in the second classifier layer')
    parser.add_argument('--classes', default=15, type=int, help='Number of classes to classify')

    # Setting batch size, number of epochs and learning rate
    parser.add_argument('--learning_rate', default=0.01, type=float, help='Initial learning rate to use for the model')
    parser.add_argument('--epochs', default=5, type=int, help='Number of epochs to train the model')

    args = parser.parse_args()
    main(args)