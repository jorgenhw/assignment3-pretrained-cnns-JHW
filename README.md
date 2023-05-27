<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h1 align="center">Cultural Datascience 2023</h1> 
  <h2 align="center">Assignment 3</h2> 
  <h3 align="center">Visual Analytics</h3> 


  <p align="center">
    Jørgen Højlund Wibe<br>
    Student number: 201807750
  </p>
</p>


<!-- ABOUT THE PROJECT -->
## About the project
This assignment deals with image classification within the domain of fashion using a pretrained convolutional neural network (CNN). We are training a classifier that enables us to identify categories for different types of clothing items: 15 different categories distributed over 106.000 images.

First, we augment the images (to have more images to train on and avoid overfitting) and load them into a data generator. A data generator provides an efficient way to load and preprocess data on-the-fly, which is especially useful when working with large datasets that may not fit into memory. It also reduces training time.

Then we load the pretrained CNN, VGG16, into our environment. Since VGG16 is a pretrained CNN we can take advantage of the feature extraction capabilities it was pretrained on. However, we are removing the top layer of the model to finetune it on our new classification task at hand.

After these steps, we train our model on the data and saves to the drive 1) a plot showing the training and validation process during training and 2) a classification report showing how the model performed on a test set.

<!-- USAGE -->
## Usage

To use or reproduce the results you need to adopt the following steps.

**NOTE:** There may be slight variations depending on the terminal and operating system you use. The following example is designed to work using the Visual Studio Code version 1.77.3 (Universal). The terminal code should therefore work using a unix-based bash. The avoid potential package conflicts, the ```setup.sh``` bash files contains the steps necesarry to create a virtual environment, install libraries and run the project.

1. Clone repository
2. Fetch data
2. Run setup.sh

### Clone repository

Clone repository using the following lines in the unix-based bash:

```bash
git clone https://github.com/jorgenhw/assignment3-pretrained-cnns-JHW.git
cd assignment3-pretrained-cnns-JHW
```

### Data
Due to the large size of the dataset, we are unable to include it in this repository. However, you can download the dataset from Kaggle by following [this link](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset). Remember to adjust the data directory paths in the ```main.py``` script to point to the location of the data.

### Run ```setup.sh```

To replicate the results, I have included a bash script that automatically 

1. Creates a virtual environment for the project
2. Activates the virtual environment
3. Installs the correct versions of the packages required
4. Runs the script
5. Deactivates the virtual environment

Run the code below in your bash terminal:

```bash
bash setup.sh
```

### Changing arguments via ```argparse```
To provide more flexibility and enable the user to change the parameters of the script from the command line, we have implemented argparse in our script. This means that by running the script with specific command line arguments, you can modify parameters such as the batch size, the number of epochs to train the model, and the learning rate.

To see all the available arguments, simply run the command:

```bash
python main.py --help
```
This will display a list of all the available arguments and their default values.



## Inspecting results

A classification report and training and validation plots are located in the folder ```out```. Here one can inspect the results.

<!-- REPOSITORY STRUCTURE -->
## Repository structure

This repository has the following structure:
```
│   main.py
│   README.md
│   requirements.txt
│   setup.sh
│
├───data
│       empty folder [put your data here]
│
├───out
│       classification_report.txt
│       training_and_validation_plots.png
│
└──src
        data_wrangling.py
        classifier.py
        evaluation.py
```


<!-- DATA -->

## Data
The image data used in this assignment was originally part of a research paper by Pranjal Singh Rajput, Shivangi Aneja which can be found [here](https://arxiv.org/abs/2104.02830). It is now freely available through [Kaggle](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset).

Due to the large size of the dataset, we are unable to include it in this repository. However, you can download the dataset from Kaggle by following the link provided above. Remember to adjust the data directory paths in the ```main.py``` script to point to the location of the data.

<!-- RESULTS -->

## Remarks on findings
Overall, this project demonstrates the use of a pretrained CNN for image classification and the effectiveness of data augmentation and data generators in improving training efficiency. However, it should be clearly stated that this project project is for educational purposes only. Nothing has been done to find the optimal hyper parameters such as batch size, number of epochs, or learning rate.
