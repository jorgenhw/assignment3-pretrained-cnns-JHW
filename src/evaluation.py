#############################
########## IMPORTS ##########
#############################

# For plotting / visualization
import matplotlib.pyplot as plt
import numpy as np

# For classification report
from sklearn.metrics import classification_report

#############################
########## FUNCTIONS ########
#############################

# Save the training and validation plots on drive:
def plot_history(H, epochs):
    plt.style.use("seaborn-colorblind")

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    #plt.show()
    plt.savefig("out/training_and_validation_plots.png")

# Function for getting the predictions from the model.
def get_predictions(model, test_gen):
    predictions = model.predict(test_gen)
    return predictions

# Function for getting the classification report.
def save_classification_report(test_gen, predictions, test_df):
    report = classification_report(test_gen.classes,
                          predictions.argmax(axis=1),
                          target_names=test_df['class_label'].unique())
    # save classification report to txt file
    with open("out/classification_report.txt", "w") as file:
        file.write(report)