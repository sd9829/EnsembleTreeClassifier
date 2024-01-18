# EnsembleTreeClassifier

EnsembleTreeClassifier is a Python script that implements a combination of Decision Trees and AdaBoost for classification tasks. This ensemble approach enhances the performance of decision trees through boosting.

# Usage

To train the model, use the following command:

  bash: python EnsembleTreeClassifier.py train training_data.txt trained_hypothesis.pkl ada/dt

training_data.txt: The file containing the training data.
trained_hypothesis.pkl: The output file where the trained model will be saved.
ada/dt: Specify the learning type as ada for AdaBoost or dt for a regular Decision Tree.

Prediction

To make predictions using the trained model, use the following command:

  bash: python EnsembleTreeClassifier.py predict trained_hypothesis.pkl test_data.txt

trained_hypothesis.pkl: The pre-trained model file generated during training.
test_data.txt: The file containing the data to be classified.

# Training Example

  bash: python EnsembleTreeClassifier.py train training_data.txt trained_hypothesis.pkl ada

This command trains the model using AdaBoost on the data in training_data.txt and saves the trained model to trained_hypothesis.pkl.

# Prediction Example

  bash: python EnsembleTreeClassifier.py predict trained_hypothesis.pkl test_data.txt

This command uses the pre-trained model in trained_hypothesis.pkl to classify the data in test_data.txt.

# Notes

The script relies on external libraries, so make sure to have them installed (math, pickle, sys).
The effectiveness of the model depends on the quality of the training data and the chosen learning type (ada or dt).
Ensure that the provided data files (training_data.txt and test_data.txt) have the required format for the script to work correctly.

Feel free to experiment with different datasets and parameters to customize the model for your specific classification tasks.
