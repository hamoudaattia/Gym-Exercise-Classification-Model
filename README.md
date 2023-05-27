# Gym-Exercise-Classification-Model
# Exercise Classification Model

This repository contains code for training an exercise classification model using Flux in Julia. The model can classify exercises into different body parts.

## Description

The exercise classification model is built using the Flux machine learning library in Julia. It takes a dataset of exercise names and their corresponding body parts as input, and learns to classify exercises into different body parts based on their names.

The code follows the following steps:

1. Data Preprocessing: The dataset is loaded from a CSV file ("megaGymDataset.csv"). Any necessary preprocessing steps, such as handling missing values or encoding categorical variables, are performed.

2. Feature Extraction: The exercise names are one-hot encoded to represent them as numerical features. The target body parts are encoded as integers.

3. Model Training: The data is split into training and testing sets. The model architecture is defined using dense layers. The model is trained using the training data and the cross-entropy loss function.

4. Model Evaluation: The trained model is evaluated on the testing set to calculate the accuracy of the predictions.

5. Model Deployment: The trained model can be used to make predictions on new exercise names. An example exercise name is provided, and the model predicts the corresponding body part.

## Dataset

The code assumes the availability of a CSV dataset file named "megaGymDataset.csv". The dataset should contain exercise names in the "Title" column and corresponding body parts in the "BodyPart" column.



