# -*- coding: utf-8 -*-
"""
Created on Sat May  3 17:34:28 2025

@author: cuih1
"""

import os
import pandas as pd
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the provided CSV files
train_df = pd.read_csv('train.csv')
taxonomy_df = pd.read_csv('taxonomy.csv')
sample_submission_df = pd.read_csv('sample_submission.csv')

# Merge train data with taxonomy data using 'primary_label'
train_merged_df = pd.merge(train_df, taxonomy_df, on='primary_label', how='left')

# Helper function to extract audio features using librosa
def extract_audio_features(file_path):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # Extract chromagram features (You can add more features if needed)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Extract MFCC features (Optional, can be used as additional features)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        # Combine features into a single vector
        return np.hstack([chroma_mean, mfcc_mean])
    except Exception as e:
        print(f"Error extracting features for {file_path}: {e}")
        return np.zeros(12)  # Return a zero vector in case of error

# Extract features from the audio files in the training dataset
features = []
labels = []

# Assuming that audio files are stored in a directory 'audio_train/'
for idx, row in train_merged_df.iterrows():
    file_path = os.path.join('audio_train', row['filename'])
    audio_features = extract_audio_features(file_path)
    features.append(audio_features)
    labels.append(row['scientific_name_x'])  # You can change the label if needed

# Convert features and labels to NumPy arrays
features = np.array(features)
labels = np.array(labels)

# Encode labels using LabelEncoder (so we can train a classifier)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, encoded_labels, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Evaluate the model
y_pred = classifier.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

# Prepare the submission file
submission = sample_submission_df.copy()

# Assuming test data is stored in a 'test_audio' folder and using the same extraction method
test_features = []
for idx, row in sample_submission_df.iterrows():
    file_path = os.path.join('audio_test', row['row_id'] + '.ogg')  # Assuming test audio filenames match row_id
    test_audio_features = extract_audio_features(file_path)
    test_features.append(test_audio_features)

# Convert test features to NumPy array
test_features = np.array(test_features)

# Predict on the test set
test_predictions = classifier.predict(test_features)

# Map predictions back to species names
predicted_species = label_encoder.inverse_transform(test_predictions)

# Fill the submission file with predictions
for idx, species in enumerate(predicted_species):
    submission.iloc[idx, 1:] = species  # The columns for predictions start from index 1 onward

# Save the submission file
submission.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")
