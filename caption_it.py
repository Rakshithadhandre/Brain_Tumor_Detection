import os
import librosa
import numpy as np
from keras.models import load_model
import os
import numpy as np

# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import tensorflow_hub as hub

# Load the saved model
model_path = '1dcnnmodel.h5'  # Replace with your model's path
loaded_model = load_model(model_path)

model = load_model(('brain_tumor_detector.h5'), custom_objects={'KerasLayer': hub.KerasLayer})

# Define a function to extract and preprocess features from a new audio file
def extract_and_preprocess_features(audio_path, expected_shape):
    try:
        img_path="./static/Y1.jpg"
        test_image = image.load_img(img_path, target_size = (240,240))
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255
        test_image = np.expand_dims(test_image, axis=0)
        print(test_image)
        result = model.predict(test_image)

        print(result)
        if result <= 0.5:
            result = "The Person has no Brain Tumor"
        else:
            result = "The Person has Brain Tumor"

        return result
        # Load the new audio file
        audio_data, sample_rate = librosa.load(audio_path, sr=None)

        # Extract MFCCs, ZCR, Mel spectrogram, and Chroma features (adjust as needed)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=20)
        zcr = librosa.feature.zero_crossing_rate(y=audio_data)
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)

        # Calculate the mean of each feature
        mfcc_mean = np.mean(mfccs, axis=1)
        zcr_mean = np.mean(zcr, axis=1)
        mel_mean = np.mean(mel_spectrogram, axis=1)
        chroma_mean = np.mean(chroma, axis=1)


        # Combine the extracted features into a single feature vector
        combined_features = np.concatenate((mfcc_mean, zcr_mean, mel_mean, chroma_mean))

        # Ensure that the feature vector shape matches the expected shape
        if combined_features.shape[0] < expected_shape[0]:
            # Pad the feature vector with zeros
            padding = np.zeros(expected_shape[0] - combined_features.shape[0])
            combined_features = np.concatenate((combined_features, padding))

        return combined_features
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# Replace 'new_audio_path' with the path to your new audio file
new_audio_path = 'C:/Users/Gk-09/Desktop/bird1.wav'

# Define the expected input shape of the model
expected_input_shape = (169, 1)  # Update with your model's expected input shape

# Extract and preprocess features from the new audio file
new_audio_features = extract_and_preprocess_features(new_audio_path, expected_input_shape)

if new_audio_features is not None:
    # Make predictions using the loaded model
    predictions = loaded_model.predict(new_audio_features[np.newaxis, :])

    # Get the predicted class labels (assuming it's a classification problem)
    predicted_class_labels = np.argmax(predictions, axis=1)

    # Create a mapping of class codes to class labels
    class_label_mapping = {
        0: 'giloma_tumor',
        1: 'pituitary_tumor',
        2: 'meningioma_tumor',
        3: 'no_tumor'
    }

    # Map the class code to class label for the predicted class
    predicted_class_name = class_label_mapping[predicted_class_labels[0]]

    # Print the predicted class label for the new audio file
    print(f"Predicted Class Label for {new_audio_path}: {predicted_class_name}")
