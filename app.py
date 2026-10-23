from flask import Flask, render_template, request
import numpy as np
import os
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load trained model
model = load_model('model/plant_model.h5')

# Class names (same order as training)
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
               'Apple___healthy', 'Corn_(maize)___Common_rust_', 
               'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
               'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
               'Tomato___Early_blight', 'Tomato___Late_blight', 
               'Tomato___Leaf_Mold', 'Tomato___healthy']

# Simple treatment suggestions
treatments = {
    'Apple___Apple_scab': 'Apply fungicide spray and remove infected leaves.',
    'Apple___Black_rot': 'Prune infected branches and apply copper fungicide.',
    'Apple___Cedar_apple_rust': 'Remove nearby cedar trees and apply fungicide.',
    'Apple___healthy': 'No treatment needed. Keep up the good care!',
    'Corn_(maize)___Common_rust_': 'Apply fungicide and rotate crops yearly.',
    'Corn_(maize)___Northern_Leaf_Blight': 'Use resistant seed varieties and fungicide.',
    'Corn_(maize)___healthy': 'No treatment needed. Keep up the good care!',
    'Potato___Early_blight': 'Apply fungicide and avoid overhead watering.',
    'Potato___Late_blight': 'Remove infected plants and apply fungicide immediately.',
    'Potato___healthy': 'No treatment needed. Keep up the good care!',
    'Tomato___Early_blight': 'Remove affected leaves and apply fungicide.',
    'Tomato___Late_blight': 'Remove infected plants immediately to prevent spread.',
    'Tomato___Leaf_Mold': 'Improve air circulation and apply fungicide.',
    'Tomato___healthy': 'No treatment needed. Keep up the good care!'
}

UPLOAD_FOLDER = 'static/uploads'
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 75

def is_image_blurry(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return blur_score < 100

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Check image quality
    if is_image_blurry(filepath):
        return render_template('result.html', 
                                error="Image is too blurry. Please upload a clearer photo.",
                                image_path=filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    predictions = model.predict(img_array)[0]
    top_3_idx = predictions.argsort()[-3:][::-1]
    top_confidence = predictions[top_3_idx[0]] * 100

    if top_confidence < CONFIDENCE_THRESHOLD:
        return render_template('result.html',
                                error="Could not confidently recognize this leaf. Please upload a clearer plant leaf image.",
                                image_path=filepath)

    results = []
    for i in top_3_idx:
        results.append((class_names[i], round(predictions[i]*100, 2)))

    predicted_class = class_names[top_3_idx[0]]
    treatment = treatments.get(predicted_class, "No specific treatment found.")

    return render_template('result.html',
                            results=results,
                            treatment=treatment,
                            image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)