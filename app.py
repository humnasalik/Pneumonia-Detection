from flask import Flask, request, render_template
import numpy as np
import cv2
import tensorflow as tf
import pickle
from skimage import transform

# Load the trained model from pickle file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Class labels
map_characters = {0: 'No Pneumonia', 1: 'Yes Pneumonia'}

# Initialize Flask app
app = Flask(__name__)

# Define routes
@app.route('/')
def index():
    return render_template('index.html')  # HTML for file upload

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded!"
    
    file = request.files['file']
    if file:
        # Read the image file
        img = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        
        # Preprocess the image
        img = transform.resize(img, (150, 150, 3))
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        # Make prediction
        prediction = model.predict(img)
        class_idx = np.argmax(prediction, axis=1)[0]
        result = map_characters[class_idx]
        
        # Pass result to the template
        return render_template('prediction_result.html', prediction=result)


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
