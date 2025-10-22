import os
import base64
from io import BytesIO
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = load_model('autism.h5') 

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 
    return img_array

def preprocess_image_from_base64(base64_string):
    # Remove the data URL prefix if present
    if 'base64,' in base64_string:
        base64_string = base64_string.split('base64,')[1]
    
    # Decode base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(BytesIO(img_data))
    
    # Resize and convert to array
    img = img.resize((256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files.get('image')
        
        if not file:
            return jsonify({'error': 'No file provided'}), 400
        
        # Create uploads directory if it doesn't exist
        os.makedirs('static/uploads', exist_ok=True)
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        timestamp = str(int(np.random.random() * 1000000))
        filename = f'uploaded_{timestamp}_{filename}'
        img_path = os.path.join('static/uploads', filename)
        file.save(img_path)
        
        # Preprocess and predict
        img_array = preprocess_image(img_path)
        prediction = model.predict(img_array)
        
        if prediction > 0.015:
            result = 'Non Autistic'
        else:
            result = 'Autistic'
        
        return jsonify({
            'result': result,
            'filename': filename,
            'img_path': f'uploads/{filename}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Create uploads directory if it doesn't exist
        os.makedirs('static/uploads', exist_ok=True)
        
        # Process the image
        img_array = preprocess_image_from_base64(image_data)
        
        # Save the captured image
        timestamp = str(int(np.random.random() * 1000000))
        filename = f'captured_{timestamp}.jpg'
        img_path = os.path.join('static/uploads', filename)
        img_array.save(img_path)
        
        # Preprocess for prediction
        img_array_processed = image.img_to_array(img_array.resize((256, 256)))
        img_array_processed = np.expand_dims(img_array_processed, axis=0)
        img_array_processed /= 255.0
        
        # Make prediction
        prediction = model.predict(img_array_processed)
        
        if prediction > 0.015:
            result = 'Non Autistic'
        else:
            result = 'Autistic'
        
        return jsonify({
            'result': result,
            'filename': filename,
            'img_path': f'uploads/{filename}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
