# app.py - Flask API for Fruit/Vegetable Classification
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter app

# Load your trained model
try:
    model = tf.keras.models.load_model('models/CNN-model.keras')
    print("✅ Model loaded successfully!")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Class names (36 fruits and vegetables)
class_names = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum',
    'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant',
    'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce',
    'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple',
    'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn',
    'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

def preprocess_image(image_bytes):
    """Preprocess image for model input"""
    try:
        # Convert bytes to image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size (224x224 for MobileNetV2)
        image = image.resize((224, 224))
        
        # Convert to numpy array and normalize
        image_array = np.array(image) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        raise Exception(f"Error preprocessing image: {e}")

@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'status': 'running',
        'message': 'Fruit & Vegetable Classification API',
        'model_loaded': model is not None,
        'classes': len(class_names)
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Check if image is provided
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        # Get image file
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        # Preprocess image
        processed_image = preprocess_image(image_bytes)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get class name
        predicted_class = class_names[predicted_class_idx]
        
        # Get all predictions
        all_predictions = [
            {
                'class': class_names[i],
                'confidence': float(predictions[0][i])
            }
            for i in range(len(class_names))
        ]
        
        # Sort by confidence
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Prepare response
        response = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_predictions': all_predictions
        }
        
        print(f"✅ Prediction: {predicted_class} ({confidence*100:.2f}%)")
        
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get list of supported classes"""
    return jsonify({
        'classes': class_names,
        'total': len(class_names)
    })

if __name__ == '__main__':
    print("=" * 50)
    print("🍎 Fruit & Vegetable Classification API")
    print("=" * 50)
    print(f"📦 Classes: {len(class_names)}")
    print(f"🤖 Model: {'Loaded' if model else 'Not Loaded'}")
    print("=" * 50)
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)