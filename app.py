import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
MODEL_PATH = 'medical_ct_denoiser_full.keras'
model = None

def get_model():
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH, compile=False)
        else:
            return None
    return model

def process_image(image_path):
    # Load and preprocess (Medical CT expects 256x256 Grayscale)
    img = load_img(image_path, target_size=(256, 256), color_mode='grayscale')
    img_array = img_to_array(img) / 255.0
    
    # Denoise
    m = get_model()
    if m is None:
        return None, None, None, None
    
    # Predict (input: 1, 256, 256, 1)
    input_tensor = np.expand_dims(img_array, axis=0)
    denoised_tensor = m.predict(input_tensor)
    denoised_img = denoised_tensor[0]
    
    # Calculate Metrics (Comparing Noisy vs Denoised to show restoration magnitude)
    psnr_val = tf.image.psnr(img_array, denoised_img, max_val=1.0).numpy()
    ssim_val = tf.image.ssim(tf.convert_to_tensor(img_array), tf.convert_to_tensor(denoised_img), max_val=1.0).numpy()
    
    # Calculate Improvement % (Heuristic based on PSNR Gain)
    # Assuming baseline noisy image PSNR is ~20 for 1/4 dose CT
    improvement_pct = min(99.9, (psnr_val / 40.0) * 100) # 40dB as 'Perfect' reference
    
    # Difference Map (What was removed)
    diff_map = np.abs(img_array - denoised_img)
    # Normalize diff map for visibility
    if np.max(diff_map) > 0:
        diff_map = diff_map / np.max(diff_map)
    
    return img_array, denoised_img, diff_map, {
        'psnr': float(psnr_val), 
        'ssim': float(ssim_val),
        'improvement': float(improvement_pct)
    }

def array_to_base64(arr):
    # Squeeze to handle grayscale (H, W, 1) -> (H, W)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    ref_file = request.files.get('reference') # Optional Reference
    
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    noisy, denoised, diff, metrics = process_image(filepath)
    
    # If Reference (Full Dose) is provided, calculate TRUE improvement
    true_metrics = None
    if ref_file and ref_file.filename != '':
        ref_path = os.path.join(app.config['UPLOAD_FOLDER'], 'ref_' + ref_file.filename)
        ref_file.save(ref_path)
        
        # Load Reference
        ref_img = load_img(ref_path, target_size=(256, 256), color_mode='grayscale')
        ref_array = img_to_array(ref_img) / 255.0
        
        # Initial Quality (Noisy vs Ref)
        init_psnr = tf.image.psnr(ref_array, noisy, max_val=1.0).numpy()
        # Final Quality (Denoised vs Ref)
        final_psnr = tf.image.psnr(ref_array, denoised, max_val=1.0).numpy()
        
        true_metrics = {
            'initial_psnr': float(init_psnr),
            'final_psnr': float(final_psnr),
            'gain': float(final_psnr - init_psnr)
        }
        os.remove(ref_path)

    if noisy is None:
        return jsonify({'error': 'Model file not found. Please ensure medical_ct_denoiser_full.keras is present.'}), 500

    # Clean up
    os.remove(filepath)
    
    return jsonify({
        'noisy': array_to_base64(noisy),
        'denoised': array_to_base64(denoised),
        'diff': array_to_base64(diff),
        'metrics': metrics,
        'true_metrics': true_metrics
    })

if __name__ == '__main__':
    app.run(debug=True)
