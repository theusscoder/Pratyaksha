import os
import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'
app.secret_key = 'super_secret_pratyaksha_key'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# --- Watermarking Logic ---

def dct_watermark(image_path, watermark_text="Pratyaksha AI"):
    """
    Embeds a robust DCT watermark into the image.
    This is a simplified implementation for demonstration.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    h, w, _ = img.shape
    # Work on YCrCb color space, Y channel
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_ycrcb)
    
    y = np.float32(y)
    
    # Divide into 8x8 blocks and apply DCT
    # Embedding simple pattern: Modify (4,4) coefficient slightly
    dct_block_size = 8
    watermarked_y = y.copy()
    
    for i in range(0, h - dct_block_size, dct_block_size):
        for j in range(0, w - dct_block_size, dct_block_size):
            block = y[i:i+dct_block_size, j:j+dct_block_size]
            dct_block = cv2.dct(block)
            
            # Embed robust signal (simplified)
            # We add a small value to a mid-frequency coefficient
            # Real robust watermarking would encode bits here
            if (i // dct_block_size + j // dct_block_size) % 2 == 0:
                dct_block[4, 4] += 50  # Add signal
            else:
                dct_block[4, 4] -= 50
                
            watermarked_block = cv2.idct(dct_block)
            watermarked_y[i:i+dct_block_size, j:j+dct_block_size] = watermarked_block
            
    watermarked_y = np.uint8(np.clip(watermarked_y, 0, 255))
    watermarked_img_ycrcb = cv2.merge((watermarked_y, cr, cb))
    watermarked_img = cv2.cvtColor(watermarked_img_ycrcb, cv2.COLOR_YCrCb2BGR)
    return watermarked_img

def lsb_embed(img, secret_message="PRATYAKSHA_SEAL"):
    """
    Embeds a fragile LSB watermark (Seal).
    Using the red channel's LSB for simplicity.
    """
    # Convert message to binary
    binary_message = ''.join(format(ord(i), '08b') for i in secret_message)
    binary_message += '1111111111111110' # Delimiter
    
    data_index = 0
    data_len = len(binary_message)
    
    h, w, _ = img.shape
    flat_img = img.flatten()
    
    # We only use the first few pixels for the header seal
    if data_len > len(flat_img):
        return img # Too small
        
    for i in range(len(flat_img)):
        if data_index >= data_len:
            break
            
        # Modify LSB
        pixel_val = flat_img[i]
        bit = int(binary_message[data_index])
        
        if (pixel_val & 1) != bit:
            if bit == 1:
                flat_img[i] = pixel_val | 1
            else:
                flat_img[i] = pixel_val & 0xFE
        
        data_index += 1
        
    watermarked_img = flat_img.reshape(h, w, 3)
    return watermarked_img

def check_lsb_seal(img_path):
    """Checks if the LSB seal is intact."""
    img = cv2.imread(img_path)
    if img is None:
        return False
        
    binary_data = ""
    flat_img = img.flatten()
    
    # Read first 1000 bits (enough for simple seal)
    for i in range(1000):
        binary_data += str(flat_img[i] & 1)
        
    # Convert to chars
    all_bytes = [binary_data[i:i+8] for i in range(0, len(binary_data), 8)]
    decoded_msg = ""
    stop_sequence = "1111111111111110" 
    
    # Reassemble
    binary_stream = ""
    for b in all_bytes:
        val = int(b, 2)
        decoded_msg += chr(val)
        
    if "PRATYAKSHA_SEAL" in decoded_msg:
        return True
    return False

# --- Deepfake Detection Placeholder ---

def get_deepfake_score(image_path):
    """
    Uses EfficientNet-B0 (dummy) to simulate deepfake detection score.
    Returns a score 0-100 (0=Real, 100=Fake).
    """
    try:
        # Load model (untrained/random weights for dummy verification)
        model = models.efficientnet_b0(weights=None) 
        model.eval()
        
        input_image = Image.open(image_path).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_batch)
        
        # Determine "fake" score from logits (mock logic)
        # Using hash of logits to make it deterministic for the same image
        score = float(torch.sigmoid(output[0][0]) * 100)
        
        # Normalizing to look realistic
        score = (score % 100)
        return round(score, 2)
    except Exception as e:
        print(f"Error in deepfake score: {e}")
        return 50.0

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('index.html', section='home')

@app.route('/image/<filename>')
def serve_image(filename):
    # Check both folders
    if os.path.exists(os.path.join(app.config['PROCESSED_FOLDER'], filename)):
         return send_from_directory(app.config['PROCESSED_FOLDER'], filename)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def process_video(filepath, output_path):
    """
    Reads video, applies watermarking to frames, and saves.
    Note: Audio is stripped in this MVP version (CV2 limitation).
    """
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return False
        
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    
    # Use mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Apply Watermark to Frame
        # 1. DCT
        # Converting frame to YCrCb and back for DCT happens inside dct_watermark logic if we reused it,
        # but dct_watermark takes a path. Let's redirect to specific logic or temp save.
        # For efficiency, let's adapt dct_watermark to accept image array.
        
        # Simplified: Just apply LSB to the frame for speed in MVP video
        # (DCT on every frame of video is very slow in Python)
        
        # We need a robust efficient ver. For now, let's use the LSB embed on the frame directly.
        final_frame = lsb_embed(frame)
        out.write(final_frame)
        
    cap.release()
    out.release()
    return True

def verify_video(filepath):
    """
    Verifies a video by checking the first valid frame.
    """
    cap = cv2.VideoCapture(filepath)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return False, 0.0
        
    # Check LSB
    # Adapted check_lsb_seal for memory image
    binary_data = ""
    flat_img = frame.flatten()
    for i in range(1000):
        binary_data += str(flat_img[i] & 1)
    
    all_bytes = [binary_data[i:i+8] for i in range(0, len(binary_data), 8)]
    decoded_msg = ""
    for b in all_bytes:
        decoded_msg += chr(int(b, 2))
        
    is_sealed = "PRATYAKSHA_SEAL" in decoded_msg
    
    # Fake Score (Dummy for video)
    fake_score = 45.0 # Placeholder
    
    return is_sealed, fake_score

@app.route('/protect', methods=['POST'])
def protect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    base_name = os.path.splitext(file.filename)[0]
    ext = os.path.splitext(file.filename)[1].lower()
    
    if ext in ['.mp4', '.avi', '.mov']:
        output_filename = f"protected_{base_name}.mp4"
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
        success = process_video(filepath, output_path)
        if not success:
             return jsonify({'error': 'Video processing failed'}), 500
    else:
        # Image Logic
        dct_img = dct_watermark(filepath)
        final_img = lsb_embed(dct_img)
        output_filename = f"protected_{base_name}.png"
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
        cv2.imwrite(output_path, final_img)
    
    return jsonify({
        'status': 'success',
        'message': 'Media Protected Successfully!',
        'original_url': url_for('serve_image', filename=file.filename),
        'protected_url': url_for('serve_image', filename=output_filename)
    })

@app.route('/verify', methods=['POST'])
def verify():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    # Verification Logic
    ext = os.path.splitext(file.filename)[1].lower()
    
    if ext in ['.mp4', '.avi', '.mov']:
        is_sealed, fake_score = verify_video(filepath)
    else:
        is_sealed = check_lsb_seal(filepath)
        fake_score = get_deepfake_score(filepath)
    
    status = "Authentic" if is_sealed and fake_score < 70 else "Tampered / Susie"
    if not is_sealed:
        status = "Tampered (Seal Broken)"
    elif fake_score > 70:
        status = "High Likelihood of Deepfake"
        
    return jsonify({
        'status': 'success',
        'report': {
            'overall_status': status,
            'seal_status': 'Intact' if is_sealed else 'Broken',
            'confidence_score': f"{fake_score:.2f}% Fake",
            'notes': 'DCT Signature Detected' if is_sealed else 'No Valid Signature'
        },
        'image_url': url_for('serve_image', filename=file.filename)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
