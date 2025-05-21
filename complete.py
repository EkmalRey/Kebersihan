from flask import Flask, request, jsonify
from ultralytics import YOLO
import os
import cv2
import numpy as np
import io
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Create upload folder at startup

# Model paths - all models in model folder
BOLT_MODEL_PATH = "model/baut.pt"
POSE_MODEL_PATH = "model/pose.pt"
CLEANLINESS_MODEL_PATH = "model/kebersihan-yolov11n-v5.pt"

# Global model variables
bolt_model = None
pose_model = None
cleanliness_model = None

# Cleanliness class mapping
CLEANLINESS_TARGET_CLASSES = {
    'tanaman_liar': 'tanaman_liar',
    'lumut': 'lumut',
    'genangan_air': 'genangan_air',
    'sampah': 'sampah',
    'noda_dinding': 'noda_dinding',
    'retakan': 'retakan',
    "sampah_daun": "sampah_daun",
}

# Cleanliness recommendation mapping
CLEANLINESS_RECOMMENDATIONS = {
    'tanaman_liar': 'Bersihkan tanaman liar di area tersebut.',
    'lumut': 'Hilangkan lumut dari permukaan.',
    'genangan_air': 'Keringkan genangan air untuk mencegah jentik nyamuk.',
    'sampah': 'Bersihkan sampah yang terlihat.',
    'noda_dinding': 'Bersihkan noda pada dinding.',
    'retakan': 'Perbaiki retakan untuk mencegah kerusakan lebih lanjut.',
    "sampah_daun": "Bersihkan daun-daun yang berserakan."
}

def load_models():
    """Load all required models at application startup"""
    global bolt_model, pose_model, cleanliness_model
    try:
        bolt_model = YOLO(BOLT_MODEL_PATH)
        pose_model = YOLO(POSE_MODEL_PATH)
        cleanliness_model = YOLO(CLEANLINESS_MODEL_PATH)
        print("All models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {str(e)}")

# =================== BOLT DETECTION FUNCTIONS ===================

def process_bolt_detection(image_path):
    """Process bolt detection and return count and status"""
    try:
        results = bolt_model(image_path)
        
        # Count BT objects
        bt_count = 0
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    class_name = result.names[int(boxes.cls[i].item())]
                    if class_name == "BT":
                        bt_count += 1
        
        # Determine if there are enough bolts (>20 considered complete)
        status = bt_count > 20
        
        return {
            "total_BT": bt_count,
            "status_complete": status
        }
    except Exception as e:
        return {"error": f"Error in bolt detection: {str(e)}"}

# =================== POSE ESTIMATION FUNCTIONS ===================

def infer_pose(image_path):
    """Detect tower poses in an image"""
    results = pose_model.predict(image_path, imgsz=640)
    detections = results[0]
    conf_threshold = 0.4
    valid_detections = []

    for i in range(len(detections.boxes)):
        box = detections.boxes.xywh[i].tolist()
        conf = float(detections.boxes.conf[i])

        if conf < conf_threshold:
            continue

        # Calculate bounding box coordinates
        x, y, w, h = box
        x1, y1 = x - w / 2, y - h / 2
        x2, y2 = x + w / 2, y + h / 2
        bbox = [x1, y1, x2, y2, conf]

        # Get keypoint data
        keypoint_coords = detections.keypoints.xyn[i].tolist()  # normalized
        keypoint_confs = detections.keypoints.conf[i].tolist()

        keypoints = []
        for (x, y), c in zip(keypoint_coords, keypoint_confs):
            keypoints.append([x, y, c])

        if len(keypoints) >= 2:
            keypoints_pair = [keypoints[0], keypoints[1]]  # top & bottom
            valid_detections.append((bbox, keypoints_pair))

    return valid_detections

def classify_pose(keypoints):
    """Classify tower pose based on keypoints"""
    top_x, top_y, top_conf = keypoints[0]
    bottom_x, bottom_y, bottom_conf = keypoints[1]

    # Check if keypoints are detected with sufficient confidence
    if top_conf < 0.3 or bottom_conf < 0.3:
        return {
            "pose": "Keypoints tidak terdeteksi dengan baik",
            "top_confidence": round(float(top_conf), 3),
            "bottom_confidence": round(float(bottom_conf), 3),
            "angle_deviation": None
        }

    # Calculate vertical angle
    delta_x = bottom_x - top_x
    delta_y = bottom_y - top_y
    angle = abs(np.degrees(np.arctan(delta_y / delta_x))) if delta_x != 0 else 90
    vertical_angle = 90 - angle

    # Classify pose based on vertical angle
    if vertical_angle < 70:
        pose = "Berdiri miring"
    elif vertical_angle < 80:
        pose = "Kurang berdiri lurus dan tegak"
    else:
        pose = "Berdiri lurus dan tegak"

    return {
        "pose": pose,
        "top_confidence": round(float(top_conf), 3),
        "bottom_confidence": round(float(bottom_conf), 3),
        "angle_deviation": round(float(vertical_angle), 2)
    }

def analyze_tower_structure(detections):
    """Analyze each detected tower's structure"""
    results = []
    for i, (_, keypoints) in enumerate(detections):
        if len(keypoints) >= 2:
            result = classify_pose(keypoints)
            results.append({
                "pose_analysis": result
            })
    return results

def process_pose_analysis(image_path):
    """Main function to process tower pose analysis"""
    if not os.path.exists(image_path):
        return {"error": "Gambar tidak ditemukan"}
    
    try:
        detections = infer_pose(image_path)
        
        if not detections:
            return {"message": "Tidak ada tower terdeteksi"}
            
        return analyze_tower_structure(detections)
    except Exception as e:
        return {"error": f"Error processing pose analysis: {str(e)}"}

# =================== CLEANLINESS DETECTION FUNCTIONS ===================

def process_cleanliness_detection(image_files):
    """Process cleanliness detection for multiple images"""
    # Initialize variables for combined results
    detections_by_filename = {}  # Dictionary to group detections by filename
    combined_counts = {key: 0 for key in CLEANLINESS_TARGET_CLASSES}
    total_found_combined = 0

    for image_file in image_files:
        filename = image_file.filename
        
        # Initialize detection list for this filename if not exists
        if filename not in detections_by_filename:
            detections_by_filename[filename] = []
            
        try:
            image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        except Exception as e:
            return {"error": f"Invalid image {filename}: {str(e)}"}, 400

        results = cleanliness_model.predict(image, verbose=False, classes=[0, 1, 2, 3, 4, 5, 9])
        result = results[0]

        boxes = result.boxes
        class_names = cleanliness_model.names

        for box in boxes:
            class_id = int(box.cls[0])
            class_name = class_names[class_id]

            if class_name in CLEANLINESS_TARGET_CLASSES:
                combined_counts[class_name] += 1
                total_found_combined += 1

                xyxy = box.xyxy[0].tolist()
                bbox = [round(float(coord), 2) for coord in xyxy]

                detection = {
                    "class": class_name,
                    "bbox": bbox
                }
                
                # Add this detection to the appropriate filename group
                detections_by_filename[filename].append(detection)

    # Convert dictionary to list format for output
    grouped_detections = []
    for filename, detections in detections_by_filename.items():
        grouped_detections.append({
            "filename": filename,
            "detections": detections
        })

    # Determine overall classification based on combined total
    combined_classification = "unclean" if total_found_combined >= 5 else "clean"

    # Generate recommendations based on detected issues
    recommendations = []
    for class_name, count in combined_counts.items():
        if count > 0 and class_name in CLEANLINESS_RECOMMENDATIONS:
            recommendations.append(CLEANLINESS_RECOMMENDATIONS[class_name])

    if not recommendations:
        recommendations = ["Tidak ada rekomendasi yang diperlukan."]

    # Create response with detections grouped by filename
    response = {
        "type": "kebersihan",
        "output": {
            "classification": combined_classification,
            "counts": combined_counts,
            "files": grouped_detections,
            "recommendations": recommendations
        }
    }

    return response

# =================== API ROUTES ===================

@app.route('/')
def home():
    """API home route"""
    return jsonify({
        "status": "running", 
        "available_endpoints": [
            "/predict_baut", 
            "/predict_pose", 
            "/predict_kebersihan"
        ]
    })

@app.route('/predict_baut', methods=['POST'])
def predict_bolt():
    """Endpoint for bolt detection"""
    # Check if image is provided
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Save and process the image
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process bolt detection
        results = process_bolt_detection(filepath)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

@app.route('/predict_pose', methods=['POST'])
def predict_pose():
    """Endpoint for pose estimation"""
    # Check if image is provided
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Save and process the image
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process pose analysis
        results = process_pose_analysis(filepath)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

@app.route('/predict_kebersihan', methods=['POST'])
def predict_kebersihan():
    """Endpoint for cleanliness detection"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    # Get all uploaded files under the 'image' key
    image_files = request.files.getlist('image')
    if not image_files or all(file.filename == '' for file in image_files):
        return jsonify({"error": "No valid images uploaded"}), 400
    
    try:
        # Process cleanliness detection
        results = process_cleanliness_detection(image_files)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

if __name__ == '__main__':
    load_models()  # Load models at startup
    app.run(host='0.0.0.0', port=5000, debug=True)