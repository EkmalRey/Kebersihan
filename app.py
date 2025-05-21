from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)

# Load YOLOv11 model
model = YOLO('E:\PROJECT\Github\Kebersihan\model\kebersihan-yolov11n-v5.pt')  # path relative to project root

# Class label mapping
target_classes = {
    'tanaman_liar': 'tanaman_liar',
    'lumut': 'lumut',
    'genangan_air': 'genangan_air',
    'sampah': 'sampah',
    'noda_dinding': 'noda_dinding',
    'retakan': 'retakan',
    "sampah_daun": "sampah_daun",
}

@app.route('/predict_kebersihan', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    # Get all uploaded files under the 'image' key
    image_files = request.files.getlist('image')

    # Initialize variables for combined results
    detections_by_filename = {}  # Dictionary to group detections by filename
    combined_counts = {key: 0 for key in target_classes}
    total_found_combined = 0

    for image_file in image_files:
        filename = image_file.filename
        
        # Initialize detection list for this filename if not exists
        if filename not in detections_by_filename:
            detections_by_filename[filename] = []
            
        try:
            image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        except Exception:
            return jsonify({'error': f"Invalid image: {filename}"}), 400

        results = model.predict(image, verbose=False, classes=[0, 1, 2, 3, 4, 5, 9])
        result = results[0]

        boxes = result.boxes
        class_names = model.names

        for box in boxes:
            class_id = int(box.cls[0])
            class_name = class_names[class_id]

            if class_name in target_classes:
                combined_counts[class_name] += 1
                total_found_combined += 1

                xyxy = box.xyxy[0].tolist()
                bbox = [round(coord, 2) for coord in xyxy]

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

    # Mapping each class to a recommendation
    recommendation_map = {
        'tanaman_liar': 'Bersihkan tanaman liar di area tersebut.',
        'lumut': 'Hilangkan lumut dari permukaan.',
        'genangan_air': 'Keringkan genangan air untuk mencegah jentik nyamuk.',
        'sampah': 'Bersihkan sampah yang terlihat.',
        'noda_dinding': 'Bersihkan noda pada dinding.',
        'retakan': 'Perbaiki retakan untuk mencegah kerusakan lebih lanjut.',
        "sampah_daun": "Bersihkan daun-daun yang berserakan."
    }

    recommendations = []
    for class_name, count in combined_counts.items():
        if count > 0 and class_name in recommendation_map:
            recommendations.append(recommendation_map[class_name])

    # Create response with detections grouped by filename
    response = {
        "type": "kebersihan",
        "output": {
            "classification": combined_classification,
            "counts": combined_counts,
            "files": grouped_detections,
            "recommendations": recommendations if recommendations else ["Tidak ada rekomendasi yang diperlukan."]
        }
    }

    return jsonify(response)


@app.route('/')
def home():
    return "Kebersihan is running."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
