from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO('model/kebersihan-yolov11n-v5.pt')  # path relative to project root

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

    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read())).convert('RGB')

    results = model.predict(image, verbose=False)
    result = results[0]

    detections_list = []
    output_counts = {key: 0 for key in target_classes}
    total_found = 0

    boxes = result.boxes
    class_names = model.names

    for box in boxes:
        class_id = int(box.cls[0])
        class_name = class_names[class_id]

        if class_name in target_classes:
            output_counts[class_name] += 1
            total_found += 1

            xyxy = box.xyxy[0].tolist()
            bbox = [round(coord, 2) for coord in xyxy]

            detections_list.append({
                "class": class_name,
                "bbox": bbox
            })

    classification = "unclean" if total_found > 5 else "clean"

    response = {
        "type": "kebersihan",
        "output": {
            "classification": classification,
            **output_counts,
            "detections": detections_list
        }
    }

    return jsonify(response)

@app.route('/')
def home():
    return "Kebersihan is running."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
