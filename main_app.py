from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import cv2
from face_utils import register_face, load_face_database, get_best_match_strict, encode_face_from_image
from PIL import Image
import io

app = Flask(__name__)

def read_image(file_storage):
    image = Image.open(file_storage.stream)
    return face_recognition.load_image_file(io.BytesIO(file_storage.read()))

@app.route('/register_face', methods=['POST'])
def register():
    file = request.files.get('image')
    name = request.form.get('name')

    if not file or not name:
        return jsonify({"error": "Missing image or name"}), 400

    image = face_recognition.load_image_file(file)
    success, message = register_face(image, name)
    return jsonify({"success": success, "message": message})

@app.route('/detect_face', methods=['POST'])
def detect():
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "Missing image"}), 400

    image = face_recognition.load_image_file(file)
    encodings = face_recognition.face_encodings(image)

    if not encodings:
        return jsonify({"name": "Unknown", "confidence": None}), 200

    face_db = load_face_database()
    name, confidence = get_best_match_strict(encodings[0], face_db)

    return jsonify({"name": name, "confidence": confidence})

if __name__ == '__main__':
    app.run(debug=True)
