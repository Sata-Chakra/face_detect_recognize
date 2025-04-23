import os
import pickle
import face_recognition
import numpy as np

DB_FILE = 'face_db.pkl'

def load_face_database():
    if not os.path.exists(DB_FILE):
        return []
    with open(DB_FILE, 'rb') as f:
        return pickle.load(f)


def save_face_database(face_db):
    with open(DB_FILE, 'wb') as f:
        pickle.dump(face_db, f)


def encode_face_from_image(image):
    encodings = face_recognition.face_encodings(image)
    if encodings:
        return encodings[0]
    return None


def register_face(image, name):
    face_db = load_face_database()
    encoding = encode_face_from_image(image)
    if encoding is None:
        return False, "No face found"

    face_db.append({"name": name, "encoding": encoding})
    save_face_database(face_db)
    return True, f"{name} registered successfully"


def get_best_match_strict(encoding, face_db, threshold=0.55, min_gap=0.10):
    if not face_db:
        return "Unknown", None

    known_encodings = np.array([p["encoding"] for p in face_db])
    distances = np.linalg.norm(known_encodings - encoding, axis=1)
    sorted_indices = np.argsort(distances)

    best_idx = sorted_indices[0]
    second_best_idx = sorted_indices[1] if len(sorted_indices) > 1 else None

    best_distance = distances[best_idx]
    second_best_distance = distances[second_best_idx] if second_best_idx is not None else None

    if best_distance < threshold and (
            second_best_distance is None or second_best_distance - best_distance > min_gap
    ):
        return face_db[best_idx]["name"], best_distance
    else:
        return "Unknown", best_distance
