import cv2
import pickle
import numpy as np
import face_recognition

def load_face_database(db_file='face_db.pkl'):
    with open(db_file, 'rb') as f:
        return pickle.load(f)

def get_best_match_strict(encoding, face_db, threshold=0.55, min_gap=0.10):
    """
    - threshold: max allowed distance for a face to be considered a match. as we are using eucleid distance we need this metric to be between 0.45-0.6
    - min_gap: distance gap between best and second-best matches , to drop unknowns and ambuiguius facess from detections
    """
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

face_db = load_face_database()

cap = cv2.VideoCapture(0)

print("[INFO] Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for i, face_encoding in enumerate(face_encodings):
        name, distance = get_best_match_strict(face_encoding, face_db, threshold=0.55, min_gap=0.08)

        top, right, bottom, left = face_locations[i]

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        label = f"{name} ({distance:.2f})" if distance is not None else name
        cv2.putText(frame, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if name != "Unknown" and distance > 0.5:
            cv2.putText(frame, "⚠ Low confidence", (left, bottom + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Strict Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
