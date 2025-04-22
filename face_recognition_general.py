import cv2
import pickle
import numpy as np
import face_recognition

def load_face_database(db_file='face_db.pkl'):
    with open(db_file, 'rb') as f:
        return pickle.load(f)

def get_best_match(encoding, face_db, threshold=0.60):
    if not face_db:
        return "Unknown"
    ############### currently uing eucledean distance
    known_encodings = np.array([person["encoding"] for person in face_db])
    distances = np.linalg.norm(known_encodings - encoding, axis=1)

    min_distance_idx = np.argmin(distances)
    if distances[min_distance_idx] < threshold:
        return face_db[min_distance_idx]["name"]
    else:
        return "Unknown"

face_db = load_face_database()

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("sample_videos/tectes_demo_sample.mp4")

cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)

print("[INFO] Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for i, encoding in enumerate(face_encodings):
        name = get_best_match(encoding, face_db)

        top, right, bottom, left = face_locations[i]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
