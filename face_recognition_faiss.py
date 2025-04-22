import cv2
import pickle
import numpy as np
import face_recognition
import faiss
from ultralytics import YOLO

index = faiss.read_index("face_index.faiss")
with open("face_metadata.pkl", "rb") as f:
    known_names = pickle.load(f)

# cap = cv2.VideoCapture("sample_videos/tectes_demo_sample.mp4")

cap = cv2.VideoCapture(0)

cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)

face_detection_threshold = 0.75
distance_threshold = 0.85

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for i, face_encoding in enumerate(face_encodings):
        vec = np.array(face_encoding, dtype="float32").reshape(1, -1)
        D, I = index.search(vec, k=1)

        if D[0][0] < distance_threshold:
            name = known_names[I[0][0]]
        else:
            name = "Unknown"

        # Draw box and name
        top, right, bottom, left = face_locations[i]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
