import os
import pickle
import face_recognition

############ here I am accessing a
def create_face_database(input_dir='known_faces_gallery', output_file='face_db.pkl'):
    face_db = []

    for file in os.listdir(input_dir):
        path = os.path.join(input_dir, file)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            face_db.append({
                "name": os.path.splitext(file)[0],
                "encoding": encodings[0]
            })
            print(f"[INFO] Encoded {file}")
        else:
            print(f"[WARNING] No face found in {file}")

    with open(output_file, 'wb') as f:
        pickle.dump(face_db, f)
        print(f"[DONE] Saved {len(face_db)} face encodings to {output_file}")

if __name__ == "__main__":
    create_face_database()
