import os
import pickle
import face_recognition
import numpy as np
import faiss

def dump_faiss_face_index(input_folder="known_faces_gallery", index_path="face_index.faiss", meta_path="face_metadata.pkl"):
    encodings = []
    names = []

    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)
        image = face_recognition.load_image_file(image_path)
        enc_list = face_recognition.face_encodings(image)

        if enc_list:
            encodings.append(enc_list[0])
            name = os.path.splitext(filename)[0]
            names.append(name)
        else:
            print(f"[WARNING] No face found in {filename}, skipping.")

    if not encodings:
        print("No valid face encodings found. Aborting.")
        return

    encodings_np = np.array(encodings).astype("float32")

    # Creatind the faiss indecing for usage
    index = faiss.IndexFlatL2(encodings_np.shape[1])
    index.add(encodings_np)
    faiss.write_index(index, index_path)

    with open(meta_path, "wb") as f:
        pickle.dump(names, f)

    print(f"[INFO] Saved index with {len(encodings)} faces.")
    print(f"[INFO] FAISS index saved to '{index_path}'")
    print(f"[INFO] Metadata (names) saved to '{meta_path}'")

if __name__ == "__main__":
    dump_faiss_face_index()
