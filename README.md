# 🎯 Real-Time Face Recognition System (Using face_recognition Library)
Built with face_recognition (Ageitgey/Dlib). Uses NumPy for fast, strict distance-based matching (Euclidean distance, threshold, optional gap check). Features confidence visualization &amp; real-time webcam labeling.
This face recognition system is built entirely on top of the face_recognition library by Facebook (Ageitgey's wrapper for Dlib) and uses simple but fast NumPy-based vector comparison for matching. It avoids over-reliance on out-of-the-box classifiers or approximate libraries like FAISS by implementing a strict distance-based matching policy:

*  Euclidean distance is used to compare new face embeddings to known faces.

*  A threshold determines whether a match is valid (lower = more strict).

*  An optional gap check ensures the best match is significantly better than others, preventing false positives.

*  Includes confidence visualization and real-time labeling in webcam streams.

# The Project includes:

- 🧠 Face encoding with name tagging from a folder of images
- 🎥 Real-time recognition via webcam
- ⚡ Efficient in-memory matching with customizable distance metric
- 🛡️ Strict logic to avoid misclassification of unknown faces
- 🧪 Support for visual confidence display and tuning of thresholds

---

# 📁 Project Structure, the one we are interested in

```bash
.
├── known_faces_gallery/          # Folder of images of known people
├── face_db.pkl                   # Saved encodings of known faces
├── encode_faces.py              # Script to encode known faces
├── recognize_faces_strict.py    # Real-time face recognition script (webcam)
├── requirements.txt
└── README.md
```

# 🚀 Quick Start

## 1. Install Dependencies

pip install -r requirements.txt

🧩 Requirements

* face_recognition>=1.3.0
* dlib>=19.22.0
* numpy>=1.21.0
* opencv-python>=4.5.3

If face_recognition fails to build on Windows, refer to dlib installation guide or try:

```bash
pip install cmake
pip install dlib
pip install face_recognition
```

## 2. Prepare Your Known Faces

* Place clear, front-facing images of known people inside:
```bash
known_faces_gallery/
├── alice.jpg
├── bob.png
└── charlie.jpeg
```

* Filenames (without extension) will be used as names in recognition.


## 3. Encode Faces
```
python face_embeddings_creation_general.py
```


This will create face_db.pkl, which stores name and embeddings.

## 4. Start Real-Time Recognition
```
python face_recognition_strict_general.py
```
It will open your webcam and annotate detected faces in real-time.


# 🌐 Flask API Usage
The system exposes two REST endpoints:

## 1. Register a New Face

* POST /register_face

* Form Data:
```
image: JPEG/PNG file

name: string name for the person
```

## 2. Detect Face from Image
* POST /detect_face

* Form Data:
```
image: JPEG/PNG file
```

# ⚙️ Configuration

You can tweak these values in face_recognition_strict_general.py:

* threshold = 0.55 [Lower = stricter (e.g. 0.45 to 0.60)]
* min_gap = 0.10    [# Ensures best match is much better than 2nd best]

# Conclusion
For each match, you'll see:

* 🟩 Name + match distance (if found)

* 🟥 "Unknown" label for non-matches

* ⚠️ Warning if match confidence is borderline
