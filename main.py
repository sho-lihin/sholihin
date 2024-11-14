import cv2
import os
from PIL import Image

# Inisialisasi webcam
camera = 0
video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)

# Memeriksa apakah video capture berhasil
if not video.isOpened():
    print("Error: Could not open video.")
    exit()

# Memuat Haar Cascade untuk deteksi wajah
faceDeteksi = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('Dataset/training.xml')

# Mendefinisikan nama-nama pengguna
names = {
    1: 'sholihin',
    1.1: 'sholihin',
    2: 'hafiz',
    3: 'faisal',
    3.1: 'faisal',
    4: 'azlan',
    5: 'Syamil',
    6: 'ananda',
    

    
}

while True:
    # Membaca frame dari webcam
    check, frame = video.read()
    
    # Memastikan frame berhasil dibaca
    if not check:
        print("Error: Could not read frame.")
        break
    
    abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = faceDeteksi.detectMultiScale(abu, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in wajah:
        id, conf = recognizer.predict(abu[y:y + h, x:x + w])
        
        # Menentukan ID berdasarkan confidence threshold
        if conf < 100:  # Sesuaikan threshold sesuai kebutuhan
            id = names.get(id, 'Unknown')
        else:
            id = 'Unknown'
        
        # Menentukan warna kotak berdasarkan ID
        color = (0, 0, 255) if id == 'Unknown' else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, str(id), (x + 40, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, color)

    cv2.imshow("Face Recognition", frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

# Melepaskan kamera dan menutup jendela
video.release()
cv2.destroyAllWindows()
