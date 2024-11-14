import cv2
import os
import numpy as np
from PIL import Image

# Membuat variabel recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
# Untuk detector menggunakan file haarcascade_frontalface_default.xml
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Fungsi untuk mengambil gambar dan label
def getImagesWithLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    faceSamples = []
    Ids = []
    
    for imagePath in imagePaths:
        # Membaca gambar dan mengonversi ke grayscale
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        
        # Mendapatkan ID dari nama file
        Id = int(os.path.split(imagePath)[-1].split(".")[1])  # Sesuaikan dengan format nama file Anda
        faces = detector.detectMultiScale(imageNp)
        
        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y+h, x:x+w])
            Ids.append(Id)

    return faceSamples, Ids

# Mengambil wajah dan ID dari dataset
faces, Ids = getImagesWithLabels('Dataset')
recognizer.train(faces, np.array(Ids))

# Menyimpan model pelatihan
recognizer.save('Dataset/training.xml')

