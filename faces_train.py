import os
import numpy as np
import cv2
from PIL import Image
import pickle # to save labels

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

face_cascade = cv2.CascadeClassifier('cascade/data/haarcascade_frontalface_alt2.xml') #only front of a face

recognizer = cv2.face.LBPHFaceRecognizer_create()


image_dir = os.path.join(BASE_DIR, "dataset")
current_id = 0
labels_id = {} #dictionary
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
	for file in files:
		if(file.endswith("png") or file.endswith("jpg")):
			path = os.path.join(root, file)
			label = os.path.basename(root).replace(" ","-").lower()
			#print(label)
			if label in labels_id:
				pass
			else:
				labels_id[label] = current_id
				current_id = current_id + 1

			id_ = labels_id[label]
			#print(labels_id)
			#y_labels.append(label)
			#x_train.append(path)
			pil_image = Image.open(path).convert("L") #python image library (L-> Grayscale xD)
			image_array =  np.array(pil_image, "uint8")
			#print(image_array)  #Every pixel into numbers
			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

			for (x, y, w, h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_labels.append(id_)


#print(y_labels)
#print(x_train)


with open("labels.pickle",'wb') as f:
	pickle.dump(labels_id, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")


