import numpy as np
import cv2
import pickle #for label naming

#Rb -> Read Byte
labels = {}
with open("labels.pickle",'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()} #talk to me about this ;) (key, value pair conversion)



face_cascade = cv2.CascadeClassifier('cascade/data/haarcascade_frontalface_alt2.xml') #only front of a face
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
cap = cv2.VideoCapture(0)

while(True):
	ret, frame = cap.read()


	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
	#scaleFactor aghe peeche krke dekho

	#Bounding Box of Face
	for (x, y, w, h) in faces:
		print(x, y, w, h)
		roi_gray = gray[y:y+h, x:x+w] #roi -> Region of Interest
		roi_color = frame[y:y+h, x:x+w]
		#Recognition part
		id_, conf = recognizer.predict(roi_gray) #id's and confidence level

		if(conf>=45 and conf<=85):
			print(id_)
			print(labels[id_])
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			color = (255,255,255)
			stroke = 2
			cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

		#Recognition part end
		img_item = "my-image.png"
		cv2.imwrite(img_item, roi_gray)

		#drawing a rectangle to recognize
		color = (255, 0, 0) #BGR
		stroke = 2 

		end_cord_x = x+w
		end_cord_y = y+h
		cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke) 

		#Now Recognition Part to predict







	cv2.imshow('frame', frame) 


	if(cv2.waitKey(20) & 0xFF == ord('q')):
		break


cap.release()
cv2.destroyAllWindows()