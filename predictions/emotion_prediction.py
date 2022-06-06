import cv2
import numpy as np
from keras.models import model_from_json

emotion_dict = {0 : "angry",1 : "disgusted",2 : "fearful",3 : "happy",4 : "neutral",5 : "sad",6 : "surprised"}

# on charge notre fichier json et on creer notre model
json_file = open("Emotion_model/emotion_model.json","r")
loaded_model_json = json_file.read()
json_file.close()
# on charge le model à partir du fichier json
emotion_model = model_from_json(loaded_model_json)
# puis on applique les métriques calculés grace à l'apprentissage
emotion_model.load_weights("Emotion_model/emotion_model.h5")
print("Chargement du model à partir du disque effectué")
# le model est maintenant pret à faire des prédictions

# video sur laquelle on va travailler
cap = cv2.VideoCapture(0)#pour la webcam

#on va utiliser une autre video pour commmencer

#cap = cv2.VideoCapture("lien de la video")

while True:

    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280,720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('Haarcascades/haarcascades_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # pour detecter les visages presents sur la camera

    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3,minNeighbors=5)

    # on prend maintenant chaque face disponible et on la traite

    for(x,y,w,h) in num_faces:
        # pour tracer un rectangle autour des faces
        cv2.rectangle(frame,(x,y-50),(x+w,y+h+10),(0,255,0),4)
        roi_gray_frame = gray_frame[y:y+h, x:x+w]
        # on recupère l'image de la frame à analyser sur la frame convertie en gray
        cropped_image = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48,48)),-1),0)

        # prédiction des émotions
        emotion_prediction = emotion_model.predict(cropped_image)
        emotion_index = int(np.argmax(emotion_prediction))
        cv2.putText(frame,emotion_dict[emotion_index],(x+5,y-20),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,0,0),2,cv2.LINE_AA)

    cv2.imshow("Détection d'émotions",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()