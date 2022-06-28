from tkinter import W
import cv2, pafy
import cvlib as cv
import numpy as np
from keras.models import model_from_json
from keras.models import load_model
from keras.preprocessing.image import img_to_array


# from predictions.functionalities.camera_detection import read_from_camera

class Detection_robot:

    def __init__(self):
        self.frame = None
        self.ret = False
        self.gender_model = None
        self.age_model = None
        self.emotion_model = None

        self.MODEL_MEAN_VALUE = (78.4263377603,87.7689143744,114.895847746)

        self.gender_list = ['Male','Female']
        self.emotion_list = {0 : "angry",1 : "disgusted",
        2 : "fearful",3 : "happy",
        4 : "neutral",5 : "sad",
        6 : "surprised"}
        
        self.emotion_colors = {0 : (0,0,255),1:(226,43,138),
        2 :  (0, 165, 255) ,3 : (0,255,0),
        4 : (255,255,255),5 :(255,0,0),
        6 : (255,255,0)}


    #le role de cette fonction c'est de charger tous les modules et les fichiers
    #de configuration et aussi de la data dont elles notre Ai aura besoin
    #pour faire la détection de Visage|Age|Sex|Emotion
    def start(self):
        # on charge nos modeles age et gender 
        self.age_model = load_model('models/age_model-005.h5')
        self.gender_model = load_model('models/gender_model-005.model')

        # on charge notre fichier json et on creer notre model
        json_file = open("models/emotion_model.json","r")
        loaded_model_json = json_file.read()
        json_file.close()
        # on charge le model à partir du fichier json
        self.emotion_model = model_from_json(loaded_model_json)
        # puis on applique les métriques calculés grace à l'apprentissage
        self.emotion_model.load_weights("models/emotion_model.h5")
    
    def read_from_camera(self):

        cap = cv2.VideoCapture(0)
        
        # on boucle sur les images générées par notre camera
        while True:

            self.ret,self.frame = cap.read()

            self.detection_process()

            cv2.imshow('frame',self.frame)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

    
    def read_from_image(self,imageLink):    
        self.frame = cv2.imread(imageLink)
        self.detection_process()

        cv2.imshow('frame',self.frame)
        cv2.waitKey(0)
        
    #cette fonction s'occupe de la detection des visages
    #à partir d'un védio youtube
    def read_from_youtube(self,video_url):
        
        try:
            url   = video_url
            video = pafy.new(url)
            best  = video.getbest()
            capture = cv2.VideoCapture(best.url)
            
            while True:
                self.ret, self.frame = capture.read()

                self.detection_process()

                cv2.imshow('frame',self.frame)
                if cv2.waitKey(1) & 0xff == ord('q'):
                    break

        except:
            print("there is something wrong with that video try again")

            
    #cette fonction s'occupe de la detection des visages
    # qui se trouvent dans une image
    def face_detection(self):
        faces, confidence = cv.detect_face(self.frame)
        return faces

    #cette fonction s'occupe de la totalité des processus qui s'applique
    #sur une image afin de faire la détection souhaitée
    def detection_process(self):
        font = cv2.FONT_HERSHEY_SIMPLEX
        # pour chaque image on essaye de détecter les visages
        #faces = self.face_detection()
        faces = self.face_detection()

        # si il y a au moins un visage détecté on applique nos processus de prédiction
        # d'age et le sex et aussi les emotion de ce visage la
        if(len(faces)>0) :
            # on boucle sur les visages détectés afin d'appliquer les processus de prédiction
            # pour chaque visage
            for (startX,startY,endX,endY) in faces :

                face_crop = np.copy(self.frame[startY:endY,startX:endX])
                age = self.age_detection(face_crop)
                gender = self.gender_detection(face_crop)
                (emotion, emotion_color) = self.emotion_detection(face_crop)

            
                cv2.rectangle(self.frame, (startX,startY), (endX,endY), (0,255,0), 2)

                overlay_text = gender +' | '+ age
                cv2.putText(self.frame,
                overlay_text,(startX,startY),font,1,
                emotion_color,2,cv2.LINE_AA)
                cv2.putText(self.frame,emotion,
                (startX+5,startY-40),
                font,1.2,emotion_color,2,cv2.LINE_AA)

    #cette fonction s'occupe de la detection de sex
    #à partir d'un visage passé en parametre
    def gender_detection(self, face_crop):
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)
        conf = self.gender_model.predict(face_crop)[0]

        idx = np.argmax(conf)
        gender = self.gender_list[idx]

        return gender

    #cette fonction s'occupe de la detection d'age
    #à partir d'un visage passé en parametre
    def age_detection(self, face_crop):

        face_crop = cv2.resize(face_crop,(200,200),interpolation=cv2.INTER_AREA)

        age_predict = self.age_model.predict(np.array(face_crop).reshape(-1,200,200,3))
        age = round(age_predict[0,0])
        age_estimation = (age // 5) * 5
        age_label ="("+str(age_estimation)+","+str(age_estimation + 5)+")"

        return age_label

    #cette fonction s'occupe de la detection de l'emotion
    #à partir d'un visage passé en parametre
    def emotion_detection(self, face_image):
        face_image_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        cropped_image = np.expand_dims(
            np.expand_dims(
            cv2.resize(face_image_gray, (48,48)),-1),0
        )
        emotion_prediction = self.emotion_model.predict(cropped_image)
        
        emotion_index = int(np.argmax(emotion_prediction))

        return (self.emotion_list[emotion_index], self.emotion_colors[emotion_index])