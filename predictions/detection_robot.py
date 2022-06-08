import cv2

from predictions.functionalities.init import init
import numpy as np
from keras.models import model_from_json

# from predictions.functionalities.camera_detection import read_from_camera

class Detection_robot:

    def __init__(self):
        self.frame = None
        self.ret = False
        self.gender_net = None
        self.age_net = None
        self.MODEL_MEAN_VALUE = (78.4263377603,87.7689143744,114.895847746)
        self.age_list = ['(0,2)','(4,6)','(8,12)','(15,20)','(25,32)',
                        '(38,43)','(48,53)','(60,100)']
        self.gender_list = ['Male','Female']
        self.emotion_list = {0 : "angry",1 : "disgusted",
        2 : "fearful",3 : "happy",
        4 : "neutral",5 : "sad",
        6 : "surprised"}
        
        self.emotion_list = {0 : "red",1 : "violet",
        2 : "orange",3 : "green",
        4 : "white",5 : "blue",
        6 : "yellow"}

        self.face_cascade = None
        self.emotion_model = None

    #le role de cette fonction c'est de charger tous les modules et les fichiers
    #de configuration et aussi de la data dont elles notre Ai aura besoin
    #pour faire la détection de Visage|Age|Emotion
    def start(self):
        (self.age_net,self.gender_net) = init()
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')     
        # on charge notre fichier json et on creer notre model
        json_file = open("Emotion_model/emotion_model.json","r")
        loaded_model_json = json_file.read()
        json_file.close()
        # on charge le model à partir du fichier json
        self.emotion_model = model_from_json(loaded_model_json)
        # puis on applique les métriques calculés grace à l'apprentissage
        self.emotion_model.load_weights("Emotion_model/emotion_model.h5")
        print("Chargement des models à partir du disque effectué")
    

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
        self.detection_proccess()

        cv2.imshow('frame',self.frame)
        cv2.waitKey(0)
        

    def read_from_youtube(video_url):
        pass

    def face_detection(self):
        gray = cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)
        faces =  self.face_cascade.detectMultiScale(gray,1.1,5)

        return faces

    
    def detection_process(self):
        font = cv2.FONT_HERSHEY_SIMPLEX
        gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        # pour chaque image on essaye de détecter les visages
        faces = self.face_detection()

        # si il y a au moins un visage détecté on applique nos processus de prédiction
        # d'age et le sex et aussi les emotion de ce visage la
        if(len(faces)>0) :

            # on boucle sur les visages détectés afin d'appliquer les processus de prédiction
            # pour chaque visage
            for (x,y,w,h) in faces :
                face_image = self.frame[y:y+h,h:h+w].copy()
                face_image_blob = cv2.dnn.blobFromImage(
                    face_image,1,(227,227),
                    self.MODEL_MEAN_VALUE,
                    swapRB=False)

                age = self.age_detection(face_image_blob)
                gendre = self.gendre_detection(face_image_blob)
                (emotion_index,emotion) = self.emotion_detection(face_image)
                cv2.rectangle(self.frame,(x,y),(x+w,y+h),(255,0,0),2)

                overlay_text = gendre +' | '+ age
                cv2.putText(self.frame,overlay_text,(x,y),font,1,(255,255,255),1,cv2.LINE_AA)


    def gendre_detection(self, face_image_blob):
        self.gender_net.setInput(face_image_blob)
        gender_preds = self.gender_net.forward()
        gender = self.gender_list[gender_preds[0].argmax()]

        return gender

    def age_detection(self, face_image_blob):
        self.age_net.setInput(face_image_blob)
        age_preds = self.age_net.forward()
        age = self.age_list[age_preds[0].argmax()]
        
        return age