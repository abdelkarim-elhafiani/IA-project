import numpy as np
import cv2
import os
from keras.preprocessing.image import ImageDataGenerator

class PrepareDataForTraining:

    def __init__(self, data_name):
        self.data_name = data_name

    def get_emotion_data():
        # pre-traitement des images
        train_data_gen = ImageDataGenerator(rescale=1. / 255)
        test_data_gen = ImageDataGenerator(rescale=1. / 255)

        # recuperer toutes les images et les prétraiter

        #pour les données d'entrainement
        train_generator = train_data_gen.flow_from_directory(
            'Dataset/train',
            target_size=(48, 48),
            batch_size=64,
            color_mode='grayscale',
            class_mode='categorical'
        )

        #pour les données de tests
        test_generator = test_data_gen.flow_from_directory(
            'Dataset/test',
            target_size=(48, 48),
            batch_size=64,
            color_mode='grayscale',
            class_mode='categorical'
        )
        return train_generator,test_generator

    def get_age_or_gender_data():
        path = "Data_Used_For_Training"
        images = []
        age = []
        gender = []

        for img in os.listdir(path):
            ages = img.split("_")[0]
            genders = img.split("_")[1]
            img = cv2.imread(str(path)+"/"+str(img))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            images.append(np.array(img))
            age.append(np.array(ages))
            gender.append(np.array(genders))

        if data_name == "age_data":
            return age,images

        elif data_name == "gender_data":
            return gender,images