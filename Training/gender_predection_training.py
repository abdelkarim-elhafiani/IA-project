import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from keras.models import Sequential,load_model,Model
from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,BatchNormalization,Flatten,Input
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint


from Training.prepare_data_for_training import PrepareDataForTraining


dataHandler = PrepareDataForTraining("gender_data")

gender,images = dataHandler.prepare_data_for_training()

gender = np.array(gender,dtype=np.int64)
images = np.array(images)

x_train_gender, x_test_gender, y_train_gender, y_test_gender = train_test_split(images, gender, random_state=42)


gender_model = Sequential()

gender_model.add(Conv2D(36, kernel_size=3, activation='relu', input_shape=(200,200,3)))

gender_model.add(MaxPool2D(pool_size=3, strides=2))
gender_model.add(Conv2D(64, kernel_size=3, activation='relu'))
gender_model.add(MaxPool2D(pool_size=3, strides=2))

gender_model.add(Conv2D(128, kernel_size=3, activation='relu'))
gender_model.add(MaxPool2D(pool_size=3, strides=2))

gender_model.add(Conv2D(256, kernel_size=3, activation='relu'))
gender_model.add(MaxPool2D(pool_size=3, strides=2))

gender_model.add(Conv2D(512, kernel_size=3, activation='relu'))
gender_model.add(MaxPool2D(pool_size=3, strides=2))

gender_model.add(Flatten())
gender_model.add(Dropout(0.2))
gender_model.add(Dense(512, activation='relu'))
gender_model.add(Dense(1, activation='sigmoid', name='gender'))

gender_model.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])

check_point = ModelCheckpoint("models/gender_model-{epoch:03d}.model",
                                monitor="val_loss",verbose=0,
                                save_best_only=True,mode="auto")

gender_model.fit(x_train_gender, y_train_gender,
                        epochs=5, callbacks=[check_point],validation_split=0.1)