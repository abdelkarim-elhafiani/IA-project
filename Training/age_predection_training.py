import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential,load_model,Model
from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,BatchNormalization,Flatten,Input
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

from Training.prepare_data_for_training import PrepareDataForTraining

dataHandler  =  PrepareDataForTraining("age_data")
age,images = dataHandler.get_data()

age = np.array(age,dtype=np.int64)
images = np.array(images) 

x_train_age, x_test_age, y_train_age, y_test_age = train_test_split(images, age, random_state=42)

#######################################
#La creation de notre reseaux de neuron
#######################################

age_model = Sequential()
age_model.add(Conv2D(128, kernel_size=3, activation='relu', input_shape=(200,200,3)))
age_model.add(MaxPool2D(pool_size=3, strides=2))

for i in range(3):
    age_model.add(Conv2D(128, kernel_size=3, activation='relu'))
    age_model.add(MaxPool2D(pool_size=3, strides=2))

age_model.add(Flatten())
age_model.add(Dropout(0.2))
age_model.add(Dense(512, activation='relu'))

age_model.add(Dense(1, activation='linear', name='age'))
              
age_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                           
check_point = ModelCheckpoint("models/age_model-{epoch:03d}.h5",
                                monitor="val_loss",verbose=0,
                                save_best_only=True,mode="auto")
                                
age_model.fit(x_train_age, y_train_age,
                    epochs=5, callbacks=[check_point],validation_split=0.1)
