from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
#from keras.optimizers import Adam
import tensorflow as tf
from prepare_data_for_training import PrepareDataForTraining

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dataHandler = PrepareDataForTraining("emotion_data")

train_generator,test_generator = dataHandler.get_emotion_data()

emotion_model = keras.Sequential()  #on utilise le model sequentiel

# creation du model
# on ajoute des couches de convolutions
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))  # pour eviter l'overfitting

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.25))
emotion_model.add(Dense(7, activation='softmax'))

emotion_model.compile(loss='categorical_crossentropy',
 optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=1e-6),
  metrics=['accuracy'])


# entrainement du model
emotion_model_info = emotion_model.fit_generator(
    train_generator,
    steps_per_epoch = 28709//64,
    epochs = 10,
    validation_data = test_generator,
    validation_steps = 7178//64
)

# enregistrement du model dans un fichier json
model_json = emotion_model.to_json()
with open("models/emotion_model.json", "w") as json_file:
    json_file.write(model_json)

# enregistrer le model dans un fichier .h5
emotion_model.save_weights("models/emotion_model.h5")






