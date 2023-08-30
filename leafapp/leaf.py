import warnings
warnings.filterwarnings("ignore")
import os
import shutil
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from PIL import Image

import tensorflow as tf
from keras.layers import Dense,Flatten
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

def prdct(fileloc):


    base_model=InceptionV3(input_shape=(256,256,3),include_top=False)

    for layer in base_model.layers:
        layer.trainable=False

    X= Flatten()(base_model.output)
    X= Dense(units=8,activation='softmax')(X)

    #model details


    model=Model(inputs=base_model.input, outputs=X)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()

    train_datagen=ImageDataGenerator(featurewise_center=True,rotation_range=0.4,width_shift_range=0.3,horizontal_flip=True,preprocessing_function=preprocess_input,zoom_range=0.4, shear_range=0.4)

    train_data = train_datagen.flow_from_directory(
    directory='./static/Dataset/Train',  # Path to the train data directory
    target_size=(256, 256),
    batch_size=36
    )

    train_data.class_indices

    t_img,label=train_data.next()

    def plotImages(img_arr,label):
        for idx ,img in enumerate(img_arr):
            if idx<=10:
                plt.figure(figsize=(5,5))
                plt.imshow(img)
                plt.title(img.shape)
                plt.axis=False
                plt.show()
    
    #plotImages(t_img,label)

    from keras.callbacks import ModelCheckpoint, EarlyStopping
    mc=ModelCheckpoint(filepath="./best_model.h5",
                  monitor="accuracy",
                  verbose=1,
                  save_best_only=True)

    es=EarlyStopping(monitor="accuracy",
                min_delta=0.01,
                patience=5,
                verbose=1)
    cb=[mc,es]


    from keras.models import load_model
    model=load_model("./best_model.h5")

    path= fileloc
    img=load_img(path,target_size=(256,256))

    i=img_to_array(img)
    i=preprocess_input(i)

    input_arr=np.array([i])
    input_arr.shape

    pred= np.argmax(model.predict(input_arr))

    if pred==0:
        result="It's Bacterial spot leaf disease"
    if pred==1:
        result="It's Black rot leaf disease"
    if pred==2:
        result="It's Early blight leaf disease"
    if pred==3:
        result="It's Healthy leaf"
    if pred==4:
        result="It's Late blight leaf disease"
    if pred==5:
        result="It's Mosaic virus leaf disease"
    if pred==6:
        result="It's Powdery leaf disease"
    if pred==7:
        result="It's Rust leaf disease"
 
    return result