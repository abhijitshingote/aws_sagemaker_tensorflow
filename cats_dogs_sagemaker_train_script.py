import argparse
import os
import json
import tensorflow as tf
import numpy as np
import pandas as pd


if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--gibberish', type=int, default=10)
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    args, _ = parser.parse_known_args()
    
    print(args)
    
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),activation='relu',input_shape=(128,128,3)))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
    model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
    model.add(tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
    model.add(tf.keras.layers.Conv2D(filters=256,kernel_size=(2,2),activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
    model.add(tf.keras.layers.Conv2D(filters=512,kernel_size=(2,2),activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
#     model.add(tf.keras.layers.Conv2D(filters=1024,kernel_size=(3,3),activation='relu'))
#     model.add(tf.keras.layers.MaxPooling2D((2,2)))
#     model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100,activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2,activation='softmax'))
    print(model.summary())
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    
    
#     model=tf.keras.models.Sequential()
#     model.add(tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False,pooling='avg',weights='imagenet',input_shape=(128,128,3)))
#     model.add(tf.keras.layers.Dropout(0.5))
#     model.add(tf.keras.layers.Dense(2,activation='softmax'))
#     model.layers[0].trainable= False
#     model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    def get_train_data(location):
        print(location)
#         X = np.load(os.path.join(location, 'training.npz'))['xtrain']
#         y=np.load(os.path.join(location, 'training.npz'))['ytrain']
        datagen=tf.keras.preprocessing.image.ImageDataGenerator(samplewise_center=True,horizontal_flip=True,validation_split=0.3)
        datagen_train_flow_object=datagen.flow_from_directory(os.path.join(location, 'train'),target_size=(128,128),batch_size=12)
        return datagen_train_flow_object

    
    
    datagen_flow_object=get_train_data(args.train)
#     print(f'Xarray {X} type :{X.dtype} shape: {X.shape}')
#     print(f'Yarray {y} type :{y.dtype} shape: {y.shape}')
    r=model.fit_generator(datagen_flow_object,epochs=args.epochs)
#     print(model.weights)
#     model.save(os.path.join(args.sm_model_dir, '000000001'), 'my_model.h5')