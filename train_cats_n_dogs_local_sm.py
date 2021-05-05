# import numpy as np
# import pandas as pd
import argparse

import tensorflow as tf
import tensorflow.python.keras.backend as K

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument(
            '--training', type=str, default='cats_n_dogs/train')
    parser.add_argument(
        '--validation', type=str, default='cats_n_dogs/validation')
    parser.add_argument(
        '--model-dir', type=str, default='modelartefacts') 

    args, _ = parser.parse_known_args()

    epochs=args.epochs
    training=args.training
    validation=args.validation
    model_dir=args.model_dir

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(
        3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(
        filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(
        filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    # print(model.summary())
    # print('epcohs',epochs)
    # print('train:', training)
    # print('validation_dir:', validation)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        samplewise_center=True,
        horizontal_flip=True,
        validation_split=0.3
    )

    datagen_flow_object = datagen.flow_from_directory(
        training,
        target_size=(150, 150),
        batch_size=64
    )

    r = model.fit_generator(
        datagen_flow_object, epochs=epochs,
        validation_data=datagen.flow_from_directory(validation,
                                                    target_size=(150, 150),
                                                    batch_size=64)
    )


    model.save(model_dir)
    # print(model)
