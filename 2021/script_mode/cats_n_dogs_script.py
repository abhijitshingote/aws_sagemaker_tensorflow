import argparse
import os

import tensorflow as tf
# import tensorflow.python.keras.backend as K

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])


    args, _ = parser.parse_known_args()

    epochs=args.epochs
    training=os.path.join(args.training,'train')
    validation=os.path.join(args.training,'validation')
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

#     save_history(args.model_dir + "/history.p", history)
    
    # create a TensorFlow SavedModel for deployment to a SageMaker endpoint with TensorFlow Serving
    model.save(args.model_dir + '/1')

