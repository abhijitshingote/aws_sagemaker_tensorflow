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
    args, _ = parser.parse_known_args()
    
    print(args)
    
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(100,input_shape=(1,),activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    

    def get_train_data(location):
        print(location)
#         X = np.load(os.path.join(location, 'training.npz'))['xtrain']
#         y=np.load(os.path.join(location, 'training.npz'))['ytrain']
        df=pd.read_csv(os.path.join(location, 'salary.csv'))
        X,y=df.iloc[:,0],df.iloc[:,1]
        X=np.array(X).astype('float32').reshape(-1,1)
        y=np.array(y).astype('float32')
        return X,y

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),loss='mse',metrics=['accuracy'])
    
    X,y=get_train_data(args.train)
    r=model.fit(X,y,epochs=args.epochs)