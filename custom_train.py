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
    
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1,input_shape=[1,]))
    print(model.summary())
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),loss='mae')
    

    def get_train_data(location):
        print(location)
#         X = np.load(os.path.join(location, 'training.npz'))['xtrain']
#         y=np.load(os.path.join(location, 'training.npz'))['ytrain']
        df=pd.read_csv(os.path.join(location, 'salary.csv'))
        X,y=df.iloc[:,0],df.iloc[:,1]
        X=np.array(X).astype('float32').reshape(-1,1)
        y=np.array(y/100000).astype('float32').reshape(-1,1)
        return X,y

    
    
    X,y=get_train_data(args.train)
    print(f'Xarray {X} type :{X.dtype} shape: {X.shape}')
    print(f'Yarray {y} type :{y.dtype} shape: {y.shape}')
    r=model.fit(X,y,epochs=args.epochs)
    print(model.weights)
    model.save(os.path.join(args.sm_model_dir, '000000001'), 'my_model.h5')