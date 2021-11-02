#region  Import Libraries
from __future__ import print_function
import keras
import os
import cv2
import csv
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, Reshape, concatenate
#from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Reshape, Input
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix,classification_report
from keras.preprocessing import image
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint

import numpy as np
import matplotlib.pyplot as plt
#endregion

IMG_HEIGHT, IMG_WIDTH  = 111, 72
CHANNELS=3
num_classes = 1
seq_steps = 2
input_shape = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

DATASET_PATH ='E:\\Graduate_CS\\Thesis_Work\\CNN\\Riddhi\\argylestreet_20190328\\2019\\03\\GCP_Temp\\dec_frame_size_FG\\'

def load_model_data(data_file):
    
    if os.path.exists(DATASET_PATH +'/'+ data_file):
        with open(DATASET_PATH +'/'+ data_file, newline='') as csvfile:
            labelsfile = list(csv.reader(csvfile))
    else:
        labelsfile = [[]]

    data_file = labelsfile          # For .csv file

    X_ordered = np.empty((len(data_file), seq_steps, IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=float)
    time_diff_ordered = np.empty((len(data_file), seq_steps, 1), dtype=float)
    labels = np.empty((len(data_file), 1), dtype=float)
    images_lst = list()
    i = 0
    
    for i_row in data_file:
        if i_row[0].endswith('.jpg') and i_row[1].endswith('.jpg') and i_row[2].endswith('.jpg') and i_row[3].endswith('.jpg'):
            
            X_ordered[i][0] = cv2.normalize(cv2.imread(DATASET_PATH +'/'+ i_row[2], cv2.IMREAD_COLOR), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # uint8 image
            X_ordered[i][1] = cv2.normalize(cv2.imread(DATASET_PATH +'/'+ i_row[3], cv2.IMREAD_COLOR), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # uint8 image
    
            #X_ordered[i][0] = cv2.imread(DATASET_PATH +'/'+ i_row[1], cv2.IMREAD_COLOR)
            #X_ordered[i][1] = cv2.imread(DATASET_PATH +'/'+ i_row[2], cv2.IMREAD_COLOR)
    
            time_diff_ordered[i][0] = i_row[7]
            time_diff_ordered[i][1] = i_row[8]
            
            i+=1

    labels = [item[4] for item in data_file]
    labels = np.array(labels).astype(float)

    return X_ordered, labels, time_diff_ordered

# Set the seed value of the random number generator
random_seed = 2
np.random.seed(random_seed)

#(x_train, y_train), (x_test, y_test) = mnist.load_data()
X_test_ordered, Y_test_ordered, TD_test_ordered = load_model_data('Test_formatted.csv')
X_train_ordered, Y_train_ordered, TD_train_ordered = load_model_data('Train_formatted.csv')
X_val_ordered, Y_val_ordered, TD_val_ordered = load_model_data('Val_formatted.csv')


#X_train, X_val, Y_train, Y_val, TD_train, TD_val = train_test_split(X_train_ordered, Y_train_ordered, TD_train_ordered, test_size=0.20, shuffle = True)
#X_train, X_test, Y_train, Y_test, TD_train, TD_test = train_test_split(X_train, Y_train, TD_train, test_size=0.25, shuffle = True)
X_test, Y_test, TD_test = X_test_ordered, Y_test_ordered, TD_test_ordered
X_train, Y_train, TD_train = X_train_ordered, Y_train_ordered, TD_train_ordered
X_val, Y_val, TD_val = X_val_ordered, Y_val_ordered, TD_val_ordered



X_train_T0 = X_train[:,0,:,:,:]
X_test_T0 = X_test[:,0,:,:,:]
X_val_T0 = X_val[:,0,:,:,:]
print(X_train_T0.shape)

X_train_T1 = X_train[:,1,:,:,:]
X_test_T1 = X_test[:,1,:,:,:]
X_val_T1 = X_val[:,1,:,:,:]
print(X_train_T1.shape)

TD_train_T1 = TD_train[:,1]
TD_test_T1 = TD_test[:,1]
TD_val_T1 = TD_val[:,1]

print("No of Training data: ",len(X_train))
print("No of Test data: ",len(X_val))
print("No of Validation data: ",len(X_test))

# convert class vectors to binary class matrices
# Y_train = keras.utils.to_categorical(Y_train, num_classes)
# Y_val = keras.utils.to_categorical(Y_val, num_classes)
# Y_test = keras.utils.to_categorical(Y_test, num_classes)

Y_train = Y_train.reshape(len(Y_train),1)
Y_test = Y_test.reshape(len(Y_test),1)
Y_val = Y_val.reshape(len(Y_val),1)

#region
"""#Step 2 - Configure the neural network architecture (graph)
The Sequential model assumes that there is one longstack, with no branching.

filters gives us the number of filters in the layer,the more filters we have, the more information we can learn

kernel_size is the size of the convolution filter activation is the activation function on each node, we use relu, could also use sigmoid

input_shape is the shape of the image. We reshaped the data above to get it in the right shape. The 1
represents a grayscale image. If you had a colour image (RGB), the last dimension would be 3.
"""
print('X_train shape:', X_train.shape, X_test.shape, X_val.shape)
print('Y_train shape:', Y_train.shape, Y_test.shape, Y_val.shape)

batch_size = 25
epochs = 100
lr_rate = 0.00015
momentum =  0.8

def create_convolution_layers(input_img):

    #model = Sequential()

    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', 
                 input_shape=(input_shape))(input_img)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(x)
    #x = Conv2D(filters=256, kernel_size=(1, 1), activation='relu')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    #x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu')(x)
    #x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu')(x)
    #x = Conv2D(filters=512, kernel_size=(1, 1), activation='relu')(x)
    #x = LeakyReLU(alpha=0.1)(x)
    #x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    return x

frame_T0 = Input(shape=input_shape)
T0_model = create_convolution_layers(frame_T0)

frame_T1 = Input(shape=input_shape)
T1_model = create_convolution_layers(frame_T1)

x = concatenate([T0_model, T1_model])
x = Conv2D(filters=256, kernel_size=(1, 1), activation='relu')(x)
x = Conv2D(filters=256, kernel_size=(1, 1), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)


TD_T1_data = Input(shape=(1,))
#x = concatenate([x, TD_T1_data])

x = concatenate([TD_T1_data, x])

x = Dense(7000, activation='relu')(x)
x = Dropout(0.2)(x)
#x = LeakyReLU(alpha=0.1)(x)
x = Dense(4000, activation='relu')(x)
#x = LeakyReLU(alpha=0.1)(x)
x = Dropout(0.2)(x)
#x = Dense(100, activation='linear')(x)
output = Dense(1, activation='linear')(x)
#model.summary()   # Show a summary of the network architecture

model = Model(inputs=[TD_T1_data, frame_T0, frame_T1], outputs=[output])
model.summary()
# Stochastic Gradient Descent with momentum and a validation set to prevent overfitting

adam = tf.keras.optimizers.Adam(lr=lr_rate, beta_1=0.8, beta_2=0.9)
SGD = tf.keras.optimizers.SGD(lr=lr_rate, momentum=momentum)
Adadelta = keras.optimizers.Adadelta(lr=lr_rate, rho=0.85)

model.compile(loss=keras.losses.mean_squared_error,
           optimizer=Adadelta,
#           optimizer="adam",
#           optimizer=Adagrad,
            metrics=['mae'])

earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=epochs, verbose=True, mode= 'min')
checkpoint = tf.keras.callbacks.ModelCheckpoint('./vgg_cnn_modelcp.ckpt', monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=True, mode='auto', period=1)
# The history structure keeps tabs on what happened during the session

history = model.fit([TD_train_T1, X_train_T0, X_train_T1], Y_train,
          batch_size = batch_size,
          epochs = epochs,
          verbose = 1,
          validation_data = ([TD_val_T1, X_val_T0, X_val_T1], Y_val),
          callbacks=[earlystopper, checkpoint])

model.load_weights('./vgg_cnn_modelcp.ckpt')
score = model.evaluate([TD_test_T1, X_test_T0, X_test_T1], Y_test, verbose=0)

print(X_test_T0.shape)
print(X_test_T1.shape)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

predictions = model.predict([TD_test_T1, X_test_T0, X_test_T1], verbose=0)

cm = confusion_matrix(Y_test.argmax(axis=1), predictions.argmax(axis=1))
print(cm)

#target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5']
#print(classification_report(Y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=target_names))

for i in range(X_test_T0.__len__()):
    # subplt = plt.subplot(int(i / 10) + 1, 10, i + 1)
    # no sense in showing labels if they don't match the letter
    #predicted_cars = np.argmax(predictions[i])
    #actual_cars = np.argmax(Y_test[i])
    print(predictions[i], end = ',')
    print(Y_test[i])

print('Test loss:', score[0])
print('Test MAE:', score[1])

plt.ylim(0,6)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Loss', 'val loss'], loc='upper left')
plt.show()