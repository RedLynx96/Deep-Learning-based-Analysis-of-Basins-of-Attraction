from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from joblib import load

scaler = load('df_scaler.bin')

while True:
    print('Train a new CNN (1) or use the pre trained ResNet50 (2) ?')
    _value1 = int(input())
    if _value1 == 1:
        Training = True
        break
    elif _value1 == 2:
        Training = False
        break
    else:
        print('Please, choose between 1 or 2.')

while True:
    print('Which metric would you like to train/estimate: Fractal dimension (1), Basin entropy (2), Boundary basin entropy (3), or Wada property (4).')
    _value1 = int(input())
    if _value1 == 1:
        metric_return = 'FDim'
        break
    elif _value1 == 2:
        metric_return = 'Sb'
        break
    elif _value1 == 3:
        metric_return = 'Sbb'
        break
    elif _value1 == 4:
        metric_return = 'Wada'
        break
    else:
        print('Please, choose between options 1, 2, 3, or 4.')

train_df = []
val_df = []
test_df = []

DuffingPath = ''
Duffing_df = pd.read_csv(DuffingPath)
train_df.append(Duffing_df)

HHPath = ''
HH_df = pd.read_csv(HHPath)
val_df.append(HH_df)

NewtonPath = ''
Newton_df = pd.read_csv(NewtonPath)
train_df.append(Newton_df)

PAFPath = ''
PAF_df = pd.read_csv(PAFPath)
val_df.append(PAF_df)

MagneticPendulumPath = ''
MagneticPendulum_df = pd.read_csv(MagneticPendulumPath)
test_df.append(MagneticPendulum_df)

train_df = pd.concat(train_df).sample(frac=1).reset_index(drop=True)
val_df = pd.concat(val_df).sample(frac=1).reset_index(drop=True)
test_df = pd.concat(test_df).sample(frac=1).reset_index(drop=True)

train_df[['FDim','Sb','Sbb','Wada']] = scaler.fit_transform(train_df[['FDim','Sb','Sbb','Wada']])
val_df[['FDim','Sb','Sbb','Wada']] = scaler.fit_transform(val_df[['FDim','Sb','Sbb','Wada']])
test_df[['FDim','Sb','Sbb','Wada']] = scaler.fit_transform(test_df[['FDim','Sb','Sbb','Wada']])

print("There are", len(train_df), "basins for training which makes", len(train_df)/(len(train_df)+len(val_df)+len(test_df)), "of the total")
print("There are", len(val_df), "basins for validation which makes", len(val_df)/(len(train_df)+len(val_df)+len(test_df)), "of the total")
print("There are", len(test_df), "basins for testing which makes", len(test_df)/(len(train_df)+len(val_df)+len(test_df)), "of the total")

from Image_Generator import custom_generator
batch_size = 16
    
train_set= custom_generator(train_df, batch_size, Apply_DataAug = True, metric_return = metric_return)
valid_set = custom_generator(val_df, batch_size, Apply_DataAug = True, metric_return = metric_return)
train_steps = len(train_df) // batch_size 
valid_steps = len(val_df) // batch_size

#----------------------------------------- Architecture and training of the CNN -----------------------------------------------------------
# The architectures of the CNNs are defined on Architecture.py

import Architecture
from tensorflow.keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard, TerminateOnNaN, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from time import time
import os

sv_chk_path = 'results/checkpoints/'
sv_path = 'results/'
file_name = 'ResNet50_1Branch_' + metric_return

optimizer = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.00, amsgrad=True, clipnorm=1.0, clipvalue=0.5)

if metric_return != 'Wada':
    model = Architecture.ResNet50.ResNet50(input_shape=(333, 333, 1),outputs=1, activation='linear') 
    model.compile(optimizer=optimizer, loss='mse', metrics='mse')

    model_checkpoint = ModelCheckpoint(sv_chk_path+file_name+'model.hdf5', monitor='val_mse', verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau('val_mse', factor=0.999, patience=int(10), cooldown=0, min_lr=1e-6, verbose=1)
    early_stop = EarlyStopping('val_mse', patience=30, verbose=1)
else: 
    model = Architecture.ResNet50.ResNet50(input_shape=(333, 333, 1),outputs=2, activation='softmax') 
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics='accuracy')

    model_checkpoint = ModelCheckpoint(sv_chk_path+file_name+'model.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau('val_accuracy', factor=0.999, patience=int(10), cooldown=0, min_lr=1e-6, verbose=1)
    early_stop = EarlyStopping('val_accuracy', patience=30, verbose=1)
    
terminate = TerminateOnNaN()
tensorboard = TensorBoard(log_dir=sv_path+'logs/'+file_name.format(time()), histogram_freq=1, write_graph=True, write_images=True)
callbacks = [model_checkpoint, early_stop, terminate, tensorboard, reduce_lr]

model.summary()

# Fit the model

if Training == True:
    
    epochs = 100
    history = model.fit(train_set,
                        epochs = epochs, 
                        batch_size = batch_size,
                        steps_per_epoch = train_steps,
                        validation_data= valid_set,
                        validation_steps = valid_steps,
                        callbacks=callbacks)    

    model.save(sv_path+file_name+'modellast.h5')
    np.save(sv_chk_path+file_name+'history1.npy',history.history)
    
else:
    model.load_weights(sv_chk_path+file_name+'model.hdf5')
    print(file_name + ' is ready!')

#------------------------------------------------Image and results plot-----------------------------------------------------------------------------
import seaborn as sn
from sklearn.metrics import confusion_matrix
from Image_Generator import Image_Loader

test_df= test_df.sample(frac = 1).reset_index(drop=True)
test_image_set, FDim_label, Sb_label, Sbb_label, Wada_label = Image_Loader(test_df, 0, int(len(test_df)/10))

test_predictions = model.predict(test_image_set, verbose = 1)

if metric_return != 'Wada':
    
    test_final_df = pd.DataFrame({'FDim': FDim_label, 'Sb': Sb_label, 'Sbb': Sbb_label, 'fakecolumn': Sb_label})
    test_predictions_df = pd.DataFrame({'if_FDim': test_predictions[:,0], 'if_Sb': test_predictions[:,0], 'if_Sbb': test_predictions[:,0], 'Metriccc': test_predictions[:,0]})

    test_final_scaled_df = scaler.inverse_transform(test_final_df)
    test_predictions_scaled_df = scaler.inverse_transform(test_predictions_df)

    if metric_return == 'FDim':
        Metric = 0
    elif metric_return == 'Sb':
        Metric = 1
    elif metric_return == 'Sbb':
        Metric = 2
    elif metric_return == 'Wada':
        Metric = 3

    Test_Fdim = []
    for i in range(0,FDim_label.shape[0]):
        Test_Fdim.append(test_predictions_scaled_df[i][Metric])
        
    ErrFdim = []; 
    for ii in range(0,len(test_predictions_scaled_df)):
        ErrFdim.append(test_predictions_scaled_df[ii,Metric]-test_final_scaled_df[ii,Metric])

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(Test_Fdim,test_final_scaled_df[:,Metric])
    ax1.plot([1,2], [1,2], 'tab:red')
    ax1.set_title(metric_return)
    ax2.hist(np.array(ErrFdim), bins = 100)
    ax2.set_title('Err ' + metric_return)
    plt.show()

else:
    predicted_class = np.round(test_predictions)
    actual_class = Wada_label

    Matrix = confusion_matrix(
        actual_class.argmax(axis=1),predicted_class.argmax(axis=1))

    ax= plt.subplot()
    sn.heatmap(Matrix/np.sum(Matrix), xticklabels= ['Wada','No Wada'],  yticklabels= ['Wada','No Wada'], annot=True, 
                fmt='.2%', cmap='Blues')
    plt.rcParams.update({'font.size': 20})
    label_font = {'size':'16'}  # Adjust to fit
    ax.set_xlabel('Predicted labels', fontdict=label_font)
    ax.set_ylabel('Observed labels', fontdict=label_font)

    title_font = {'size':'16'}  # Adjust to fit
    ax.set_title('Confusion Matrix', fontdict=title_font)

    ax.tick_params(axis='both', which='major', labelsize=16)  # Adjust to fit
    ax.xaxis.set_ticklabels(['No Wada', 'Wada'])
    ax.yaxis.set_ticklabels(['No Wada', 'Wada'])

    plt.show()