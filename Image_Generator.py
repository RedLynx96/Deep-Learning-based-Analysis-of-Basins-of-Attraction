from keras.preprocessing.image import ImageDataGenerator
import random
from matplotlib import image
import numpy as np
import scipy.io as sp
import tensorflow as tf
from tensorflow.keras.utils import to_categorical #Since labels must be strings

# Create an empty data generator
datagen = ImageDataGenerator(horizontal_flip=True)

def custom_generator(dataframe, batch_size, Apply_DataAug, metric_return):
    i = 0
    while True:
        batch = {'images': [], 'FDim_label': [], 'Sb_label': [], 'Sbb_label': [], 'Wada_label': []}
        dataframe = dataframe.sample(frac = 1).reset_index(drop=True)
        
        for b in range(batch_size):    
            if i == batch_size-1:
                i = 0
                
            # Get the "image" from the dataframe and store it as the original arrayy
            mat_contents = sp.loadmat(dataframe["Basin"][b])
            mat_array = mat_contents.get(list(mat_contents.keys())[3])/10
            image = mat_array.reshape(333,333,1)
            
            # Data augmentation
            if Apply_DataAug == True:
                if random.random() > 0.5:
                    image = datagen.apply_transform(image,transform_parameters={'flip_vertical': True})
                if random.random() > 0.5: 
                    image = datagen.apply_transform(image,transform_parameters={'flip_horizontal': True})
            
            image = tf.keras.preprocessing.image.img_to_array(image)

            # Read data from csv using the name of current image
            mat_label_Fdim = dataframe["FDim"][b]
            mat_label_Sb = dataframe["Sb"][b]
            mat_label_Sbb = dataframe["Sbb"][b]
            mat_label_Wada = dataframe["Wada"][b]

            batch['images'].append(image)
            batch['FDim_label'].append([mat_label_Fdim])
            batch['Sb_label'].append([mat_label_Sb])
            batch['Sbb_label'].append([mat_label_Sbb])
            batch['Wada_label'].append([mat_label_Wada])

            i += 1
            
        batch['images'] = np.asarray(batch['images']).reshape(-1,333,333,1)
        batch['FDim_label'] = np.array(batch['FDim_label'])
        batch['Sb_label'] = np.array(batch['Sb_label'])
        batch['Sbb_label'] = np.array(batch['Sbb_label'])
        batch['Wada_label'] = to_categorical(np.array(batch['Wada_label']))

        if metric_return == 'FDim':
            yield batch['images'], batch['FDim_label']
        if metric_return == 'Sb':
            yield batch['images'], batch['Sb_label']
        if metric_return == 'Sbb':
            yield batch['images'], batch['Sbb_label']
        if metric_return == 'Wada':
            yield batch['images'], batch['Wada_label']        

def Image_Loader(dataframe, range_0, range_length):
   eval_set = []
   FDim_label = []
   Sb_label = []
   Sbb_label = [] 
   Wada_label = []

   for i in range(range_0,range_length):
      # Get the "image" from the dataframe and stores it as the original arrayy
      
      mat_contents = sp.loadmat(dataframe["Basin"][i])
      mat_array = mat_contents.get(list(mat_contents.keys())[3])/10
      image = mat_array.reshape(333,333,1)
      
      mat_label_Fdim = dataframe["FDim"][i]
      mat_label_Sb = dataframe["Sb"][i]
      mat_label_Sbb = dataframe["Sbb"][i]
      mat_label_Wada =  dataframe["Wada"][i]

      eval_set.append(image)
      FDim_label.append(mat_label_Fdim)
      Sb_label.append(mat_label_Sb)
      Sbb_label.append(mat_label_Sbb)
      Wada_label.append(mat_label_Wada)
            
   eval_set = np.asarray(eval_set)  
   FDim_label = np.array(FDim_label)
   Sb_label = np.array(Sb_label)
   Sbb_label = np.array(Sbb_label)
   Wada_label = to_categorical(np.array(Wada_label))
   
   return eval_set, FDim_label, Sb_label, Sbb_label, Wada_label