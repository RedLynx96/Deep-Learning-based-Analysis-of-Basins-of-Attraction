from tkinter import Tk, Label, Button, filedialog, messagebox, ttk, simpledialog
import tkinter as tk
from pandastable import Table
import scipy.io as sp
from PIL import Image, ImageTk

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy.io as sp
import Architecture
from tensorflow.keras.optimizers import Adam

from joblib import load
scaler = load('df_scaler.bin')

import warnings
warnings.filterwarnings("ignore")

class Basin_Metrics_GUI:
    def __init__(self, master):
        
        self.master = master
        master.title("Basin Metrics Calculator")

        self.label = Label(master, text="Add a '.csv' file from a folder to calculate the metrics")
        self.label.pack()

        self.matrix_button = Button(master, text="Open file .csv", command = self.open_csv)
        self.matrix_button.pack()

        self.metrics_button = Button(master, text="Compute basin metrics", command=self.compute_Metrics)
        self.metrics_button.pack()
        
        self.visualize_basin_button = Button(master, text="Visualize basin", command=self.visualize_basin)
        self.visualize_basin_button.pack()

    def open_csv(self):
        file_path = filedialog.askopenfilename()
        
        if file_path:
            Basins_df = pd.read_csv(file_path, header=None)
            
        self.Basins_Characterized_df = pd.DataFrame(columns = ['Name', 'FDim', 'Sb', 'Sbb', 'Wada'])
        self.Basins_Characterized_df.Name = Basins_df[0].str.split('\\').str[-1]
        
        test_set = []

        for i in range(0,len(Basins_df)):
            
            # Get the "image" from the dataframe and stores it as the original arrayy
            mat_contents = sp.loadmat(Basins_df[0][i])
            mat_array = mat_contents.get(list(mat_contents.keys())[3])/10
            image = mat_array.reshape(333,333,1)

            test_set.append(image)
            
        self.eval_set = np.asarray(test_set) 
        
        matrix = self.eval_set[0].reshape(self.eval_set[0].shape[0],self.eval_set[0].shape[1]) #We reshape the matrix to be an image with 1 channel (grayscale) that us input in the CNN
        arrmax, arrmin = matrix.max(), matrix.min()
        norm_matrix = (matrix - arrmin) / (arrmax - arrmin)
        
        img = Image.fromarray(np.uint8(mpl.colormaps['viridis'].resampled(len(np.unique(norm_matrix))+1)(norm_matrix)*255))
        #img = Image.open(file_path)
        img.thumbnail((500, 500))  # Ajusta el tamaño de la imagen si es muy grande
        photo = ImageTk.PhotoImage(img)
        self.label.config(image=photo)
        self.label.image = photo
        
        messagebox.showinfo(message="Basins loaded succesfully", title="Basins loaded")
            
    def Load_CNN(self):
        self.CNN_path = 'results\checkpoints'
        
        self.Basin_metrics = ['FDim','Sb','Sbb','Wada']
        self.optimizer = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=True, clipnorm=1.0, clipvalue=0.5)

        self.Basin_Predictor = {} #We create a dictionary to store the trained models

        for i in self.Basin_metrics:
            if i != 'Wada':
                model = Architecture.ResNet50.ResNet50(input_shape=(333, 333, 1),outputs=1, activation='linear') 
                model.compile(optimizer=self.optimizer, loss='mse', metrics='mse')
                model.load_weights(self.CNN_path + '\ResNet50_1Branch_' + i + 'model.hdf5')
            else:
                model = Architecture.ResNet50.ResNet50(input_shape=(333, 333, 1),outputs=2, activation='softmax')
                model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics='accuracy')
                model.load_weights(self.CNN_path + '\ResNet50_1Branch_' + i + 'model.hdf5')
                
            self.Basin_Predictor[i] = model
    
    def compute_Metrics(self):
        
        prediction_label = Label(root, text= "Loading CNN...")
        prediction_label.pack()
        
        progressbar = ttk.Progressbar()
        progressbar.pack()
        self.update()
            
        self.Load_CNN()
        progressbar.step(10)
        prediction_label.config(text = 'Predicting fractal dimension...')
        self.update()
        
        self.Basins_Characterized_df.FDim = self.Basin_Predictor['FDim'].predict(self.eval_set)
        
        progressbar.step(20)
        prediction_label.config(text = 'Predicting basin entropy...')
        
        self.update()
        self.Basins_Characterized_df.Sb = self.Basin_Predictor['Sb'].predict(self.eval_set)
        
        progressbar.step(20)
        prediction_label.config(text = 'Predicting boundary basin entropy...')
        
        self.update()
        self.Basins_Characterized_df.Sbb = self.Basin_Predictor['Sbb'].predict(self.eval_set)
        
        progressbar.step(20)
        prediction_label.config(text = 'Predicting the pressence of the Wada property...')
        
        self.update()
        Wada = self.Basin_Predictor['Wada'].predict(self.eval_set)        
        self.Basins_Characterized_df.Wada = Wada.argmax(axis=1)
        progressbar.step(20)
        self.update()

        self.Basins_Characterized_df[["FDim","Sb","Sbb","Wada"]] = scaler.inverse_transform(self.Basins_Characterized_df[["FDim","Sb","Sbb","Wada"]])
        self.Basins_Characterized_df.Wada[np.where(self.Basins_Characterized_df.Wada == 0)] = 'No Wada'
        self.Basins_Characterized_df.Wada[np.where(self.Basins_Characterized_df.Wada == 1)] = 'Wada'

        self.Basins_Characterized_df.to_csv('Basins_Characterized.csv', index=False) 
        prediction_label.config(text = 'Done!')
        progressbar.step(9)
        self.update()
        self.Load_Table()
  
        
    def Load_Table(self):
        pt = Table(frame, dataframe=self.Basins_Characterized_df, showtoolbar=True,showstatusbar=True )
        pt.show()
        
        self.visualize_basin.pack()
       
    def visualize_basin(self):   
        Basin_Name = simpledialog.askstring('Basin visualizer', 'Enter the name of the basin you want to visualize (without the .mat extension):')
        Basin_index = np.where(self.Basins_Characterized_df.Name == str(Basin_Name) + '.mat')[0][0]
        
        matrix = self.eval_set[Basin_index].reshape(self.eval_set[Basin_index].shape[0],self.eval_set[Basin_index].shape[1]) #We reshape the matrix to be an image with 1 channel (grayscale) that us input in the CNN
        arrmax, arrmin = matrix.max(), matrix.min()
        norm_matrix = (matrix - arrmin) / (arrmax - arrmin)
        
        img = Image.fromarray(np.uint8(mpl.colormaps['viridis'].resampled(len(np.unique(norm_matrix))+1)(norm_matrix)*255))
        #img = Image.open(file_path)
        img.thumbnail((500, 500))  # Ajusta el tamaño de la imagen si es muy grande
        photo = ImageTk.PhotoImage(img)
        self.label.config(image=photo)
        self.label.image = photo
        
                
    def update(self):
        self.master.update_idletasks()
        self.master.update()

root = Tk()
my_gui = Basin_Metrics_GUI(root)
frame = tk.Frame(root)
frame.pack()

root.mainloop()