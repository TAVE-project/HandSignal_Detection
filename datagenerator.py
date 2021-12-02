# Reference : https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly 


import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image

class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, list_IDs, labels, batch_size=64, dim=(224,224,3), 
                 shuffle=True, n_classes=6):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.n_classes = n_classes
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size :
                               (index+1)*self.batch_size]
        
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        X, y = self.__data_generation(list_IDs_temp)
        
        return X, y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # X : (n_samples, *dim)
        #Initialization
        
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, self.n_classes))
        
        for i, ID in enumerate(list_IDs_temp):
            img = image.load_img(ID, target_size=(224,224,3))
            img = image.img_to_array(img)
            img = img/255
            X[i, ] = img

            y[i] = self.labels[ID]
            
        return X, y
    
    
    