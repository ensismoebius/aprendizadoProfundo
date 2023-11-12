import numpy as np
import pandas as pd
from scipy.io import loadmat
import torch
from torch.utils.data import DataLoader, Dataset

class EEGDataset(Dataset):
    
    def __init__(self, path):
        
        ####################
        ### Loading data ###
        ####################
        
        # '/home/ensismoebius/Documentos/UNESP/doutorado/databases/Base de Datos Habla Imaginada/S01/S01_EEG.mat',
        mat = loadmat(path, struct_as_record=True, squeeze_me=True, mat_dtype=False)
        
        #################################
        ### Setting up the properties ###
        #################################
        
        self.estimuli = {
                1 : "A",
                2 : "E",
                3 : "I",
                4 : "O",
                5 : "U",
                6 : "Arriba",
                7 : "Abajo",
                8 : "Adelante",
                9 : "Atrás",
                10 : "Derecha",
                11 : "Izquierda"
            }
        
        # Modalities
        self.modalities = {
            1 : "Imaginada",
            2 : "Falada" 
        }
        
        # Artfacts
        self.artfacts = {
            -1 : "Indiferente",  
            1 : "Com artefato", 
            2 : "Sem artefato" 
        }

        self.dataframe = pd.DataFrame(mat['EEG'])

        # Creates 6 channels of EEG data
        self.channels = 6
        self.channelsLength = 4096
        self._joinIntoArray(0, 4096, 'F3', self.dataframe)
        self._joinIntoArray(0, 4096, 'F4', self.dataframe)
        self._joinIntoArray(0, 4096, 'C3', self.dataframe)
        self._joinIntoArray(0, 4096, 'C4', self.dataframe)
        self._joinIntoArray(0, 4096, 'P3', self.dataframe)
        self._joinIntoArray(0, 4096, 'P4', self.dataframe)
        
        # Creates 3 more channels:
        # labels 'Estímulo'
        # artifacts 'Modalidade' and 'Artefatos'
        self._joinIntoValue(0, 1, 'Modalidade', self.dataframe)
        self._joinIntoValue(0, 1, 'Estímulo', self.dataframe)
        self._joinIntoValue(0, 1, 'Artefatos', self.dataframe)
        
        self.filteredData = self.dataframe[(self.dataframe['Modalidade'] == 1) & (self.dataframe['Artefatos'] == 1)]
        
        self.labels = self.filteredData['Estímulo'].values
        self.data = self.filteredData[['F3','F4','C3','C4','P3','P4']].values
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        sample_data =  torch.tensor(self.data[idx].tolist(),dtype=torch.float) # Convert numpy arrays to tensors
        sample_label = torch.tensor(self.labels[idx].tolist(),dtype=torch.float)  # Ensure labels are also tensors
        return sample_data, sample_label

    # Auxiliary methods            
    def _joinIntoArray(self, start_col, end_col, newColumn, dataframe):
        cols_to_join = dataframe.iloc[:, start_col:end_col].columns
        dataframe[newColumn] = dataframe[cols_to_join].apply(lambda x: np.array(pd.to_numeric(x, errors='coerce')), axis=1)
        dataframe.drop(cols_to_join, axis=1, inplace=True)
    
    def _joinIntoValue(self, start_col, end_col, newColumn, dataframe):
        cols_to_join = dataframe.iloc[:, start_col:end_col].columns
        dataframe[newColumn] = dataframe[cols_to_join].apply(lambda x: pd.to_numeric(x, errors='coerce'), axis=1)
        dataframe.drop(cols_to_join, axis=1, inplace=True)