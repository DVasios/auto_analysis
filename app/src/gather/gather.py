# Basic
import os
import pandas as pd
import numpy as np
from scipy.io import arff

# Other
import ijson

class Gather:
    def __init__(self, filepath, target):
        self.filepath = filepath 
        self.target = target

    # File Characteristics
    def _descr (self): 
        descr = {
            'FileType': os.path.splitext(self.filepath)[1],
            'FileSize': os.path.getsize(self.filepath),
        }
        return descr

    # Converters | File type to Dataframe
    def _convert (self, file_descr, nrows = 20000):

        # JSON
        if (file_descr['FileType'] == '.json'):

            # Open File and Convert it to JSON Object
            with open(self.filepath, 'r') as file:
                data = ijson.items(file, 'item')
                json_object = []
                count = 0
                for line in data:
                    json_object.append(line)
                    count = count + 1
                    if(count == 1000): break
            return pd.DataFrame(json_object)

        # CSV
        elif (file_descr['FileType'] == '.csv'):

            # Check whether there is a header | TODO
            return pd.read_csv(self.filepath, nrows=nrows)
        
        # Arff
        elif (file_descr['FileType'] == '.arff'):
            data = arff.loadarff(self.filepath)
            df = pd.DataFrame(data[0])
            for i in df.columns:
                if (df[i].dtype.name == 'object'):
                    df[i] = df[i].apply(lambda x: int(x.decode('utf-8')))
            return df

        else: 
            return False
        
    # Gather Data
    def gather(self):

        # File Characteristics
        file_characteristics = self._descr()

        # Dataframe Initialization 
        df = self._convert(file_characteristics)

        return df