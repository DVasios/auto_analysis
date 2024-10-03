# Basic
import os
import pandas as pd
from scipy.io import arff

# Other
import ijson

class Gather:
    def __init__(self, filepath):
        self.__filepath = filepath 

    # File Characteristics
    def __descr (self): 
        descr = {
            'FileType': os.path.splitext(self.__filepath)[1],
            'FileSize': os.path.getsize(self.__filepath),
        }
        return descr

    # Converters | File type to Dataframe
    def __convert (self, file_descr, nrows = 10000):

        # JSON
        if (file_descr['FileType'] == '.json'):

            # Open File and Convert it to JSON Object
            with open(self.__filepath, 'r') as file:
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

            # Check whether there is a header 
            return pd.read_csv(self.__filepath, nrows=nrows)
        
        # Arff
        elif (file_descr['FileType'] == '.arff'):
            data = arff.loadarff(self.__filepath)
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
        file_characteristics = self.__descr()

        # Dataframe Initialization 
        df = self.__convert(file_characteristics)

        return df