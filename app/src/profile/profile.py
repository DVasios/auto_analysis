# Basic Libs
import pandas as pd

# Other Libs
from visions.functional import detect_type
from visions.typesets import StandardSet
import visions

# profile.py
class Profile:
    def __init__(self, df):
        self.df = df 
        self.typeset = StandardSet()

    # Features, Rows
    def df_shape(self):
        return {
            'features': self.df.shape[1],
            'rows': self.df.shape[0]
        }

    # Data Type | Standard Set | Integer,  Float, Boolean, Categorical, Complex, DateTime, Object, String
    def f_data_type(self, f : pd.DataFrame):

        return str(self.typeset.infer_type(f.astype(str)))

    # Feature Type | Categorical, Numerical, Alphanumerical | {Data Type, Unique values Ratio, Thres}
    def f_feature_type(self, data_type, unique_values_ratio, thres = 0.1): 

        if ((data_type == 'String' or 
            data_type == 'Integer' or 
            data_type == 'Float' or 
            data_type == 'Boolean') and unique_values_ratio < thres):
            return 'Categorical'
        elif (data_type == 'Integer' or data_type == 'Float'):
            return 'Numerical'
        else: 
            return 'Alphanumerical'
        
    # Qualitative Characteristics | Nominal, Interval, Ordinal, Binary
    def f_qual(self, data_type):
        if (data_type == 'String'):
            return 'Nominal'
        elif (data_type == 'Int' or data_type == 'Float'):
            return 
        
    # Role | Input (Independant), ID (Unique Identifier)
    def f_role(self, f, f_target):
        
        f_unique_values = len(f.unique())
        f_total_values = len(f)

        if (f_unique_values / f_total_values) > 0.9:
            return 'id'
        elif (f.name == f_target): 
            return 'target'
        else:
            return 'input' 
        
    ## Feature Profiling  | {Feature, Objective, Unique Values Threshold}
    def f_profile(self, f, f_objective, unique_values_thres):
        
        # Data Type
        data_type = self.f_data_type(f)

        # Role
        role = self.f_role(f, f_objective['target_feature'])

        # Unique Values Ration
        unique_values = len(f.value_counts()) 
        unique_values_ratio = unique_values / f.count()

        # Feature Type
        feature_type = self.f_feature_type(data_type, unique_values_ratio, unique_values_thres)

        return {
            'data_type' : data_type,
            'role' : role,
            'unique_values' : unique_values,
            'feature_type' : feature_type
        }

    ## All feature profiling
    def df_profile(self, df_objective, unique_values_ratio):
        df_profile = {}

        # Name
        df_profile['name'] = 'Titanic'

        # Dataset
        df_profile['dataset'] = self.df_shape()

        # Target Feature
        df_profile['target_feature'] = df_objective['target_feature']

        # Features
        df_features = {}
        for f in self.df.columns: 
            df_features[f] = self.f_profile(self.df[f], df_objective, unique_values_ratio)
        df_profile['features'] = df_features

        return df_profile 

