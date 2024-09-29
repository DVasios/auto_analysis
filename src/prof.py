# Libraries

# Visions | Data Type Detection
from visions.typesets import StandardSet

class Profile: 

    def __init__(self, d, target, params):
        self.__d = d
        self.__target = target
        self.__params = params

    ## Dataset

    # Features, RowRandom Search Optimizations
    def __shape(self):
        return {
            'features': self.__d.shape[1],
            'rows': self.__d.shape[0]
        }
    
    # Duplicates | Check if they exist
    def __duplicates(self):
        return {
            'exist' : bool(self.__d.duplicated().any()),
            'sum' : int(self.__d.duplicated().sum())
        } 
    
    # Dataset
    def dataset(self):

        dataset = {}

        # Dataset
        dataset = self.__shape()

        # Duplicates
        dataset['duplicates'] = self.__duplicates()

        return dataset

    ## Features Profile

    # Data Type | Standard Set | Integer,  Float, Boolean, DateTime, String
    def __data_type(self, f):

        typeset = StandardSet()
        return str(typeset.infer_type(f.astype(str)))

    # Feature Type | Categorical, Numerical, Alphanumerical | {Data Type, Unique values Ratio, Thres}
    def __feature_type(self, f, data_type): 

        # Unique Values Ratio
        unique_values = len(f.value_counts()) 
        unique_values_ratio = unique_values / f.count()

        if ((f.dtype.name == 'object') and unique_values_ratio < self.__params['cat_thres']):
            return 'Categorical'
        elif (data_type == 'Integer' or data_type == 'Float'):
            return 'Numerical'
        else: 
            return 'Alphanumerical'
        
    # Role | Input (Independant), ID (Unique Identifier)
    def __role(self, f, data_type):
        
        f_unique_values = len(f.unique())
        f_total_values = len(f)

        if ((data_type != 'Float' and data_type != 'Integer') and f_total_values > 0 and (f_unique_values / f_total_values)) > self.__params['id_thres']: 
            return 'id'
        elif (f.name == self.__target): 
            return 'target'
        else:
            return 'input' 
        
    ## Features Profiling  | {Feature, Objective, Unique Values Threshold}
    def feature_profile(self, f):
        
        # Data Type
        data_type = self.__data_type(f)

        # Role
        role = self.__role(f, data_type)

        # Feature Type
        feature_type = self.__feature_type(f, data_type)

        return {
            'data_type' : data_type,
            'role' : role,
            'feature_type' : feature_type
        }
        
    ## Features Univariate

    # Statistics
    def __statistics(self, f, d_type, f_type):

        f_statistics = {}
        if f_type == 'Numerical' and d_type != 'String':

            descr = f.describe()
            f_statistics = {
                'count' : int(descr['count']),
                'mean' : float(round(descr['mean'], 2)),
                'std' : float(round(descr['std'], 2)),
                'min' : float(round(descr['min'], 2)),
                'max' : float(round(descr['max'], 2))
            }
        elif f_type == 'Categorical': 
            for i, y in f.value_counts().items():
                
                # Calculate Frequency of each categorical value
                freq = round(y / len(f), 2)
                f_statistics[i] = {
                    'value': y,
                    'frequency': int(freq)
                }

        return f_statistics

    # Missing Values | Return: {Total Missing Values, Percentage}
    def __missing_data(self, f):

        # Null Values
        null_values = f.isnull().sum() 

        # Empty Values 
        empty_values = f.isin(['']).sum()

        # Missing Values
        missing_values = null_values + empty_values

        ## Percentage
        percentage = round((missing_values / len(f)), 2)
        return {
            'missing_values': bool(missing_values),
            'percentage': float(percentage)
        }

    ## Univariate | Feature | {Dataframe_Feature, Data Type, Feature Type}
    def feature_univariate(self, f, d_type, f_type): 
        f_univariate = {}

        # Statistics
        f_univariate['statistics'] = self.__statistics(f, d_type, f_type)

        # Missing Values
        f_univariate['missing_data'] = self.__missing_data(f)

        return f_univariate

    ## All feature profiling
    def data_profile(self):

        df_profile = {}

        # Dataset
        df_profile['dataset'] = self.dataset()

        # Target Feature
        df_profile['target_feature'] = self.__target

        # Features Profiling
        df_features = {}
        for f in self.__d.columns: 
            df_features[f] = self.feature_profile(self.__d[f])
        df_profile['features'] = df_features

        # Features Univariate 
        for f, d in df_profile['features'].items():
            df_profile['features'][f]['eda'] = self.feature_univariate(self.__d[f], d['data_type'], d['feature_type'])

        return df_profile 