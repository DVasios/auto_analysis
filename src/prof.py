# Libraries

# Visions | Data Type Detection
from visions.typesets import StandardSet

class Profile: 

    def __init__(self, d, target, params):
        self.d = d
        self.target = target
        self.params = params

    ## Dataset

    # Features, Rows
    def shape(self):
        return {
            'features': self.d.shape[1],
            'rows': self.d.shape[0]
        }
    
    ## Duplicates | Check if they exist
    def duplicates(self):
        return {
            'exist' : self.d.duplicated().any(),
            'sum' : self.d.duplicated().sum()
        } 

    ## Features Characteristics

    # Data Type | Standard Set | Integer,  Float, Boolean, DateTime, String
    def data_type(self, f):

        typeset = StandardSet()
        return str(typeset.infer_type(f.astype(str)))

    # Feature Type | Categorical, Numerical, Alphanumerical | {Data Type, Unique values Ratio, Thres}
    def feature_type(self, f, data_type): 

        # Unique Values Ratio
        unique_values = len(f.value_counts()) 
        unique_values_ratio = unique_values / f.count()

        if ((f.dtype.name == 'object') and unique_values_ratio < self.params['cat_thres']):
            return 'Categorical'
        elif (data_type == 'Integer' or data_type == 'Float'):
            return 'Numerical'
        else: 
            return 'Alphanumerical'
        
    # Role | Input (Independant), ID (Unique Identifier)
    def role(self, f, data_type):
        
        f_unique_values = len(f.unique())
        f_total_values = len(f)

        if ((data_type != 'Float' and data_type != 'Integer') and f_total_values > 0 and (f_unique_values / f_total_values)) > self.params['id_thres']: 
            return 'id'
        elif (f.name == self.target): 
            return 'target'
        else:
            return 'input' 
        
    ## Feature Profiling  | {Feature, Objective, Unique Values Threshold}
    def profile(self, f):
        
        # Data Type
        data_type = self.data_type(f)

        # Role
        role = self.role(f, data_type)

        # Feature Type
        feature_type = self.feature_type(f, data_type)

        return {
            'data_type' : data_type,
            'role' : role,
            'feature_type' : feature_type
        }
        
    ## Features Univariate

    # Statistics
    def statistics(self, f, d_type, f_type):

        f_statistics = {}
        if f_type == 'Numerical' and d_type != 'String':

            descr = f.describe()
            f_statistics = {
                'count' : int(descr['count']),
                'mean' : round(descr['mean'], 2),
                'std' : round(descr['std'], 2),
                'min' : round(descr['min'], 2),
                'max' : round(descr['max'], 2)
            }
        elif f_type == 'Categorical': 
            for i, y in f.value_counts().items():
                
                # Calculate Frequency of each categorical value
                freq = round(y / len(f), 2)
                f_statistics[i] = {
                    'value': y,
                    'frequency': freq
                }

        return f_statistics

    # Missing Values | Return: {Total Missing Values, Percentage}
    def missing_data(self, f):

        # Null Values
        null_values = f.isnull().sum() 

        # Empty Values 
        empty_values = f.isin(['']).sum()

        # Missing Values
        missing_values = null_values + empty_values

        ## Percentage
        percentage = round((missing_values / len(f)), 2)
        return {
            'missing_values': missing_values,
            'percentage': percentage
        }

    ## Univariate | Feature | {Dataframe_Feature, Data Type, Feature Type}
    def univariate(self, f, d_type, f_type): 
        f_univariate = {}

        # Statistics
        f_univariate['statistics'] = self.statistics(f, d_type, f_type)

        # Missing Values
        f_univariate['missing_data'] = self.missing_data(f)

        return f_univariate

    ## All feature profiling
    def df_profile(self):

        # unique_values_ratio | Param

        df_profile = {}

        # Dataset
        df_profile['dataset'] = self.shape()

        # Duplicates
        df_profile['dataset']['duplicates'] = self.duplicates()

        # Target Feature
        df_profile['target_feature'] = self.target

        # Features
        df_features = {}
        for f in self.d.columns: 
            df_features[f] = self.profile(self.d[f])
        df_profile['features'] = df_features

        # Univariate
        for f, d in df_profile['features'].items():
            df_profile['features'][f]['eda'] = self.univariate(self.d[f], d['data_type'], d['feature_type'])

        return df_profile 