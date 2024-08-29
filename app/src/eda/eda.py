
from sklearn.preprocessing import LabelEncoder

class Eda:
    def __init__(self, df, profile):
        self.df = df 
        self.profile = profile

    ## Duplicates | Check if they exist
    def df_duplicates(self):
        return {
            'exist' : self.df.duplicated().any(),
            'sum' : self.df.duplicated().sum()
        } 
    
    ## Univariate Analysis

    # statistics
    def f_statistics(self, f, d_type, f_type):

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
                
                # calculate frequency of each categorical value
                freq = round(y / len(f), 2)
                f_statistics[i] = {
                    'value': y,
                    'frequency': freq
                }

        return f_statistics

    # missing values | return: {total missing values, percentage}
    def f_missing_data(self, f):

        # null values
        null_values = f.isnull().sum() 

        # empty values 
        empty_values = f.isin(['']).sum()

        # missing values
        missing_values = null_values + empty_values

        ## percentage
        percentage = round((missing_values / len(f)), 2)
        return {
            'missing_values': missing_values,
            'percentage': percentage
        }

    ## univariate | feature | {dataframe_feature, data type, feature type}
    def f_univariate(self, f, d_type, f_type): 
        f_univariate = {}

        # statistics
        f_univariate['statistics'] = self.f_statistics(f, d_type, f_type)

        # missing values
        f_univariate['missing_data'] = self.f_missing_data(f)

        return f_univariate

    # univariate | dataframe | {dataframe, dataframe_profiling}
    def df_univariate(self, df_prof):

        df_univariate = {}
        for f, d in df_prof['features'].items():
            df_univariate[f] = self.f_univariate(self.df[f], d['data_type'], d['feature_type'])

        df_univariate['features'] = df_univariate

        return df_univariate
    
    ## Bivariate Analysis

    # Correlation Analysis | Each feature with target variable
    def f_corr (self, f, corr_matrix):
        if f in corr_matrix:
            return round(corr_matrix[f], 2)
        else:
            return ''
        
    # Bivariate Analysis | Dataframe
    def df_bivariate(self, df_prof):

        # Check if target feature is in String format | Label Encoding
        if (df_prof['features'][df_prof['target_feature']]['data_type'] == 'String'):

            label_encoder = LabelEncoder()
            self.df[df_prof['target_feature']] = label_encoder.fit_transform(self.df[df_prof['target_feature']])

        df_bivariate = {}

        # Correlation Analysis
        corr_matrix = self.df.corr(numeric_only=True)[df_prof['target_feature']]
        for f in df_prof['features']:
            df_bivariate[f] = self.f_corr(f, corr_matrix)
        
        return df_bivariate
    
    # EDA  | {Datframe, Dataframe Profiling}
    def df_eda (self):

        df_eda = {}

        # Duplicates
        df_eda['duplicates'] = self.df_duplicates()

        # Univariate
        df_eda['univariate'] = self.df_univariate(self.profile)

        # Bivariate
        df_eda['bivariate'] = self.df_bivariate(self.profile)
    
        return df_eda
    
    # Dataset Description | {Dataframe, Objective, Unique Values Ratio}
    def df_describe(self):

        # Statistics
        df_stat = self.df_eda()

        # Duplicates
        self.profile['dataset']['duplicates'] = df_stat['duplicates']

        # Univariate
        for f, i in self.profile['features'].items():
            self.profile['features'][f]['eda'] = df_stat['univariate'][f]

        return self.profile