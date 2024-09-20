# Libraries

# Core Libraries
import pandas as pd

# Sk learn 
from sklearn.impute import SimpleImputer

# App Libs
from .prof import Profile

# Class Clean
class Clean:

    # Constructor
    def __init__(self, train, test, profile, params):
        self.train = train
        self.test = test
        self.profile = profile
        self.params = params

    # Getters & Setters
    @property
    def get_train(self):
        return self.train

    @property
    def get_test(self):
        return self.test

    # Drop Duplicates
    def drop_duplicates(self):

        self.train.drop_duplicates()
        self.test.drop_duplicates()

    # Drop Features | IDs, Missing Values > Thres 
    def drop(self, drop_perc): 

        # Drop Features with missing percentage > thres
        for f, details in self.profile['features'].items():

            # If missing values are above threshold | Drop
            if (details['eda']['missing_data']['percentage'] >= drop_perc):

                self.train = self.train.drop(columns=[f])
                self.test = self.test.drop(columns=[f])

    # Data Imputation | {Dataframe, Profile, EDA}
    def impute(self, num_type):

        # Split Categorical and Numerical Columns
        numerical_columns = []
        categorical_columns = []
        alphanumerical_columns = []

        # For each feature
        for f, p in self.profile['features'].items():

            # Split | If missing percentage is above 0
            if (p['eda']['missing_data']['missing_values'] > 0.0):
                if (p['feature_type'] == 'Numerical'):
                    numerical_columns.append(f)

                elif (p['feature_type'] == 'Categorical'):
                    categorical_columns.append(f)


                elif (p['feature_type'] == 'Alphanumerical'):
                    alphanumerical_columns.append(f)

        # Imputers
        numerical_imputer = SimpleImputer(strategy=num_type)
        categorical_imputer = SimpleImputer(strategy='most_frequent')

        # Numerical
        if (len(numerical_columns) > 0):

            # Train
            self.train[numerical_columns] = numerical_imputer.fit_transform(self.train[numerical_columns])

            # Test
            self.test[numerical_columns] = numerical_imputer.transform(self.test[numerical_columns])

        # Mode | Categorical
        if (len(categorical_columns) > 0):

            # Train
            self.train[categorical_columns] = categorical_imputer.fit_transform(self.train[categorical_columns])

            # Test
            self.test[categorical_columns] = categorical_imputer.transform(self.test[categorical_columns])

        # Empty Space | Alphanumerical
        if (len(alphanumerical_columns) > 0):

            # Train
            self.train[alphanumerical_columns] = self.train[alphanumerical_columns].fillna('')

            # Test
            self.test[alphanumerical_columns] = self.test[alphanumerical_columns].fillna('')

    # Handle Outliers | {Dataframe, Profile, EDA, Threshold}
    def handle_outliers(self, iqr_thres : float = 1.5):

        # Numerical Columns
        numerical_columns = []
        for f, p in self.profile['features'].items():
            if (p['feature_type'] == 'Numerical'):
                numerical_columns.append(f)

        for f in numerical_columns: 

            # Boundaries on train set
            Q1 = self.train[f].quantile(0.25)
            Q3 = self.train[f].quantile(0.75)

            IQR = Q3 - Q1 

            # Train
            outliers_train = ((self.train[f] < (Q1 - iqr_thres * IQR)) | (self.train[f] > (Q3 + iqr_thres * IQR)))
            self.train = self.train[~outliers_train]

            # Test
            # outliers_test = ((self.test[f] < (Q1 - iqr_thres * IQR)) | (self.test[f] > (Q3 + iqr_thres * IQR)))
            # self.test = self.test[~outliers_test]

    # Dataframe Clean | {Dataframe, Description, MV Thres, Outlier Thres, Unique Value Thres}
    def clean (self, profile_params):

        # Drop Duplicates
        self.drop_duplicates()

        # Drop Features
        self.drop(self.params['drop_thres'])

        ## Describe
        profile = Profile(self.train, self.profile['target_feature'], profile_params)
        self.profile = profile.df_profile()

        # Data Imputation 
        self.impute(self.params['num_type'])

        ## Describe
        profile = Profile(self.train, self.profile['target_feature'], profile_params)
        self.profile = profile.df_profile()

        # Drop Outliers 
        if (self.params['outlier_thres'] != 'none'):
            self.handle_outliers(self.params['outlier_thres'])