# Libraries

# Sk learn 
from sklearn.impute import SimpleImputer

# App Libs
from .prof import Profile

class Clean:

    # Constructor
    def __init__(self, train, test, target, params):
        self.__train = train
        self.__test = test
        self.__target = target
        self.__params = params

        self.__profile =  None

    # Getters & Setters
    @property
    def get_train(self):
        return self.__train

    @property
    def get_test(self):
        return self.__test

    # Drop Duplicates
    def __drop_duplicates(self):

        self.__train.drop_duplicates()
        self.__test.drop_duplicates()

    # Drop Features | IDs, Missing Values > Thres 
    def __drop_features(self): 

        # Drop Features with missing percentage > thres
        for f, details in self.__profile['features'].items():

            # If missing values are above threshold | Drop
            if (details['eda']['missing_data']['percentage'] >= 0.7):

                self.__train = self.__train.drop(columns=[f])
                self.__test = self.__test.drop(columns=[f])

    # Data Imputation | {Dataframe, Profile, EDA}
    def __impute(self):

        # Split Categorical and Numerical Columns
        numerical_columns = []
        categorical_columns = []
        alphanumerical_columns = []

        # For each feature
        for f, p in self.__profile['features'].items():

            # Split | If missing percentage is above 0
            if (p['eda']['missing_data']['missing_values'] > 0.0):
                if (p['feature_type'] == 'Numerical'):
                    numerical_columns.append(f)

                elif (p['feature_type'] == 'Categorical'):
                    categorical_columns.append(f)


                elif (p['feature_type'] == 'Alphanumerical'):
                    alphanumerical_columns.append(f)

        # Imputers
        numerical_imputer = SimpleImputer(strategy=self.__params['clean']['impute_type'])
        categorical_imputer = SimpleImputer(strategy='most_frequent')

        # Numerical
        if (len(numerical_columns) > 0):

            # Train
            self.__train.loc[:, numerical_columns] = numerical_imputer.fit_transform(self.__train[numerical_columns])

            # Test
            self.__test.loc[:, numerical_columns] = numerical_imputer.transform(self.__test[numerical_columns])

        # Mode | Categorical
        if (len(categorical_columns) > 0):

            # Train
            self.__train.loc[:, categorical_columns] = categorical_imputer.fit_transform(self.__train[categorical_columns])

            # Test
            self.__test.loc[:, categorical_columns] = categorical_imputer.transform(self.__test[categorical_columns])

        # Empty Space | Alphanumerical
        if (len(alphanumerical_columns) > 0):

            # Train
            self.__train.loc[:, alphanumerical_columns] = self.__train[alphanumerical_columns].fillna('')

            # Test
            self.__test.loc[:, alphanumerical_columns] = self.__test[alphanumerical_columns].fillna('')

    # Handle Outliers | {Dataframe, Profile, EDA, Threshold}
    def __handle_outliers(self):

        # Numerical Columns
        numerical_columns = []
        for f, p in self.__profile['features'].items():
            if (p['feature_type'] == 'Numerical'):
                numerical_columns.append(f)
        # IQR 
        if (self.__params['clean']['outlier_type'] == 'iqr'):

            for f in numerical_columns: 

                # Boundaries on train set
                Q1 = self.__train[f].quantile(0.25)
                Q3 = self.__train[f].quantile(0.75)

                IQR = Q3 - Q1 

                # Train
                outliers = ((self.__train[f] < (Q1 - 1.5 * IQR)) | (self.__train[f] > (Q3 +  1.5 * IQR)))
                self.__train = self.__train[~outliers]

        # Z-Score
        if (self.__params['clean']['outlier_type'] == 'z-score'):

            for f in numerical_columns: 

                mean = self.__train[f].mean()
                std_dev = self.__train[f].std()

                z_scores = (self.__train[f] - mean) / std_dev

                threshold = 3
                outliers = (z_scores > threshold) | (z_scores < -threshold)
                self.__train = self.__train[~outliers]

    # Dataframe Clean | {Dataframe, Description, MV Thres, Outlier Thres, Unique Value Thres}
    def clean (self):

        ## Describe
        profile = Profile(self.__train, self.__target)
        self.__profile = profile.data_profile()

        # Drop Duplicates
        self.__drop_duplicates()

        # Drop Features
        self.__drop_features()

        ## Describe
        profile = Profile(self.__train, self.__target)
        self.__profile = profile.data_profile()

        # Data Imputation 
        self.__impute()

        ## Describe
        profile = Profile(self.__train, self.__target)
        self.__profile = profile.data_profile()

        # Drop Outliers 
        if (self.__params['clean']['outlier_type'] != 'none'):
            self.__handle_outliers()