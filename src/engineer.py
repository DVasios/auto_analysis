# Libraries

# Core Libraries
import pandas as pd

# Sk learn 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import mutual_info_classif

# App Libs
from .prof import Profile

class Engineer:

    # Constructor
    def __init__(self, train, test, target, params):
        self.__train = train
        self.__test = test
        self.__target = target
        self.__params = params

    # Getters
    @property
    def get_train(self):
        return self.__train
    
    @property
    def get_test(self):
        return self.__test

    ## Feature Extraction | {Series, Extraction Type, Frequency Threshold}
    def __extract(self):
        
        # Feature Extraction
        for f, details in self.profile['features'].items():

            # Extract Text Features
            if (details['feature_type'] == 'Alphanumerical' and details['data_type'] != 'DateTime' and details['role'] != 'target'):

                # TFID Vectorizer
                tfidf_vectorizer = TfidfVectorizer()

                # Train
                x_tfidf_train = tfidf_vectorizer.fit_transform(self.__train[f])
                f_extracted_train = pd.DataFrame(x_tfidf_train.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

                # Test
                x_tfidf_test = tfidf_vectorizer.transform(self.__test[f])
                f_extracted_test = pd.DataFrame(x_tfidf_test.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

                # Keep specific columns
                term_document_frequency = (f_extracted_train > 0).sum(axis=0)
                term_document_frequency_ratio = term_document_frequency / self.__train[f].count()
                columns_to_keep = term_document_frequency[term_document_frequency_ratio >= 0.15].index

                # Filter
                f_extracted_train_filtered = f_extracted_train[columns_to_keep]
                f_extracted_test_filtered = f_extracted_test[columns_to_keep]

                # Change to 0 or 1

                # Change Names of encoded df
                for c in f_extracted_train_filtered.columns: 
                    name = str(f) + '_' + str(c)
                    f_extracted_train_filtered = f_extracted_train_filtered.rename(columns={c : name})

                # Change Names of encoded df
                for c in f_extracted_test_filtered.columns: 
                    name = str(f) + '_' + str(c)
                    f_extracted_test_filtered = f_extracted_test_filtered.rename(columns={c : name})

                # Concat to existing df
                X_train = self.__train.drop(columns=[f])
                X_train = X_train.reset_index(drop=True)
                f_extracted_train_filtered = f_extracted_train_filtered.reset_index(drop=True)
                self.__train = pd.concat([X_train, f_extracted_train_filtered], axis=1, ignore_index=False)

                X_test = self.__test.drop(columns=[f])
                X_test = X_test.reset_index(drop=True)
                f_extracted_test_filtered = f_extracted_test_filtered.reset_index(drop=True)
                self.__test = pd.concat([X_test, f_extracted_test_filtered], axis=1, ignore_index=False)
    
    ## Feature Encoding | {Feature, Encoding Type}
    def __encode(self):

        # Feature Encoding
        for f, details in self.profile['features'].items():
            if ((details['feature_type'] == 'Categorical' or details['data_type'] == 'DateTime') and 
                details['role'] != 'id' and 
                details['role'] != 'target' and 
                details['data_type'] != 'Integer' and
                details['data_type'] != 'Float'):

                # Dataset Current Series
                f_set_train = self.__train[f]
                f_set_test = self.__test[f]

                # Encoded
                encoded = False

                # DateTime
                if (details['data_type']  == 'DateTime'):

                    # DateTime Encode | Train
                    f_set_train = pd.to_datetime(f_set_train)

                    # 
                    f_encoded_train = pd.DataFrame()
                    f_encoded_train['year'] = f_set_train.dt.year
                    f_encoded_train['month'] = f_set_train.dt.month
                    f_encoded_train['day'] = f_set_train.dt.day

                    # DateTime Encode | Test
                    f_set_test = pd.to_datetime(f_set_test)

                    # 
                    f_encoded_test = pd.DataFrame()
                    f_encoded_test['year'] = f_set_test.dt.year
                    f_encoded_test['month'] = f_set_test.dt.month
                    f_encoded_test['day'] = f_set_test.dt.day

                    # Encoded True
                    encoded = True

                # Categorical | one-hot
                elif (details['feature_type'] == 'Categorical' and self.__params['engineer']['encode_type'] == 'one-hot'):

                    # Train
                    f_encoded_train = pd.get_dummies(f_set_train).astype(int)

                    # Test
                    f_encoded_test = pd.get_dummies(f_set_test).astype(int)
                    f_encoded_test = f_encoded_test.reindex(columns=f_encoded_train.columns, fill_value=0)

                    encoded = True

                # Categorical | label
                elif (details['feature_type'] == 'Categorical' and self.__params['engineer']['encode_type'] == 'label'):

                    # Combine train and test sets to fit the encoder on all categories
                    combined_set = pd.concat([f_set_train, f_set_test])

                    # Fit the encoder on the combined set
                    encoder = LabelEncoder()
                    encoder.fit(combined_set)

                    # Transform both train and test sets using the same encoder
                    f_encoded_train = pd.DataFrame(encoder.transform(f_set_train))
                    f_encoded_test = pd.DataFrame(encoder.transform(f_set_test))

                    encoded = True

                if (encoded):
                    # Change Names of encoded df
                    for c in f_encoded_train.columns: 
                        name = str(f_set_train.name) + '_' + str(c)
                        f_encoded_train = f_encoded_train.rename(columns={c : name})

                    # Change Names of encoded df
                    for c in f_encoded_test.columns: 
                        name = str(f_set_test.name) + '_' + str(c)
                        f_encoded_test = f_encoded_test.rename(columns={c : name})

                    # Concat to existing df
                    X_train = self.__train.drop(columns=[f])
                    X_train = X_train.reset_index(drop=True)
                    f_encoded_train = f_encoded_train.reset_index(drop=True)
                    self.__train = pd.concat([X_train, f_encoded_train], axis=1, ignore_index=False)

                    X_test = self.__test.drop(columns=[f])
                    X_test = X_test.reset_index(drop=True)
                    f_encoded_test = f_encoded_test.reset_index(drop=True)
                    self.__test = pd.concat([X_test, f_encoded_test], axis=1, ignore_index=False)

    ## Feature Scaling
    def __scale(self):

        # Scale features
        if (self.__params['engineer']['scale_type'] == 'std'):
            scaler = StandardScaler()

        elif (self.__params['engineer']['scale_type'] == 'min-max'):
            scaler = MinMaxScaler(feature_range=(0,1))
        
        elif (self.__params['engineer']['scale_type'] == 'robust'):
            scaler = RobustScaler()

        # Targets
        y_train = self.__train[self.__target]
        y_test = self.__test[self.__target]

        # Variables
        x_train = self.__train.drop(columns=[self.__target])
        x_test = self.__test.drop(columns=[self.__target])

        scaled_train = scaler.fit_transform(x_train)
        scaled_test = scaler.transform(x_test)

        # Convert to DF
        self.__train = pd.DataFrame(scaled_train, columns=x_train.columns, index=x_train.index)
        self.__test = pd.DataFrame(scaled_test, columns=x_test.columns, index=x_test.index)

        # Concatenate
        self.__train = pd.concat([self.__train, y_train], axis=1, ignore_index=False)
        self.__test = pd.concat([self.__test, y_test], axis=1, ignore_index=False)

    ## Feature Selection
    def __select(self):

        # Targets
        y_train = self.__train[self.__target]
        y_test = self.__test[self.__target]

        # Variables
        x_train = self.__train.drop(columns=[self.__target])
        x_test = self.__test.drop(columns=[self.__target])

        if (self.__params['engineer']['select_type'] == 'variance_thres'):

            # Variance Threshold
            selector = VarianceThreshold()

            # 
            selected_train =  selector.fit_transform(x_train)
            selected_test =  selector.transform(x_test)

            # Convert to DF
            selected_columns = x_train.columns[selector.get_support()]
            self.__train = pd.DataFrame(selected_train, columns=selected_columns, index=x_train.index)
            self.__test = pd.DataFrame(selected_test, columns=selected_columns, index=x_test.index)

            # Concatenate
            self.__train = pd.concat([self.__train, y_train], axis=1, ignore_index=False)
            self.__test = pd.concat([self.__test, y_test], axis=1, ignore_index=False)
        
        elif(self.__params['engineer']['select_type'] == 'univariate'):

            k = int(len(self.__train.columns) * self.__params['engineer']['select_perc'])

            # Select K Best
            selector = SelectKBest(score_func=f_classif, k=k)

            # 
            selected_train =  selector.fit_transform(x_train, y_train)
            selected_test =  selector.transform(x_test)

            # Convert to DF
            selected_columns = x_train.columns[selector.get_support()]
            self.__train = pd.DataFrame(selected_train, columns=selected_columns, index=x_train.index)
            self.__test = pd.DataFrame(selected_test, columns=selected_columns, index=x_test.index)

            # Concatenate
            self.__train = pd.concat([self.__train, y_train], axis=1, ignore_index=False)
            self.__test = pd.concat([self.__test, y_test], axis=1, ignore_index=False)

        elif(self.__params['engineer']['select_type'] == 'mi'):
            k = int(len(self.__train.columns) * self.__params['engineer']['select_perc'])

            selector = SelectKBest(score_func=mutual_info_classif, k=k) 

            selected_train =  selector.fit_transform(x_train, y_train)
            selected_test =  selector.transform(x_test)

            # Get the selected feature indices
            selected_features = x_train.columns[selector.get_support()]

            # Convert to DataFrame
            self.__train = pd.DataFrame(selected_train, columns=selected_features, index=x_train.index)
            self.__test = pd.DataFrame(selected_test, columns=selected_features, index=x_test.index)

            # Concatenate with the target variable
            self.__train = pd.concat([self.__train, y_train], axis=1)
            self.__test = pd.concat([self.__test, y_test], axis=1)

    # Feature Engineering | {Dataframe, Profile, EDA}
    def engineer(self): 

        ## Describe
        profile = Profile(self.__train, self.__target)
        self.profile = profile.data_profile()

        # Feature Extraction
        self.__extract()

        ## Describe
        profile = Profile(self.__train, self.__target)
        self.profile = profile.data_profile()

        # Feature Encoding
        self.__encode()

        # Feature Scaling
        if (self.__params['engineer']['scale_type'] != 'none'):
            self.__scale()

        # Feature Select
        if (self.__params['engineer']['select_type'] != 'none'):
            self.__select()