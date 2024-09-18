# Libraries

# Core Libraries
import numpy as np
import pandas as pd

# Sk learn 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from skopt.space import Real, Categorical, Integer
from skopt import gp_minimize

from xgboost import XGBClassifier 

# Helpers
import random

# App Libs
from gather.gather import Gather
from prof.profile import Profile
from clean.clean import Clean
from feature_engineering.engineer import Engineering

class Automate:

    # X_train, X_test
    X_train = None
    X_test = None

    # Preprocessed
    X_train_preprocessed = None
    X_test_preprocessed = None

    # DFs

    df = None
    df_profile = None

    df_cleaned = None
    df_cleaned_profile = None

    df_egineered = None
    df_egineered_profile = None

    # Other
    iter = 1

    def __init__(self, auto_params):
        self.auto_params = auto_params

    # Preprocess
    def preprocess(self, params):

        # Profile
        prof = Profile(self.X_train, self.auto_params['target'], params['profile'])
        profile = prof.df_profile()
        
        ## Save
        self.df = self.X_train
        self.df_profile = profile

        # Clean
        clean = Clean(self.X_train, self.X_test, profile, params['clean'])
        clean.clean(params['profile'])

        # Cleaned Profile
        prof = Profile(clean.get_train, self.auto_params['target'], params['profile'])
        cleaned_profile = prof.df_profile()

        ## Save
        self.df_cleaned = clean.get_train
        self.df_cleaned_profile = cleaned_profile

        # Feature Engineer
        engineer = Engineering(clean.get_train, clean.get_test, cleaned_profile, self.auto_params['target'], params['engineering'])
        engineer.engineer(params['profile'])

        # Describe | Feature Engineered
        prof = Profile(engineer.get_train, self.auto_params['target'], params['profile'])
        engineered_profile = prof.df_profile()

        ## Save
        self.df_engineered = engineer.get_train
        self.df_engineered_profile = engineered_profile

        return engineer.train, engineer.test

    # Model 
    def train_model(self, m):

        # Labels
        y_train = self.X_train_preprocessed[self.auto_params['target']]
        y_test = self.X_test_preprocessed[self.auto_params['target']]

        # Transform Labels if they are strings
        if (y_train.dtype.name == 'object'):
            label_encoder = LabelEncoder()
            label_encoder.fit(y_train)

            # Transform both train and test sets using the same encoder
            y_train = pd.DataFrame(label_encoder.transform(y_train))
            y_test = pd.DataFrame(label_encoder.transform(y_test))

        # Drop Targets
        x_train = self.X_train_preprocessed.drop(columns=[self.auto_params['target']])
        x_test = self.X_test_preprocessed.drop(columns=[self.auto_params['target']])

        # Model
        if (m == 'lg'):
            model = RandomForestClassifier(random_state=42)
        
        elif (m == 'rf'):
            model = LogisticRegression(random_state=42)

        elif(m == 'xgb'):
            model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

        # Scores
        scores = cross_val_score(model, x_train, y_train, cv=5)

        # Fit data to model
        model_fitted = model.fit(x_train, y_train)

        # Predict 
        y_pred = model_fitted.predict(x_test)

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy

    # Objective
    def objective(self, params):

        if (self.auto_params['opt_method'] == 'bayesian'):

            # Params
            profile_params = {
                'cat_thres' : params[0],
                'id_thres' : 0.9
            }               

            # Clean Parameters
            clean_params = {
                'drop_thres' : params[1],
                'outlier_thres' : params[2],
                'num_type' : params[3] 
            }

            # Feature Engineering Parameters
            engineering_params = {
                'freq_thres' : params[4],
                'encode_type' : params[5],
                'scale_type' : params[6],
                'select_type' : params[7],
                'select_perc' : params[8]
            }

            ## Params
            params = {
                'profile' : profile_params,
                'clean' : clean_params,
                'engineering' : engineering_params,
            }

        # Preprocess
        self.X_train_preprocessed, self.X_test_preprocessed = self.preprocess(params)

        # Train model
        if (self.auto_params['model'] == 'ensemble'):
            lg_acc = self.train_model('lg')
            rf_acc = self.train_model('rf')
            xgb_acc = self.train_model('xgb')
            accuracy =  round((lg_acc + rf_acc + xgb_acc) / 3, 4)
        else:
            accuracy = round(self.train_model(self.auto_params['model']), 4)


        # Log
        print(f"\rIteration: {self.iter}/{self.auto_params['n_iter']}", end="")
        # print(f"\rAccuracy: {accuracy}", end="")
        self.iter += 1

        if (self.auto_params['opt_method'] == 'random'):
            return accuracy
        elif(self.auto_params['opt_method'] == 'bayesian'):
            return 1-accuracy

    # Optimization Method
    def random_search(self, n_iter):

        best_accuracy = -np.inf
        best_params = None

        accuracies = []
        opt_accuracies = []
    
        for i in range(n_iter):

            # Params
            profile_params = {
                'cat_thres' : random.choice([0.1, 0.15, 0.2]),
                'id_thres' : 0.9
            }

            # Clean Parameters
            clean_params = {
                'drop_thres' : random.choice([0.5, 0.6, 0.7]),
                'outlier_thres' : random.choice([1.5, 2, 3, 4]),
                'num_type' : random.choice(['mean', 'median'])
            }

            # Feature Engineering Parameters
            engineering_params = {
                'freq_thres' : random.uniform(0.05, 0.15),
                'encode_type' : random.choice(['one-hot', 'label']),
                'scale_type' : random.choice(['none', 'std', 'min-max', 'robust']),
                'select_type' : random.choice(['none', 'variance_thres', 'univariate', 'mi']),
                'select_perc' : random.choice([0.4, 0.6, 0.8])
            }

            ## Params
            params = {
                'profile' : profile_params,
                'clean' : clean_params,
                'engineering' : engineering_params,
            }

            # Evaluate the model with the sampled parameters
            accuracy = self.objective(params)

            # Push Accuracy
            accuracies.append(accuracy)
            
            # Check if this is the best score so far
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = [ profile_params['cat_thres'], profile_params['id_thres'], clean_params['drop_thres'], clean_params['outlier_thres'], clean_params['num_type'], engineering_params['freq_thres'], engineering_params['encode_type'], engineering_params['scale_type'], engineering_params['selection_type']]

            opt_accuracies.append(best_accuracy)

        return best_params, best_accuracy, accuracies, opt_accuracies
    
    def bayesian(self, n_iter):

        search_space = [
            Real(0.1, 0.3, name='cat_thres'),
            Real(0.4, 0.7, name='drop_thres'),
            Categorical(['none', 1,  1.5, 2, 3], name='outlier_thres'),
            Categorical(['mean', 'median'], name='num_team'),
            Real(0.05, 0.15, name='freq_thres'),
            Categorical(['one-hot', 'label'], name='encode_type'),
            Categorical(['none', 'std', 'min-max', 'robust'], name='scale_type'),
            Categorical(['none', 'variance_thres', 'univariate', 'mi'], name='select_type'),
            Categorical([0.4, 0.5, 0.75, 1], name='select_perc'),
        ]

        result = gp_minimize(self.objective, search_space, n_initial_points=10, n_calls=n_iter, n_jobs=-1)

        return result

    # Optimize  
    def optimize(self):

        if (self.auto_params['opt_method'] == 'random'):
            return self.random_search(self.auto_params['n_iter'])
        
        elif (self.auto_params['opt_method'] == 'bayesian'):
            res = self.bayesian(self.auto_params['n_iter'])
            return res

    # Automated Data Preprocessing | {File Source, Problem Type, Target}
    def auto_preproc (self):

        # Gather Data from source file
        gather = Gather(self.auto_params['filepath'], self.auto_params['target'])
        df = gather.gather()

        # Split Train, Test
        self.X_train, self.X_test = train_test_split(df, test_size=0.2, random_state=42)

        # Run Optimization Method
        result = self.optimize()

        return result
    
    # Print Report
    def report(self):
        print('hi')

    # Extract Results
    def extract(self):
        print('extract')
    