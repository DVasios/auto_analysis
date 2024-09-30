# Libraries

# Core Libraries
import numpy as np
import pandas as pd

# Skicit Learn 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Skicit Optimize
from skopt import gp_minimize
from skopt.space import Real, Categorical

# Helpers
import random
import time
import csv

# App Libs
from .gather import Gather
from .prof import Profile
from .clean import Clean
from .engineer import Engineer
from .utils import plot_convergence, has_missing_data

# Warnings# App Libs
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore')
# warnings.filterwarnings('ignore', category=RuntimeWarning)
# warnings.filterwarnings('ignore', category=ConvergenceWarning)

class Automate:

    # X_train, X_test
    X_train = None
    X_test = None

    # Preprocessed
    X_train_preprocessed = None
    X_test_preprocessed = None

    # DFs
    df_start = None

    df = None
    df_profile = None

    df_cleaned = None
    df_cleaned_profile = None

    df_egineered = None
    df_egineered_profile = None

    # Best params
    best_params = None

    # Other
    iter = 1
    result = None
    run_time = None
    simple_acc = None
    adv_acc = None

    def __init__(self, auto_params):
        self.__auto_params = auto_params

    # Preprocess
    def __preprocess(self, params):

        # Profile
        prof = Profile(self.X_train, self.__auto_params['target'], params['profile'])
        profile = prof.data_profile()
        
        ## Save
        self.df = self.X_train
        self.df_profile = profile

        # Clean
        clean = Clean(self.X_train, self.X_test, self.__auto_params['target'], params)
        clean.clean()

        # Cleaned Profile
        prof = Profile(clean.get_train, self.__auto_params['target'], params['profile'])
        cleaned_profile = prof.data_profile()

        ## Save
        self.df_cleaned = clean.get_train
        self.df_cleaned_profile = cleaned_profile

        # Feature Engineer
        engineer = Engineer(clean.get_train, clean.get_test, self.__auto_params['target'], params)
        engineer.engineer()

        # Describe | Feature Engineered
        prof = Profile(engineer.get_train, self.__auto_params['target'], params['profile'])
        engineered_profile = prof.data_profile()

        ## Save
        self.df_engineered = engineer.get_train
        self.df_engineered_profile = engineered_profile

        return engineer.get_train, engineer.get_test

    # Model 
    def __train_model(self, m):

        # Labels
        y_train = self.X_train_preprocessed[self.__auto_params['target']]
        y_test = self.X_test_preprocessed[self.__auto_params['target']]

        # If label has only one value
        if (y_train.nunique() == 1):
            return 0 
        
        # If y_train and y_test labels doesn't match
        if set(y_train.unique()) != set(y_test.unique()):
            return 0

        # Transform Labels if they are strings
        if (y_train.dtype.name == 'object'):
            label_encoder = LabelEncoder()
            label_encoder.fit(y_train)

            # Transform both train and test sets using the same encoder
            y_train = pd.DataFrame(label_encoder.transform(y_train))
            y_test = pd.DataFrame(label_encoder.transform(y_test))

        # Drop Targets
        x_train = self.X_train_preprocessed.drop(columns=[self.__auto_params['target']])
        x_test = self.X_test_preprocessed.drop(columns=[self.__auto_params['target']])

        # Model
        if (m == 'lg'):
            model = LogisticRegression(n_jobs=-1, random_state=42)
        
        elif (m == 'dt'):
            model = DecisionTreeClassifier(random_state=42)

        elif(m == 'nb'):
            model = GaussianNB()

        elif(m =='ensemble'):

            model1 = DecisionTreeClassifier(random_state=42)
            model2 = LogisticRegression(random_state=42)
            model3 = GaussianNB()
            model = VotingClassifier(
                estimators=[
                    ('dt', model1),
                    ('lr', model2),
                    ('nb', model3)
                ],
                voting='hard',
                n_jobs=-1
                )

        # Fit data to model
        model_fitted = model.fit(x_train, y_train)

        # Predict 
        y_pred = model_fitted.predict(x_test)

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy

    # Objective
    def __objective(self, params):

        if (self.__auto_params['opt_method'] == 'bayesian'):

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
                'engineer' : engineering_params,
            }

        # Accuracies
        accuracies = []

        kf = KFold(n_splits=self.__auto_params['cv'], shuffle=True, random_state=None)

        for train_index, test_index in kf.split(self.df_start):

            # Split Train, Test
            self.X_train, self.X_test = self.df_start.iloc[train_index], self.df_start.iloc[test_index]

            # Preprocess
            self.X_train_preprocessed, self.X_test_preprocessed = self.__preprocess(params)

            # Train model
            accuracies.append(round(self.__train_model(self.__auto_params['model']), 4))

        # Final Accuracy
        accuracy = np.mean(accuracies)

        # Best accuracy and best params
        if (accuracy > self.simple_acc):
            self.simple_acc = accuracy
            self.best_params = params

        # Log
        print(f"\rIteration: {self.iter}/{self.__auto_params['n_iter']}", end="")
        self.iter += 1

        # Accuracy   
        if(self.__auto_params['opt_method'] == 'bayesian'):
            accuracy = 1 - accuracy

        return accuracy

    # Optimization Method
    def __random_search(self, n_iter):

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
                'outlier_thres' : random.choice(['none', 1.5, 2, 3, 4]),
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
                'engineer' : engineering_params,
            }

            # Evaluate the model with the sampled parameters
            accuracy = self.__objective(params)

            # Push Accuracy
            accuracies.append(accuracy)
            
            # Check if this is the best score so far
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = [ profile_params['cat_thres'], 
                                clean_params['drop_thres'], 
                                clean_params['outlier_thres'],
                                clean_params['num_type'], 
                                engineering_params['freq_thres'], 
                                engineering_params['encode_type'], 
                                engineering_params['scale_type'], 
                                engineering_params['select_type'], 
                                engineering_params['select_perc']]

            opt_accuracies.append(best_accuracy)

        return best_params, best_accuracy, accuracies, opt_accuracies
    
    def __bayesian(self, n_iter):

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

        result = gp_minimize(self.__objective, search_space, n_calls=n_iter, n_jobs=-1)

        return result

    # Optimize  
    def __optimize(self):

        # Initialize Variables
        self.simple_acc = 0

        if (self.__auto_params['opt_method'] == 'random'):
            return self.__random_search(self.__auto_params['n_iter'])
        
        elif (self.__auto_params['opt_method'] == 'bayesian'):
            res = self.__bayesian(self.__auto_params['n_iter'])
            return res

    # Automated Data Preprocessing | {File Source, Problem Type, Target}
    def auto_analysis (self):

        ## Start Time
        start_time = time.time()

        # Gather Data from source file
        gather = Gather(self.__auto_params['filepath'])
        df = gather.gather()
        self.df_start = df

        # Run Optimization Method
        result = self.__optimize()

        ## End Time
        end_time = time.time()

        ## Elapsed Time
        elapsed_time = end_time - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)

        ## Running Time
        if (minutes > 0):
            self.run_time =  f"{minutes} minutes and {seconds} seconds"
        else:
            self.run_time =  f"{seconds} seconds"

        # Store Result
        self.result = result

    # Model + HP Optimization
    def hp_optization(self, n_iter):

        # Split Train, Test
        self.X_train, self.X_test = train_test_split(self.df_start, test_size=0.2, random_state=42)
        y_train = self.X_train[self.__auto_params['target']]
        y_test = self.X_test[self.__auto_params['target']]


        # Transform Labels if they are strings
        if (y_train.dtype.name == 'object'):
            label_encoder = LabelEncoder()
            label_encoder.fit(y_train)

            # Transform both train and test sets using the same encoder
            y_train = label_encoder.transform(y_train)
            y_test = label_encoder.transform(y_test)

        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)

        # Preprocess
        self.X_train_preprocessed, self.X_test_preprocessed = self.__preprocess(self.best_params)

        # Drop Targets
        x_train = self.X_train_preprocessed.drop(columns=[self.__auto_params['target']])
        x_test = self.X_test_preprocessed.drop(columns=[self.__auto_params['target']])

        # Define the model
        xgb_model = XGBClassifier()

        # Define the hyperparameters to search over
        param_distributions = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }

        # Initialize RandomizedSearchCV
        random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_distributions, n_iter=n_iter, cv=5, verbose=2, n_jobs=-1)

        # Fit the model with the training data
        random_search.fit(x_train, y_train)

        # Get the best model and parameters
        best_xgb_model = random_search.best_estimator_

        # Predict on test data
        y_pred = best_xgb_model.predict(x_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        self.adv_acc = accuracy
        print(f"Best XGBoost Accuracy: {accuracy:.4f}")
        print("Best Parameters:", random_search.best_params_)

    # Print Report
    def report(self):
        if (self.__auto_params['opt_method'] == 'bayesian'):
            acc = 1 - self.result['fun']
            pip = self.result['x']
        else:
            acc = self.result[1]
            pip = self.result[0]

        print(f"---- Auto Analysis Report ----\n")

        print(f"-- Details -- \n")
        print(f"Features: {self.df_profile['dataset']['features']}")
        print(f"Rows: {self.df_profile['dataset']['rows']}")
        print(f"Missing Data: {has_missing_data(self.df_profile)}")
        print(f"Duplicates: {self.df_profile['dataset']['duplicates']['exist']}\n")

        # Metrics
        print(f"-- Metrics -- \n")
        print(f"Best Accuracy: {round(acc, 4)}")
        print(f"Total Running Time: {self.run_time}\n")

        # Suggested Pipeline
        print(f"-- Suggested Pipeline -- \n")

        # Outliers
        outlier_thres = pip[2]
        print(f"Outlier IQR Threshold: {outlier_thres}")

        # Imputation
        imputation_method = pip[3].title()
        print(f"Imputation Type: {imputation_method}")

        # Encode 
        encode_type = 'One-hot' if pip[5] == 'one-hot' else 'Label'
        print(f"Encoding Type: {encode_type}")

        # Scale
        scale_type = pip[6].title()
        print(f"Scale Type: {scale_type}")

        # Selection
        selection_type = pip[7].title()
        print(f"Selection Type: {selection_type}")

        # Selection Percentage
        if (selection_type != 'None'):
            selection_perc = pip[8] * 100
            print(f"Selection Percentage: {selection_perc} %")

    # Convergence
    def plot(self):

        if (self.__auto_params['opt_method'] == 'bayesian'):
            plot_convergence('bayesian', self.result['func_vals'])
        else:
            plot_convergence('random', self.result[3])

    # Extract Results
    def save(self, filepath):
        if (self.__auto_params['opt_method'] == 'bayesian'):
            acc = 1 - self.result['fun']
            pip = self.result['x']
        else:
            acc = self.result[1]
            pip = self.result[0]

        # Data
        data = [
            [self.__auto_params['filepath'], 
             self.df_profile['dataset']['features'],
             self.df_profile['dataset']['rows'],
             has_missing_data(self.df_profile),
             self.df_profile['dataset']['duplicates']['exist'],
             self.__auto_params['model'],
             round(acc, 4),
             round(self.adv_acc, 4),
             self.run_time,
             self.__auto_params['n_iter'],
             pip[2],
             pip[3].title(),
             pip[5].title(),
             pip[6].title(),
             pip[7].title(),
             pip[8],
            ],
        ]

        # Writing to CSV
        with open(filepath, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(data)