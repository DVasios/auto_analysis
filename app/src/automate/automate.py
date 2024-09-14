# Libraries

# Core Libraries
import numpy as np

# Sk learn 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from skopt.space import Real, Categorical, Integer
from skopt import gp_minimize

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
        prof = Profile(clean.get_train, self.auto_params['target'], params['profile'])
        engineered_profile = prof.df_profile()

        ## Save
        self.df_engineered = engineer.get_train
        self.df_engineered_profile = engineered_profile

        return engineer.train, engineer.test

    # Model 
    def train_model(self, params):

        # Labels
        y_train = self.X_train_preprocessed[self.auto_params['target']]
        y_test = self.X_test_preprocessed[self.auto_params['target']]

        # Drop Targets
        x_train = self.X_train_preprocessed.drop(columns=[self.auto_params['target']])
        x_test = self.X_test_preprocessed.drop(columns=[self.auto_params['target']])

        # Model
        if (params['model'] == 'lg'):
        
            # Define the parameter grid
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [None, 10],
                'min_samples_split': [2, 5, 10],
            }

            model = RandomForestClassifier(random_state=42)
            # grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
        
        elif (params['model'] == 'rf'):

            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [100, 200, 300]
            }

            model = LogisticRegression(random_state=42)

        elif(params['model'] == 'svm'):

            model = SVC(kernel='linear', C=1.0)
            # grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)


        scores = cross_val_score(model, x_train, y_train, cv=5)
        print(scores.mean())

        # Fit data to model
        model_fitted = model.fit(x_train, y_train)

        # Best Model
        # best_model = grid_search.best_estimator_

        # Predict 
        y_pred = model_fitted.predict(x_test)

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"accuracy: {accuracy}")
        print(f"precision: {precision}")
        print(f"recall: {recall}")
        print(f"f1: {f1}")

        return accuracy

    # Objective
    def objective(self, params):

        if (self.auto_params['opt_method'] == 'bayesian'):

            # Params
            profile_params = {
                'unique_values_thres' : params[0],
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
                'scale' : params[6],
                'scale_type' : params[7],
                'select' : params[8],
                'select_type' : params[9]
            }

            # Models Parameters
            model_params = {
                'model' : params[10] 
            }

            ## Params
            params = {
                'profile' : profile_params,
                'clean' : clean_params,
                'engineering' : engineering_params,
                'model' : model_params
            }

        # Preprocess
        self.X_train_preprocessed, self.X_test_preprocessed = self.preprocess(params)

        # Train model
        accuracy = self.train_model(params['model'])

        if (self.auto_params['opt_method'] == 'random'):
            return accuracy
        elif(self.auto_params['opt_method'] == 'bayesian'):
            return 1-accuracy


    # Optimization Method
    def random_search(self, n_iter):

        best_accuracy = -np.inf
        best_params = None

        accuracies = []
        for i in range(n_iter):

            # Params
            profile_params = {
                'unique_values_thres' : random.choice([0.1, 0.15, 0.2]),
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
                'scale' : random.choice([True, False]),
                'scale_type' : random.choice(['std', 'min-max', 'robust']),
                'select' : random.choice([True, False]),
                'select_type' : random.choice(['variance_thres', 'PCA', 'univariate'])
            }

            # Models Parameters
            model_params = {
                'model' : random.choice(['lg', 'rf', 'svm'])
            }

            ## Params
            params = {
                'profile' : profile_params,
                'clean' : clean_params,
                'engineering' : engineering_params,
                'model' : model_params
            }

            print(params)

            # Evaluate the model with the sampled parameters
            accuracy = self.objective(params)

            # Push Accuracy
            accuracies.append(accuracy)
            
            # Check if this is the best score so far
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = (params, model_params)

            print(f"Iteration: {i+1}, Accuracy: {accuracy}" )
           
        
        return best_params, best_accuracy, accuracies
    
    def bayesian(self, n_iter):

        search_space = [
            Categorical([0.1, 0.15, 0.2], name='unique_values_thres'),
            Categorical([0.5, 0.6, 0.7], name='drop_thres'),
            Categorical([1.5, 2, 3, 4], name='outlier_thres'),
            Categorical(['mean', 'median'], name='num_team'),
            Real(0.05, 0.15, name='freq_thres'),
            Categorical(['one-hot', 'label'], name='encode_type'),
            Categorical([True, False], name='scale'),
            Categorical(['std', 'min-max', 'robust'], name='scale_type'),
            Categorical([True, False], name='select'),
            Categorical(['variance_thres', 'PCA', 'univariate'], name='select_type'),
            Categorical(['lg', 'rf', 'svm'], name='model'),
        ]


        result = gp_minimize(self.objective, search_space, n_calls=n_iter, random_state=42)

        return result


    # Optimize  
    def optimize(self):

        if (self.auto_params['opt_method'] == 'random'):
            return self.random_search(self.auto_params['n_iter'])
        
        elif (self.auto_params['opt_method'] == 'bayesian'):
            res = self.bayesian(self.auto_params['n_iter'])
            return 1 - res.fun, res.x

    # Automated Data Preprocessing | {File Source, Problem Type, Target}
    def auto_preproc (self):

        # Gather Data from source file
        gather = Gather(self.auto_params['filepath'])
        df = gather.gather()

        # Split Train, Test
        self.X_train, self.X_test = train_test_split(df, test_size=0.2, random_state=42)

        # Run Optimization Method
        result = self.optimize()

        return resultp
    