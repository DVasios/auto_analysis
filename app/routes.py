# app/routes.py
from app import app
from flask import render_template, request, jsonify
from werkzeug.utils import secure_filename
from app.src.profile.profile import Profile
from app.src.gather.gather import Gather
from app.src.eda.eda import Eda
import os

# Config
upload_folder = "files/"
app.config['upload_folder'] = upload_folder

# Index
@app.route('/')
def index():
    return render_template('index.html')

# About 
@app.route('/about')
def about():
    return render_template('about.html')

# Examples 
@app.route('/examples')
def example():
    return render_template('example.html')

# EDA
@app.route('/eda', methods=['POST'])
def eda():
    file_path = os.path.dirname(__file__) + '/' + app.config['upload_folder']
    file = request.files['file']
    filename = secure_filename(file.filename)

    # Save File
    filepath = os.path.join(file_path, filename)
    file.save(filepath)

    # Gather 
    df_gather = Gather(filepath)
    df = df_gather.gather()

    ## Objective (temp)
    objective = {
        'problem_type' : '',
        'target_feature' : 'Survived' 
    }

    # Profile
    file_profile = Profile(df)
    profile = file_profile.df_profile(objective, 0.1)

    # EDA
    file_eda = Eda(df, profile)
    eda = file_eda.df_describe()

    return render_template('eda.html', eda=eda)
