# app/routes.py
from app import app, db
from flask import render_template, request, jsonify, Blueprint
from werkzeug.utils import secure_filename
from app.src.prof.profile import Profile
from app.src.gather.gather import Gather
from app.src.eda.eda import Eda
from app.utils import generate_unique_filename, delete_all_files

# Models
from app.models import UserCase

# Other
import os

# Blueprint
bp = Blueprint('main', __name__)

# Index
@bp.route('/')
def index():

    # Delete Files
    file_path = os.path.dirname(__file__) + '/' + app.config['upload_folder']
    delete_all_files(file_path)

    return render_template('index.html')

# About 
@bp.route('/about')
def about():
    return render_template('about.html')

# Examples 
@bp.route('/examples')
def example():
    return render_template('example.html')

# Choose Target Feature
@bp.route('/target_feature', methods=['POST'])
def target_feature():

    # Save File
    file_path = os.path.dirname(__file__) + '/' + app.config['upload_folder']
    file = request.files['file']
    filename = generate_unique_filename(file.filename)
    new_filename = secure_filename(filename)
    filepath = os.path.join(file_path, new_filename)
    file.save(filepath)

    # Save UserCase
    new_user_case = UserCase(path=filepath, problem_type='Classification')
    db.session.add(new_user_case)
    db.session.commit()

    # Gather 
    df_gather = Gather(filepath)
    df = df_gather.gather()

    # Features 
    feature_names = df.columns.tolist()
    
    # Render Template
    return render_template('target_feature.html', user_case=new_user_case, feature_names=feature_names)

# EDA
@bp.route('/eda', methods=['POST'])
def eda():

    # Form Data
    user_case_id = request.form.get('id')
    target_feature = request.form.get('target-select')

    # Find User Case Instance
    user_case_instance = UserCase.query.get(user_case_id)
    user_case_instance.target_feature = target_feature
    db.session.commit()

    # Gather 
    df_gather = Gather(user_case_instance.path)
    df = df_gather.gather()

    # Profile
    file_profile = Profile(df, user_case_instance.problem_type, user_case_instance.target_feature)
    profile = file_profile.df_profile(0.1)

    # EDA
    file_eda = Eda(df, profile)
    eda = file_eda.df_describe()

    return render_template('eda.html', eda=eda)
