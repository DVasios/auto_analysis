from app import app
import os

class Config:

    # Database Configuration
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DB_URL')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # General
    upload_folder = "files/"
    app.config['upload_folder'] = upload_folder