from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv

# Load dotenv
load_dotenv()

# Create Flask Application
app = Flask(__name__)
app.config.from_object('config.Config')

# Database 
db = SQLAlchemy(app)

# Models
from .models import UserCase

# Routes
from app import routes
app.register_blueprint(routes.bp)
