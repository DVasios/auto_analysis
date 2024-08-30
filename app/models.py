from app import db

# Define a User model
class UserCase(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    path = db.Column(db.String(255), unique=True, nullable=False)
    problem_type = db.Column(db.String(255), unique=False, nullable=False)
    target_feature = db.Column(db.String(255), unique=False, nullable=True)
    
    # String representation of the object
    def __repr__(self):
        return f'<Case {self.id}>'