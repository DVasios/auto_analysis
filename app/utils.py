from datetime import datetime
import uuid
import os
import glob

def generate_unique_filename(filename):
    ext = filename.rsplit('.', 1)[1].lower()
    unique_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{unique_id}_{timestamp}.{ext}"

def delete_all_files(directory):
    # Create a pattern for all files in the directory
    file_pattern = os.path.join(directory, '*')
    
    # Iterate over all files matching the pattern
    for file_path in glob.glob(file_pattern):
        try:
            # Check if it's a file and delete it
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted {file_path}")
            else:
                print(f"{file_path} is not a file")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")