from datetime import datetime
import uuid

def generate_unique_filename(filename):
    ext = filename.rsplit('.', 1)[1].lower()
    unique_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{unique_id}_{timestamp}.{ext}"
