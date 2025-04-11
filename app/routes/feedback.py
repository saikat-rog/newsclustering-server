from flask import Blueprint, request, jsonify
import csv, os
from datetime import datetime

feedback_bp = Blueprint('feedback', __name__)

@feedback_bp.route('/feedback_training', methods=['POST'])
def feedback_training():
    data = request.get_json()
    all_algo = data.get('all', False)
    preferred_algo = data.get('preferred')
    now = datetime.now().isoformat()

    file_path = 'feedback_training.csv'
    file_exists = os.path.exists(file_path)

    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['timestamp', 'all_algorithms', 'preferred_algorithm'])
        writer.writerow([now, all_algo, preferred_algo])

    return jsonify({"message": "Feedback saved!"})
