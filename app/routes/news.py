from flask import Blueprint, request, jsonify
import numpy as np
from app.services.services import extract_text_from_url, summarize_text, generate_clustering_metrics
from app.models.models import predict_category, analyze_sentiment
from app.services.services import analyze_news_by_country
news_bp = Blueprint('news', __name__)

def convert_numpy(obj):
    """Recursively convert NumPy types to native Python types."""
    if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy(item) for item in obj)
    else:
        return obj

@news_bp.route('/summary', methods=['POST'])
def get_summary():
    try:
        data = request.get_json()
        if not data or "url" not in data:
            return jsonify({"error": "Missing 'url' parameter"}), 400

        url = data["url"]
        text = extract_text_from_url(url)
        
        if text.startswith("Error"):
            return jsonify({"error": text}), 400

        category = predict_category(text)
        summary = summarize_text(text)
        sentiment = analyze_sentiment(text)
        metrics = generate_clustering_metrics(text)

        metrics = convert_numpy(metrics)

        return jsonify({
            "category": category,
            "summary": summary,
            "sentiment": sentiment,
            "clustering_metrics": metrics
        })

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@news_bp.route('/getnewsbycountry', methods=['POST'])
def getnewsbycountry():
    try:
        data = request.get_json()
        if not data or "country" not in data:
            return jsonify({"error": "Missing 'country' parameter"}), 400

        country = data["country"]
        response = analyze_news_by_country(country)
        
        # if response.startswith("Error"):
        #     return jsonify({"error": response}), 400

        return jsonify({
            "response": response
        })

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500