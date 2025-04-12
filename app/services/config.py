from flask import Flask
from dotenv import load_dotenv
import os

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

load_dotenv(dotenv_path='.env', override=True)

def create_app():
    app = Flask(__name__)
    app.json.sort_keys = False

    from app.routes import register_blueprints
    register_blueprints(app)

    return app
