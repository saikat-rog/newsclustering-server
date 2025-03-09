from flask import Blueprint
from .news import news_bp

def register_blueprints(app):
    app.register_blueprint(news_bp, url_prefix = '/news')