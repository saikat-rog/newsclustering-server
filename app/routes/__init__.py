from flask import Blueprint
from .news import news_bp
from flask import Flask
from app.routes.news import news_bp
from app.routes.feedback import feedback_bp

def register_blueprints(app):
    app.register_blueprint(news_bp, url_prefix = '/news')
    app.register_blueprint(feedback_bp, url_prefix = '/feedback')
