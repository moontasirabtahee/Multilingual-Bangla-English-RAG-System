from flask import Flask
from api.routes import rag_blueprint

def create_app():
    app = Flask(__name__)
    app.register_blueprint(rag_blueprint)
    return app