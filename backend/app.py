from flask import Flask, request, jsonify, session, make_response, send_from_directory
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager, jwt_required,
    create_access_token,
    get_jwt_identity, get_jwt, unset_jwt_cookies
)

from werkzeug.local import LocalProxy
from dotenv import load_dotenv
import os

from helpers.utils_helper import *
from datetime import datetime
# from mongodb.db import *


def create_app():
    load_dotenv('.flaskenv')

    app = Flask(__name__)
    jwt = JWTManager(app)

    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
    app.config['DEBUG'] = os.getenv('DEBUG')
    app.config['MONGO_URI'] = os.getenv('DB_SRV_CONNECTOR')
    app.config['JWT_TOKEN_LOCATION'] = ["headers"]
    app.config['JWT_COOKIE_SECURE'] = False  # Set to True in production
    app.config['SESSION_COOKIE_HTTPONLY'] = False  # Set to True in production
    app.config['JWT_COOKIE_CSRF_PROTECT'] = False  # Set to True in production
    app.config['SESSION_COOKIE_SAMESITE'] = "None"  # Set to True in production
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = 3600  # 1 hour
    # db = LocalProxy(get_db)
#     CORS(app)
    CORS(app, origins='*', methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'], allow_headers=['Content-Type', 'Authorization'], supports_credentials=True)

#     @app.before_request
#     def before_request():
#         print(request.headers)
        # print(request.json)

#     @app.after_request
#     def after_request(response):
#         print(response)

    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve_vue(path):
        if path != "" and os.path.exists(os.path.join('static', path)):
            return send_from_directory('static', path)
        else:
            return send_from_directory('static', 'index.html')

    @app.route('/api/predict', methods=['POST'])
    def predict():
        data = request.json
        print(data)
        pred = predictor(data['messageText'])

        return jsonify({**data,'prediction': str(pred[0])}),201

    return app