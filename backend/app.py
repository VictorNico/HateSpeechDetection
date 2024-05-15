from flask import Flask, request, jsonify, session, make_response, send_from_directory
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager, jwt_required,
    create_access_token,
    get_jwt_identity, get_jwt, unset_jwt_cookies
)

from passlib.hash import pbkdf2_sha256

import bson
from bson.json_util import dumps
from bson.son import SON
from bson.objectid import ObjectId

from functools import wraps
from dotenv import load_dotenv
import os

# from peewee import SqliteDatabase, Model, CharField
from mongodb.db import *
from helpers.utils_helper import *
from datetime import datetime

# Custom JSON serializer for User object
def user_encoder(user):
    return {
        att: str(user[att]) for att in user.keys()
    }

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
    db = LocalProxy(get_db)
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
    
    @app.route('/api/register', methods=['POST'])
    def register():
        name = request.json['name']
        if not name:
            return jsonify({'message': 'Missing name'}), 400

        if db.users.find_one({'name': name}):
            return jsonify({'message': 'name already exists'})

        # Create a new user document
        timestamp = datetime.now()
        user = {
            'name': name,
            'createdAt': timestamp,
            'updatedAt': timestamp
        }
        # Insert the user into the database
        user_id = db.users.insert_one(user)

        return jsonify({'message': 'User registered successfully', 'user_id': str(user_id)}), 201

    @app.route('/api/login', methods=['POST'])
    def login():
        email = request.json['email']
        password = request.json['password']

        if not email or not password:
            return jsonify({'message': 'Missing email or password'}), 400
        

        # Find the user in the database
        user = db.users.find_one({'email': email})
        # print(type(user))
        if user and pbkdf2_sha256.verify(password, user['password']):
            timestamp = datetime.now()
            conn = {
                'userId': user_encoder(user)['_id'],
                'online': True,
                'createdAt': timestamp,
                'updatedAt': timestamp
            }
            # Insert the connexion into the database
            conn = db.connexions.insert_one(conn)
#           print('hj-peewee')
            # Generate a JWT token
            payload = {'conn':str(conn.inserted_id),**user}
            # access_token = create_access_token(identity=User(str(user['_id']),user['username'],user['email']))
            access_token = create_access_token(identity=user_encoder(user)['_id'], additional_claims=user_encoder(payload))
            resp = make_response(jsonify({'message': 'Login successful', 'token':access_token}), 200)
            resp.set_cookie('access_token_cookie', access_token)
            # resp.cache_control.no_cache = True
            return resp
        else:
            return make_response('Invalid email or password', 401, {'WWW-Authenticate': 'Basic realm="Login Required"'})
        

    @app.route('/api/logout', methods=['POST'])
    @jwt_required()
    def logout():
#         print('kkk')
        jti = get_jwt_identity()  # Update get_jwt_identity function usage
#         print(jti)
#         print(get_jwt())
        resp = make_response(jsonify({'message': 'Logout successful'}), 200)  # Create a response object
        unset_jwt_cookies(resp)
        return resp

    @app.route('/api/predict', methods=['POST'])
    @jwt_required()
    def predict():
        data = request.json
        timestamp = datetime.now()
        dat = {key:data[key][0] for key in data.keys()}
        stud = {
            **dat,
            'createdAt': timestamp,
            'updatedAt': timestamp
        }
        # Insert the student into the database
        student = db.students.insert_one(stud)
#         del data["_id"]
#         print(data)
        dataframe = pd.DataFrame(data)
#         print(dataframe)
        pred = predictor(preprocessing(dataframe.drop(["matricule"],axis=1)))
        pred['proba']["matricule"]= list(dataframe["matricule"].values)
        pred['proba'] = pred['proba'].to_dict()
        print(list(
                      set(pred['args'].keys()) - set([
                      key for key in pred['args'].keys()
                      if (sum([col in key for col in [*list(dataframe['Sexe'].values),*list(dataframe['Classe'].values)]]) == 0) and (('Sexe' in key) or ('Classe' in key))
                       ])
                      ))
        pred['args'] = {el:pred['args'][el] for el in list(
                                                                            set(pred['args'].keys()) - set([
                                                                            key for key in pred['args'].keys()
                                                                            if (sum([col in key for col in [*list(dataframe['Sexe'].values),*list(dataframe['Classe'].values)]]) == 0) and (('Sexe' in key) or ('Classe' in key))
                                                                             ])
                                                                            )}
        timestamp = datetime.now()
        # Insert the student into the database
        print(type(student.inserted_id))
        data ={
            'idUser': str(get_jwt_identity()),
            'idStudent': str(student.inserted_id),
            'idConn':str(get_jwt()['conn']),
            'prediction': pred['prediction'][0],
            'createdAt': timestamp,
            'updatedAt': timestamp
        }
        student = db.predictions.insert_one(data)

        return jsonify(pred)

    @app.route("/api/history", methods=["GET"])
    @jwt_required()
    def history():
        jti = get_jwt_identity()  # get the session authentification id
        # Find the all prediction belong to this user in the database
#         print(jti)
#         print('----')
        predictions = db.predictions.find({'idUser': jti})
        predictions = list(predictions)

        for index,pred in enumerate(predictions):
            print(index,pred)
            temp = db.students.find_one({'_id': ObjectId(pred['idStudent'])})
            print(temp)
            predictions[index]['student'] = temp

#         print(predictions)
        return jsonify(dumps(predictions))

    @app.route("/api/updateProfile", methods=["POST"])
    @jwt_required()
    def updateProfile():
        jti = get_jwt_identity()  # get the session authentification id
        data = request.json
        query = {'_id': ObjectId(jti)}
        update_data = {'$set': data}

        result = db.users.find_one_and_update(query, update_data, return_document=True)

        if result:
            return make_response(jsonify({'message': 'Document updated successfully'}), 200)
        else:
            return make_response(jsonify({'message': 'Document not found'}), 200)

    @app.route('/api/predictions/count-by-month', methods=['GET'])
    @jwt_required()
    def count_predictions_by_month():
        jti = get_jwt_identity()  # get the session authentification id
        pipeline = [
            {
                '$match': {
                    'idUser': jti
                }
            },
            {
                '$project': {
                    'year': {'$year': '$createdAt'},
                    'month': {'$month': '$createdAt'}
                }
            },
            {
                '$group': {
                    '_id': {'year': '$year', 'month': '$month'},
                    'count': {'$sum': 1}
                }
            },
            {
                '$sort': SON([('_id.year', 1), ('_id.month', 1)])
            }
        ]
        result = list(db.predictions.aggregate(pipeline))
        return dumps(result)

    @app.route('/api/connexion/count-by-month', methods=['GET'])
    @jwt_required()
    def count_pconnexions_by_month():
        jti = get_jwt_identity()  # get the session authentification id
        pipeline = [
            {
                '$match': {
                    'userId': jti
                }
            },
            {
                '$project': {
                    'year': {'$year': '$createdAt'},
                    'month': {'$month': '$createdAt'}
                }
            },
            {
                '$group': {
                    '_id': {'year': '$year', 'month': '$month'},
                    'count': {'$sum': 1}
                }
            },
            {
                '$sort': SON([('_id.year', 1), ('_id.month', 1)])
            }
        ]
        result = list(db.connexions.aggregate(pipeline))
        return dumps(result)

    return app

