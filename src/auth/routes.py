from flask import Blueprint, jsonify, request, current_app
from flask_jwt_extended import jwt_required  
from auth.service.authenticate import login, logout,refresh
from auth.schemas.entities import LoginRequest

def register_routes(name:str, prefix:str):
    auth_bp = Blueprint(name=name, import_name=__name__, url_prefix=prefix)


    @auth_bp.route('/login',  methods=['post'])
    def authenticate():
        params = LoginRequest.model_validate(request.get_json())
        return login(userinfo=params).model_dump(),200

    @auth_bp.route('/logout',  methods=['post'])
    @jwt_required()
    def quit():
        return jsonify(logout()),200

    @auth_bp.route('/refresh',  methods=['post'])
    @jwt_required(refresh=True)
    def update_token():
        return refresh().model_dump(),200
    
    return auth_bp
