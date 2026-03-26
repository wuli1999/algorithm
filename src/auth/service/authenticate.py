
import hashlib
from flask import jsonify
from flask_jwt_extended import (
    create_access_token,     
    create_refresh_token,
    get_jwt_identity
)

from auth.schemas.entities import LoginRequest,LoginResponse, RefreshResponse
from common.business_validation import BusinessValidationError,ErrorCode

USERS={
    "algorithm":hashlib.sha256(b"Vixtel815@@@@815").hexdigest()
}

def login(userinfo:LoginRequest)->LoginResponse:    
    username = userinfo.username
    password = userinfo.password
    if username in USERS and USERS[username] == hashlib.sha256(password.encode()).hexdigest():
        return LoginResponse(access_token=create_access_token(identity=username),
                                 refresh_token=create_refresh_token(identity=username))

    raise BusinessValidationError(message='Invalid username or password.', error_code=ErrorCode.PERMISSION_DENIED)


def logout():
    current_user = get_jwt_identity()
    return {'message':f'user:{current_user} logout successfully'}

def refresh()->RefreshResponse:
    current_user = get_jwt_identity()
    new_access_token = create_access_token(identity=current_user)
    return RefreshResponse(access_token=new_access_token)
