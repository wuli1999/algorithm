from pydantic import BaseModel,Field

class LoginRequest(BaseModel):
    username:str = Field(..., description='用户名')
    password:str = Field(..., description='密码')

class LoginResponse(BaseModel):
    access_token:str = Field(..., description='access token')
    refresh_token:str = Field(..., description='refresh_token')

class RefreshResponse(BaseModel):
    access_token:str = Field(..., description='access token')
