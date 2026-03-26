from flask import Blueprint, request
from flask_jwt_extended import jwt_required
from correlation.core.detect import correlation_detect, correlation_detect_file
from correlation.core.mixed_regression import regression,fregression
from correlation.schema.entities import DetectParams, DetectRequest,FDetectRequest,RegressionRequest,FRegressionRequest,Params
from common.business_validation import BusinessValidationError, ErrorCode
import json


def register_routes(name:str, prefix:str):
    cs_bp = Blueprint(name=name, import_name=__name__, url_prefix=prefix)
    @cs_bp.before_request
    @jwt_required()
    def before_request():
        pass

    @cs_bp.route('/detect',  methods=['post'])
    def detect():
        params = DetectRequest.model_validate(request.get_json())
        return correlation_detect(request = params).model_dump(), 200

    @cs_bp.route('/fdetect', methods=['post'])
    def file_detect():
        if 'params' in request.form:
            try:
                params = json.loads(request.form['params'])
            except json.JSONDecodeError:
                raise BusinessValidationError(error_code=ErrorCode.INVALID_PARAMETER, message="params参数格式错误--非法JSON字符串")
        else:
            raise BusinessValidationError(error_code=ErrorCode.INVALID_PARAMETER, message="params参数缺失")

        if 'file' not in request.files:
            raise BusinessValidationError(error_code=ErrorCode.INVALID_PARAMETER, message="file参数缺失")

        file = request.files['file']
        p = FDetectRequest(file=file, params=DetectParams(**params))

        return correlation_detect_file(request=p).model_dump(), 200

    @cs_bp.route('/regression', methods=['post'])
    def fit():
        params = RegressionRequest.model_validate(request.get_json())
        return regression(params).model_dump(), 200

    @cs_bp.route('/fregression', methods=['post'])
    def file_regression():
        if 'params' in request.form:
            try:
                params = json.loads(request.form['params'])
            except json.JSONDecodeError:
                raise BusinessValidationError(error_code=ErrorCode.INVALID_PARAMETER, message="params参数格式错误--非法JSON字符串")    
        else:
            raise BusinessValidationError(error_code=ErrorCode.INVALID_PARAMETER, message="params参数缺失")
        
        if 'file' not in request.files:
            raise BusinessValidationError(error_code=ErrorCode.INVALID_PARAMETER, message="file参数缺失")
        
        file = request.files['file']
        p = FRegressionRequest(file=file, params=Params(**params))

        return fregression(p).model_dump(), 200
    
    return cs_bp