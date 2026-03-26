from flask import Blueprint, jsonify, request, url_for, redirect,session
from flask_jwt_extended import jwt_required

from common.evaluation import evaluate_model
from timeseries.utils.preprocessing import explore
from timeseries.utils.explore_report import interpret_summary
from timeseries.core.arima import train as arima_train, infer as arima_infer
from timeseries.core.lstmts import train as lstm_train, infer as lstm_infer
from timeseries.core.prophet import train as prophet_train, infer as prophet_infer
from common.business_validation import BusinessValidationError, ErrorCode
import json

models={
    'arima':{
        'train':arima_train,
        'infer':arima_infer,
        },
    'lstm':{
        'train':lstm_train, 
        'infer':lstm_infer,
        },
    'prophet':{
        'train':prophet_train,
        'infer':prophet_infer,
    },
}

methods={
    'explore':explore,
    'evaluate':evaluate_model,
    'interpret':interpret_summary,
}

def parse_form_data(request):
    try:
        params = json.loads(request.form['params'])
        file = request.files['file']
        #if file.content_length <= 0:
        #    raise Exception('无效文件')
        return file, params
    except Exception as e:
        raise BusinessValidationError(message=f'参数格式或内容解析错误:{str(e)}', error_code=ErrorCode.INVALID_PARAMETER)

def direct_blueprints():
    direct_bp = Blueprint('direct', __name__, url_prefix='/')   
    
    @direct_bp.route('/train/<string:model>', methods=['post'])
    def timeseries_train(model:str):
        return models[model]['train'](request.get_json()).model_dump(exclude_none=True),200

    @direct_bp.route('/infer/<string:model>', methods=['post'])
    def timeseries_infer(model:str):
         return models[model]['infer'](request.get_json()).model_dump(exclude_none=True),200

    @direct_bp.route('/<string:method>', methods=['post'])
    def timeseries_methods(method:str):
        m = methods[method]
        res = m(request.get_json())
        if isinstance(res, str):
            from flask import Response
            return Response(res,mimetype="text/plain; charset=utf-8"),200
        return res,200
    
    return direct_bp

def file_blueprints():
    file_bp = Blueprint('file', __name__, url_prefix='/')
    
    @file_bp.before_request
    def before_file_request():
        import pandas
        file, params = parse_form_data(request)
        
        try:
            data_config = params.get('data_config')
            header = data_config.get('header')
            if isinstance(header, bool):
                header = 0 if header else None
            else:
                header = 'infer'
                        
            data = pandas.read_csv(file,header=header)
            if header == 'infer':
                header = None if data.columns.to_list() == list(range(len(data.columns))) else 0
            
            if header is None:   
                params['data'] = data.values.tolist()
            else:
                params['data'] = [data.columns.to_list()] + data.values.tolist()
                params['data_config']['header'] = True
        except Exception:
            raise BusinessValidationError(message='文件解析失败,内容或参数错误', error_code=ErrorCode.DATA_ERROR)            
        
        request.custom_data = params

    @file_bp.route('/ftrain/<string:model>', methods=['post'])
    def timeseries_train(model:str):
        body = request.custom_data
        return models[model]['train'](body).model_dump(exclude_none=True),200

    @file_bp.route('/finfer/<string:model>', methods=['post'])
    def timeseries_infer(model:str):
        body = request.custom_data
        return models[model]['infer'](body).model_dump(exclude_none=True),200

    @file_bp.route('/f<string:method>', methods=['post'])
    def timeseries_methods(method:str):
        m = methods[method]
        body = request.custom_data
        res = m(body)
        if isinstance(res, str):
            from flask import Response
            return Response(res,mimetype="text/plain; charset=utf-8"),200
        return res,200
    
    return file_bp

def register_routes(name:str, prefix:str):
    ts_bp = Blueprint(name=name, import_name=__name__, url_prefix=prefix)
   
    direct_bp = direct_blueprints()
    file_bp = file_blueprints()

    ts_bp.register_blueprint(direct_bp)
    ts_bp.register_blueprint(file_bp)

    @ts_bp.before_request
    @jwt_required()  
    def validate_url():
        model = request.view_args.get('model')
        method = request.view_args.get('method')

        if model is not None:
            if model not in models.keys():
                raise BusinessValidationError(message=f'不支持模型:{model}', error_code=ErrorCode.RESOURCE_NOT_FOUND)            
        elif method is not None:
            if method not in methods.keys():
                raise BusinessValidationError(message=f'不支持方法:{method}', error_code=ErrorCode.RESOURCE_NOT_FOUND) 
        else:
            raise BusinessValidationError(message='无效URL', error_code=ErrorCode.RESOURCE_NOT_FOUND)

    return ts_bp