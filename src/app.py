import os
import logging
from logging.handlers import RotatingFileHandler
from logging import Filter
from flask import Flask, json,jsonify,request, g

from flask_jwt_extended import JWTManager, jwt_required
from datetime import timedelta
from pydantic import ValidationError

from common.business_validation import BusinessValidationError
from common.serializable_tools import conv_nan_inf_to_null
from flask_apscheduler import APScheduler
from config import Config
import uuid
import time

def set_logger(app):
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.handlers.clear()

    if app.debug:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        root_logger.setLevel(logging.DEBUG)
    else:
        log_dir = app.config['LOG_DIR']
        handler = RotatingFileHandler(filename=os.path.join(log_dir, "app.log"),
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        handler.setLevel(logging.INFO)
        root_logger.setLevel(logging.INFO)
   
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s %(module)s: %(message)s')
    handler.setFormatter(formatter)
  
    root_logger.addHandler(handler)

def create_flask_app():
    app = Flask(__name__)    
    app.config.from_object(Config)
    set_logger(app)

    from routes.blueprints import create_blueprints
    create_blueprints(app)

    @app.errorhandler(ValidationError)
    def handle_validation_error(e):
        app.logger.error(f"ValidationError: {e.errors()}")
        return jsonify(e.errors()), 422  # 统一返回 422

    @app.errorhandler(BusinessValidationError)
    def handle_business_validation_error(e):
        response = {
            'message': e.message,
            'error_code': e.error_code,
            'error_type': e.error_type,
        }
        app.logger.error(f"BusinessValidationError: {response}")
        return jsonify(response), 422  # 统一返回 422
    
    @app.before_request
    def before_request():
        trace_id = request.headers.get('X-Trace-ID', request.headers.get('X-Request-ID', uuid.uuid4().hex))
        g.trace_id = trace_id
        g.start_time = time.time()
        app.logger.info(f'开始处理请求:{request.path},trace_id:{trace_id}')
 
    @app.after_request
    def inject_trace_id(response):
        response.headers['X-Trace-ID'] = g.trace_id
        process_time = time.time() - g.start_time
        response.headers['X-Processing-Time'] = f'{process_time:.3f}'
        app.logger.info(f'处理完成,trace_id:{g.trace_id},status_code:{response.status_code}')
        return response

    @app.after_request
    def standardise_json(response):
        if response.content_type == 'application/json':
            try:
                data = json.loads(response.get_data(as_text=True))
                cleaned_data = conv_nan_inf_to_null(data)
                response.set_data(json.dumps(cleaned_data))
            except Exception as e:
                app.logger.error(f"Error processing JSON response: {str(e)}")
        return response
    
    @app.route('/download/<filename>', methods=['GET'])
    @jwt_required()
    def download_file(filename):
        from flask import send_from_directory
        directory = app.config['DATA_DIR']
        return send_from_directory(directory, filename, as_attachment=True)


    return app

app = create_flask_app()
JWTManager(app)

scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()

if __name__== "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)