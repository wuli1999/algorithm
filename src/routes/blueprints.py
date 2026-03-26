from flask import Blueprint
import importlib

BLUEPRINTS_CONFIG={
    'timeseries':{
    'name':'timeseries',
    'import_name':'timeseries.api.routes',
    'url_prefix':'/ts',
    },
    'correlation':{
        'name':'correlation',
        'import_name':'correlation.api.routes',
        'url_prefix':'/cs',
    },
    'auth':{
        'name':'auth',
        'import_name':'auth.routes',
        'url_prefix':'/auth'
    }
}

def create_blueprints(app):
    bp_root = Blueprint(name = 'root', import_name='app', url_prefix='/')     
    for k,v in BLUEPRINTS_CONFIG.items():
        module = importlib.import_module(v['import_name'])
        func = getattr(module, 'register_routes')
        bp = func(v['name'], v['url_prefix'])
        bp_root.register_blueprint(bp)

    app.register_blueprint(bp_root)