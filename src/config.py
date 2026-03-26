import os

class Config:
    JWT_SECRET_KEY =  '00eff3fc3eb3e4c4320d1379095f643ec9eb03d2dd9d8bbb875160ee6371f862'
    JWT_ACCESS_TOKEN_EXPIRES = 3600  # 1小时
    JWT_REFRESH_TOKEN_EXPIRES = 7 * 24 * 3600  # 7天

    SECRET_KEY = '5935144adecc7b9b65d8aa5d73073a21e766df9df187dde7758853343e16957e'

    LOG_DIR = os.getenv('LOG_DIR', os.path.join(os.getcwd(), 'logs'))
    DATA_DIR = os.getenv('DATA_DIR', os.path.join(os.getcwd(), 'data'))
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # APScheduler配置
    SCHEDULER_API_ENABLED = True
    SCHEDULER_TIMEZONE = 'Asia/Shanghai'
    JOBS = [
        {
            'id': 'file_cleanup',
            'func': 'common.filemgr:clean_expired_files',  # 格式: '模块路径:函数名'
            'args': (DATA_DIR,),  # 传递数据目录作为参数
            'trigger': 'interval',
            'seconds': 300  # 每5分钟执行一次
        }
    ]
    # 执行器配置
    SCHEDULER_EXECUTORS = {
        'default': {'type': 'threadpool', 'max_workers': 1}
    }

class VxdfsConfig:
    config ={
        'host':'10.168.1.61',
        'username':'default',
        'password':'Vixtelnpm805@153_1104',
        'port':8123,
        'settings':{
            'readonly': 1,  # 会话级别只读
            'allow_ddl': 0,  # 禁止DDL
        }
    }