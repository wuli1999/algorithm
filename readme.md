
pyenv activate algorithm-env
#python 版本3.12.3
# 设置环境变量
export FLASK_APP=app.py
export FLASK_ENV=production  # 或 development
export FLASK_RUN_PORT=8080
export FLASK_RUN_HOST=0.0.0.0
export FLASK_DEBUG=0  # 禁用调试
export DATA_DIR=数据文件路径，未指定时为cwd的data目录
export LOG_DIR=日志文件路径，未指定时为cwd的logs目录

# 然后启动
flask run