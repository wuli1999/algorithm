import os
from datetime import datetime, timedelta
import logging
import csv
logger = logging.getLogger(__name__)

from flask import current_app
from pandas import DataFrame
import uuid

def clean_expired_files(directory):
    now = datetime.now()
    expiration_time = timedelta(minutes=10)  # 设置文件过期时间为10分钟
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            if now - file_mod_time > expiration_time:
                try:
                    os.remove(file_path)
                    logger.info(f"Removed expired file: {filename}")
                except Exception as e:
                    logger.error(f"Error removing file {filename}: {str(e)}")

def write_df_to_file(data: DataFrame, header: bool = True, filename: str = None, index: bool = False)-> str:
    filename = filename or f'{uuid.uuid4()}.csv'
    file_path = os.path.join(current_app.config['DATA_DIR'], filename)
    try:
        data.to_csv(file_path, index=index, header=data.columns.tolist() if header else False)
        logger.info(f"DataFrame successfully written to file: {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error writing DataFrame to file {filename}: {str(e)}")
    return None