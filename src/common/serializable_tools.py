from pydantic import BaseModel, field_serializer, ConfigDict
import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import Any

class SerializableModel(BaseModel):
    """
    一个增强的 BaseModel，可自动将 NumPy / Pandas 类型转换为可 JSON 序列化的格式。
    所有需要此功能的模型都应继承此类。
    """
    model_config = ConfigDict(arbitrary_types_allowed=True,  # 允许非 Pydantic 类型作为字段
                              )
    @field_serializer('*')
    def serialize_field(self, value: Any) -> Any:
        """通用字段序列化器，递归处理所有字段。"""
        # 处理 NumPy 类型
        if isinstance(value, np.integer):
            return int(value)
        elif isinstance(value, np.floating):
            return float(value)
        elif isinstance(value, np.bool_):
            return bool(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        
        # 处理 Pandas 类型
        elif isinstance(value, pd.Timestamp):
            return value.isoformat()
        elif isinstance(value, pd.Timedelta):
            return value.isoformat()
        elif isinstance(value, pd.Series):
            # 使用 name 作为字典的键，若无则用 'value'
            return {value.name or "value": value.tolist()} if not value.empty else {}
        elif isinstance(value, pd.DataFrame):
            return value.to_dict(orient="records")
        
        # 递归处理 Pydantic 模型
        elif isinstance(value, BaseModel):
            return value.model_dump()
        
        # 递归处理容器类型
        elif isinstance(value, dict):
            return {k: self.serialize_field(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple, set)):
            return [self.serialize_field(item) for item in value]
        
        # 处理日期和时间
        elif isinstance(value, (datetime, date)):
            return value.isoformat()
        
        # 兜底：返回原值
        return value
    
import json
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

class NumpyPandasEncoder(json.JSONEncoder):
    """自定义 JSON 编码器，用于处理 NumPy 和 Pandas 对象[1,3](@ref)"""
    
    def default(self, obj):
        # 处理 NumPy 标量类型
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return {
                "__numpy_ndarray__": True,
                "data": obj.tolist(),
                "dtype": str(obj.dtype)
            }
        
        # 处理 NumPy 通用标量类型
        if isinstance(obj, np.generic):
            return obj.item()

        # 处理 Pandas Series
        if isinstance(obj, Series):
            return {
                "__pandas_series__": True,
                "name": obj.name,
                "index": obj.index.tolist(),
                "data": obj.values.tolist(),
                "dtype": str(obj.dtype)
            }

        # 处理 Pandas DataFrame
        if isinstance(obj, DataFrame):
            return {
                "__pandas_dataframe__": True,
                "columns": obj.columns.tolist(),
                "index": obj.index.tolist(),
                "data": obj.values.tolist(),
                "dtypes": [str(dtype) for dtype in obj.dtypes]
            }

        # 其他类型交给默认编码器处理
        return super().default(obj)

def numpy_pandas_decoder(dct):
    """自定义 JSON 解码器，将字典恢复为 NumPy 和 Pandas 对象[1,3](@ref)"""
    if "__numpy_ndarray__" in dct:
        return np.array(dct["data"], dtype=dct["dtype"])
    
    if "__pandas_series__" in dct:
        return Series(
            data=dct["data"],
            index=dct["index"],
            name=dct["name"],
            dtype=dct["dtype"]
        )
    
    if "__pandas_dataframe__" in dct:
        return DataFrame(
            data=dct["data"],
            columns=dct["columns"],
            index=dct["index"]
        )
    
    return dct

def serialize_to_json(data, filepath=None):
    """
    序列化数据到 JSON 字符串或文件
    Args:
        data: 要序列化的数据
        filepath: 文件路径，如果为 None 则返回字符串
    Returns:
        如果 filepath 为 None，返回 JSON 字符串
    """
    if filepath:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, cls=NumpyPandasEncoder, ensure_ascii=False, indent=2)
    else:
        return json.dumps(data, cls=NumpyPandasEncoder, ensure_ascii=False, separators=(',', ':'))

def deserialize_from_json(filepath_or_str):
    """
    从 JSON 文件或字符串反序列化数据
    Args:
        filepath_or_str: 文件路径或 JSON 字符串
    Returns:
        反序列化后的数据
    """
    if isinstance(filepath_or_str, str) and '\n' in filepath_or_str or filepath_or_str.startswith('{'):
        # 如果是 JSON 字符串
        return json.loads(filepath_or_str, object_hook=numpy_pandas_decoder)
    else:
        # 如果是文件路径
        with open(filepath_or_str, 'r', encoding='utf-8') as f:
            return json.load(f, object_hook=numpy_pandas_decoder)


def conv_nan_inf_to_null(obj):
    """递归清理NaN/Inf"""
    import math
    
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    elif isinstance(obj, dict):
        return {k: conv_nan_inf_to_null(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [conv_nan_inf_to_null(v) for v in obj]
    elif hasattr(obj, '__dict__'):  # 对象
        return conv_nan_inf_to_null(obj.__dict__)
    return obj