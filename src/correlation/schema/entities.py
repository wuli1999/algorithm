from typing import List, Any, Tuple,Optional,Dict
from pydantic import BaseModel, Field
from common.serializable_tools import SerializableModel

class DetectParams(BaseModel):
    header:Optional[bool] = Field(False, description='数据是否包含表头，默认为False')
    columns:Optional[List[int]] = Field(None, description='需要检测的列索引列表，默认为None表示检测所有列')

class DetectRequest(BaseModel):
    # 数据集
    data: List[List[Any]]
    params: DetectParams

class FDetectRequest(BaseModel):
    file:Any
    params: DetectParams

class DetectResponse(SerializableModel):
    correlation_matrix:Any
    pvalue_matrix:Any

class Params(BaseModel):
    trials:Optional[int] = Field(100, description='随机试验次数，默认为100')
    header:Optional[bool] = Field(False, description='数据是否包含表头，默认为False')
    column_x:Optional[int] = Field(0, description='自变量列索引')
    column_y:Optional[int] = Field(1, description='因变量列索引')
    save_to_file:Optional[bool] = Field(False, description='是否保存文件，默认为False')
    confidence:Optional[float] = Field(0.90, description='包络线置信水平，默认为0.90')

class RegressionRequest(BaseModel):
    data:List[List[Any]]
    params:Params

class FRegressionRequest(BaseModel):
    file:Any
    params:Params

class SpearmanCeoff(BaseModel):
    """
    贝尔曼相关系数
    coefficient:相关系数值
    p_value:相关系数的p值
    """
    coefficient:float
    p_value:float

class Metrics(BaseModel):
    mse:float
    r2_score:float

class RegressionResponse(SerializableModel):
    data:Any
    model_params:Dict[str,Any]
    metrics:Metrics
    s_ceof:SpearmanCeoff
    filename:Optional[str] = None