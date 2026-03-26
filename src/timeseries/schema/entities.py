from typing import List,Tuple,Any,Optional,Dict,Literal,Union
from pydantic import BaseModel,Field,model_validator,ConfigDict
from common.business_validation import BusinessValidationError,ErrorCode
from timeseries.utils.tools import validate_freq

class NumericSummary(BaseModel):    
    """
    数值摘要

    :count: 数量
    :mean: 算数平均值
    :std: 标准差
    :mode: 众数
    :invalid: 无效值数量
    :five_number_summary: 五数摘要，分别对应[最小值,25分位数,中位数,75分位数,最大值]
    :histogram: 直方图数组
    """
    count:int|None=None
    mean:float|None=None
    std:float|None = None
    mode:List[float]|None=None
    invalid:int |None = None
    five_number_summary:Tuple[float,float,float,float,float] |None=None
    histogram:Any|None=None

class SeriesSummary(BaseModel):
    """
    时序数据摘要

    :missed: 缺失点数量
    :freq: 频率
    :time_range: 时间（unix second timestamp）范围
    :seasonal: 检测到的季节周期列表
    """
    missed:int = Field(0, description='缺失点数量')
    freq:Optional[str]|None = Field(None, description='数据频率')
    time_range:Optional[Tuple[int, int]]|None = Field(None, description='时间范围')
    seasonal:Optional[List[int]]|None = Field(None, description='检测到的季节周期列表')

class ExploreResponse(BaseModel):
    """
    时序数据探索返回信息

    :numeric_summary: 数值摘要信息
    :series_summary: 时序数据摘要信息
    """
    numeric_summary:Optional[NumericSummary]|None = None
    series_summary:Optional[SeriesSummary]|None = None   


class HistogramConfig(BaseModel):
    """
    直方图请求参数

    :bins: 分箱数量
    :left_edge: 下边界
    :right_edge: 上边界
    :overflow: 是否允许溢出箱
    """
    bins:int = Field(15, description='数量')
    left_edge:Optional[float] = Field(None, description='下边界')
    right_edge:Optional[float] = Field(None, description='上边界')
    overflow:Optional[bool] = Field(True, description='是否允许溢出箱')

class DataSummaryConfig(BaseModel):
    histogram:Optional[HistogramConfig]|None = None

class DataConfig(BaseModel):
    """
    数据集配置映射类
    """
    header:Optional[bool] = Field(None, description='数据是否包含表头')
    time_col: int = Field(0, ge=0, description='时间列索引位置')
    data_col: int = Field(1, ge=0, description='数据列索引位置')
    freq:Optional[str] = Field(default=None, description='时序间隔周期')
    interpolate:Optional[bool] = Field(False, description='对缺失数据是否插值,默认False')
    data_range:Optional[Tuple[float,float]] = Field(None, description='数据有效范围，左闭右开,异常数据过滤以及拟合范围限制')
    summary_config:Optional[DataSummaryConfig]|None = None
    save_to_file:Optional[bool] = Field(False, description='是否将结果数据保存为文件，默认False')

    @model_validator(mode='before')
    @classmethod
    def _normalize_config(cls, data: Any) -> Any:
        if isinstance(data, dict):
            v = data.get('freq')
        else:
            v = getattr(data, 'freq', None)
        
        if v is None or v == 'auto':
            data['freq'] = 'auto'
        else:
            validate_freq(v, f'data_config/freq:{v} value error.')
        
        return data

class EnvelopeConfig(BaseModel):
    """
    包络线配置映射类
    """
    method:Literal['sigma','iqr','quantile','native'] = Field('native', description='阈值包络线方法')
    windows:int = Field(0, description='滑动窗口大小')
    confidence:float = Field(0.95, description='置信范围', ge=0.0, le=1.0)

class ForecastConfig(BaseModel):
    steps:int = Field(..., ge=0, description='预测步数')
       
    @model_validator(mode='before')
    @classmethod
    def _normalize_config(cls, data: Any) -> Any:
        if isinstance(data, dict):
            s = data.get('steps')
        else:
            s = getattr(data, 'steps', None)
        
        if s is None:
            data['steps'] = 0

        return data

class InferConfig(BaseModel):
    """
    算法配置参数映射类
    """
    forecast:Optional[ForecastConfig] = Field(None, description='预测参数')
    envelope:Optional[EnvelopeConfig] = Field(None, description='包络线参数')
    
    @model_validator(mode='before')
    @classmethod
    def _normalize_config(cls, data: Any) -> Any:
        if isinstance(data, dict):
            e = data.get('envelope')
            f = data.get('forecast')
        else:
            e = getattr(data, 'envelope', None)
            f = getattr(data, 'forecast', None)

        if e is None:
            data['envelope'] = {'emthod':'sigma', 'params':[0.95,0] }

        if f is None:
            data['forecast'] = {}

        return data

class TrainConfig(BaseModel):
    train_ratio:float = Field(0.8,  description='训练集占比', ge=0.1, le=1.0)
    forecast_steps:int = Field(..., description='预测步数', ge=0)
    forecast_detail:bool = Field(False, description='是否返回多步预测详情')
    max_trial_count:int = Field(20, lt=100, gt=0, description='最大训练次数')
    total_trial_timeout:int = Field(-1, description='训练超时秒数')
    per_trial_timeout:int = Field(-1, description='单次训练超时秒数')

class TimeSeriesRequest(BaseModel):
    """
    时序算法请求参数映射类
    """
    # 数据集
    data: List[List[Any]] = Field(None, description='时序数据')
    # 数据集参数
    data_config:DataConfig    
    envelope:EnvelopeConfig = Field(EnvelopeConfig(), description='包络线参数')

class TimeSeriesTrainRequest(TimeSeriesRequest):
    """
    时序算法训练请求参数
    """
    train_config:TrainConfig = Field(..., description='训练参数')

class TimeSeriesInferRequest(TimeSeriesRequest):
    """
    时序算法预测/拟合请求参数
    """
    infer_config:InferConfig = Field(..., description='预测/拟合参数')

class ModelMetrics(BaseModel):
    MAE:Optional[float]|None=None
    MSE:Optional[float]|None=None
    RMSE:Optional[float]|None=None
    MAPE:Optional[float]|None=None
    SAMP_STD:Optional[float]|None=None
    SAMP_MEAN:Optional[float]|None=None
    RESID_STD:Optional[float]|None=None
    RESID_MEAN:Optional[float]|None=None
    LB_PVALUE:Optional[float]|None=None
    SW_PVALUE:Optional[float]|None=None

class TimeSeriesResponse(BaseModel):
    """
    时序算法返回类
    """
    data:Any
    numeric_summary:Optional[NumericSummary]|None = None
    inference_metrics:Optional[ModelMetrics]|None = None
    # 预测模型评估指标
    forecast_metrics:Optional[List[ModelMetrics]]|None = None
    series_summary:Optional[SeriesSummary]|None = None    
    model_summary:Optional[Dict]|None = None
    # 训练预测结果
    forecast_results:Optional[List[Any]]|None = None
    filename:Optional[str]|None = None

class ARIMAConfig(BaseModel):
    """
    ARIMA算法配置参数映射类
    """
    #p,d,q参数 
    order:Optional[Tuple[int, int, int]] = Field(None, description='order参数,为空时自动检测')
    
    @model_validator(mode='after')
    @classmethod
    def _normalize_config(cls, data: Any) -> Any:
        if isinstance(data, dict):
            order = data.get('order')
            #season_order = data.get('season_order')
        else:
            order = getattr(data, 'order', None)
            season_order = getattr(data, 'season_order', None)
        
        if order is not None:
            if sum(i < 0 or i >5 for i in order) > 0:
                raise BusinessValidationError(message=f'arima_onfig/order:{order}  0≤p,d,q≤5', error_code=ErrorCode.INVALID_PARAMETER)
            if sum(order) > 10:
                raise BusinessValidationError(message=f'arima_onfig/order:{order}  sum p,d,q value > 10)', error_code=ErrorCode.INVALID_PARAMETER)
        return data

   

class ARIMARequest(TimeSeriesInferRequest):
    """
    ARIMA算法请求参数映射类
    """
    model_config = ConfigDict(strict=False)
    arima_config:ARIMAConfig