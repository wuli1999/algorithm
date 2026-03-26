from typing import List, Dict, Any
from pydantic import field_validator, Field,BaseModel
from common.business_validation import BusinessValidationError,ErrorCode
from timeseries.schema.entities import TimeSeriesTrainRequest,TimeSeriesInferRequest

class LSTMTrainConfig(BaseModel):
    lookback:List[int]=Field(..., description='lookback config')

    @field_validator('lookback')
    def valid_lookback(cls, v:List[int])->range:
        if len(v) < 2 or len(v) > 3:
            raise BusinessValidationError(message='The lookback config should be specified in the format [start, end[, step]]', error_code=ErrorCode.INVALID_PARAMETER)
        if any(not isinstance(x, int) for x in v):
            raise BusinessValidationError(message='The lookback config all elements must be integer', error_code=ErrorCode.INVALID_PARAMETER)
        
        return range(*v)    

class LSTMTrainRequest(TimeSeriesTrainRequest):
    '''
    LSTM时序训练请求参数映射类
    '''
    lstm_config:LSTMTrainConfig = Field(..., description='LSTM时序训练请求参数')
        

class LSTMInferRequest(TimeSeriesInferRequest):
    """
    lstm推理算法请求参数映射类
    """
    lstm_config:Dict[str,Any] = Field(..., description='LSTM模型配置信息')
    @field_validator('lstm_config')
    @classmethod
    def check_required_keys(cls, v:Dict[str,Any]) -> Dict[str,Any]:
        required_keys={'model_dumps'}
        missing = required_keys - v.keys()
        if missing:
            raise BusinessValidationError(message=f'/lstm_config missing required kyes:{missing}', error_code=ErrorCode.INVALID_PARAMETER)
        return v
 