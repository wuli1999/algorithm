import json
import pandas as pd
from pandas import DataFrame
from common.business_validation import BusinessValidationError,ErrorCode
from timeseries.schema.entities import TimeSeriesRequest
from timeseries.schema.entities import (ExploreResponse, 
                              NumericSummary, 
                              SeriesSummary, 
                              TimeSeriesRequest)

from timeseries.utils.tools import (histogram,
                                   is_continuous_auto,
                                   find_periodicity,
                                   compare_freq)
import numpy as np

def explore(body:json)->json:

     request = TimeSeriesRequest.model_validate(body)
     _, data_summary, series_summary = data_prepare(request=request, analysis_data=True, analysis_series=True)
     return ExploreResponse(numeric_summary=data_summary, 
                            series_summary=series_summary).model_dump(exclude_none=True)


def data_prepare(request, analysis_data:bool = True, analysis_series:bool = True):
    tcol = request.data_config.time_col
    dcol = request.data_config.data_col
    header = request.data_config.header

    summary_config = request.data_config.summary_config
    data_range = request.data_config.data_range
    freq:str = request.data_config.freq

    data_summary:NumericSummary = None
    series_summary:SeriesSummary = None

    ###数据加载与清洗
    try:
        if header:
            fdata = DataFrame(request.data[1:], columns=request.data[0])
        else:
            fdata = DataFrame(request.data)                                 
    except Exception as e:
        raise BusinessValidationError(error_code=ErrorCode.DATA_ERROR, message=f'数据解析错误: {str(e)}')

    #设置索引列
    try:
        is_timestamp =  isinstance(fdata.iloc[:, tcol][0], (int, float, np.integer, np.floating))
        ts = pd.to_datetime(fdata.iloc[:, tcol], errors='raise', unit='s' if is_timestamp else None) #验证时间列数据格式
        fdata.set_index(ts, inplace=True)
        #清除时间列索引无效数据,包括重复值、空值，并排序
        fdata = fdata[~fdata.index.duplicated(keep='first')]
        fdata = fdata[~fdata.index.isna()]
        fdata.sort_index(inplace=True)
    except Exception as e:
        raise BusinessValidationError(error_code=ErrorCode.DATA_ERROR, message=f'时间列数据格式错误: {str(e)}')
    
    #校验数据列是否存在非数值数据
    try:
        pd.to_numeric(fdata.iloc[:, dcol], errors='raise')
    except Exception as e:
        raise BusinessValidationError(error_code=ErrorCode.DATA_ERROR, message=f'数据列包含非数值数据: {str(e)}')
    
    
    #将数据列无效值替换为NaN
    invalid_count = 0
    if data_range is not None:
        (l,u) = data_range
        fdata.iloc[:, dcol] = fdata.iloc[:, dcol].where((fdata.iloc[:, dcol] >= l) & (fdata.iloc[:, dcol] < u))
        invalid_count = fdata.iloc[:, dcol].isna().sum().item()
    
    #检查时间是否连续，并推断频率
    continuous, freq_infer = is_continuous_auto(fdata.index.to_series())
    if compare_freq(freq_infer, '1min') < 0:
        raise BusinessValidationError(error_code=ErrorCode.DATA_ERROR, message=f'数据频率过高(无效时间列？): {freq_infer}, 最小支持1分钟频率')
    
    #如果数据不连续，进行重采样并插值
    if not continuous or invalid_count > 0:
        fdata = fdata.asfreq(freq_infer) #重采样为等频数据，缺失部分会自动补NaN       
        fdata[fdata.columns[dcol]] = fdata[fdata.columns[dcol]].interpolate(method='time') #时间序列插值
    
    fdata.index.freq = freq_infer #设置频率属性，方便后续分析使用

    datac = fdata[fdata.columns[dcol]]
    """
    数据摘要信息
    """
    if summary_config is not None and analysis_data:
        data_summary = NumericSummary(count = fdata.shape[0], 
            mean=datac.mean().item(), 
            std=datac.std().item(),
            mode= datac.mode().tolist(),
            invalid = invalid_count,
            five_number_summary=[datac.min().item(), 
                datac.quantile(0.25).item(),
                datac.median().item(),
                datac.quantile(0.75).item(),
                datac.max().item()])
            
        if summary_config.histogram is not None:
                """
                生成直方图数据
                """
                data_summary.histogram = histogram(datac, 
                    bins = summary_config.histogram.bins,
                    lower_edge= summary_config.histogram.left_edge,
                    upper_edge= summary_config.histogram.right_edge,
                    overflow=summary_config.histogram.overflow).values.tolist()
        
    # 原始数据频率、连续性推断
    if freq is None:
        pass

    if analysis_series:
        series_summary = SeriesSummary()
        series_summary.missed = fdata[fdata.columns[tcol]].isna().sum().item()
        series_summary.freq = freq_infer
        series_summary.time_range = (int(fdata.index[0].timestamp()), int(fdata.index[-1].timestamp()))
        series_summary.seasonal = find_periodicity(datac)

    return fdata, data_summary, series_summary