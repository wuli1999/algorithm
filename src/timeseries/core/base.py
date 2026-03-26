from typing import List, Any, Tuple,Dict
import pandas as pd
import numpy as np

from pandas import DataFrame
from abc import ABC, abstractmethod

import sys
from timeseries.schema.entities import (TimeSeriesRequest,
                              EnvelopeConfig,
                              TimeSeriesResponse,
                              ModelMetrics,
                              TimeSeriesTrainRequest,
                              TimeSeriesInferRequest)
from common.business_validation import BusinessValidationError,ErrorCode
import logging

from timeseries.utils.preprocessing import data_prepare
from timeseries.utils.tools import validate_freq
logger =logging.getLogger(__name__)

from common.filemgr import write_df_to_file as save_dataframe_to_csv

class TimeSeriesAlgorithmBase(ABC):
    """
    时序算法基类,负责数据加载、插值等操作
    """
    def __init__(self, request:TimeSeriesRequest)->None:
        """
        成员变量：
        fdata: 经过预处理，包含输入数据的所有列以及时间索引、插值后的数据
        data: 处理后的数据，包含时间索引和'y'列

        data_summary: 数值摘要
        series_summary: 时序摘要
        data_range: 数据范围
        freq: 数据频率
        data_config: 数据配置
        envelope: 包络线配置
        """
        (self.fdata, self.data_summary, self.series_summary) = data_prepare(request)

        """将时序数据转换为统一格式，列名为['y']，索引名为'time'"""
        self.data = self.fdata.iloc[:, [request.data_config.data_col]].copy()
        self.data.columns = ['y']
        self.data.index.name = 'time'

        ### 数据范围
        self.data_range:Tuple[float, float] = request.data_config.data_range 

        ###    
        self.data_config = request.data_config

        ### 频率
        self.freq:str = self.data.index.freq

        ###包络线
        self.envelope:EnvelopeConfig = request.envelope

        ###模型摘要和评估指标
        self.model_summary:Dict = {}

        ###预测评估指标
        self.inference_metrics:Dict = None

        self.data[['y_hat', 'y_lower', 'y_upper']] = np.nan

    def _add_model_summary(self, key:str, value:Any)->None:
        self.model_summary[key] = value 
    
    def merge_result_data(self)->DataFrame:
        """
        合并原始数据和预测数据，形成统一结果：增加预测值'y_hat'以及包络线'y_lower'和'y_upper'列
        """
        column_name = self.fdata.columns[self.data_config.data_col]
        #避免列名冲突，对data的列重新命名
        result = self.data[['y_hat','y_lower','y_upper']]
        result = result.rename(columns={'y_hat': f'{column_name}_hat', 
                                        'y_lower': f'{column_name}_lower', 
                                        'y_upper': f'{column_name}_upper'})

        result = pd.concat([self.fdata, result], axis =1)
        #将原始数据时间列用索引替代
        result[result.columns[self.data_config.time_col]] = result.index.strftime('%Y-%m-%d %H:%M:%S')
        return result

    
    def _envelope(self)->None:
        """
        计算包络线
        """
        # 滑动窗口,置信范围
        method = self.envelope.method
        if method == 'native':
            return

        w = self.envelope.windows
        ci=self.envelope.confidence

        #计算残差
        resid = self.data['y'] - self.data['y_hat']
        resid_sigma = resid.std(ddof=1)

        if method == 'sigma':
            from scipy import stats
            k = stats.norm.ppf(ci)
            if w > 0 and w < len(self.data['y_hat']):
                resid_sigma = resid.rolling(w, min_periods=w).std()
                nav = resid_sigma.isna()
                resid_sigma[nav] = resid[nav].std()
            else:
                resid_sigma = resid.std()

            upper = self.data['y_hat'] + k * resid_sigma
            lower = self.data['y_hat'] - k * resid_sigma

            self.data['y_upper'] = upper
            self.data['y_lower'] = lower
        elif method == 'iqr':
            pass
        elif method == 'quantile':
            pass 


    def response(self)->TimeSeriesResponse:
        ###合并结果数据，包含原始数据、预测数据以及包络线数据
        result = self.merge_result_data()

        ###如果配置了保存文件，则将结果数据保存为csv文件，并返回文件名
        filename = None
        if self.data_config.save_to_file:
            filename = save_dataframe_to_csv(result)

        data = result.values.tolist()        
        if self.data_config.header is True:
            data = [result.columns.tolist()] + data
            
       
        return TimeSeriesResponse(data=data, 
                        filename=filename,
                        numeric_summary=self.data_summary,
                        series_summary=self.series_summary,
                        inference_metrics=self.inference_metrics if hasattr(self, 'inference_metrics') else None,
                        forecast_metrics=self.forecast_metrics if hasattr(self, 'forecast_metrics') else None,
                        model_summary=self.model_summary)      

    
    def extend_forecast_result(self, forecast:DataFrame)->None:
        """
        扩展预测步数,将预测结果追加到self.data中
        预测结果forecast应包含'y_hat'列
        """
        
        if len(forecast) <=0:
            return
        
        last_time = self.data.index[-1]
        dindex=pd.date_range(last_time, freq = self.freq, periods=len(forecast)+1)[1:]        

        forecast.index = pd.Index(dindex, name='time')
        self.data = pd.concat([self.data, forecast], ignore_index=False,axis=0)

    def compute_metrics(self, y, y_hat)->ModelMetrics:
        """
        计算预测指标
        参数：
            y: pd.Series
                真实值
            y_hat: pd.Series
                预测值
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from statsmodels.stats.diagnostic import acorr_ljungbox
        import scipy.stats as stats

        mask = y.notna() & y_hat.notna()
        y = y[mask]
        y_hat = y_hat[mask]

        lags = 1 if len(y) <= 4 else min((len(y)//2-2), 40)
        lb_test_result = acorr_ljungbox(y - y_hat, lags=lags, return_df=True)
        stat,p_value = stats.shapiro(y - y_hat)
 
        return ModelMetrics(
            SAMP_STD = np.std(y).item(),
            SAMP_MEAN= np.mean(y).item(),
            MAE=mean_absolute_error(y, y_hat),
            MSE=mean_squared_error(y, y_hat),
            RMSE=np.sqrt(mean_squared_error(y, y_hat)).item(),
            MAPE=np.mean(np.abs((y - y_hat)/y)).item(),
            RESID_STD=np.std(y - y_hat).item(),
            RESID_MEAN=np.mean(y - y_hat).item(),
            LB_PVALUE=lb_test_result['lb_pvalue'].values[0].item(),
            SW_PVALUE=p_value.item(),
        ) 
        
    
    def eval_model_metrics(self, len:int)->None:
        """
        评估模型指标
        """
        historical_data = self.data[:len].dropna(subset=['y_hat', 'y'])
        self.inference_metrics = self.compute_metrics(historical_data['y'], historical_data['y_hat'])

    @property
    def confidence(self):
        return self.envelope.confidence

class TimeSeriesAlgorithmTrain(TimeSeriesAlgorithmBase):
    """
    时序数据训练基类
    """
    def __init__(self, request:TimeSeriesTrainRequest):
        super().__init__(request)

        self.train_config = request.train_config
        if self.train_config.total_trial_timeout <=0:
            self.train_config.total_trial_timeout = sys.maxsize
        
        if self.train_config.per_trial_timeout <= 0:
            self.train_config.per_trial_timeout = sys.maxsize

        self.train_len:int = int(self.data.shape[0] * self.train_ratio)
        self.test_len:int = self.data.shape[0] - self.train_len  

        if self.test_len < self.forecast_steps:
            raise BusinessValidationError(message = f"训练数据集长度不足，无法进行{self.forecast_steps}步预测，当前测试集长度为{self.test_len}.",error_code=ErrorCode.INVALID_PARAMETER)

        self.forecast_metrics:List[ModelMetrics] = []
        self.forecast_results:np.ndarray = None
        self.forecast_detail:bool = request.train_config.forecast_detail
    
    @abstractmethod
    def _train():
        pass
    
    def set_forecast_results(self, fcasts:np.ndarray)->None:
        """
        设置多步预测结果
        参数：
            fcasts: np.ndarray
                包含多步预测结果的ndarray,每行对应一个时间点的多步预测结果（y_hat）
        """
        self.forecast_results = fcasts

    def compute_forecast_metrics(self):
        """
        评估多步预测效果
        """
        try:
            if self.forecast_results is not None and len(self.forecast_results) > 1:
                # 滚动预测的轮次总和
                groups = len(self.forecast_results)
                for i in range(self.forecast_steps):
                    y_start = self.train_len + i
                    y_end = y_start + groups

                    y_true = self.data['y'].iloc[y_start:y_end].dropna().values
                    y_hat = self.forecast_results[:,i:i+1].reshape(-1)

                    y_true = pd.Series(y_true)
                    y_hat = pd.Series(y_hat)

                    metrics_by_step = self.compute_metrics(y=y_true,y_hat=y_hat)
                    self.forecast_metrics.append(metrics_by_step)                   

        except Exception as e:
            logger.error(f"Error computing forecast metrics: {e}")

    def eval_model_metrics(self, len:int):
        super().eval_model_metrics(len=self.train_len)
        self.compute_forecast_metrics() 

    def response(self):
        resp = super().response()
        if self.forecast_results is not None and len(self.forecast_results) > 0:
            resp.forecast_metrics = self.forecast_metrics
            if self.forecast_detail:
                resp.forecast_results = self.forecast_results.tolist()

        
        return resp
    
    @property
    def forecast_steps(self):
        return self.train_config.forecast_steps
    
    @property
    def train_ratio(self):
        return self.train_config.train_ratio
    
    @property
    def max_trial_count(self):
        return self.train_config.max_trial_count
    
    @property
    def total_trial_timeout(self):
        return self.train_config.total_trial_timeout
    
    @property
    def per_trial_timeout(self):
        return self.train_config.per_trial_timeout

    def train(self):
        self._train()
        self._envelope()
        self.eval_model_metrics(len=self.train_len)
        return self.response()

class TimeSeriesAlgorithmInfer(TimeSeriesAlgorithmBase):
    """
    时序数据预测/拟合基类
    """
    def __init__(self, request:TimeSeriesInferRequest):
        super().__init__(request)
        self.forecast_steps = request.infer_config.forecast.steps

    @abstractmethod
    def _infer():
        pass

    def infer(self):
        self._infer()
        self._envelope()
        self.eval_model_metrics(len=len(self.data))
        return self.response()