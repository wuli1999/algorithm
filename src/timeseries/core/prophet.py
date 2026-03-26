
import pandas as pd
from prophet import Prophet
from typing import Any,Tuple,Dict
from timeseries.core.base import TimeSeriesAlgorithmInfer,TimeSeriesAlgorithmTrain
from timeseries.schema.entities import TimeSeriesInferRequest,TimeSeriesTrainRequest
import json
import logging
from pydantic import BaseModel, Field

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").disabled = True

def infer(input:json, file=None)->json:    
    req = ProphetInferRequest.model_validate(input)
    req.data = file or req.data
    model = ProphetInfer(req)
    return model.infer()

def train(request:json, file=None)->json:
    req = TimeSeriesTrainRequest.model_validate(request)
    req.data = file or req.data
    model = ProphetTrain(req)   
    return model.train()


class ProphetConfig(BaseModel):
    """
    Prophet算法配置参数映射类
    """
    changepoint_prior_scale:float = Field(default=0.05, description="趋势变化点的先验尺度，控制趋势变化的灵活性")
    seasonality_mode:str = Field(default="additive", description="季节性模式，可以是'additive'或'multiplicative'")
    seasonality_prior_scale:float = Field(default=10.0, description="季节性先验尺度")
    holidays_prior_scale:float = Field(default=10.0, description="节假日先验尺度")
    changepoint_range:float = Field(default=0.8, description="趋势变化点范围，控制趋势变化点的分布")
    interval_width:float = Field(default=0.8, description="预测区间的宽度，表示不确定性的范围")


class ProphetInferRequest(TimeSeriesInferRequest):
    prophet_config:ProphetConfig

class ProphetInfer(TimeSeriesAlgorithmInfer):
    """Prophet推理类"""
    def __init__(self, request:ProphetInferRequest):
        super().__init__(request)
        self.prophet_config = request.prophet_config
        self.prophet_config.interval_width = self.confidence

    def _infer(self):
        model = Prophet(**self.prophet_config.model_dump())
        df = self.data[['y']].copy()
        df['ds'] = self.data.index

        model.fit(df)
        #通过model.seasonalities可以查看自动检测到的季节成分
        future = model.make_future_dataframe(periods=self.forecast_steps, freq=self.freq)
        if self.data_range is not None:
            future['floor'] = self.data_range[0]
            future['cap'] = self.data_range[1]

        fcast = model.predict(future)

        
        fcast_df = fcast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        fcast_df.rename(columns={'ds':'time', 'yhat':'y_hat', 'yhat_lower':'y_lower', 'yhat_upper':'y_upper'}, inplace=True)
        fcast_df.set_index('time', inplace=True)
        
        self.data.update(fcast_df)
        extra_rows = fcast_df.loc[~fcast_df.index.isin(self.data.index)]
        if not extra_rows.empty:
            self.data = pd.concat([self.data, extra_rows], ignore_index=False,axis=0).sort_index()

class ProphetTrain(TimeSeriesAlgorithmTrain):
    """Prophet训练类"""
    def __init__(self, request:TimeSeriesTrainRequest):
        super().__init__(request)
        #针对prophet不做滚动预测，测试集长度设为预测步长
        self.test_len = self.forecast_steps
        self.train_len = len(self.data) - self.test_len

    def _train(self):
        optimizer = TimeSeriesProphetOptimizer(
            data=self.data['y'],
            train_len=self.train_len,
            test_len=self.test_len,
            freq=self.freq
        )
        best_params = optimizer.optimize_hyper_params(n_trials=self.max_trial_count)
        best_params['interval_width'] = self.confidence

        result = optimizer.inference(data=self.data['y'], params=best_params)

        self.data[['y_hat','y_upper','y_lower']] = result[['y_hat','y_upper','y_lower']]
        info = {
                'model_params':best_params,
                'train_len':self.train_len,
                'test_len':self.test_len
        }
        self._add_model_summary('Prophet', info)
    
    def compute_forecast_metrics(self):
        if self.test_len <=0:
            return

        pred = self.data['y_hat'].iloc[self.train_len:]
        true = self.data['y'].iloc[self.train_len:]

        m = self.compute_metrics(y=true,y_hat=pred)
        self.forecast_metrics.append(m)

import optuna
class TimeSeriesProphetOptimizer:
    """Prophet参数优化器"""
    def __init__(self, data:pd.Series, train_len:int, test_len:int, freq:str):
        self.data = data
        self.train_len = train_len
        self.test_len = test_len
        self.freq = freq

    def objective(self, trial:optuna.Trial)->float:
        model, params = self.create_trial_model(trial)

        df = self.data.iloc[:self.train_len].to_frame(name='y').copy()
        df['ds'] = df.index

        model.fit(df)

        future = model.make_future_dataframe(periods=self.test_len, freq=self.freq)
        fcast = model.predict(future)

        fcast_df = fcast[['ds', 'yhat']].copy()
        fcast_df.rename(columns={'ds':'time', 'yhat':'y_hat'}, inplace=True)
        fcast_df.set_index('time', inplace=True)

        if self.test_len > 0:
            pred = fcast_df['y_hat'].iloc[-self.test_len:]
            true = self.data.iloc[self.train_len:self.train_len + self.test_len]
        else:
            pred = fcast_df['y_hat']
            true = self.data

        mae = (pred - true).abs().mean()
        return mae
    
    def inference(self, data:pd.Series, params:Dict[str, Any])->pd.DataFrame:
        model = self.create_model(params)

        df = data.to_frame(name='y').copy()
        df['ds'] = df.index

        model.fit(df)

        future = model.make_future_dataframe(periods=self.test_len, freq=self.freq)
        fcast = model.predict(future)

        fcast_df = fcast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        fcast_df.rename(columns={'ds':'time', 'yhat':'y_hat', 'yhat_lower':'y_lower', 'yhat_upper':'y_upper'}, inplace=True)
        fcast_df.set_index('time', inplace=True)

        return fcast_df

    def create_trial_model(self, trial:optuna.Trial)->Tuple[Prophet, Dict[str, Any]]:
        params:Dict[str, Any] = {
            'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative']),
            'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.001, 0.05),
            'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.01, 10.0),
            'holidays_prior_scale': trial.suggest_float('holidays_prior_scale', 0.01, 10.0),
            'changepoint_range': trial.suggest_float('changepoint_range', 0.7, 0.95),
            'growth': 'logistic',
        }
        return self.create_model(params), params
    
    def create_model(self,params:dict[str, Any])->Prophet:
        model = Prophet(
            seasonality_mode=params['seasonality_mode'],
            changepoint_prior_scale=params['changepoint_prior_scale'],
            seasonality_prior_scale=params['seasonality_prior_scale'],
            holidays_prior_scale=params['holidays_prior_scale'],
            changepoint_range=params['changepoint_range'],
            interval_width=params.get('interval_width',0.8)
        )
        return model

    def optimize_hyper_params(self, n_trials=20, n_jobs=-1, timeout=3600)->dict[str, Any]:
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)
        return study.best_params