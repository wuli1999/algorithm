import numpy as np
import pandas as pd
import json

import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA

from typing import Tuple
from common.business_validation import BusinessValidationError,ErrorCode


from timeseries.schema.entities import TimeSeriesTrainRequest,ARIMARequest
from timeseries.core.base import TimeSeriesAlgorithmInfer, TimeSeriesAlgorithmTrain

def infer(input:json, file=None)->json:    
    req = ARIMARequest.model_validate(input)
    req.data = file or req.data
    model = ARIMAInfer(req)
    return model.infer()

def train(request:json, file = None)->json:
    req = TimeSeriesTrainRequest.model_validate(request)
    req.data = file or req.data
    model = ARIMATrain(req)   
    return model.train()

'''算法实现'''
class ARIMAInfer(TimeSeriesAlgorithmInfer):
    
    def __init__(self, request:ARIMARequest)->None:
        super().__init__(request)
        if len(self.data) > 1000 or len(self.data) < 30:
            raise BusinessValidationError(message=f'ARIMA模型数据规模限制:30-1000,当前数据长度:{len(self.data)}', error_code=ErrorCode.DATA_ERROR)

        self.order = request.arima_config.order

    def _infer(self):
        """
        ARIMA拟合/预测
        返回: (拟合Series,预测DataFrame,算法拟合/预测信息)
        """ 
        """
        season = self.fit_config.season
        if season is not None:
            '''SARIMA'''
            if season.period >= self.data.shape[0]:
                raise ValidationError(f'data length:{self.data.shape[0]} < period steps:{season.period}')

            if self.order is None or self.season_order is None:
                '''只要其中一个是空就自动获取'''
                am = pm.auto_arima(self.data['y'].iloc[:400], seasonal=True,           # 启用季节性
                        m=season.period,                                    # 季节周期
                        start_p=0,                                          # p的起始值
                        max_p=3,                                            # p的最大值
                        start_q=0,                                          # q的起始值
                        max_q=3,                                            # q的最大值
                        d=1,                                             # 自动确定差分阶数
                        D=1,                                             # 自动确定季节性差分阶数
                        start_P=0,                                          # P的起始值
                        max_P=1,                                            # P的最大值
                        start_Q=0,                                          # Q的起始值
                        max_Q=1,                                            # Q的最大值
                        max_D = 1,
                        max_order=5,                                        # (p+q+P+Q)的最大值
                        trace=True,                                         # 打印搜索过程
                        error_action='ignore',                              # 忽略无效参数组合
                        suppress_warnings=True,                             # 抑制警告
                        #stepwise=True,                                      # 使用逐步搜索（更快）
                        n_fits=50,                                          # 最多尝试50个模型
                        method='lbfgs',
                        n_jobs=16)                                         
                self.order = am.order
                self.season_order = am.season_order
            return self.SARIMA_fit_forecast()
        else:
        """
        '''ARIMA'''
        if self.order is None:
            am = pm.auto_arima(self.data['y'], seasonal=False, 
                                start_p=0, 
                                start_q=0, 
                                max_p=5, 
                                max_q=5, 
                                stepwise=True, 
                                n_fits=50, 
                                error_action='ignore', 
                                trace=False,
                                max_order=10)
            self.order = am.order                
        self.ARIMA_infer()
             
    def SARIMA_infer(self):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        model = SARIMAX(self.data['y'], 
                        order = self.order, 
                        seasonal_order=self.season_order)
        res = model.fit(disp=False)
        # 预测
        fcast_df = None
        if self.forecast_steps > 0:
            fcast = res.get_forecast(steps = self.forecast_steps)
            fcast_df = fcast.conf_int(alpha =  1 - self.confidence)
            fcast_df['y_hat'] = fcast.predicted_mean
            fcast_df.rename(columns={'upper y':'y_upper', 'lower y':'y_lower'}, inplace=True)
        
        return res.fittedvalues, fcast_df, {'order':self.order, 'season_order':self.season_order, 'name':'SARIMA'}
    
    def ARIMA_infer(self):
        # 拟合ARIMA并获取残差
        model = ARIMA(self.data['y'], order=self.order, freq=self.data.index.freq)
        model_fit = model.fit()
        p = model_fit.get_prediction()

        ci = p.conf_int(alpha = 1 - self.confidence)
        self.data['y_hat'] = p.predicted_mean
        self.data['y_lower'] = ci['lower y']
        self.data['y_upper'] = ci['upper y']

        self.data.iloc[0, self.data.columns.get_loc('y_hat')] = np.nan
        self.data.iloc[0, self.data.columns.get_loc('y_lower')] = np.nan
        self.data.iloc[0, self.data.columns.get_loc('y_upper')] = np.nan  

        # 预测
        if self.forecast_steps > 0:
            fcast = model_fit.get_forecast(steps = self.forecast_steps)
            fcast_df = fcast.conf_int(alpha =  1 - self.confidence)
            fcast_df['y_hat'] = fcast.predicted_mean
            fcast_df.rename(columns={'upper y':'y_upper', 'lower y':'y_lower'}, inplace=True)
            self.extend_forecast_result(fcast_df)
        
class ARIMATrain(TimeSeriesAlgorithmTrain):
    """ARIMA训练类"""
    def __init__(self, request:TimeSeriesTrainRequest):
        super().__init__(request)

        if len(self.data) > 1000 or len(self.data) < 30:
            raise BusinessValidationError(message=f'ARIMA模型数据规模限制:30-1000,当前数据长度:{len(self.data)}', error_code=ErrorCode.DATA_ERROR)

        self.order:Tuple[int, int, int] = None
        self.update_frequency:int = 1  # 每步都更新
        if self.forecast_steps == 0:
            self.train_len = len(self.data)
            self.test_len = 0

    def rolling_forecast(self)->np.ndarray:
        """
        每次预测未来多步，按指定频率更新
        返回: 
            np.ndarray: 形状为(预测轮数, 预测步数, 3(y_hat,y_lower,y_upper))的数组            
        """
        train_data = self.data['y'].iloc[:self.train_len]
        test_data = self.data['y'].iloc[self.train_len:]

        history = list(train_data)
        predictions = []
        forecast_df = None
        
        i = 0
        while i < len(test_data):
            # 训练模型
            model = ARIMA(history, order=self.order)
            try:
                model_fit = model.fit()
            except Exception as e:
                raise BusinessValidationError(message=f'训练失败，检查数据可用性：{str(e)}', error_code=ErrorCode.DATA_ERROR)
            
            # 计算本次需要预测的步数
            #steps_ahead = min(self.forecast_steps, len(test_data) - i)
            steps_ahead = self.forecast_steps
            
            # 预测未来多步
            forecast = model_fit.get_forecast(steps=steps_ahead)
            conf_int = forecast.conf_int(alpha =  1 - self.confidence)
            
            # 将预测结果加入结果列表
            f = np.array(conf_int)
            f = np.column_stack((np.array(forecast.predicted_mean).reshape(-1, 1), f))
            predictions.append(f)
            
            # 更新历史数据
            if self.update_frequency == 1:
                # 每步都更新：使用真实值
                history.append(test_data.iloc[i])
                i += 1
            else:
                # 每N步更新一次
                if i + self.update_frequency <= len(test_data):
                    history.extend(test_data.iloc[i:i+self.update_frequency].tolist())
                i += self.update_frequency
            
            # 保持历史数据长度合理
            history = history[self.update_frequency:]

        return np.stack(predictions,axis=0)

    def _train(self):    

        am = pm.auto_arima(self.data['y'].iloc[:self.train_len], seasonal=False, 
                            start_p = 0, 
                            start_q=0, 
                            max_p=5, 
                            max_q=5, 
                            stepwise=True, 
                            n_fits=50, 
                            error_action='ignore', 
                            trace=False,
                            max_order=10)
        
        self.order = am.order
        
        # 拟合ARIMA(测试集)并获取残差
        model = ARIMA(self.data['y'].iloc[:self.train_len], order=self.order)
        model_fit = model.fit()

        p = model_fit.get_prediction()        
        ci = p.conf_int(alpha = 1 - self.confidence)

        result_data = pd.DataFrame([p.predicted_mean, ci['lower y'], ci['upper y']]).T
        result_data.columns = ['y_hat','y_lower','y_upper']
        result_data.iloc[0, result_data.columns.get_loc('y_hat')] = np.nan
        result_data.iloc[0, result_data.columns.get_loc('y_lower')] = np.nan
        result_data.iloc[0, result_data.columns.get_loc('y_upper')] = np.nan  

        
        if self.forecast_steps > 0:
            """预测未来多步,评估预测效果"""
            forecast = self.rolling_forecast()
            
            #取预测数据第一步预测的y,y_lower,y_upper
            forecast_1st_steps = pd.DataFrame(forecast[:self.test_len - self.forecast_steps,0,:],
                                               columns=['y_hat','y_lower','y_upper'])

            #最后一个完整步数的预测结果
            last_fullstep_forecast = pd.DataFrame(forecast[self.test_len - self.forecast_steps],
                                                  columns=['y_hat','y_lower','y_upper'])
            
            # 合并历史数据和预测数据
            result_data = pd.concat([result_data,forecast_1st_steps, last_fullstep_forecast], ignore_index=False,axis=0)
            self.set_forecast_results(fcasts=forecast[:,:,0])
        
        self.data.loc[self.data.index, ['y_hat','y_lower','y_upper']] = result_data.values 
        self._add_model_summary('ARIMA',{
            'model_params':{'order':self.order},
            'train_len':self.train_len, 
            'test_len':self.test_len})