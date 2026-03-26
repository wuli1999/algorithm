import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Input,Dropout,Bidirectional
from tensorflow.keras import optimizers,regularizers
from tensorflow.keras.callbacks import EarlyStopping,Callback
from numpy.lib.stride_tricks import sliding_window_view
import optuna
from optuna.integration import TFKerasPruningCallback

from typing import Any, Tuple, Dict,List
from timeseries.core.base import TimeSeriesAlgorithmInfer,TimeSeriesAlgorithmTrain
from timeseries.schema.lstm_entities import LSTMInferRequest, LSTMTrainRequest
from common.business_validation import BusinessValidationError,ErrorCode
import json
import pickle
import gzip
import base64

import logging
logger =logging.getLogger(__name__)

def infer(input:json, file=None)->json:    
    req = LSTMInferRequest.model_validate(input)
    req.data = file or req.data
    model = LSTMInfer(req)
    return model.infer()

def train(request:json, file=None)->json:
    req = LSTMTrainRequest.model_validate(request)
    req.data = file or req.data
    model = LSTMTrain(req)   
    return model.train()

class LSTMInfer(TimeSeriesAlgorithmInfer):
    """LSTM推理类"""
    def __init__(self, request:LSTMInferRequest):
        super().__init__(request)
        self.lstm_config = request.lstm_config
        try:
            #recovered_binary:bytes = bytes.fromhex(self.lstm_config['model_dumps'])
            recovered_binary:bytes = base64.b85decode(self.lstm_config['model_dumps'])
            self.model_info = pickle.loads(gzip.decompress(recovered_binary))       
        except Exception as e:
            logger.error(f"Failed to recover model info from dumps: {e}")
            raise BusinessValidationError(message=f"Failed to recover model info from dumps: {e}", error_code=ErrorCode.INVALID_PARAMETER)

    def _infer(self)->Any:
        model:Sequential = Sequential.from_config(self.model_info['model_config'])
        model.set_weights(self.model_info['model_weights'])
        
        lookback = model.input_shape[1]
        if self.forecast_steps != model.output_shape[-1] - lookback:
            msg = f'request forecast steps:{self.forecast_steps} not equal model output:{model.output_shape[-1] - lookback}'
            raise BusinessValidationError(message=msg, error_code=ErrorCode.INVALID_PARAMETER)
        
        values = self.data['y'].values.reshape(-1, 1)
        if len(values) < lookback:
            raise BusinessValidationError(message=f'data length:{len(values)} less than model input size:{lookback}', error_code=ErrorCode.INVALID_PARAMETER)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(values)

        samples = len(values) // lookback
        start_pos = len(values) % lookback
        inputs = scaled_data[start_pos:].reshape(samples, lookback, -1)
        if start_pos > 0:
            head = scaled_data[:lookback].reshape(1, lookback, -1)
            inputs = np.concatenate((head, inputs))

        # 预测全数据集
        predictions = model.predict(inputs, verbose=0)
        # 反归一化预测结果  
        predictions = scaler.inverse_transform(predictions)

        history = predictions[:, :lookback].reshape(-1)
        if start_pos > 0:
            # 处理头部数据
            history = np.concatenate((history[:start_pos], history[lookback:]))

        
        self.data['y_hat'] = history.tolist()

        if self.forecast_steps > 0:
            self.extend_forecast_result(forecast=pd.DataFrame(
                predictions[-1:, -self.forecast_steps:].reshape(-1), columns=['y_hat']))
               
        

class LSTMTrain(TimeSeriesAlgorithmTrain):
    def __init__(self, request: LSTMTrainRequest):
        super().__init__(request)
        self.envelope.method='sigma'
        self.lookback = request.lstm_config.lookback

    def _train(self)->None:
        """
        获取最佳参数
        """
        values = self.data['y'].values.reshape(-1, 1)  #仅用'y'列进行训练，转换为列向量
        optimizer = TimeSeriesLstmOptmizer(data = values,
                                           train_len=self.train_len, 
                                           forecast_steps=self.forecast_steps, 
                                           lookback=self.lookback)
        model_info =  optimizer.optimize_hyper_params(n_trials=self.max_trial_count, 
                                                      n_jobs=2, 
                                                      timeout=self.total_trial_timeout,
                                                      per_timeout=self.per_trial_timeout)
        
        '''利用model_info直接恢复模型及权重'''
        #optimizer.train_best_model()

        history, forecasts = optimizer.inference(data = values, params=model_info)
        y_hat = history.reshape(-1)

        if len(forecasts) > 0:
            self.set_forecast_results(forecasts)
            if self.test_len > self.forecast_steps:
                y_hat = np.concatenate((y_hat, forecasts[:self.test_len-self.forecast_steps,:1].reshape(-1)))
            y_hat = np.concatenate((y_hat, forecasts[-1].reshape(-1)))

        self.data['y_hat'] = y_hat.tolist()
        
        # 序列化模型信息        
        dumps = base64.b85encode(gzip.compress(pickle.dumps(model_info,protocol=pickle.HIGHEST_PROTOCOL))).decode('utf-8')
        info = {
            'model_params':{'params':model_info['params'],'dumps':dumps},
            'train_len':self.train_len,
            'test_len':self.test_len
        }
        self._add_model_summary('LSTM', info)

class TimeSeriesLstmOptmizer:
    def __init__(self, data:np.ndarray, train_len:int, forecast_steps:int, lookback:range):
        self.origin_data:np.ndarray = data
        self.forecast_steps:int = forecast_steps
        self.lookback_range:range = lookback
        self.n_features = data.shape[-1]

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled_data = self.scaler.fit_transform(data)
        self.train_len = train_len

        '''自己保存最佳模型'''
        self.best_model = {
            'trial':-1,
            'score' : np.inf,
            'params':{},
            'model_config':{},
            'model_weights':{}
        }

        self.early_stopping_callback = EarlyStopping(monitor='loss',
                                                patience=5,
                                                restore_best_weights=True, 
                                                mode='min',
                                                verbose=0)


    def prepare_train_data(self, lookback:int, forecast:int)->\
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
       # 创建序列数据
       windows = sliding_window_view(self.scaled_data, window_shape=(lookback + forecast,), axis=0)
        
       #重塑输入数据为 [样本数, 时间步长, 特征数]，适应LSTM输入要求
       windows = windows.reshape(len(windows), lookback + forecast, -1)
       X = windows[:, :-forecast]  if forecast > 0 else windows #输入序列
       #Y = windows[:,-forecast:]  #目标值
       Y=windows

       #划分训练集和验证集
       X_train,X_test = X[:self.train_len],X[self.train_len:]
       Y_train,Y_test = Y[:self.train_len],Y[self.train_len:]
       return X_train, X_test, Y_train, Y_test
    
    def create_trial_model(self, trial)->Tuple[Sequential, Dict[str, Any]]:
        """根据 trial 建立 Keras 模型"""
        params:Dict[str, Any] = {}
        params['lookback'] = trial.suggest_int("lookback", self.lookback_range.start, self.lookback_range.stop, step=self.lookback_range.step)
        # LSTM 层数
        params['n_layers'] = trial.suggest_int("n_layers", 1, 2)
        # LSTM 单元数
        params['lstm_units'] = trial.suggest_int("lstm_units", 32, 128)
        # Dropout 比例
        params['dropout_rate'] = trial.suggest_float("dropout_rate", 0.1, 0.5)
        # 全连接层维度
        params['dense_units'] = trial.suggest_int("dense_units", 16, 64)
        # 训练参数
        params['learning_rate'] = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True) # 学习率
        # 正则化强度超参数
        params['weight_decay'] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        # 是否使用双向 LSTM,让模型同时利用过去和未来的上下文信息，对捕捉周期性、趋势性有帮助
        params['use_bidirectional'] = trial.suggest_categorical("use_bidirectional", [False, True])
        
        params['batch_size']= trial.suggest_categorical("batch_size", [32, 64, 128])
        params['epochs'] = trial.suggest_int("epochs", 10, 100) 

        model = self.create_model(params=params)
        return model, params

    def create_model(self, params:Dict[str,Any])->Sequential:
        model = Sequential()
        model.add(Input(shape=(params['lookback'], self.n_features)))
        # LSTM 层数
        n_layers = params["n_layers"]
        # LSTM 单元数
        lstm_units = params["lstm_units"]
        # Dropout 比例
        dropout_rate = params["dropout_rate"]
        # 全连接层
        dense_units = params["dense_units"]
        # 训练参数
        learning_rate = params["learning_rate"] # 学习率
        # 正则化强度超参数
        weight_decay = params["weight_decay"]
        use_bidirectional = params["use_bidirectional"]

        for i in range(n_layers):
            return_sequences=(i < n_layers - 1)
            if use_bidirectional:
                layer = Bidirectional(LSTM(lstm_units,
                    return_sequences=return_sequences,
                    name=f"lstm_{i}",
                    kernel_regularizer=regularizers.l2(weight_decay)))
            else:
                layer = LSTM(lstm_units,
                    return_sequences=return_sequences,
                    name=f"lstm_{i}",
                    kernel_regularizer=regularizers.l2(weight_decay))
            model.add(layer)
            model.add(Dropout(dropout_rate, name=f"dropout_{i}"))

        # 添加全连接层
        model.add(Dense(dense_units, activation="relu", name="dense_1",kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Dropout(dropout_rate, name="dropout_dense"))

        # 输出层
        #model.add(Dense(self.forecast_steps, name="output"))
        model.add(Dense(self.forecast_steps + params['lookback'], name="output"))
        model.compile(
            loss="mse",
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            metrics=['mae', 'mse']
        )
        return model
    
    def objective(self, trial, timeout)->float:
        """Optuna 目标函数"""        
        model, params = self.create_trial_model(trial)                
        #决定模型“看多长”的历史数据来做预测
        lookback = model.input_shape[1]

        # 创建剪枝回调
        pruning_callback = OptunaPruningCallback(trial, monitor='loss', timeout=timeout)
        x_train,x_test,y_train,y_test = self.prepare_train_data(lookback=lookback, forecast=self.forecast_steps)
        """
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
            .batch(params['batch_size'])\
            .repeat()
                
        valid_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
            .batch(params['batch_size'])\
            .repeat()
        """
        
        try:
            history = model.fit(
                x_train, y_train,
                validation_data=(x_test, y_test),
                epochs=params['epochs'], 
                steps_per_epoch=len(x_train) // params['batch_size'],
                validation_steps=len(x_test) // params['batch_size'],               
                callbacks=[pruning_callback, self.early_stopping_callback],
                verbose=0
            )
            # 返回验证集上的最佳指标值
            val_loss = min(history.history["loss"])

            if val_loss < self.best_model['score']:
                self.best_model['score'] = val_loss
                self.best_model['trial'] = trial._trial_id
                self.best_model['params'] = params
                self.best_model['model_config'] = model.get_config()
                self.best_model['model_weights'] = model.get_weights()

            return val_loss        
        except optuna.TrialPruned:
            # 试验被剪枝
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return float('inf')
    
    
    def optimize_hyper_params(self, n_trials=20, n_jobs=1, timeout=sys.maxsize, per_timeout=sys.maxsize)->Dict[str, Any]:
        """运行超参数优化"""
        logger.info(f"开始Optuna超参数优化,试验次数: {n_trials}")
        
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,     # 前5个trial不进行剪枝
            n_warmup_steps=5,       # 每个trial训练5个epoch后再开始考虑剪枝
            interval_steps=1        # 每隔1个epoch检查一次
        )

        # 创建study
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42),
            study_name='timeseries_lstm_opt',
            pruner=pruner
        )

        # 设置日志回调
        def log_callback(study, trial):
            if trial.state == optuna.trial.TrialState.COMPLETE:
                logger.info(f"Trial {trial.number} finished with value: {trial.value:.4f}")
                logger.info(f'started at:{trial.datetime_start},finished at:{trial.datetime_complete}, duration:{trial.duration}')
        
        # 运行优化
        study.optimize(
            lambda trial:self.objective(trial,per_timeout),
            n_trials=n_trials,
            n_jobs=n_jobs,
            timeout=timeout,
            callbacks=[log_callback]
        )

        if self.best_model['score'] == np.inf:
            raise BusinessValidationError('训练失败,请检查数据质量并合理设置超时、实验次数等参数值.', error_code=ErrorCode.PROCESS_FAILURE)

        return self.best_model
    
    def train_best_model(self):
        """
        """        
        params = self.best_params
        model = self.create_model(params=params)
        #决定模型“看多长”的历史数据来做预测
        lookback = model.input_shape[1]
        x_train,x_test,y_train,y_test = self.prepare_train_data(lookback=lookback, forecast=self.forecast_steps)

        history = model.fit(
            x_train, y_train,

            validation_data=(x_test, y_test),
            batch_size=params['batch_size'],
            epochs=params['epochs'],
            callbacks=[self.early_stopping_callback],
            verbose=0
        )
        return model.get_config(),model.get_weights()

    def eval_forecast_metrics(self, model:Sequential, scaled_data:np.ndarray)->np.ndarray:
        """
        通过在测试集上滚动预测，在最佳模型上评估预测指标
         1. 通过滚动窗口方式，逐步预测测试集
         2. 计算并输出预测指标
         """
        lookback = model.input_shape[1]
        X_test_windows = sliding_window_view(scaled_data[self.train_len - lookback:], window_shape=(lookback,), axis=0)
        X_test_windows = X_test_windows.reshape(len(X_test_windows), lookback, -1)
        X_test_tensor = tf.convert_to_tensor(X_test_windows)
        preds = model(X_test_tensor, training=False)
        forecasts = preds.numpy()[:, -self.forecast_steps:]
        return forecasts

    def inference(self, data, params)->Tuple[np.ndarray, np.ndarray]:

        def infer(model,dataset:np.ndarray,lookback, forecast_steps)->Tuple[np.ndarray, np.ndarray]:
            result = model(tf.convert_to_tensor(dataset), training=False).numpy()
            history = result[:, :lookback]
            forecasts = result[:, -forecast_steps:] if forecast_steps >0 else np.array([])
            return history, forecasts

        """
        利用最佳模型进行推理
        1. 对历史数据进行预测，得到拟合值 y_hat
        2. 对未来数据进行预测，得到预测值 forecast  
        3. 返回 y_hat 和 forecast
        4. params:最佳模型参数 
        返回值:
            y_hat:np.ndarray, 历史数据拟合值
            forecast:np.ndarray, 滚动预测值
        """
        try:
            model:Sequential = Sequential.from_config(params['model_config'])
            model.set_weights(params['model_weights'])        
            lookback = model.input_shape[1]
        except Exception as e:
            raise BusinessValidationError(message=f'模型创建失败{e}', error_code=ErrorCode.INVALID_PARAMETER)

        if len(data) < lookback:
            raise BusinessValidationError(message=f'data length:{len(data)} less than model input size:{lookback}', error_code=ErrorCode.INVALID_PARAMETER) 

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)


        """
        历史数据拟合值
         1. 构建输入序列
         2. 预测历史数据
        """
        history_data_len = len(data) if self.forecast_steps ==0 else self.train_len
        history_groups = history_data_len // lookback
        history_dataset = scaled_data[:history_groups * lookback].reshape(history_groups, lookback, -1)

        if history_data_len > history_groups * lookback:
            # 处理尾部数据
            tail = scaled_data[history_data_len - lookback:history_data_len].reshape(1, lookback, -1)
            history_dataset = np.concatenate((history_dataset, tail))
        
        history, _ = infer(model, history_dataset, lookback, self.forecast_steps)
        history = history.reshape(-1,1)
        history = scaler.inverse_transform(history)
        if history_data_len > history_groups * lookback:
            # 处理尾部数据
            history = np.concatenate((history[:history_groups * lookback], history[-(history_data_len - history_groups * lookback):]))

        if self.forecast_steps==0:
            return history, np.array([])

        """
        未来数据滚动预测
         1. 构建输入序列
         2. 预测未来数据
        """
        test_len = len(data) - self.train_len + lookback
        test_data = scaled_data[-test_len:]
        test_windows = sliding_window_view(test_data, window_shape=(lookback + self.forecast_steps,), axis=0)[:, :,:lookback]
        test_windows = test_windows.reshape(len(test_windows), lookback, -1)
        
        _,forecasts = infer(model, test_windows, lookback, self.forecast_steps)
        forecasts = scaler.inverse_transform(forecasts)
        return history, forecasts
    
import time
class OptunaPruningCallback(Callback):
    """Optuna剪枝回调"""
    def __init__(self, trial, monitor='loss', timeout = sys.maxsize):
        super().__init__()
        self.trial = trial
        self.monitor = monitor
        self.best_score = None
        self.timeout = timeout
        self.start_time = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_score = logs.get(self.monitor)
        
        if current_score is None:
            return
        
        # 报告当前分数给Optuna
        self.trial.report(current_score, epoch)
        
        # 检查是否需要剪枝
        if self.trial.should_prune():
            logger.info(f"Trial {self.trial.number} pruned at epoch {epoch}")
            raise optuna.TrialPruned()
        
        # 更新最佳分数
        if self.best_score is None or current_score < self.best_score:
            self.best_score = current_score
        
        if self.trial._cached_frozen_trial.state == optuna.trial.TrialState.RUNNING:
            if time.time() - self.start_time > self.timeout:
                raise TimeoutError(f'trial:{self.trial._trial_id} timeout')