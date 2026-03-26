import optuna
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from common.business_validation import BusinessValidationError, ErrorCode
from correlation.schema.entities import Metrics, Params, RegressionRequest, RegressionResponse,FRegressionRequest, SpearmanCeoff
from sklearn.preprocessing import PolynomialFeatures
from common.envelope import compute_envelope
from common.filemgr import write_df_to_file
import logging
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

# 一元线性回归模型
def linear_regression_model(x_train, y_train, trial=None):
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model

# 一元线性分段回归模型（最多3段）
def piecewise_linear_regression_model(x_train, y_train, trial):
    # 选择分段数量（1-3段）
    n_segments = trial.suggest_int('n_segments', 2, 3)
    
    # 选择分割点（基于训练数据范围）
    x_min, x_max = x_train.min(), x_train.max()
    breakpoints = []
    for i in range(n_segments - 1):
        breakpoint = trial.suggest_float(f'breakpoint_{i}', x_min, x_max)
        breakpoints.append(breakpoint)
    breakpoints.sort()
    
    # 为每个分段拟合线性模型
    models = []
    segments = []
    
    # 添加边界点
    all_breakpoints = [x_min] + breakpoints + [x_max]
    
    for i in range(len(all_breakpoints) - 1):
        mask = (x_train.flatten() >= all_breakpoints[i]) & (x_train.flatten() < all_breakpoints[i + 1])
        x_segment = x_train[mask]
        y_segment = y_train[mask]
        
        if len(x_segment) > 1:  # 确保分段有足够的数据点
            model = LinearRegression()
            model.fit(x_segment, y_segment)
            models.append(model)
            segments.append((all_breakpoints[i], all_breakpoints[i + 1]))
        else:
            # 如果分段数据不足，使用整个训练集拟合一个备用模型
            model = LinearRegression()
            model.fit(x_train, y_train)
            models.append(model)
            segments.append((all_breakpoints[i], all_breakpoints[i + 1]))
    
    return models, segments

# 一元多次回归模型
def polynomial_regression_model(x_train, y_train, degree, trial=None):
    poly = PolynomialFeatures(degree=degree)
    x_poly = poly.fit_transform(x_train)
    model = LinearRegression()
    model.fit(x_poly, y_train)
    return model, poly

# 计算模型预测
def predict_model(model_type, model, x, segments=None, poly=None):
    if model_type == "linear":
        return model.predict(x)
    elif model_type == "piecewise":
        predictions = np.zeros(len(x))
        for i, (model_segment, (start, end)) in enumerate(zip(model, segments)):
            mask = (x.flatten() >= start) & (x.flatten() < end)
            if np.sum(mask) > 0:  # 确保有数据点需要预测
                predictions[mask] = model_segment.predict(x[mask]).reshape(-1)
        return predictions
    elif model_type in ["quadratic", "cubic"]:
        x_poly = poly.transform(x)
        return model.predict(x_poly)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_linear_model_params(model)->dict:
    slope = model.coef_.reshape(-1)[0]  # 斜率
    intercept = model.intercept_.reshape(-1)[0]  # 截距
    equation = f"y = {slope}x + {intercept}"
    return {"slope": slope, "intercept": intercept, "equation": equation}

def get_piecewise_model_params(models, segments):
    params = []
    segment_points = [seg[-1] for seg in segments][:-1]  # 获取分段点（不包括最后一个）
    for i,(model,segment) in enumerate(zip(models, segments)):
        expr = get_linear_model_params(model)
        params.append({'expression': expr, 'x-range': segment})
    return params,segment_points

def get_polynomial_model_params(model, poly)->dict:
    coefficients = model.coef_.reshape(-1)
    intercept = model.intercept_.reshape(-1)[0]
    feature_names = poly.get_feature_names_out()
    feature_names = [name.replace('x0', 'x') for name in feature_names]
    
    equation = f"y = {intercept}"
    for i, (coef, name) in enumerate(zip(coefficients, feature_names)):
        if i == 0:
            continue
        if coef >= 0:
            equation += f" + {coef}*x^{i}"
        else:
            equation += f" - {abs(coef)}*x^{i}"

    equation = equation.replace('x^1', 'x') 
    
    inflection_points = None
    if poly.degree ==3:
        import sympy
        x = sympy.symbols('x')
        expr = sympy.sympify(equation.replace('y = ', '').replace('^', '**'))
        f2 =sympy.diff(expr, x, 2)
        solutions = sympy.solve(f2, x)
        #获取实数解
        inflection_points = [float(sol.evalf()) for sol in solutions if sol.is_real]
    else:
        # 求二次函数顶点
        a = coefficients[2]  # 二次项系数
        b = coefficients[1]  # 一次项系数
        if a != 0:
            vertex_x = -b / (2 * a)
            vertex_y = intercept + b * vertex_x + a * vertex_x**2
            inflection_points = [(vertex_x, vertex_y)]
    
    points_key = "inflection_points" if poly.degree == 3 else "vertex"
            
    return {"intercept": intercept, "coefficients": dict(zip(feature_names[1:], coefficients[1:])), "equation": equation, points_key: inflection_points}

def regression(request:RegressionRequest)->RegressionResponse:
    params = request.params
    try:
        if params.header:
            df = pd.DataFrame(request.data[1:], columns=request.data[0])
        else:
            df = pd.DataFrame(request.data)
    except Exception as e:
        raise BusinessValidationError(message=f'数据解析失败,检查参数格式和内容:{str(e)}', error_code=ErrorCode.DATA_ERROR)

    tm:TrainModel = TrainModel(df = df, params=params)
    return tm.optimize_models(n_trials=params.trials)

def fregression(request:FRegressionRequest)->RegressionResponse:
    """
    从上传的文件中读取数据，并执行回归分析。
    """
    params = request.params
    try:
        if params.header:
            header = 0
        elif params.header is False:
            header = None
        else:
            header = 'infer'

        df = pd.read_csv(request.file,header=header)
    except Exception as e:
        raise BusinessValidationError(message=f'文件解析失败,检查格式和内容:{str(e)}', error_code=ErrorCode.DATA_ERROR)

    tm:TrainModel = TrainModel(df = df, params=params)
    return tm.optimize_models(n_trials=params.trials)

class TrainModel:
    def __init__(self, df:pd.DataFrame, params:Params):
        # 保留原始数据以便后续使用
        self.original_df = df
        self.params = params

        self.data = np.array(df.iloc[:, [params.column_x, params.column_y]].values.tolist())       
        x,y = self.data[:,0].reshape(-1,1), self.data[:,1].reshape(-1,1)

        try:
            x.astype(float)
            y.astype(float)
        except Exception as e:
            raise BusinessValidationError(message=f'数据类型错误,确保第{params.column_x}列和第{params.column_y}列为数值类型:{str(e)}', error_code=ErrorCode.DATA_ERROR)

        statistic, p_value = spearmanr(x.flatten(), y.flatten())
        self.spearman = SpearmanCeoff(coefficient=statistic, p_value=p_value)
  
        SPLIT_RATIOS = [100, 0, 0]  # 训练集100%，验证集0%，测试集0%

        train_size = len(self.data) * SPLIT_RATIOS[0] // 100
        trial_size = len(self.data) * SPLIT_RATIOS[1] // 100
        test_size = len(self.data) - train_size - trial_size
        self.X_train, self.y_train = x[:train_size], y[:train_size]
        self.X_val, self.y_val = (x[train_size:trial_size + train_size], y[train_size:trial_size + train_size])\
              if trial_size > 0 else (self.X_train.copy(), self.y_train.copy())
        
        self.X_test, self.y_test = (x[-test_size:], y[-test_size:]) \
                if test_size > 0 else (self.X_train.copy(), self.y_train.copy())

    # 目标函数（使用训练集进行模型训练）
    def objective(self, trial):
        # 使用训练集进行模型训练
        x_train, y_train = self.X_train, self.y_train
        X_val,y_val = self.X_val, self.y_val
        # 选择模型类型
        model_type = trial.suggest_categorical('model_type', [
                                                #'linear', 
                                                'piecewise', 
                                                #'quadratic',
                                                #'cubic'
                                                ])
        
        PENALTY_WEIGHTS = {
            'linear': 1.0,      # 基准
            'piecewise': 1.0,   # 基准
            'quadratic': 1.5,   # 中等惩罚
            'cubic': 2.0        # 较重惩罚
        }

        try:
            if model_type == 'linear':
                model = linear_regression_model(x_train, y_train, trial)
                # 使用验证集评估模型性能
                y_pred_val = predict_model(model_type, model, X_val)
                mse = mean_squared_error(y_val, y_pred_val)                
                
            elif model_type == 'piecewise':
                model, segments = piecewise_linear_regression_model(x_train, y_train, trial)
                y_pred_val = predict_model(model_type, model, X_val, segments=segments)
                mse = mean_squared_error(y_val, y_pred_val)
                
            elif model_type == 'quadratic' or model_type == 'cubic':
                model, poly = polynomial_regression_model(x_train, y_train, degree=2 
                                                          if model_type == 'quadratic' else 3, trial=trial)
                y_pred_val = predict_model(model_type, model, X_val, poly=poly)
                mse = mean_squared_error(y_val, y_pred_val)   
                    
            # 根据模型类型应用权重        
            weight = PENALTY_WEIGHTS[model_type]

            # 分段回归额外惩罚
            if model_type == 'piecewise':
                weight += 0.025 * (len(segments) - 1)  # 每超过1段，增加2.5%惩罚     
            
            return mse * weight
            
        except Exception as e:
            # 如果模型拟合失败，返回一个很大的MSE
            return np.inf
        
    def eval_simple_model(self):
        """
        评估简单模型（线性、二次、三次）在测试集上的性能，作为基准比较
        """
        linear_model = linear_regression_model(self.X_train, self.y_train)
        y_pred_test = predict_model("linear", linear_model, self.X_test)
        linear_mse = mean_squared_error(self.y_test, y_pred_test)

        quadratic_model, poly = polynomial_regression_model(self.X_train, self.y_train, degree=2)
        y_pred_test_quad = predict_model("quadratic", quadratic_model, self.X_test, poly=poly)
        quadratic_mse = mean_squared_error(self.y_test, y_pred_test_quad)

        cubic_model, poly_cubic = polynomial_regression_model(self.X_train, self.y_train, degree=3)
        y_pred_test_cubic = predict_model("cubic", cubic_model, self.X_test, poly=poly_cubic)
        cubic_mse = mean_squared_error(self.y_test, y_pred_test_cubic)

        scores= {
            'linear': linear_mse,
            'quadratic': quadratic_mse * 1.5,  # 二次模型增加50%惩罚
            'cubic': cubic_mse*2.0  # 三次模型增加100%惩罚
            }
        
        scores = sorted(scores.items(), key=lambda item: item[1])
        return scores[0]

    
    # 运行Optuna优化
    def optimize_models(self, n_trials=400):
        # 首先评估简单模型的性能，作为基准比较
        simple_score = self.eval_simple_model()

        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials) 

        best_trial_params = study.best_trial.params
        best_model_mse = study.best_value

        if best_model_mse >= simple_score[1]:
            model_type = simple_score[0]
            params = None
        else:
            model_type = best_trial_params['model_type']
            params = best_trial_params

        y_pred_test, mse_test, r2_test, params =  self.evaluate_best_model(model_type, params)

        y_test = self.data[-len(y_pred_test):,1]
        metrics = Metrics(r2_score = r2_test, mse=mse_test)
        
        #计算y_pred_test的包络线
        upper, lower = compute_envelope(y_pred_test, y_test,windows=0, confidence=self.params.confidence)

        self.original_df['y_hat'] = y_pred_test
        self.original_df['y_lower'] = lower
        self.original_df['y_upper'] = upper

        data = self.original_df.values.tolist()
        if self.params.header:
            data = [self.original_df.columns.tolist()] + data

        filename = None
        if self.params.save_to_file:
            filename = write_df_to_file(self.original_df, header=self.params.header, index=False)

        return RegressionResponse(data = data, model_params=params, metrics=metrics, filename=filename,s_ceof=self.spearman)
    
    def evaluate_best_model(self, model_type, params=None):
        """
        使用最佳模型在独立测试集上进行评估，并返回预测结果和评估指标。
        return: 
            y_pred_test:最佳模型在测试集上的预测结果
            mse_test:测试集上的均方误差
            r2_test:测试集上的R²分数
            best_params:最佳模型的参数
        """

        X_test, y_test, X_train, y_train = self.X_test, self.y_test, self.X_train, self.y_train
        
 
        # 使用完整训练集重新拟合最佳模型
        
        if model_type == 'linear':
            model = linear_regression_model(X_train, y_train)
            y_pred_test = predict_model(model_type, model, X_test)
            best_params = {'name': model_type, 'expression': get_linear_model_params(model)}
            
        elif model_type == 'piecewise':
            # 提取分段参数
            n_segments = params['n_segments']
            breakpoints = []
            for i in range(n_segments - 1):
                breakpoints.append(params[f'breakpoint_{i}'])
            breakpoints.sort()
            
            # 重新拟合分段模型
            trial = optuna.trial.FixedTrial(params)
            model, segments = piecewise_linear_regression_model(X_train, y_train, trial)
            y_pred_test = predict_model(model_type, model, X_test, segments=segments)

            params, segment_points = get_piecewise_model_params(model, segments)
            best_params = {'name': model_type,
                'segments': params,
                'segment_points': segment_points}
            
        elif model_type == 'quadratic' or model_type == 'cubic':
            model, poly = polynomial_regression_model(X_train, y_train, degree=2 
                                                      if model_type == 'quadratic' else 3)
            y_pred_test = predict_model(model_type, model, X_test, poly=poly)            
            best_params = {'name': model_type, 'expression': get_polynomial_model_params(model, poly)}
        
        # 计算测试集评估指标
        mse_test = mean_squared_error(y_test, y_pred_test)
        r2_test = r2_score(y_test, y_pred_test)

        return y_pred_test.reshape(-1), mse_test, r2_test, best_params