import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from correlation.schema.entities import DetectRequest,DetectResponse, FDetectRequest
from common.business_validation import BusinessValidationError, ErrorCode

def correlation_detect(request:DetectRequest)->DetectResponse:
    try:
        if request.params.header:
            df = pd.DataFrame(request.data[1:], columns=request.data[0], dtype=float) 
        else:
            df = pd.DataFrame(request.data, dtype=float)
        
        df = df.iloc[:,request.params.columns]
    except Exception as e:
        raise BusinessValidationError(message=f'数据读取失败,检查参数格式和内容:{str(e)}', error_code=ErrorCode.INVALID_PARAMETER)

    c, p = spearman_coeff(df)
    return DetectResponse(correlation_matrix=c, pvalue_matrix=p)

def correlation_detect_file(request:FDetectRequest)->DetectResponse:
    try:
        params = request.params
        if params.header:
            header = 0
        elif params.header is False:
            header = None
        else:
            header = 'infer'

        df = pd.read_csv(request.file, header=header, usecols=params.columns,dtype=float)
    except Exception as e:
        raise BusinessValidationError(message=f'数据读取失败,检查文件格式和内容:{str(e)}', error_code=ErrorCode.INVALID_PARAMETER)

    c, p = spearman_coeff(df)
    return DetectResponse(correlation_matrix=c, pvalue_matrix=p)

def spearman_coeff(df):
    if len(df) < 10:
        raise BusinessValidationError(message='数据量少于10,相关性检测无意义', error_code=ErrorCode.INVALID_PARAMETER)
    if df.shape[1] < 2:
        raise BusinessValidationError(message='数据列数小于2,相关性检测无意义', error_code=ErrorCode.INVALID_PARAMETER)

    correlation_matrix, pvalue_matrix = spearmanr(df, axis=0)
    if df.shape[1] == 2:
        m1 = np.full((2, 2), np.nan)
        m1[0, 1] = correlation_matrix
        m1[1, 0] = correlation_matrix

        m2 = np.full((2, 2), np.nan)
        m2[0, 1] = pvalue_matrix
        m2[1, 0] = pvalue_matrix

        return m1, m2
    np.fill_diagonal(correlation_matrix, np.nan)
    np.fill_diagonal(pvalue_matrix, np.nan)

    return correlation_matrix, pvalue_matrix 

def interpret_spearman_results(corr_matrix, pvalue_matrix, alpha=0.05):
    """解读斯皮尔曼相关系数结果"""
    variables = corr_matrix.index
    
    print("=== 斯皮尔曼相关系数结果解读 ===")
    for i in range(len(variables)):
        for j in range(i+1, len(variables)):
            corr = corr_matrix.iloc[i, j]
            pval = pvalue_matrix.iloc[i, j]
            var1, var2 = variables[i], variables[j]
            
            # 判断相关性强度和显著性
            strength = "极弱"
            if abs(corr) >= 0.8:
                strength = "极强"
            elif abs(corr) >= 0.6:
                strength = "强"
            elif abs(corr) >= 0.4:
                strength = "中等"
            elif abs(corr) >= 0.2:
                strength = "弱"
            
            direction = "正" if corr > 0 else "负"
            significant = "显著" if pval < alpha else "不显著"
            
            print(f"{var1} 与 {var2}: {direction}相关({corr:.3f}), {strength}相关, {significant}(p={pval:.4f})")