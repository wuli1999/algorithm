import pandas as pd
def compute_envelope(y_pred, y_true, method='sigma', windows=100, confidence=0.95, range=None)->tuple:
        """
        计算包络线
        """

        #计算残差    
        resid =  pd.Series(y_true - y_pred)
        resid_sigma = resid.std(ddof=1)
        upper,lower = None, None
        if method == 'sigma':
            from scipy import stats
            k = stats.norm.ppf(confidence)
            if windows > 0 and windows < len(y_pred):
                resid_sigma = resid.rolling(windows, min_periods=windows).std()
                nav = resid_sigma.isna()
                resid_sigma[nav] = resid[nav].std()
            else:
                resid_sigma = resid.std()

            upper = y_pred + k * resid_sigma
            lower = y_pred - k * resid_sigma
        elif method == 'iqr':
            pass
        elif method == 'quantile':
            pass
        if range is not None:
            upper = upper.clip(upper=range[1])
            lower = lower.clip(lower=range[0])
            
        return upper, lower