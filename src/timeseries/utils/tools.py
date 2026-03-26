import numpy as np
import pandas as pd
from pandas import DataFrame,Series
from pandas.tseries.frequencies import to_offset
from typing import List

from common.business_validation import BusinessValidationError,ErrorCode

def histogram(data, bins:int=15, 
              lower_edge:float= None, 
              upper_edge:float=None, 
              overflow:bool = True, 
              density:bool=True, 
              cumulative:bool=True):
    """
    直方图统计
    参数：
    data:array like
        一维数组
    bins
        箱数
    lower_edge
        下溢出边界(包含)
    upper_edge
        上溢出边界(包含)
    overflow
        是否启用上下溢出箱（在最左/最右添加开放区间
    density
        是否输出pdf
    cumulative
        是否输出cdf
    返回
        DataFrame(columns=['left','right','width', 'freq', 'rel_freq', 'cdf'])
    """
    data = np.asarray(data).ravel()
    if data.size == 0:
        raise ValueError("data is empty.")
    
    min_v = np.min(data)
    max_v = np.max(data)
    mean_v = np.mean(data)
    std_v = np.std(data)

    if lower_edge is not None:
        lower_edge = np.maximum(min_v, lower_edge)
    else:        
        lower_edge = np.maximum(min_v, mean_v - 3 * std_v)

    if upper_edge is not None:
        upper_edge = np.minimum(max_v, upper_edge)
    else:
        upper_edge = np.minimum(max_v, mean_v + 3 * std_v)

    bin_edges = np.linspace(lower_edge, upper_edge, bins + 1)
    if overflow:
        if lower_edge > min_v:
            bin_edges = np.r_[min_v, bin_edges]
        
        if upper_edge < max_v:
            bin_edges = np.r_[bin_edges, max_v]

    # 计算频次与密度/累计
    # 注意：density=True 时返回的是概率密度（面积=1），不是概率
    counts, bin_edges = np.histogram(data, bins=bin_edges, density=False)
    total = float(counts.sum())
    # 相对频次（PDF 高度）
    rel_freq = counts / total
    # 累计相对频次（CDF 高度）
    cdf = np.cumsum(rel_freq)
    # 若需要“概率密度”而非“相对频次”，按箱宽缩放
    widths = np.diff(bin_edges)
    pdf = rel_freq / widths  # 各箱高度，使得 sum(pdf * widths) = 1

    return DataFrame(list(zip(bin_edges[:-1],
                               bin_edges[:-1] + widths,
                               widths,
                               counts,
                               rel_freq,
                               cdf)), columns=['left','right','width', 'freq', 'rel_freq', 'cdf'])


def find_periodicity(data,
                     top_k:int = 3, 
                     min_period:int=3,
                     max_period_ratio:float = 0.5)->List[int]:
    """
    参数:
        data: 一维数值数组
        top_k: 需要返回的周期数量，默认为3
        min_period: 最小可接受的周期（采样点数）
        max_period_ratio: 最大周期与序列长度的最大比例，用于过滤长周期噪声
    返回:
        包含前 top_k 个周期长度的列表，按显著性（能量）降序排列。
        如果未找到足够的有效周期，则返回的列表长度会小于 top_k。
    """
    n = len(data)
    if n < 4:
        return []

    # 1. 预处理：去均值和去趋势
    y = data - np.mean(data)
    if n > 2:
        y = y - np.polyval(np.polyfit(np.arange(n), y, 1), np.arange(n))

    # 2. 执行FFT并计算幅度谱
    ft = np.fft.rfft(y)
    mags = np.abs(ft)

    # 3. 去除直流分量（0Hz）和奈奎斯特频率（最高频）
    if n % 2 == 0:
        mags[0] = 0      # 直流分量
        mags[-1] = 0      # 奈奎斯特频率
    else:
        mags[0] = 0      # 只有直流分量

    # 4. 寻找所有局部峰值
    # 寻找局部极大值：一个点比左右邻居都高
    is_peak = (np.diff(np.sign(np.diff(mags))) < 0)
    peak_indices = np.where(is_peak)[0] + 1  # 峰值索引

    # 5. 筛选有效峰值
    max_period = int(n * max_period_ratio)
    valid_peaks = [
        idx for idx in peak_indices
        if min_period <= idx <= max_period
    ]

    # 如果有效峰值不足，直接返回
    if len(valid_peaks) == 0:
        return []

    # 6. 按幅度（能量）对有效峰值进行降序排序，并取前 top_k 个
    sorted_peaks = sorted(valid_peaks, key=lambda i: mags[i], reverse=True)
    top_peaks = sorted_peaks[:top_k]

    # 7. 将频率索引转换为周期长度（采样点数）
    # 周期 T = N / f，其中 f = idx / N
    periods = [int(n // idx) for idx in top_peaks]

    return periods


def is_continuous_auto(series: Series, tol: pd.Timedelta | None = None) -> tuple[bool,str]:
    """
    根据Series时间列推断频率,并判断数据是否连续
    """
    s = series.sort_values().drop_duplicates()
    if len(s) < 2:
        return True, None

    # 1) 计算相邻间隔并取最常见（或最小）作为候选步长
    diffs =s.diff()
    mode_diff = diffs.value_counts().index[0]  # 最常见间隔
    # 可选：用最小间隔以增强鲁棒性
    # mode_diff = diffs.min()

    # 2) 将步长标准化为 pandas 偏移量（处理 '5T' 与 '5min' 等别名）
    try:
        offset = to_offset(mode_diff)
    except Exception:
        return False,None
    freq = offset.freqstr

    # 3) 生成完整网格并比对
    full = pd.date_range(s.iloc[0], s.iloc[-1], freq=freq).to_series()
    return full.equals(s), freq

def compare_freq(f1: str, f2: str) -> int:
    """
    比较两个频率字符串是否表示相同的频率
    返回:
        0: 两个频率相同
        -1: f1 < f2
        1: f1 > f2
        -1000: 无法比较（无效频率）
    """
    try: 
        v1 = to_offset(f1)
        v2 = to_offset(f2)
        if v1 == v2:
            return 0
        elif v1 < v2:
            return -1
        else:
            return 1
    except Exception:
       return -1000

def validate_freq(freq:str, msg:str)->None:
    try:
        to_offset(freq)
    except Exception:
        raise BusinessValidationError(message=msg, error_code=ErrorCode.INVALID_PARAMETER)