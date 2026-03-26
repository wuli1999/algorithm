import json
from datetime import datetime, timezone

def format_timestamp(ts, freq: str = "5min") -> str:
    """
    将时间戳按 freq 转成可读时间，freq 仅用于提示，实际按秒解析
    """
    try:
        # 兼容带时区和不带时区的时间戳
        if ts > 1e10:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        else:

            dt = datetime.utcfromtimestamp(ts)

        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)

def append_md_line(md, text):
    md.append(text)
    
def interpret_summary(summary: dict) -> str:
    lines = []
# ==================== 1. 数据概览 ====================
    num = summary.get("numeric_summary", {})
    ser = summary.get("series_summary", {})

    count = num.get("count", 0)
    invalid = num.get("invalid", 0)

    time_range = ser.get("time_range", [])
    if len(time_range) == 2:
        start_ts, end_ts = time_range
        start_str = format_timestamp(start_ts)
        end_str = format_timestamp(end_ts)
        duration_min = (end_ts - start_ts) / 60
        if duration_min < 60:
            dur_str = f"{duration_min:.1f} 分钟"
        elif duration_min < 1440:
            dur_str = f"{duration_min / 60:.1f} 小时"
        else:
            dur_str = f"{duration_min / 1440:.1f} 天"
    else:
        start_str = end_str = "未知"
        dur_str = "未知"

    freq = ser.get("freq", "")
    if freq:
        if freq.endswith("min"):
            freq_desc = f"{freq}（较高频监控数据，适合精细分析）"
        elif freq.endswith("h"):
            freq_desc = f"{freq}（小时级数据，适合观察日内/日间变化）"
        elif freq.endswith("d"):
            freq_desc = f"{freq}（天级数据，适合观察长期趋势）"
        else:
            freq_desc = freq
    else:
        freq_desc = "未知"

    lines.append("##📊数据概览")
    append_md_line(lines,f"**数据量：**共 {count} 个有效样本，无效或缺失值 {invalid} 条。")
    append_md_line(lines,f"**时间跨度：**约 {dur_str}，从 {start_str} 至 {end_str}，数据{'完整连续' if ser.get('missed', 0) == 0 else '<font color="red">存在缺失</font>'}。")
    append_md_line(lines,f"**采样频率：**{freq_desc}。")

    # ==================== 2. 数值分布特征 ====================
    mean = num.get("mean")
    std = num.get("std")
    five_num = num.get("five_number_summary", [])
    mode = num.get("mode", [])

    lines.append("##📈数值分布特征")

    if mean is not None and std is not None:
        append_md_line(lines,
            f"**集中趋势：**平均值 ({mean:.3f}) "
            f"{'高于' if mean > five_num[2] else '低于' if mean < five_num[2] else '接近'}中位数 ({five_num[2]:.3f})，"
            f"说明数据分布呈{'轻微的右偏（正偏态）' if mean > five_num[2] else '轻微的左偏（负偏态）' if mean < five_num[2] else '大致对称'}。"
        )
        append_md_line(lines,f"-**离散程度：**标准差 ({std:.3f}) {'较大' if std > 0.5 * mean else '中等' if std > 0.2 * mean else '较小'}，数据波动{'明显' if std > 0.5 * mean else '适中' if std > 0.2 * mean else '平缓'}。")

    # 粗略给出主要集中区间
    if len(five_num) == 5:
        low, q1, q2, q3, high = five_num
        append_md_line(lines,
            f"**形态解读：**约 50% 的数据落在 [{q1:.2f}, {q3:.2f}] 区间内，"
            f"整体数值主要集中在 {low:.2f} ~ {high:.2f} 范围。"
        )

    if mode:
        mode_str = ", ".join(f"{m:.3f}" for m in mode)
        append_md_line(lines,f"**众数：**{mode_str}（出现频率最高的几个值）。")

    # ==================== 3. 分位数与极值 ====================
    append_md_line(lines,"##📉 分位数与极值")

    if len(five_num) == 5:
        q1, q2, q3 = five_num[1], five_num[2], five_num[3]
        iqr = q3 - q1
        low, high = five_num[0], five_num[4]
        append_md_line(lines,f"**中位数(Q2)：**{q2:.3f}")
        append_md_line(lines,f"**四分位数：**<br>   Q1 (下四分位): {q1:.3f}<br>   Q3 (上四分位): {q3:.3f}")
        append_md_line(lines,f"**四分位距(IQR)：**{iqr:.3f}，代表数据中间 50% 部分的波动范围。")
        append_md_line(lines,f"**极值：**<br>   最小值：{low:.3f}<br>   最大值：{high:.3f}")

        append_md_line(lines,"**业务含义：**")
        append_md_line(lines,f"  **核心区间：**约 50% 的数据落在 [{q1:.2f}, {q3:.2f}] 内。")
        append_md_line(lines,f"  **正常范围：**约 75% 的数据小于 {q3:.2f}。可将 [{low:.2f}, {q3:.2f}] 视为常规运行区间。")
        if high > q3 + 1.5 * iqr:
            append_md_line(lines,f"  **异常识别：**{high:.3f} 这个值明显高于上边界（Q3 + 1.5×IQR），很可能是需要重点排查的异常点或业务高峰。")
        else:
            append_md_line(lines,f"  **异常识别：**未发现明显远离主体的极端异常点。")

    # ==================== 4. 直方图形态 ====================
    append_md_line(lines,"##📊 直方图形态")

    hist = num.get("histogram", [])
    if hist:
        max_bin = max(hist, key=lambda x: x[3])
        append_md_line(lines,
            f"**主峰：**在 [{max_bin[0]:.3f}, {max_bin[1]:.3f}) 区间内存在一个明显的主峰，"
            f"是数据最密集的区域，可视为系统的“典型负载”。"
        )

        peaks = [h for h in hist if h[3] > count * 0.05]
        if len(peaks) > 1:
            peak_ranges = [f"[{h[0]:.2f}, {h[1]:.2f})" for h in peaks]
            append_md_line(lines,
                f"**多峰性：**在 {', '.join(peak_ranges)} 等区间存在较明显的峰，"
                f"表明数据可能来自不同模式或状态的叠加（如不同业务场景、工作日/周末等）。"
            )

        # 看右侧长尾
        right_tail_bins = [h for h in hist if h[1] > q3]
        if right_tail_bins and sum(h[3] for h in right_tail_bins) > 0.1 * count:
            append_md_line(lines,
                "**长尾：**在较高数值区间仍有较多样本分布，说明数据存在一定右长尾特征，"
                "可能对应偶发的业务高峰或极端事件。"
            )
        else:
            append_md_line(lines,"长尾：高值区间样本较少，数据分布相对集中在主体区间内。")

    # ==================== 5. 季节性/周期性 ====================
    append_md_line(lines,"##🔄 季节性/周期性")

    seasonal = ser.get("seasonal", [])
    if seasonal:
        append_md_line(lines,
            f"**周期模式：**{seasonal} 揭示了数据存在固定的周期性模式，"
            f"其周期为 {seasonal[0]} 个采样点。"
        )
        if freq and freq.endswith("min"):
            try:
                minutes_per_point = int(freq.replace("min", ""))
                period_minutes = seasonal[0] * minutes_per_point
                if period_minutes == 1440:
                    period_str = "1 天"
                elif period_minutes % 60 == 0:
                    period_str = f"{period_minutes // 60} 小时"
                else:
                    period_str = f"{period_minutes} 分钟"
                append_md_line(lines,
                    f"**周期换算：**由于采样频率为 {freq}，一个完整周期时长为：\n"
                    f"   {seasonal[0]} × {minutes_per_point} 分钟 = {period_minutes} 分钟 = {period_str}"
                )
                append_md_line(lines,
                    f"**模式解读：**这表明数据具有明显的“{period_str}”特征。数组 {seasonal} 可能表示在一个周期内，"
                    f"不同状态或模式持续的时间长度（以 {freq} 为单位），例如："
                )
                sub_periods = [f"   **模式{i+1}：**持续 {x * minutes_per_point} 分钟" for i, x in enumerate(seasonal)]
                append_md_line(lines,"\n".join(sub_periods))
                append_md_line(lines,
                    "这种模式常见于具有固定作息或业务高峰（如白天高、夜间低）的场景。"
                )
            except ValueError:
                append_md_line(lines,"周期换算：无法解析采样频率，仅给出采样点周期。")
    else:
        append_md_line(lines,"当前统计信息未检测到明显周期性，序列可能以趋势或随机波动为主。")

    # ==================== 6. 综合解读与后续建议 ====================
    append_md_line(lines,"##💡 综合解读与后续建议")

    # 自动综合解读
    interpretation_parts = []
    if ser.get("missed", 0) == 0:
        interpretation_parts.append("数据整体质量较好，时间序列完整无缺失")
    else:
        interpretation_parts.append(f"数据存在{ser['missed']}个缺失点，建议检查数据采集或传输环节")

    if mean is not None and len(five_num) == 5:
        if mean > five_num[2]:
            interpretation_parts.append("数值分布呈右偏，说明存在少量相对较大的异常值或业务高峰")
        elif mean < five_num[2]:
            interpretation_parts.append("数值分布呈左偏，说明存在少量相对较小的值拉低了整体水平")
        else:
            interpretation_parts.append("数值分布大致对称")

        if high > q3 + 1.5 * iqr:
            interpretation_parts.append("最大值明显偏高，可能存在极端异常点")

    if seasonal:
        if freq and freq.endswith("min"):
            try:
                minutes_per_point = int(freq.replace("min", ""))
                period_minutes = seasonal[0] * minutes_per_point
                if period_minutes == 1440:
                    period_desc = "日周期"
                elif period_minutes % 60 == 0:
                    period_desc = f"{period_minutes // 60}小时周期"
                else:
                    period_desc = f"{period_minutes}分钟周期"
                interpretation_parts.append(f"具有明显的 {period_desc} 特征")
            except ValueError:
                interpretation_parts.append("检测到周期性模式")

    if dur_str != "未知":
        if "天" in dur_str:
            duration_desc = "中长期"
        elif "小时" in dur_str:
            duration_desc = "中期"
        else:
            duration_desc = "短期"
        interpretation_parts.append(f"覆盖时间跨度约 {dur_str}，属于{duration_desc}观测")

    append_md_line(lines,
        "该序列" + "、".join(interpretation_parts) + "的时间序列。"
    )

    # 自动建议
    suggestions = []

    if ser.get("missed", 0) > 0:
        suggestions.append(
            "对缺失值进行合理填补（如前向填充、插值或基于模型的预测），以减少对后续分析和建模的影响。"
        )
    else:
        suggestions.append(
            "数据完整性较好，可直接用于时序建模与预测分析。"
        )

    if len(five_num) == 5 and high > q3 + 1.5 * iqr:
        suggestions.append(
            "对明显偏高的值进行标注和排查，结合业务日志判断是正常业务高峰还是异常事件，必要时可考虑截断或稳健统计方法。"
        )
    else:
        suggestions.append(
            "极端异常点不明显，可重点关注常规波动和周期性变化。"
        )

    if seasonal:
        suggestions.append(
            "序列存在明显周期性，可进一步绘制按周期切分的子序列图（如按天/周），观察模式差异，并在建模时引入季节性特征或周期编码。"
        )
    else:
        suggestions.append(
            "当前统计信息未检测到明显周期性，可尝试通过自相关图、谱分析等方法进一步探查是否存在局部周期或趋势变化。"
        )

    if count < 100 or ("天" in dur_str and float(dur_str.replace(" 天", "")) < 7):
        suggestions.append(
            "数据量相对有限，可优先进行探索性分析和简单模型（如移动平均、指数平滑）尝试，不建议直接采用参数复杂的模型。"
        )
    else:
        suggestions.append(
            "数据量较为充足，时间跨度适中，适合采用ARIMA/SARIMA、Prophet、LSTM等模型进行系统建模和预测。"
        )

    if mean is not None and std is not None and mean != 0:
        cv = std / mean
        if cv > 1:
            suggestions.append(
                "变异系数较大，数据波动剧烈，建议在建模前考虑对数据进行对数或 Box-Cox 变换，以提升模型稳定性。"
            )
        else:
            suggestions.append(
                "数据波动相对温和，可直接使用原始尺度进行建模，也可尝试标准化处理以方便与多指标联合分析。"
            )

    suggestions.append(
        "建议绘制时间序列曲线、直方图、箱线图及（若有周期）周期子图，以直观检查趋势、季节性、异常点和分布特征。"
    )

    append_md_line(lines,"**后续分析建议：**")
    for i, s in enumerate(suggestions, 1):
        append_md_line(lines,f"{i}) {s}")

    return "\n".join(lines)