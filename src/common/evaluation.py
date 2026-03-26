import json
from typing import Dict, Any
def evaluate_model(metrics: Dict[str, Any], alpha: float = 0.05) -> str:
    """
    根据输入的模型指标字典，生成一份 Markdown 格式的评估报告

    参数:
    metrics (dict): 包含模型指标的字典。
    alpha (float): 显著性水平，默认 0.05

    返回:
    str: 生成的 Markdown 格式报告内容
    """
    
    # 辅助函数：根据 p 值输出判断
    def judge(p: float) -> str:
        return "显著" if p < alpha else "不显著"

    lines = []
    lines.append("# 模型评估报告\n")

    # ==================== 1. 整体预测精度 ====================
    lines.append("## 1. 整体预测精度\n")
    mae = metrics.get("MAE")
    mape = metrics.get("MAPE")
    mse = metrics.get("MSE")
    rmse = metrics.get("RMSE")

    if all(v is not None for v in [mae, mape, mse, rmse]):
        lines.append("| 指标 | 值 | 说明 |")
        lines.append("| :--- | :--- | :--- |")
        lines.append(f"| MAE (平均绝对误差) | {mae:.4f} | 平均每次预测偏离约 {mae:.2f} 个单位。 |")
        lines.append(f"| MAPE (平均绝对百分比误差) | {mape:.4%} | 平均预测偏差约为实际值的 {mape:.2%}。 |")
        lines.append(f"| MSE (均方误差) | {mse:.4f} | 对较大误差更敏感，可作为优化目标参考。 |")
        lines.append(f"| RMSE (均方根误差) | {rmse:.4f} | 与原始数据同量纲，平均偏离约 {rmse:.2f} 个单位。 |\n")

        if mape < 0.05:
            lines.append("- **精度非常高**：MAPE < 5%。\n")
        elif mape < 0.10:
            lines.append("- **精度较高**：MAPE 介于 5%~10%。\n")
        elif mape < 0.20:
            lines.append("- **精度一般**：MAPE 介于 10%~20%。\n")
        else:
            lines.append("- **精度有待提升**：MAPE ≥ 20%。\n")
    else:
        lines.append("部分精度指标缺失，无法完整评估预测精度。\n")

    # ==================== 2. 数据分布与残差分析 ====================
    lines.append("## 2. 数据分布与残差分析\n")
    samp_mean = metrics.get("SAMP_MEAN")
    samp_std = metrics.get("SAMP_STD")
    resid_mean = metrics.get("RESID_MEAN")
    resid_std = metrics.get("RESID_STD")

    if all(v is not None for v in [samp_mean, samp_std, resid_mean, resid_std]):
        lines.append("| 指标 | 值 | 说明 |")
        lines.append("| :--- | :--- | :--- |")
        lines.append(f"| 样本均值 (SAMP_MEAN) | {samp_mean:.4f} | 测试集真实值的平均水平。 |")
        lines.append(f"| 样本标准差 (SAMP_STD) | {samp_std:.4f} | 测试集真实值的波动范围。 |")
        lines.append(f"| 残差均值 (RESID_MEAN) | {resid_mean:.4f} | 预测误差的平均值。 |")
        lines.append(f"| 残差标准差 (RESID_STD) | {resid_std:.4f} | 残差的波动幅度。 |\n")

        if abs(resid_mean) < 0.01 * samp_std:
            lines.append("- **无明显系统性偏差**：残差均值接近 0。\n")
        else:
            bias_dir = "高估" if resid_mean > 0 else "低估"
            lines.append(f"- **存在轻微系统性偏差**：残差均值偏离 0，模型整体有轻微 **{bias_dir}** 倾向。\n")

        if samp_std > 0 and resid_std > 0:
            r2_like = 1 - (resid_std ** 2) / (samp_std ** 2)
            lines.append(f"- **残差波动解释比例**: {r2_like:.2%}")
            if r2_like > 0.8:
                lines.append("  - 模型已捕捉数据大部分波动。\n")
            elif r2_like > 0.5:
                lines.append("  - 模型对数据波动有一定解释力。\n")
            else:
                lines.append("  - 模型对数据波动解释不足。\n")
    else:
        lines.append("部分数据分布或残差指标缺失，无法完整分析。\n")

    # ==================== 3. 统计显著性检验 ====================
    lines.append("## 3. 统计显著性检验\n")
    lb_p = metrics.get("LB_PVALUE")
    sw_p = metrics.get("SW_PVALUE")

    if lb_p is not None:
        lines.append(f"- **Ljung-Box 检验 p 值 (LB_PVALUE)**: `{lb_p:.4e}`")
        lines.append(f"  - 在 α={alpha} 水平下，检验结论：**{judge(lb_p)}**。")
        if lb_p < alpha:
            lines.append("  - 说明残差中仍存在显著的自相关结构，模型可能未完全提取信息。\n")
        else:
            lines.append("  - 说明残差在整体上看接近白噪声，模型拟合较充分。\n")
    else:
        lines.append("- **LB_PVALUE 缺失**，无法执行 Ljung-Box 检验解读。\n")

    if sw_p is not None:
        lines.append(f"- **Shapiro-Wilk 正态性检验 p 值 (SW_PVALUE)**: `{sw_p:.4e}`")
        lines.append(f"  - 在 α={alpha} 水平下，检验结论：**{judge(sw_p)}**。")
        if sw_p < alpha:
            lines.append("  - 说明残差显著偏离正态分布，可能存在厚尾或偏斜。\n")
        else:
            lines.append("  - 说明残差分布与正态分布无显著差异，满足正态性假设。\n")
    else:
        lines.append("- **SW_PVALUE 缺失**，无法执行 Shapiro-Wilk 检验解读。\n")

    # ==================== 4. 综合评估 ====================
    lines.append("## 4. 综合评估\n")
    good_accuracy = mape is not None and mape < 0.10
    good_resid_mean = resid_mean is not None and abs(resid_mean) < 0.01 * samp_std
    lb_ok = lb_p is not None and lb_p >= alpha
    sw_ok = sw_p is not None and sw_p >= alpha

    if good_accuracy and good_resid_mean and lb_ok and sw_ok:
        level = "非常优秀"
    elif good_accuracy and good_resid_mean and (not lb_ok or not sw_ok):
        level = "整体良好，但残差或自相关结构仍有优化空间"
    elif good_accuracy:
        level = "预测精度较好，但模型假设（如残差正态性、无自相关）存在一定问题"
    else:
        level = "预测精度一般或较差，需重点优化模型结构或特征工程"

    lines.append(f"综合来看，该模型在测试集上的表现：**{level}**。\n")
    lines.append("### 主要结论：")
    lines.append(f"- {'✅ 预测误差较小，精度较高。' if good_accuracy else '⚠️ 预测误差偏大，精度有待提升。'}")
    lines.append(f"- {'✅ 模型无明显系统性偏差。' if good_resid_mean else '⚠️ 模型存在一定系统性高估或低估。'}")
    lines.append(f"- {'✅ 残差整体接近白噪声。' if lb_ok else '⚠️ 残差中仍有可建模的自相关结构。'}")
    lines.append(f"- {'✅ 残差分布接近正态。' if sw_ok else '⚠️ 残差分布偏离正态，可能对部分统计推断有影响。'}\n")

    lines.append("### 建议：")
    if not good_accuracy:
        lines.append("- 尝试引入更多特征或调整模型结构以提升精度。")
    if not lb_ok:
        lines.append("- 考虑增加模型复杂度（如季节性、非线性）或引入 ARIMA 等模型。")
    if not sw_ok:
        lines.append("- 若后续需进行严格的统计推断，可尝试对残差做变换或选用更稳健的模型。")

    report_md = "\n".join(lines)

    return report_md