#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF四周期量化打分系统
====================
基于"年K定战略 -> 月K定方向 -> 周K定节奏 -> 日K找买点"框架
可每日执行，对指定ETF列表进行四周期打分并生成操作信号

依赖安装: pip install akshare pandas numpy openpyxl
运行方式: python ETF四周期量化打分系统.py
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 配置区域 - 用户可根据需要修改
# ============================================================

# ETF列表（代码 + 名称）
ETF_LIST = [
    {"code": "513350", "name": "标普油气ETF富国"},
    {"code": "159920", "name": "恒生ETF华夏"},
    {"code": "159915", "name": "创业板ETF易方达"},
    {"code": "159941", "name": "纳指ETF广发"},
    {"code": "159985", "name": "豆粕ETF华夏"},
    {"code": "518880", "name": "黄金ETF华安"},
    {"code": "513520", "name": "日经ETF华夏"},
    {"code": "513030", "name": "德国ETF华安"},
    {"code": "159981", "name": "建信能源化工期货ETF"},
    {"code": "159980", "name": "有色ETF大成"},
    {"code": "588000", "name": "科创50ETF"},
    {"code": "513180", "name": "恒生科技ETF"},
    {"code": "513500", "name": "标普500ETF博时"},
    {"code": "512890", "name": "红利低波ETF华泰柏瑞"},
    {"code": "159201", "name": "自由现金流ETF"},
]

# 输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 数据缓存目录
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# 权重配置
WEIGHTS = {
    "year": 0.25,   # 年K权重25%
    "month": 0.20,  # 月K权重20%
    "week": 0.30,   # 周K权重30%
    "day": 0.25,    # 日K权重25%
}

# 评分维度权重（每个周期内各维度的满分）
# 年K: 均线位置10分 + 趋势形态10分 + 估值分位5分 = 25分
# 月K: 均线位置10分 + 趋势排列5分 + 形态信号5分 = 20分
# 周K: 均线位置10分 + MACD信号10分 + 量价配合10分 = 30分
# 日K: 均线位置10分 + 指标信号10分 + K线形态5分 = 25分

# ============================================================
# 数据获取函数
# ============================================================

def fetch_etf_data(code: str, days: int = 365 * 5) -> pd.DataFrame:
    """
    获取ETF历史日K数据
    返回: DataFrame with columns [date, open, close, high, low, volume, amount]
    """
    cache_file = os.path.join(CACHE_DIR, f"{code}_daily.csv")

    # 尝试从缓存读取
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, parse_dates=["日期"])
            if len(df) >= days * 0.8:  # 缓存数据足够
                df = df.tail(days).reset_index(drop=True)
                return df
        except Exception:
            pass

    # 使用akshare获取数据
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")

    try:
        df = ak.fund_etf_hist(
            symbol=code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=""
        )
    except Exception as e:
        print(f"  [警告] {code} akshare获取失败: {e}, 尝试备用方法...")
        try:
            # 备用方法：使用stock_zh_index_daily
            df = ak.stock_zh_index_daily(symbol=f"sh{code}" if code.startswith("5") else f"sz{code}")
        except Exception as e2:
            print(f"  [错误] {code} 备用方法也失败: {e2}")
            return pd.DataFrame()

    if df is None or len(df) == 0:
        print(f"  [警告] {code} 未获取到数据")
        return pd.DataFrame()

    # 统一列名
    col_map = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if "日期" in col_lower or "date" in col_lower:
            col_map[col] = "日期"
        elif "开盘" in col or "open" in col_lower:
            col_map[col] = "开盘"
        elif "收盘" in col or "close" in col_lower:
            col_map[col] = "收盘"
        elif "最高" in col or "high" in col_lower:
            col_map[col] = "最高"
        elif "最低" in col or "low" in col_lower:
            col_map[col] = "最低"
        elif "成交量" in col or "volume" in col_lower:
            col_map[col] = "成交量"
        elif "成交额" in col or "amount" in col_lower:
            col_map[col] = "成交额"

    df = df.rename(columns=col_map)

    # 确保必要列存在
    required_cols = ["日期", "开盘", "收盘", "最高", "最低", "成交量"]
    for col in required_cols:
        if col not in df.columns:
            print(f"  [警告] {code} 缺少列: {col}")
            return pd.DataFrame()

    # 转换数值类型
    for col in ["开盘", "收盘", "最高", "最低", "成交量"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[required_cols].dropna().reset_index(drop=True)
    df = df.tail(days).reset_index(drop=True)

    # 保存到缓存
    try:
        df.to_csv(cache_file, index=False)
    except Exception:
        pass

    return df


def resample_to_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    将日K数据重采样为周K/月K/年K
    period: 'weekly', 'monthly', 'yearly'
    """
    if len(df) < 10:
        return df

    df_copy = df.copy()
    df_copy["日期"] = pd.to_datetime(df_copy["日期"])
    df_copy = df_copy.set_index("日期")

    if period == "weekly":
        # 周K：使用周五的数据，或一周的最后一根K线
        df_weekly = df_copy.resample("W-FRI").last()
    elif period == "monthly":
        # 月K：使用月末数据
        df_monthly = df_copy.resample("ME").last()
    elif period == "yearly":
        # 年K：使用年末数据
        df_yearly = df_copy.resample("YE-DEC").last()
    else:
        return df_copy

    # 对于重采样，OHLC需要特殊处理
    # 开盘取周期第一个，收盘取周期最后一个，最高取周期最高，最低取周期最低
    if period == "weekly":
        df_weekly_raw = df_copy.resample("W-FRI")
        df_weekly = pd.DataFrame()
        df_weekly["开盘"] = df_weekly_raw["开盘"].first()
        df_weekly["收盘"] = df_weekly_raw["收盘"].last()
        df_weekly["最高"] = df_weekly_raw["最高"].max()
        df_weekly["最低"] = df_weekly_raw["最低"].min()
        df_weekly["成交量"] = df_weekly_raw["成交量"].sum()
        df_weekly = df_weekly.dropna(subset=["收盘"])
    elif period == "monthly":
        df_monthly_raw = df_copy.resample("ME")
        df_monthly = pd.DataFrame()
        df_monthly["开盘"] = df_monthly_raw["开盘"].first()
        df_monthly["收盘"] = df_monthly_raw["收盘"].last()
        df_monthly["最高"] = df_monthly_raw["最高"].max()
        df_monthly["最低"] = df_monthly_raw["最低"].min()
        df_monthly["成交量"] = df_monthly_raw["成交量"].sum()
        df_monthly = df_monthly.dropna(subset=["收盘"])
    elif period == "yearly":
        df_yearly_raw = df_copy.resample("YE-DEC")
        df_yearly = pd.DataFrame()
        df_yearly["开盘"] = df_yearly_raw["开盘"].first()
        df_yearly["收盘"] = df_yearly_raw["收盘"].last()
        df_yearly["最高"] = df_yearly_raw["最高"].max()
        df_yearly["最低"] = df_yearly_raw["最低"].min()
        df_yearly["成交量"] = df_yearly_raw["成交量"].sum()
        df_yearly = df_yearly.dropna(subset=["收盘"])

    df_result = df_weekly if period == "weekly" else (df_monthly if period == "monthly" else df_yearly)
    df_result = df_result.reset_index()
    df_result = df_result.rename(columns={"index": "日期"})
    return df_result


# ============================================================
# 技术指标计算函数
# ============================================================

def calc_ma(series: pd.Series, period: int) -> pd.Series:
    """计算移动平均线"""
    return series.rolling(window=period, min_periods=1).mean()


def calc_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal_period: int = 9) -> tuple:
    """
    计算MACD指标
    返回: (macd_line, signal_line, histogram)
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_kdj(df: pd.DataFrame, period: int = 9) -> tuple:
    """
    计算KDJ指标
    返回: (k_line, d_line, j_line)
    """
    low_list = df["最低"].rolling(window=period, min_periods=1).min()
    high_list = df["最高"].rolling(window=period, min_periods=1).max()

    rsv = (df["收盘"] - low_list) / (high_list - low_list + 1e-10) * 100

    k_line = rsv.ewm(com=2, adjust=False).mean()
    d_line = k_line.ewm(com=2, adjust=False).mean()
    j_line = 3 * k_line - 2 * d_line

    return k_line, d_line, j_line


# ============================================================
# 四周期评分函数
# ============================================================

def score_yeark(df_daily: pd.DataFrame) -> dict:
    """
    年K评分（满分25分）
    维度1: 均线位置(10分) - 股价与20年线关系
    维度2: 趋势形态(10分) - 高低点变化
    维度3: 估值分位(5分) - PE历史分位(用价格分位替代)
    """
    # 重采样为年K
    df_yearly = resample_to_period(df_daily, "yearly")

    if len(df_yearly) < 3:
        # 数据不足，返回中性评分
        return {
            "均线位置": {"得分": 5, "说明": "年K数据不足，无法判断"},
            "趋势形态": {"得分": 3, "说明": "年K数据不足，无法判断"},
            "估值分位": {"得分": 3, "说明": "估值数据不可用，使用价格分位替代"},
            "总分": 11,
            "detail": "年K数据不足"
        }

    close = df_yearly["收盘"]
    ma20_year = calc_ma(close, 20)

    # 维度1: 均线位置(10分)
    current_price = close.iloc[-1]
    ma20_val = ma20_year.iloc[-1] if not pd.isna(ma20_year.iloc[-1]) else current_price

    if ma20_val > 0:
        ratio = current_price / ma20_val
        if ratio > 1.05:
            ma_score = 10
            ma_desc = f"股价({current_price:.2f})站稳20年线({ma20_val:.2f})上方，长期牛市格局"
        elif ratio > 0.95:
            ma_score = 5
            ma_desc = f"股价({current_price:.2f})在20年线({ma20_val:.2f})附近缠绕，长期趋势不明"
        else:
            ma_score = -10
            ma_desc = f"股价({current_price:.2f})在20年线({ma20_val:.2f})下方，长期熊市格局"
    else:
        ma_score = 5
        ma_desc = "20年线数据不可用"

    # 维度2: 趋势形态(10分)
    # 检查高低点变化
    if len(close) >= 6:
        # 取最近6个年度数据
        recent = close.tail(6)
        # 计算局部高低点
        highs = []
        lows = []
        for i in range(1, len(recent) - 1):
            if recent.iloc[i] > recent.iloc[i-1] and recent.iloc[i] > recent.iloc[i+1]:
                highs.append((i, recent.iloc[i]))
            if recent.iloc[i] < recent.iloc[i-1] and recent.iloc[i] < recent.iloc[i+1]:
                lows.append((i, recent.iloc[i]))

        if len(highs) >= 2 and len(lows) >= 2:
            high_values = [h[1] for h in highs]
            low_values = [l[1] for l in lows]

            if all(high_values[i] < high_values[i+1] for i in range(len(high_values)-1)) and \
               all(low_values[i] < low_values[i+1] for i in range(len(low_values)-1)):
                trend_score = 10
                trend_desc = "低点不断抬高、高点突破，长期上升趋势确立"
            elif all(high_values[i] > high_values[i+1] for i in range(len(high_values)-1)) and \
                 all(low_values[i] > low_values[i+1] for i in range(len(low_values)-1)):
                trend_score = -10
                trend_desc = "低点不断降低、高点下移，长期下降通道"
            else:
                trend_score = 3
                trend_desc = "横盘震荡，无明显方向"
        else:
            trend_score = 3
            trend_desc = "数据点不足，横盘震荡"
    else:
        trend_score = 3
        trend_desc = "年K数据点不足，横盘震荡"

    # 维度3: 估值分位(5分) - 用价格历史分位替代（ETF多数无PE数据）
    if len(close) >= 5:
        current = close.iloc[-1]
        # 排除最后一个值，计算历史分位
        historical = close.iloc[:-1]
        percentile = (historical < current).sum() / len(historical) * 100

        if percentile < 20:
            val_score = 5
            val_desc = f"价格处于历史低位({percentile:.0f}%分位)，安全边际高"
        elif percentile <= 80:
            val_score = 3
            val_desc = f"价格处于历史中性区({percentile:.0f}%分位)，估值合理"
        else:
            val_score = -5
            val_desc = f"价格处于历史高位({percentile:.0f}%分位)，注意风险"
    else:
        val_score = 3
        val_desc = "数据不足，无法计算分位"

    total = ma_score + trend_score + val_score

    return {
        "均线位置": {"得分": ma_score, "说明": ma_desc},
        "趋势形态": {"得分": trend_score, "说明": trend_desc},
        "估值分位": {"得分": val_score, "说明": val_desc},
        "总分": total,
    }


def score_monthk(df_daily: pd.DataFrame) -> dict:
    """
    月K评分（满分20分）
    维度1: 均线位置(10分) - 股价与20月线关系
    维度2: 趋势排列(5分) - 月线均线多头/空头排列
    维度3: 形态信号(5分) - 底部/顶部形态
    """
    # 重采样为月K
    df_monthly = resample_to_period(df_daily, "monthly")

    if len(df_monthly) < 3:
        return {
            "均线位置": {"得分": 5, "说明": "月K数据不足"},
            "趋势排列": {"得分": 0, "说明": "月K数据不足"},
            "形态信号": {"得分": 0, "说明": "月K数据不足"},
            "总分": 5,
        }

    close = df_monthly["收盘"]
    ma20_month = calc_ma(close, 20)

    # 维度1: 均线位置(10分)
    current_price = close.iloc[-1]
    ma20_val = ma20_month.iloc[-1] if not pd.isna(ma20_month.iloc[-1]) else current_price

    if ma20_val > 0:
        ratio = current_price / ma20_val
        if ratio > 1.05:
            ma_score = 10
            ma_desc = f"股价({current_price:.2f})站稳20月线({ma20_val:.2f})上方，中长期强势"
        elif ratio > 0.95:
            ma_score = 5
            ma_desc = f"股价({current_price:.2f})在20月线({ma20_val:.2f})附近争夺，方向待定"
        else:
            ma_score = -10
            ma_desc = f"股价({current_price:.2f})在20月线({ma20_val:.2f})下方，中长期弱势"
    else:
        ma_score = 5
        ma_desc = "20月线数据不可用"

    # 维度2: 趋势排列(5分) - 月线均线排列
    ma5_month = calc_ma(close, 5)
    ma10_month = calc_ma(close, 10)
    ma20_val_full = ma20_month

    if len(close) >= 20:
        ma5_last = ma5_month.iloc[-1]
        ma10_last = ma10_month.iloc[-1]
        ma20_last = ma20_val_full.iloc[-1]

        if not any(pd.isna([ma5_last, ma10_last, ma20_last])) and ma20_last > 0:
            if ma5_last > ma10_last > ma20_last:
                trend_score = 5
                trend_desc = "月线均线多头排列，中期趋势向上"
            elif ma5_last < ma10_last < ma20_last:
                trend_score = -5
                trend_desc = "月线均线空头排列，中期趋势向下"
            else:
                trend_score = 0
                trend_desc = "月线均线纠缠震荡，无明确趋势"
        else:
            trend_score = 0
            trend_desc = "均线数据不完整"
    else:
        trend_score = 0
        trend_desc = "月K数据点不足判断排列"

    # 维度3: 形态信号(5分)
    if len(close) >= 6:
        recent = close.tail(6).values

        # 检查底部形态(W底/头肩底/连续上涨)
        # 简单判断：最近几根月线是否出现底部反转
        lows = []
        highs = []
        for i in range(1, len(recent) - 1):
            if recent[i] < recent[i-1] and recent[i] < recent[i+1]:
                lows.append(recent[i])
            if recent[i] > recent[i-1] and recent[i] > recent[i+1]:
                highs.append(recent[i])

        # W底判断：两个低点相近，然后反弹
        if len(lows) >= 2:
            if abs(lows[-1] - lows[-2]) / max(abs(lows[-1]), abs(lows[-2])) < 0.05:
                if recent[-1] > recent[-2]:  # 最近一根上涨
                    form_score = 5
                    form_desc = "月线出现W底形态，中长期底部信号"
                else:
                    form_score = 0
                    form_desc = "可能有底部形态但未确认"
            else:
                form_score = 0
                form_desc = "无明显底部或顶部形态"
        elif len(highs) >= 2:
            # 顶部形态判断
            if abs(highs[-1] - highs[-2]) / max(abs(highs[-1]), abs(highs[-2])) < 0.05:
                if recent[-1] < recent[-2]:
                    form_score = -5
                    form_desc = "月线出现双顶形态，警惕顶部风险"
                else:
                    form_score = 0
                    form_desc = "可能有顶部形态但未确认"
            else:
                form_score = 0
                form_desc = "无明显底部或顶部形态"
        else:
            # 简单判断：连续三根上涨月线(红三兵)
            if len(recent) >= 3 and recent[-1] > recent[-2] > recent[-3]:
                form_score = 5
                form_desc = "连续三根上涨月线(红三兵)，底部信号"
            # 连续三根下跌
            elif len(recent) >= 3 and recent[-1] < recent[-2] < recent[-3]:
                form_score = -5
                form_desc = "连续三根下跌月线(三只乌鸦)，顶部信号"
            else:
                form_score = 0
                form_desc = "无明显形态信号"
    else:
        form_score = 0
        form_desc = "月K数据点不足判断形态"

    total = ma_score + trend_score + form_score

    return {
        "均线位置": {"得分": ma_score, "说明": ma_desc},
        "趋势排列": {"得分": trend_score, "说明": trend_desc},
        "形态信号": {"得分": form_score, "说明": form_desc},
        "总分": total,
    }


def score_weekk(df_daily: pd.DataFrame) -> dict:
    """
    周K评分（满分30分）
    维度1: 均线位置(10分) - 股价与20周线关系
    维度2: MACD信号(10分) - 周线MACD状态
    维度3: 量价配合(10分) - 放量突破/缩量震荡/放量下跌
    """
    # 重采样为周K
    df_weekly = resample_to_period(df_daily, "weekly")

    if len(df_weekly) < 5:
        return {
            "均线位置": {"得分": 5, "说明": "周K数据不足"},
            "MACD信号": {"得分": 3, "说明": "周K数据不足"},
            "量价配合": {"得分": 3, "说明": "周K数据不足"},
            "总分": 11,
        }

    close = df_weekly["收盘"]
    volume = df_weekly["成交量"]
    ma20_week = calc_ma(close, 20)

    # 维度1: 均线位置(10分)
    current_price = close.iloc[-1]
    ma20_val = ma20_week.iloc[-1] if not pd.isna(ma20_week.iloc[-1]) else current_price

    if ma20_val > 0:
        ratio = current_price / ma20_val
        if ratio > 1.05:
            ma_score = 10
            ma_desc = f"股价({current_price:.2f})站稳20周线({ma20_val:.2f})上方，中期趋势健康"
        elif ratio > 0.95:
            ma_score = 5
            ma_desc = f"股价({current_price:.2f})在20周线({ma20_val:.2f})附近，中期震荡"
        else:
            ma_score = -10
            ma_desc = f"股价({current_price:.2f})跌破20周线({ma20_val:.2f})，中期趋势走弱"
    else:
        ma_score = 5
        ma_desc = "20周线数据不可用"

    # 维度2: MACD信号(10分)
    macd_line, signal_line, histogram = calc_macd(close)

    macd_val = macd_line.iloc[-1]
    signal_val = signal_line.iloc[-1]
    macd_prev = macd_line.iloc[-2] if len(macd_line) > 1 else macd_val
    signal_prev = signal_line.iloc[-2] if len(signal_line) > 1 else signal_val

    # 判断MACD状态
    if pd.isna(macd_val) or pd.isna(signal_val):
        macd_score = 3
        macd_desc = "MACD数据不可用"
    elif macd_val > 0 and signal_val > 0:
        if macd_val > signal_val and macd_prev <= signal_prev:
            macd_score = 10
            macd_desc = "周线MACD零轴上方金叉，中期上涨动能强"
        elif macd_val > signal_val:
            macd_score = 8
            macd_desc = "周线MACD在零轴上方运行，中期偏多"
        else:
            macd_score = 5
            macd_desc = "周线MACD在零轴上方但死叉，注意回调"
    elif macd_val > -1 and macd_val < 1:  # 零轴附近(允许一定误差)
        if abs(macd_val - signal_val) < 0.5:
            macd_score = 3
            macd_desc = "周线MACD在零轴附近缠绕，中期动能不明"
        else:
            macd_score = 3
            macd_desc = "周线MACD接近零轴，方向待定"
    else:
        if macd_val < signal_val and macd_prev >= signal_prev:
            macd_score = -10
            macd_desc = "周线MACD零轴下方死叉，中期下跌动能强"
        elif macd_val < signal_val:
            macd_score = -7
            macd_desc = "周线MACD在零轴下方运行，中期偏空"
        else:
            macd_score = -5
            macd_desc = "周线MACD零轴下方但金叉，可能止跌"

    # 维度3: 量价配合(10分)
    # 比较近期成交量与历史平均成交量
    avg_volume = volume.rolling(window=20, min_periods=1).mean()
    current_volume = volume.iloc[-1]
    avg_vol_val = avg_volume.iloc[-1] if not pd.isna(avg_volume.iloc[-1]) else current_volume

    # 判断量价关系
    price_change = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] if len(close) > 1 else 0
    volume_ratio = current_volume / avg_vol_val if avg_vol_val > 0 else 1

    if volume_ratio > 1.5 and price_change > 0.02:
        # 放量上涨突破
        vol_score = 10
        vol_desc = f"放量({volume_ratio:.1f}倍)突破平台压力位，突破有效"
    elif volume_ratio < 0.8 and abs(price_change) < 0.01:
        # 缩量震荡
        vol_score = 3
        vol_desc = f"缩量({volume_ratio:.1f}倍)震荡整理，多空平衡"
    elif volume_ratio > 1.5 and price_change < -0.02:
        # 放量下跌
        vol_score = -10
        vol_desc = f"放量({volume_ratio:.1f}倍)下跌破位，资金出逃"
    elif volume_ratio > 1.2 and price_change > 0:
        vol_score = 5
        vol_desc = "温和放量上涨，量价配合良好"
    elif volume_ratio > 1.2 and price_change < 0:
        vol_score = -3
        vol_desc = "放量下跌，需警惕"
    else:
        vol_score = 3
        vol_desc = "量价关系中性"

    total = ma_score + macd_score + vol_score

    return {
        "均线位置": {"得分": ma_score, "说明": ma_desc},
        "MACD信号": {"得分": macd_score, "说明": macd_desc},
        "量价配合": {"得分": vol_score, "说明": vol_desc},
        "总分": total,
    }


def score_dayk(df_daily: pd.DataFrame) -> dict:
    """
    日K评分（满分25分）
    维度1: 均线位置(10分) - 股价与20日线关系
    维度2: 指标信号(10分) - MACD/KDJ金叉死叉
    维度3: K线形态(5分) - 看涨/看跌K线组合
    """
    if len(df_daily) < 5:
        return {
            "均线位置": {"得分": 5, "说明": "日K数据不足"},
            "指标信号": {"得分": 3, "说明": "日K数据不足"},
            "K线形态": {"得分": 0, "说明": "日K数据不足"},
            "总分": 8,
        }

    close = df_daily["收盘"]
    df_copy = df_daily.copy()
    ma20_day = calc_ma(close, 20)

    # 维度1: 均线位置(10分)
    current_price = close.iloc[-1]
    ma20_val = ma20_day.iloc[-1] if not pd.isna(ma20_day.iloc[-1]) else current_price

    if ma20_val > 0:
        ratio = current_price / ma20_val
        if ratio > 1.03:
            ma_score = 10
            ma_desc = f"股价({current_price:.2f})站稳20日线({ma20_val:.2f})上方，短期强势"
        elif ratio > 0.97:
            ma_score = 5
            ma_desc = f"股价({current_price:.2f})在20日线({ma20_val:.2f})附近，短期方向不明"
        else:
            ma_score = -10
            ma_desc = f"股价({current_price:.2f})跌破20日线({ma20_val:.2f})，短期弱势"
    else:
        ma_score = 5
        ma_desc = "20日线数据不可用"

    # 维度2: 指标信号(10分) - MACD + KDJ
    macd_line, signal_line, histogram = calc_macd(close)
    k_line, d_line, j_line = calc_kdj(df_copy)

    # 判断MACD信号
    macd_val = macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else 0
    signal_val = signal_line.iloc[-1] if not pd.isna(signal_line.iloc[-1]) else 0
    macd_prev = macd_line.iloc[-2] if len(macd_line) > 1 else macd_val
    signal_prev = signal_line.iloc[-2] if len(signal_line) > 1 else signal_val

    # 判断KDJ信号
    k_val = k_line.iloc[-1] if not pd.isna(k_line.iloc[-1]) else 50
    d_val = d_line.iloc[-1] if not pd.isna(d_line.iloc[-1]) else 50
    j_val = j_line.iloc[-1] if not pd.isna(j_line.iloc[-1]) else 50
    k_prev = k_line.iloc[-2] if len(k_line) > 1 else k_val
    d_prev = d_line.iloc[-2] if len(d_line) > 1 else d_val

    # 综合MACD和KDJ信号
    macd_signal = 0
    kdj_signal = 0

    # MACD信号判断
    if macd_val > 0 and signal_val > 0:
        if macd_val > signal_val and macd_prev <= signal_prev:
            macd_signal = 10  # 零轴上方金叉
        elif macd_val > signal_val:
            macd_signal = 5   # 零轴上方多头
        else:
            macd_signal = -5  # 零轴上方死叉
    elif macd_val < 0 and signal_val < 0:
        if macd_val < signal_val and macd_prev >= signal_prev:
            macd_signal = -10  # 零轴下方死叉
        elif macd_val < signal_val:
            macd_signal = -5   # 零轴下方空头
        else:
            macd_signal = 3    # 零轴下方金叉，可能止跌
    else:
        # 零轴附近
        if abs(macd_val - signal_val) < 0.3:
            macd_signal = 3  # 中性徘徊
        elif macd_val > signal_val:
            macd_signal = 5
        else:
            macd_signal = 0

    # KDJ信号判断
    if k_val > 20 and d_val > 20:
        if k_val > d_val and k_prev <= d_prev:
            kdj_signal = 8  # KDJ低位金叉
        elif k_val > d_val:
            kdj_signal = 3  # KDJ多头
        else:
            kdj_signal = -5 # KDJ死叉
    else:
        if k_val > d_val and k_prev <= d_prev:
            kdj_signal = 10  # KDJ超卖区金叉，强信号
        elif k_val > 80 and d_val > 80:
            kdj_signal = -5  # KDJ超买区
        else:
            kdj_signal = 0

    # 综合指标信号
    if macd_signal >= 8 and kdj_signal >= 8:
        indicator_score = 10
        indicator_desc = "MACD和KDJ同时发出金叉信号，短线入场高胜率"
    elif macd_signal >= 5 and kdj_signal >= 3:
        indicator_score = 8
        indicator_desc = "MACD和KDJ均偏多，短线看涨"
    elif macd_signal <= -5 and kdj_signal <= -5:
        indicator_score = -10
        indicator_desc = "MACD和KDJ同时发出死叉信号，短线离场"
    elif macd_signal <= -5 or kdj_signal <= -5:
        indicator_score = -5
        indicator_desc = "MACD或KDJ发出死叉信号，短线注意风险"
    elif macd_signal >= 5 or kdj_signal >= 5:
        indicator_score = 5
        indicator_desc = "MACD或KDJ偏多，短线暂可持有"
    else:
        indicator_score = 3
        indicator_desc = "指标中性徘徊，短线方向不明"

    # 维度3: K线形态(5分)
    if len(df_copy) >= 3:
        recent = df_copy.tail(3)[["开盘", "收盘", "最高", "最低"]].values

        # 看涨形态判断
        # 1. 阳包阴：今天收盘价>开盘价，且今天实体包住昨天实体
        if len(recent) >= 2:
            today = recent[-1]
            yesterday = recent[-2]

            today_green = today[1] > today[0]  # 阳线
            yesterday_red = yesterday[1] < yesterday[0]  # 阴线

            if today_green and yesterday_red:
                today_body_high = max(today[0], today[1])
                today_body_low = min(today[0], today[1])
                yesterday_body_high = max(yesterday[0], yesterday[1])
                yesterday_body_low = min(yesterday[0], yesterday[1])

                if today_body_high >= yesterday_body_high and today_body_low <= yesterday_body_low:
                    kline_score = 5
                    kline_desc = "出现阳包阴形态，短期反转信号"
                elif today[1] > yesterday[0] and today[0] < yesterday[1]:
                    kline_score = 3
                    kline_desc = "出现部分包阴形态，偏多信号"
                else:
                    kline_score = 0
                    kline_desc = "无明显看涨或看跌形态"
            else:
                # 锤子线判断
                if today[3] < today[1] * 0.98 and today[2] - today[1] < today[1] * 0.01 and today[1] - today[3] > today[2] - today[3] * 2:
                    kline_score = 5
                    kline_desc = "出现锤子线形态，短期企稳信号"
                else:
                    kline_score = 0
                    kline_desc = "无明显看涨或看跌形态"
        else:
            kline_score = 0
            kline_desc = "数据不足"
    else:
        kline_score = 0
        kline_desc = "数据不足"

    total = ma_score + indicator_score + kline_score

    return {
        "均线位置": {"得分": ma_score, "说明": ma_desc},
        "指标信号": {"得分": indicator_score, "说明": indicator_desc},
        "K线形态": {"得分": kline_score, "说明": kline_desc},
        "总分": total,
    }


# ============================================================
# 综合评分与信号判断
# ============================================================

def calc_composite_score(year_score: int, month_score: int, week_score: int, day_score: int) -> float:
    """计算综合评分（加权）"""
    # 各周期得分需要归一化到0-100范围
    # 年K满分25分，月K满分20分，周K满分30分，日K满分25分
    year_norm = (year_score / 25) * 100
    month_norm = (month_score / 20) * 100
    week_norm = (week_score / 30) * 100
    day_norm = (day_score / 25) * 100

    composite = (year_norm * WEIGHTS["year"] +
                 month_norm * WEIGHTS["month"] +
                 week_norm * WEIGHTS["week"] +
                 day_norm * WEIGHTS["day"])

    return round(composite, 1)


def get_signal(score: float) -> dict:
    """根据综合评分获取操作信号"""
    if score >= 85:
        return {
            "信号": "强烈买入",
            "仓位": "重仓(70%-100%)",
            "策略": "多周期共振向上，果断建仓或加仓",
            "风控": "设置移动止损，跟踪止盈"
        }
    elif score >= 70:
        return {
            "信号": "买入",
            "仓位": "中等仓位(40%-70%)",
            "策略": "大周期向好，中期趋势健康，可参与波段",
            "风控": "止损设在周线支撑位下方"
        }
    elif score >= 55:
        return {
            "信号": "观望",
            "仓位": "轻仓试错(10%-30%)",
            "策略": "部分周期信号矛盾，小仓位试探",
            "风控": "严格止损，破位即离场"
        }
    elif score >= 40:
        return {
            "信号": "卖出",
            "仓位": "减仓(降至10%以下)",
            "策略": "多周期发出走弱信号，应减仓或离场",
            "风控": "不要抄底，等待明确信号"
        }
    else:
        return {
            "信号": "强烈卖出",
            "仓位": "清仓(0%)",
            "策略": "多周期共振向下，全面回避",
            "风控": "不逆势抄底，等待底部信号"
        }


# ============================================================
# HTML报告生成
# ============================================================

def generate_html_report(results: list) -> str:
    """生成自包含HTML报告，可部署到GitHub Pages"""
    now = datetime.now()
    successful = [r for r in results if "error" not in r]
    error_list = [r for r in results if "error" in r]

    signal_order = ["强烈买入", "买入", "观望", "卖出", "强烈卖出"]
    signal_counts = {s: 0 for s in signal_order}
    for r in successful:
        if r["signal"] in signal_counts:
            signal_counts[r["signal"]] += 1

    sorted_results = sorted(successful, key=lambda x: x.get("composite_score", 0), reverse=True)

    sig_map = {
        "强烈买入": "badge-strong-buy",
        "买入": "badge-buy",
        "观望": "badge-watch",
        "卖出": "badge-sell",
        "强烈卖出": "badge-strong-sell",
    }

    css = """
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f1117; color: #e0e0e0; line-height: 1.6; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #1a1a2e, #16213e); color: #fff; padding: 30px; border-radius: 12px; margin-bottom: 20px; border: 1px solid #2a2a4e; }
        .header h1 { font-size: 24px; margin-bottom: 8px; }
        .header .meta { font-size: 14px; opacity: 0.7; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 25px; }
        .card { background: #1a1a2e; border: 1px solid #2a2a4e; border-radius: 10px; padding: 20px; text-align: center; }
        .card .label { font-size: 13px; color: #888; margin-bottom: 8px; }
        .card .value { font-size: 28px; font-weight: 700; }
        table { width: 100%; border-collapse: collapse; background: #1a1a2e; border-radius: 10px; overflow: hidden; border: 1px solid #2a2a4e; margin-bottom: 25px; }
        th { background: #16213e; padding: 14px 12px; text-align: left; font-size: 13px; color: #aaa; border-bottom: 1px solid #2a2a4e; white-space: nowrap; }
        td { padding: 14px 12px; border-bottom: 1px solid #2a2a4e; font-size: 14px; }
        tr:last-child td { border-bottom: none; }
        tr:hover td { background: #1f1f3a; }
        .score { font-weight: 700; font-size: 16px; }
        .score-high { color: #ff4d4f; }
        .score-mid { color: #faad14; }
        .score-low { color: #52c41a; }
        .badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; white-space: nowrap; }
        .badge-strong-buy { background: rgba(255,77,79,0.15); color: #ff4d4f; border: 1px solid rgba(255,77,79,0.3); }
        .badge-buy { background: rgba(255,120,120,0.1); color: #ff7875; border: 1px solid rgba(255,120,120,0.2); }
        .badge-watch { background: rgba(250,173,20,0.1); color: #faad14; border: 1px solid rgba(250,173,20,0.2); }
        .badge-sell { background: rgba(82,196,26,0.1); color: #73d13d; border: 1px solid rgba(82,196,26,0.2); }
        .badge-strong-sell { background: rgba(82,196,26,0.15); color: #52c41a; border: 1px solid rgba(82,196,26,0.3); }
        .score-bar { display: inline-block; width: 50px; height: 6px; background: #2a2a4e; border-radius: 3px; overflow: hidden; vertical-align: middle; margin-right: 6px; }
        .score-bar-fill { height: 100%; border-radius: 3px; }
        .detail-card { background: #1a1a2e; border: 1px solid #2a2a4e; border-radius: 10px; padding: 20px; margin-bottom: 15px; }
        .detail-card h3 { color: #e0e0e0; margin-bottom: 15px; font-size: 18px; }
        .detail-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
        .period-box { background: #16213e; border: 1px solid #2a2a4e; border-radius: 8px; padding: 15px; }
        .period-box h4 { color: #aaa; font-size: 13px; margin-bottom: 8px; }
        .period-box .period-score { font-size: 22px; font-weight: 700; margin-bottom: 12px; }
        .period-box .item { margin-bottom: 8px; font-size: 13px; }
        .period-box .item-name { color: #888; }
        .period-box .item-score { font-weight: 600; }
        .period-box .item-desc { color: #666; font-size: 12px; margin-top: 2px; }
        .strategy-box { margin-top: 15px; padding-top: 15px; border-top: 1px solid #2a2a4e; }
        .strategy-box div { margin-bottom: 6px; font-size: 13px; }
        .strategy-box .label { color: #888; display: inline-block; width: 60px; }
        .footer { text-align: center; padding: 30px 20px; color: #555; font-size: 13px; }
        @media (max-width: 768px) {
            .detail-grid { grid-template-columns: repeat(2, 1fr); }
            .container { padding: 10px; }
            th, td { padding: 8px 6px; font-size: 12px; }
            .header { padding: 20px; }
            .header h1 { font-size: 20px; }
        }
    """

    def make_bar(score, max_score):
        pct = min(abs(score) / max_score * 100, 100)
        color = "#ff4d4f" if score > 0 else ("#52c41a" if score < 0 else "#555")
        return f'<span class="score-bar"><span class="score-bar-fill" style="width:{pct:.0f}%;background:{color}"></span></span>'

    cards = []
    for sig in signal_order:
        count = signal_counts[sig]
        cards.append(f'<div class="card"><div class="label">{sig}</div><div class="value">{count}</div></div>')
    cards_html = '\n'.join(cards)

    rows = []
    for i, r in enumerate(sorted_results, 1):
        composite = r["composite_score"]
        comp_cls = "score-high" if composite >= 70 else ("score-mid" if composite >= 55 else "score-low")
        badge_cls = sig_map.get(r["signal"], "badge-watch")
        rows.append(f"""<tr>
<td>{i}</td>
<td>{r['code']}</td>
<td>{r['name']}</td>
<td>{r['latest_price']:.3f}</td>
<td>{make_bar(r['year_score'], 25)}{r['year_score']}/25</td>
<td>{make_bar(r['month_score'], 20)}{r['month_score']}/20</td>
<td>{make_bar(r['week_score'], 30)}{r['week_score']}/30</td>
<td>{make_bar(r['day_score'], 25)}{r['day_score']}/25</td>
<td><span class="score {comp_cls}">{composite}</span></td>
<td><span class="badge {badge_cls}">{r['signal']}</span></td>
<td>{r['position']}</td>
</tr>""")
    rows_html = '\n'.join(rows)

    details = []
    for r in sorted_results:
        badge_cls = sig_map.get(r["signal"], "badge-watch")
        comp_cls = "score-high" if r["composite_score"] >= 70 else ("score-mid" if r["composite_score"] >= 55 else "score-low")
        periods = [
            ("年K", r.get("year_detail", {}), 25),
            ("月K", r.get("month_detail", {}), 20),
            ("周K", r.get("week_detail", {}), 30),
            ("日K", r.get("day_detail", {}), 25),
        ]
        period_boxes = []
        for pname, detail, max_s in periods:
            total = detail.get("总分", 0)
            items = []
            for key, val in detail.items():
                if key in ("总分", "detail"):
                    continue
                if isinstance(val, dict):
                    sc = val.get("得分", 0)
                    desc = val.get("说明", "")
                    sc_color = "#ff4d4f" if sc > 0 else ("#52c41a" if sc < 0 else "#888")
                    items.append(f'<div class="item"><span class="item-name">{key}</span> <span class="item-score" style="color:{sc_color}">{sc}</span><div class="item-desc">{desc}</div></div>')
            period_boxes.append(f'<div class="period-box"><h4>{pname}</h4><div class="period-score">{total}/{max_s}</div>{"".join(items)}</div>')
        details.append(f"""<div class="detail-card">
<h3>{r['name']} ({r['code']}) <span class="badge {badge_cls}">{r['signal']}</span> <span class="score {comp_cls}">{r['composite_score']}分</span></h3>
<div class="detail-grid">
{"".join(period_boxes)}
</div>
<div class="strategy-box">
<div><span class="label">仓位</span>{r['position']}</div>
<div><span class="label">策略</span>{r['strategy']}</div>
<div><span class="label">风控</span>{r['risk_control']}</div>
</div>
</div>""")
    details_html = '\n'.join(details)

    error_cards = []
    for r in error_list:
        error_cards.append(f'<div class="detail-card"><h3>{r["name"]} ({r["code"]}) - 数据获取失败</h3></div>')
    error_html = '\n'.join(error_cards)

    meta_parts = [f"生成时间: {now.strftime('%Y-%m-%d %H:%M:%S')}", f"分析: {len(results)}只", f"成功: {len(successful)}只"]
    if error_list:
        meta_parts.append(f"错误: {len(error_list)}只")
    meta_str = " | ".join(meta_parts)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ETF四周期量化打分报告 - {now.strftime('%Y-%m-%d')}</title>
<style>
{css}
</style>
</head>
<body>
<div class="container">
<div class="header">
<h1>ETF四周期量化打分报告</h1>
<div class="meta">{meta_str}</div>
</div>
<div class="summary">
{cards_html}
</div>
<table>
<thead>
<tr><th>#</th><th>代码</th><th>名称</th><th>收盘价</th><th>年K</th><th>月K</th><th>周K</th><th>日K</th><th>综合</th><th>信号</th><th>仓位</th></tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>
<div class="detail">
{details_html}
{error_html}
</div>
<div class="footer">
ETF四周期量化打分系统 | 年K定战略 → 月K定方向 → 周K定节奏 → 日K找买点<br>
本报告仅供学习参考，不构成投资建议 | 生成于 {now.strftime('%Y-%m-%d %H:%M:%S')}
</div>
</div>
</body>
</html>"""

    html_file = os.path.join(OUTPUT_DIR, f"ETF打分报告_{now.strftime('%Y%m%d_%H%M%S')}.html")
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html)
    index_file = os.path.join(OUTPUT_DIR, "index.html")
    with open(index_file, "w", encoding="utf-8") as f:
        f.write(html)
    return html_file


# ============================================================
# 手机推送通知
# ============================================================

def send_notification(results: list):
    """发送打分摘要到手机，支持Server酱/PushPlus/Bark/ntfy/自定义webhook"""
    import urllib.request
    import urllib.parse

    push_type = os.environ.get("PUSH_TYPE", "").lower()
    push_key = os.environ.get("PUSH_KEY", "")
    push_url = os.environ.get("PUSH_URL", "")

    if not push_type and not push_url:
        print("未配置推送服务，跳过手机通知。")
        print("如需推送，设置环境变量 PUSH_TYPE (serverchan/pushplus/bark/ntfy) 和 PUSH_KEY/PUSH_URL")
        return

    successful = [r for r in results if "error" not in r]
    buy = [r for r in successful if r["signal"] in ["强烈买入", "买入"]]
    sell = [r for r in successful if r["signal"] in ["强烈卖出", "卖出"]]

    title = f"ETF打分 {datetime.now().strftime('%m-%d')} | 买{len(buy)} 卖{len(sell)}"

    lines = [
        "ETF四周期量化打分报告",
        f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"ETF: {len(successful)}/{len(results)}只成功",
        "",
        "==信号分布=="
    ]
    sig_order = ["强烈买入", "买入", "观望", "卖出", "强烈卖出"]
    sig_counts = {}
    for r in successful:
        sig_counts[r["signal"]] = sig_counts.get(r["signal"], 0) + 1
    for s in sig_order:
        if s in sig_counts:
            lines.append(f"  {s}: {sig_counts[s]}只")

    if buy:
        lines.append("\n==买入信号==")
        for r in sorted(buy, key=lambda x: -x["composite_score"]):
            lines.append(f"  {r['name']}({r['code']}): {r['composite_score']}分 {r['signal']}")

    if sell:
        lines.append("\n==卖出信号==")
        for r in sorted(sell, key=lambda x: x["composite_score"]):
            lines.append(f"  {r['name']}({r['code']}): {r['composite_score']}分 {r['signal']}")

    content = "\n".join(lines)

    try:
        if push_type == "serverchan" and push_key:
            url = f"https://sctapi.ftqq.com/{push_key}.send"
            data = urllib.parse.urlencode({"title": title, "desp": content}).encode()
            req = urllib.request.Request(url, data=data, method="POST")
            urllib.request.urlopen(req, timeout=10)
            print("Server酱推送成功")
        elif push_type == "pushplus" and push_key:
            url = "http://www.pushplus.plus/send"
            data = json.dumps({"token": push_key, "title": title, "content": content, "template": "txt"}).encode()
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=10)
            print("PushPlus推送成功")
        elif push_type == "bark" and push_key:
            bark_server = os.environ.get("BARK_SERVER", "https://api.day.app")
            url = f"{bark_server}/{push_key}"
            data = json.dumps({"title": title, "body": content, "group": "ETF"}).encode()
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=10)
            print("Bark推送成功")
        elif push_type == "ntfy" and push_url:
            data = content.encode()
            req = urllib.request.Request(push_url, data=data, headers={"Title": title, "Tags": "chart,money"}, method="POST")
            urllib.request.urlopen(req, timeout=10)
            print("ntfy推送成功")
        elif push_url:
            data = json.dumps({"title": title, "content": content}).encode()
            req = urllib.request.Request(push_url, data=data, headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=10)
            print("Webhook推送成功")
    except Exception as e:
        print(f"推送失败: {e}")


# ============================================================
# 主函数
# ============================================================

def run_scoring():
    """执行打分流程"""
    print("=" * 80)
    print("ETF四周期量化打分系统")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    results = []

    for etf in ETF_LIST:
        code = etf["code"]
        name = etf["name"]

        print(f"\n{'─' * 60}")
        print(f"正在分析: {name} ({code})")

        # 获取数据
        df_daily = fetch_etf_data(code, days=365 * 5)

        if len(df_daily) < 10:
            print(f"  [错误] {name} 数据不足({len(df_daily)}条)，跳过")
            results.append({
                "code": code,
                "name": name,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "error": "数据不足",
                "daily_data_points": len(df_daily)
            })
            continue

        print(f"  日K数据: {len(df_daily)}条")

        # 四周期评分
        print("  正在计算年K评分...")
        year_result = score_yeark(df_daily)
        print(f"    年K得分: {year_result['总分']}/25")

        print("  正在计算月K评分...")
        month_result = score_monthk(df_daily)
        print(f"    月K得分: {month_result['总分']}/20")

        print("  正在计算周K评分...")
        week_result = score_weekk(df_daily)
        print(f"    周K得分: {week_result['总分']}/30")

        print("  正在计算日K评分...")
        day_result = score_dayk(df_daily)
        print(f"    日K得分: {day_result['总分']}/25")

        # 计算综合评分
        composite = calc_composite_score(
            year_result["总分"],
            month_result["总分"],
            week_result["总分"],
            day_result["总分"]
        )

        # 获取信号
        signal = get_signal(composite)

        # 获取最新收盘价
        latest_price = df_daily["收盘"].iloc[-1]

        result = {
            "code": code,
            "name": name,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "latest_price": float(latest_price),
            "year_score": year_result["总分"],
            "month_score": month_result["总分"],
            "week_score": week_result["总分"],
            "day_score": day_result["总分"],
            "composite_score": composite,
            "signal": signal["信号"],
            "position": signal["仓位"],
            "strategy": signal["策略"],
            "risk_control": signal["风控"],
            "year_detail": year_result,
            "month_detail": month_result,
            "week_detail": week_result,
            "day_detail": day_result,
        }

        results.append(result)

        # 打印结果
        print(f"\n  【{name}({code})】打分结果")
        print(f"  最新收盘价: {latest_price:.2f}")
        print(f"  年K得分: {year_result['总分']}/25  |  月K得分: {month_result['总分']}/20  |  "
              f"周K得分: {week_result['总分']}/30  |  日K得分: {day_result['总分']}/25")
        print(f"  综合评分: {composite}/100")
        print(f"  操作信号: 【{signal['信号']}】")
        print(f"  仓位建议: {signal['仓位']}")
        print(f"  操作策略: {signal['策略']}")
        print(f"  风控要点: {signal['风控']}")

    # 打印汇总表格
    print(f"\n{'=' * 80}")
    print("四周期量化打分汇总")
    print(f"日期: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"{'=' * 80}")

    # 创建汇总表格
    table_data = []
    for r in results:
        if "error" not in r:
            table_data.append({
                "代码": r["code"],
                "名称": r["name"],
                "收盘价": r["latest_price"],
                "年K": r["year_score"],
                "月K": r["month_score"],
                "周K": r["week_score"],
                "日K": r["day_score"],
                "综合评分": r["composite_score"],
                "信号": r["signal"],
                "仓位": r["position"],
            })
        else:
            table_data.append({
                "代码": r["code"],
                "名称": r["name"],
                "收盘价": "-",
                "年K": "-",
                "月K": "-",
                "周K": "-",
                "日K": "-",
                "综合评分": "-",
                "信号": "数据错误",
                "仓位": "-",
            })

    df_table = pd.DataFrame(table_data)

    # 按综合评分排序（有效的评分）
    valid_mask = df_table["综合评分"] != "-"
    if valid_mask.any():
        df_table_sorted = df_table.copy()
        df_table_sorted.loc[valid_mask, "综合评分_num"] = pd.to_numeric(df_table_sorted.loc[valid_mask, "综合评分"], errors="coerce")
        df_table_sorted = df_table_sorted.sort_values("综合评分_num", ascending=False, na_position="last")
        # 去掉辅助列
        df_table_sorted = df_table_sorted.drop(columns=["综合评分_num"], errors="ignore")
    else:
        df_table_sorted = df_table

    # 打印表格
    print("\n")
    print(df_table_sorted.to_string(index=False))

    # 保存结果到JSON文件
    output_file = os.path.join(OUTPUT_DIR, f"打分结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    # 去除detail字段中的复杂对象以便JSON序列化
    results_serializable = []
    for r in results:
        r_copy = {k: v for k, v in r.items() if k not in ["year_detail", "month_detail", "week_detail", "day_detail"]}
        results_serializable.append(r_copy)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_serializable, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果已保存至: {output_file}")

    # 同时保存完整结果（含detail）
    output_file_full = os.path.join(OUTPUT_DIR, f"打分结果_完整_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file_full, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"完整结果(含评分明细)已保存至: {output_file_full}")

    # 生成HTML报告
    html_path = generate_html_report(results)
    print(f"HTML报告已保存至: {html_path}")
    print(f"GitHub Pages首页已更新: {os.path.join(OUTPUT_DIR, 'index.html')}")

    # 发送手机推送通知
    send_notification(results)

    return results


# ============================================================
# 定时执行功能（可选）
# ============================================================

def setup_scheduler():
    """
    设置定时执行（可选功能）
    使用方法: 将本脚本添加到系统crontab或Windows任务计划程序
    例如: 每个交易日9:30执行
    """
    print("\n定时执行配置说明:")
    print("Linux/Mac: crontab -e")
    print("  添加: 30 9 * * 1-5 cd /path/to/script && python ETF四周期量化打分系统.py")
    print("\nWindows: 任务计划程序")
    print("  创建基本任务 -> 触发器: 每天9:30 -> 操作: 启动程序 -> 程序: python.exe -> 参数: 脚本路径")
    print("\n或使用Python schedule库:")
    print("  pip install schedule")
    print("  在脚本末尾添加:")
    print("  import schedule")
    print("  schedule.every().day.at('09:30').do(run_scoring)")
    print("  while True: schedule.run_pending()")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ETF四周期量化打分系统")
    parser.add_argument("--schedule", action="store_true", help="显示定时执行配置说明")
    parser.add_argument("--code", type=str, help="只分析指定ETF代码(可多次指定)")
    parser.add_argument("--cache", action="store_true", help="使用缓存数据(加快运行速度)")

    args = parser.parse_args()

    if args.schedule:
        setup_scheduler()
        sys.exit(0)

    # 如果指定了特定代码，过滤ETF列表
    if args.code:
        codes = [c.strip() for c in args.code.split(",")]
        ETF_LIST = [etf for etf in ETF_LIST if etf["code"] in codes]
        print(f"正在分析指定ETF: {', '.join(codes)}")

    try:
        results = run_scoring()

        # 打印总结
        successful = [r for r in results if "error" not in r]
        print(f"\n{'=' * 80}")
        print(f"分析完成! 成功: {len(successful)}/{len(results)}只ETF")

        if successful:
            # 统计信号分布
            signal_counts = {}
            for r in successful:
                sig = r["signal"]
                signal_counts[sig] = signal_counts.get(sig, 0) + 1

            print("\n信号分布:")
            for sig, count in sorted(signal_counts.items(), key=lambda x: -x[1]):
                print(f"  {sig}: {count}只")

            # 打印建议买入的ETF
            buy_signals = [r for r in successful if r["signal"] in ["强烈买入", "买入"]]
            if buy_signals:
                print(f"\n【建议关注】以下ETF发出买入信号:")
                for r in buy_signals:
                    print(f"  {r['name']}({r['code']}): 综合{r['composite_score']}分 - {r['signal']}")
    except KeyboardInterrupt:
        print("\n用户中断执行")
        sys.exit(0)
    except Exception as e:
        print(f"\n[错误] 程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
