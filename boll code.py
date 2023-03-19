import pandas_ta as ta
import numpy as np
import pandas as pd

# 获取数据
daily_300 = pd.read_csv("000300.csv", header=0, names=[ "trade_date",  "high", "low","close", "pct_chg"]).sort_values('trade_date').reset_index()


# 计算上中下轨道
def calc_BBand(mkt_data, n=20, m=2):
    close = mkt_data['close']
    high = mkt_data['high']
    low = mkt_data['low']
    """ 指标计算 """
    TP = (high+low+close)/3
    MID = TP.rolling(n).mean()
    BANDUP = MID + m*TP.rolling(n).std()
    BANDDOWN = MID - m*TP.rolling(n).std()
    """ 结果赋值 """
    mkt_data['MID'] = MID
    mkt_data['BANDUP'] = BANDUP
    mkt_data['BANDDOWN'] = BANDDOWN
    return mkt_data

daily_300 = calc_BBand(daily_300)

# 计算信号
def calc_signal(mkt_data):
    BANDUP = mkt_data['BANDUP']
    BANDDOWN = mkt_data['BANDDOWN']
    close = mkt_data['close']
    """ 计算信号 """
    signals = []
    for bup, bdown, close, pre_bup, pre_bdown, pre_close in zip(BANDUP, BANDDOWN, close, BANDUP.shift(1), BANDDOWN.shift(1),close.shift(1)):
        signal = None
        if pre_close<pre_bup and close>=bup:
            signal = 1
        elif pre_close>=pre_bdown and close<bdown:
            signal = -1
        signals.append(signal)
    mkt_data['signal'] = signals
    return mkt_data

daily_300 = calc_signal(daily_300)

# 计算持仓
def calc_position(mkt_data):
    mkt_data['position'] = mkt_data['signal'].fillna(method='ffill').shift(1).fillna(0)
    return mkt_data

daily_300 = calc_position(daily_300)

# 计算结果
def statistic_performance(mkt_data, r0=0.03, data_period=1440):
    position = mkt_data['position']

    """      序列型特征 
        hold_r :      持仓收益
        hold_win :    持仓胜负
        hold_cumu_r : 累计持仓收益
        drawdown :    回撤
        ex_hold_r :   超额收益
    """
    hold_r = mkt_data['pct_chg'] / 100 * position
    hold_win = hold_r > 0
    hold_cumu_r = (1 + hold_r).cumprod() - 1
    drawdown = (hold_cumu_r.cummax() - hold_cumu_r) / (1 + hold_cumu_r).cummax()
    ex_hold_r = hold_r - r0 / (250 * 1440 / data_period)

    mkt_data['hold_r'] = hold_r
    mkt_data['hold_win'] = hold_win
    mkt_data['hold_cumu_r'] = hold_cumu_r
    mkt_data['drawdown'] = drawdown
    mkt_data['ex_hold_r'] = ex_hold_r

    """       数值型特征 
        v_hold_cumu_r：         累计持仓收益
        v_pos_hold_times：      多仓开仓次数
        v_pos_hold_win_times：  多仓开仓盈利次数
        v_pos_hold_period：     多仓持有周期数
        v_pos_hold_win_period： 多仓持有盈利周期数
        v_neg_hold_times：      空仓开仓次数
        v_neg_hold_win_times：  空仓开仓盈利次数
        v_neg_hold_period：     空仓持有盈利周期数
        v_neg_hold_win_period： 空仓开仓次数
        v_hold_period：         持仓周期数（最后一笔未平仓订单也算）
        v_hold_win_period：     持仓盈利周期数（最后一笔未平仓订单也算）
        v_max_dd：              最大回撤
        v_annual_std：          年化标准差
        v_annual_ret：          年化收益
        v_sharpe：              夏普率
    """
    v_hold_cumu_r = hold_cumu_r.tolist()[-1]

    v_pos_hold_times = 0
    v_pos_hold_win_times = 0
    v_pos_hold_period = 0
    v_pos_hold_win_period = 0
    v_neg_hold_times = 0
    v_neg_hold_win_times = 0
    v_neg_hold_period = 0
    v_neg_hold_win_period = 0
    for w, r, pre_pos, pos in zip(hold_win, hold_r, position.shift(1), position):
        # 有换仓（先结算上一次持仓，再初始化本次持仓）
        if pre_pos != pos:
            # 判断pre_pos非空：若为空则是循环的第一次，此时无需结算，直接初始化持仓即可
            if pre_pos == pre_pos:
                # 结算上一次持仓
                if pre_pos > 0:
                    v_pos_hold_times += 1
                    v_pos_hold_period += tmp_hold_period
                    v_pos_hold_win_period += tmp_hold_win_period
                    if tmp_hold_r > 0:
                        v_pos_hold_win_times += 1
                elif pre_pos < 0:
                    v_neg_hold_times += 1
                    v_neg_hold_period += tmp_hold_period
                    v_neg_hold_win_period += tmp_hold_win_period
                    if tmp_hold_r > 0:
                        v_neg_hold_win_times += 1
            # 初始化本次持仓
            tmp_hold_r = r
            tmp_hold_period = 0
            tmp_hold_win_period = 0
        else:  # 未换仓
            if abs(pos) > 0:
                tmp_hold_period += 1
                if r > 0:
                    tmp_hold_win_period += 1
                if abs(r) > 0:
                    tmp_hold_r = (1 + tmp_hold_r) * (1 + r) - 1

    v_hold_period = (abs(position) > 0).sum()
    v_hold_win_period = (hold_r > 0).sum()
    v_max_dd = drawdown.max()
    v_annual_ret = pow(1 + v_hold_cumu_r,
                       1 / (data_period / 1440 * len(mkt_data) / 250)) - 1
    v_annual_std = ex_hold_r.std() * np.sqrt(250 * 1440 / data_period)
    v_sharpe = v_annual_ret / v_annual_std

    """ 生成Performance DataFrame """
    performance_cols = ['累计收益',
                        '多仓次数', '多仓胜率', '多仓平均持有期',
                        '空仓次数', '空仓胜率', '空仓平均持有期',
                        '日胜率', '最大回撤', '年化收益/最大回撤',
                        '年化收益', '年化标准差', '年化夏普'
                        ]
    performance_values = ['{:.2%}'.format(v_hold_cumu_r),
                          v_pos_hold_times, '{:.2%}'.format(v_pos_hold_win_times / v_pos_hold_times),
                          '{:.2f}'.format(v_pos_hold_period / v_pos_hold_times),
                          v_neg_hold_times, '{:.2%}'.format(v_neg_hold_win_times / v_neg_hold_times),
                          '{:.2f}'.format(v_neg_hold_period / v_neg_hold_times),
                          '{:.2%}'.format(v_hold_win_period / v_hold_period),
                          '{:.2%}'.format(v_max_dd),
                          '{:.2f}'.format(v_annual_ret / v_max_dd),
                          '{:.2%}'.format(v_annual_ret),
                          '{:.2%}'.format(v_annual_std),
                          '{:.2f}'.format(v_sharpe)
                          ]
    performance_df = pd.DataFrame(performance_values, index=performance_cols)
    return mkt_data, performance_df

# 可视化
import datetime
from bokeh.plotting import figure, show, output_notebook
from bokeh.layouts import column, row, gridplot, layout
from bokeh.models import Span
output_notebook()


def visualize_performance(mkt_data):
    mkt_data['trade_datetime'] = mkt_data['trade_date'].apply(lambda x: datetime.datetime.strptime(str(x), '%Y%m%d'))
    dt = mkt_data['trade_datetime']

    f1 = figure(height=300, width=700,
                sizing_mode='stretch_width',
                title='Target Trend',
                x_axis_type='datetime',
                x_axis_label="trade_datetime", y_axis_label="close")
    f2 = figure(height=200, sizing_mode='stretch_width',
                title='Position',
                x_axis_label="trade_datetime", y_axis_label="position",
                x_axis_type='datetime',
                x_range=f1.x_range)
    f3 = figure(height=200, sizing_mode='stretch_width',
                title='Return',
                x_axis_type='datetime',
                x_range=f1.x_range)
    f4 = figure(height=200, sizing_mode='stretch_width',
                title='Drawdown',
                x_axis_type='datetime',
                x_range=f1.x_range)

    # 绘制行情
    close = mkt_data['close']
    cumu_hold_close = (mkt_data['hold_cumu_r'] + 1)
    f1.line(dt, close / close.tolist()[0], line_width=1)
    f1.line(dt, cumu_hold_close, line_width=1, color='red')

    # 绘制指标
    #     indi = figure(height=200, sizing_mode='stretch_width',
    #                   title='KDJ',
    #                   x_axis_type='datetime',
    #                   x_range=f1.x_range
    #                  )

    # 绘制仓位
    position = mkt_data['position']
    f2.step(dt, position)

    # 绘制收益
    hold_r = mkt_data['hold_r']
    f3.vbar(x=dt, top=hold_r)

    # 绘制回撤
    drawdown = mkt_data['drawdown']
    f4.line(dt, -drawdown, line_width=1)

    # p = column(f1,f2,f3,f4)
    p = gridplot([[f1],
                  # [indi],
                  [f2],
                  [f3],
                  [f4]
                  ])
    show(p)

# A:研报原时间段（2005-09-01至2012-09-01） N=20
# 评价和展现
#result_daily_300, performance_df = statistic_performance(daily_300)
result_daily_300, performance_df = statistic_performance(daily_300[daily_300['trade_date'].apply(lambda x: x>=20050901 and x<=20120315)])
#result_daily_300, performance_df = statistic_performance(daily_300[daily_300['trade_date'].apply(lambda x: x>=20120315)])

result_daily_300
visualize_performance(result_daily_300)
print(performance_df)

# B:研报原时间段（2005-09-01至2012-09-01） N=14
daily_300 = calc_BBand(daily_300, n=14)
daily_300 = calc_signal(daily_300)
daily_300 = calc_position(daily_300)

# 评价和展现
#result_daily_300, performance_df = statistic_performance(daily_300)
result_daily_300, performance_df = statistic_performance(daily_300[daily_300['trade_date'].apply(lambda x: x>=20050901 and x<=20120315)])
#result_daily_300, performance_df = statistic_performance(daily_300[daily_300['trade_date'].apply(lambda x: x>='20120315')])

visualize_performance(result_daily_300)
print(performance_df)

# C:研报后至今（2012-09-01至2023-03-17） N=20
daily_300 = calc_BBand(daily_300, n=14)
daily_300 = calc_signal(daily_300)
daily_300 = calc_position(daily_300)

# 评价和展现
#result_daily_300, performance_df = statistic_performance(daily_300)
#result_daily_300, performance_df = statistic_performance(daily_300[daily_300['trade_date'].apply(lambda x: x>=20050901 and x<=20120315)])
result_daily_300, performance_df = statistic_performance(daily_300[daily_300['trade_date'].apply(lambda x: x>=20120315)])

visualize_performance(result_daily_300)
print(performance_df)

# D:全局（2005-01至今） N=20
daily_300 = calc_BBand(daily_300, n=20)
daily_300 = calc_signal(daily_300)
daily_300 = calc_position(daily_300)

# 评价和展现
result_daily_300, performance_df = statistic_performance(daily_300)
#result_daily_300, performance_df = statistic_performance(daily_300[daily_300['trade_date'].apply(lambda x: x>=20050901 and x<=20120315)])
#result_daily_300, performance_df = statistic_performance(daily_300[daily_300['trade_date'].apply(lambda x: x>=20120315)])

visualize_performance(result_daily_300)
print(performance_df)
