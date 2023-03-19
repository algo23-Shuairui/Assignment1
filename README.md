
Assignment1
==========

**参考研报：**
20120509-光大证券-技术指标系列(四)：布林带的“追涨杀跌”

**1. 回测数据来源**
万得 Wind

**2. 回测样本**
沪深300指数数据（2005-01-01至2023-03-17）

**3. 策略**
计算基于Tp值的布林带上中下轨道 

首先复现研报中的20日参数策略：<br>
中轨线=N日tp的移动平均线 均值拟合：N=20 20天移动平均<br> 
上轨线=中轨线+M倍的标准差（中轨线标准差） M=2 2倍标准差 <br>
下轨线=中轨线-M倍的标准差（中轨线标准差）M=2 2倍标准差

再复现研报中的14日参数策略，当时条件下是最优：<br>
中轨线=N日tp的移动平均线 均值拟合：N=14 14天移动平均 <br>
上轨线=中轨线+M倍的标准差（中轨线标准差） M=2 2倍标准差 <br>
下轨线=中轨线-M倍的标准差（中轨线标准差）M=2 2倍标准差

**4. 策略信号**
当收盘价'Close'上穿上轨线，买入<br>
当收盘价'Close'下穿下轨线，卖空<br>
计算策略信号并赋值给持仓，并计算持仓<br>

**4. 计算指标并进行回测**

**计算以下指标变化情况：**
持仓收益
持仓胜负
累计持仓收益
回撤
超额收益

**并输出结果：**
累计收益 
多仓次数 
多仓胜率
多仓平均持有期
空仓次数
空仓胜率
空仓平均持有期
日胜率
最大回撤
年化收益/最大回撤
年化收益
年化标准差
年化夏普 

**5. 多条件对比**

A条件:N=20与N=14条件 <br>
B条件：研报原时间段（2005-09-01至2012-09-01）与研报后至今（2012-09-01至2023-03-17）

AB组合生成4种结果作为本次作业的结果进行输出。
