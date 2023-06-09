
Assignment1
==========

**参考研报：**<br>
20120509-光大证券-技术指标系列(四)：布林带的“追涨杀跌”

**1. 回测数据来源**<br>
万得 Wind

**2. 回测样本**<br>
沪深300指数数据（2005-01-01至2023-03-17）

**3. 策略**<br>
计算基于Tp值的布林带上中下轨道 

首先复现研报中的20日参数策略：<br>
中轨线=N日tp的移动平均线 均值拟合：N=20 20天移动平均<br> 
上轨线=中轨线+M倍的标准差（中轨线标准差） M=2 2倍标准差 <br>
下轨线=中轨线-M倍的标准差（中轨线标准差）M=2 2倍标准差

再复现研报中的14日参数策略，当时条件下是最优：<br>
中轨线=N日tp的移动平均线 均值拟合：N=14 14天移动平均 <br>
上轨线=中轨线+M倍的标准差（中轨线标准差） M=2 2倍标准差 <br>
下轨线=中轨线-M倍的标准差（中轨线标准差）M=2 2倍标准差

**4. 策略信号**<br>
当收盘价'Close'上穿上轨线，买入<br>
当收盘价'Close'下穿下轨线，卖空<br>
计算策略信号并赋值给持仓，并计算持仓<br>

**4. 计算指标并进行回测**<br>

**计算以下指标变化情况：**<br>
持仓收益<br>
持仓胜负<br>
累计持仓收益<br>
回撤<br>
超额收益

**并输出结果：**<br>
累计收益 <br>
多仓次数 <br>
多仓胜率<br>
多仓平均持有期<br>
空仓次数<br>
空仓胜率<br>
空仓平均持有期<br>
日胜率<br>
最大回撤<br>
年化收益/最大回撤<br>
年化收益<br>
年化标准差<br>
年化夏普<br> 

**5. 多种情况的结果**(红色为沪深三百，蓝色为布林带策略）

A:研报原时间段（2005-09-01至2012-09-01） N=20
![image](https://github.com/algo23-Shuairui/Assignment1/blob/main/IMG/A1.png)
![image](https://github.com/algo23-Shuairui/Assignment1/blob/main/IMG/A2.png)
![image](https://github.com/algo23-Shuairui/Assignment1/blob/main/IMG/A3.png)

B:研报原时间段（2005-09-01至2012-09-01） N=14
![image](https://github.com/algo23-Shuairui/Assignment1/blob/main/IMG/B1.png)
![image](https://github.com/algo23-Shuairui/Assignment1/blob/main/IMG/B2.png)
![image](https://github.com/algo23-Shuairui/Assignment1/blob/main/IMG/B3.png)

C:研报后至今（2012-09-01至2023-03-17） N=20
![image](https://github.com/algo23-Shuairui/Assignment1/blob/main/IMG/C1.png)
![image](https://github.com/algo23-Shuairui/Assignment1/blob/main/IMG/C2.png)
![image](https://github.com/algo23-Shuairui/Assignment1/blob/main/IMG/C3.png)

D:全局（2005-01至今） N=20
![image](https://github.com/algo23-Shuairui/Assignment1/blob/main/IMG/D1.png)
![image](https://github.com/algo23-Shuairui/Assignment1/blob/main/IMG/D2.png)
![image](https://github.com/algo23-Shuairui/Assignment1/blob/main/IMG/D3.png)



