import statsmodels.api as sm
import pandas_datareader.data as pdr
import yfinance as yf

import matplotlib.pyplot as plt
plt.style.use('seaborn')
from datetime import datetime


start = datetime(2020, 1, 1)
end = datetime(2022, 8, 5)

riskfree_rate = pdr.DataReader('TB4WK', 'fred', start, end)
benchmark = yf.Ticker('SPY').history(start=start, end=end)
stock = yf.Ticker('AAPL').history(start=start, end=end)

print('Benchmark', benchmark)
print('Stock', stock)

riskfree_rate = pdr.DataReader('DTB1YR', 'fred', start, end)
riskfree_rate = riskfree_rate.dropna()
daily_riskfree_rate = (1 +  riskfree_rate['DTB1YR']) ** (1/360) - 1
plt.hist(daily_riskfree_rate), plt.title('Bond 1Y USA:\n{}'.format(daily_riskfree_rate.describe()))