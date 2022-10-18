import statsmodels.api as sm
import pandas_datareader.data as pdr
import yfinance as yf

import matplotlib.pyplot as plt
plt.style.use('seaborn')
from datetime import datetime

#start = datetime(2017, 9, 14)
start = datetime(2020,1,1)
end = datetime(2022, 8, 8)

#riskfree_rate = pdr.DataReader('TB4WK', 'fred', start, end)
market = yf.Ticker('SPY').history(start=start, end=end)
stock = yf.Ticker('AAPL').history(start=start, end=end)

#riskfree_rate = pdr.DataReader('TB4WK', 'fred', start, end)
#riskfree_scaling = 28 #Need to scale the risk free rate by its maturity to get the daily risk free rate
#riskfree_rate = riskfree_rate['TB4WK'].dropna()/riskfree_scaling
#riskfree_rate.plot(), plt.title('Daily Risk Free Rate'), plt.show()
riskfree_rate = pdr.DataReader('DTB1YR', 'fred', start, end)
riskfree_rate = riskfree_rate.dropna()
daily_riskfree_rate = (1 + riskfree_rate['DTB1YR']) ** (1/360) - 1 #Daily rate based on 360 days for the calendar year
plt.hist(daily_riskfree_rate), plt.title('Bond 1Y USA Mercato secondario'), plt.xlabel('Ritorno in percentuale'), plt.ylabel('Frequenza'), plt.show()
print("Distribuzione statististica ritorno Bond 1Y USA:\n{}".format(daily_riskfree_rate.describe()))

market_return = market['Close'].pct_change(1)*100
market_return = market_return.dropna()
plt.hist(market_return), plt.title('Distribuzione ritorno giornaliero SPY'), plt.xlabel('Ritorno percentuale giornaliero'), plt.ylabel('Frequenza'), plt.show()
print("Distribuzione statistica ritorno SPY:\n{}".format(market_return.describe()))

stock_return = stock['Close'].pct_change(1)*100
stock_return = stock_return.dropna()
plt.hist(stock_return), plt.title('Distribuzione ritorno giornaliero AAPL'), plt.xlabel('Ritorno percentuale giornaliero'), plt.ylabel('Frequenza'), plt.show()
print("Distribuzione statistica ritorno AAPL:\n{}".format(stock_return.describe()))

#AAPL's Market Model
#y = stock_return - riskfree_rate.mean()
#x = market_return - riskfree_rate.mean()
y = stock_return - riskfree_rate['DTB1YR'].mean()
x = market_return - riskfree_rate['DTB1YR'].mean()
plt.scatter(x,y)
# mambojambo
x = sm.add_constant(x) # aggiunge una costante di valore 1

market_model = sm.OLS(y, x).fit()
plt.plot(x, x*market_model.params[1]+market_model.params[0])
plt.title('Market Model of AAPL'), plt.xlabel('SPY Ritorno giornaliero'), plt.ylabel('AAPL Ritorno giornaliero'), plt.show();
print("In accordo col Modello di Mercato di AAPL, l'azione ha Alpha di {0}% e Beta di {1}".format(round(market_model.params[0],2), round(market_model.params[1],2)))

print("Sommario del modello di mercato di AAPL:\n{}".format(market_model.summary()))

print(plt.plot(market_model.resid),plt.show());