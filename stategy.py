import os
import pandas as pd
import numpy as np
import talib as ta
import catboost as cb
#from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

#%%
#计算最大回撤
def Max_drawdown(net_value):
    end = np.argmax((np.maximum.accumulate(net_value)-net_value)/np.maximum.accumulate(net_value))
    if end == 0:
        return 0
    start = np.argmax(net_value[:end])
    return (net_value[start] - net_value[end]) / (net_value[start])

#%%
#计算夏普比率
def Sharpe_ratio(return_series,freq="30minute"):
    if freq=="month":
        n = 12        
    elif freq=="week":
        n = 52
    elif freq=="daily":
        n = 365 
    elif freq=='hour':
        n = 365*24
    elif freq=="minute":
        n = 365*24*60 
    elif freq=="30minute":
        n = 365*24*2
    ex_return = return_series - 0.04/n
    return (np.sqrt(n) * ex_return.mean() / ex_return.std())

#%%
###计算策略表现
def Strategy_performance(df, benchmark="equal_weight",freq="30minute"):  
    import statsmodels.api as sm
    if freq=="month":
        n = 12        
    elif freq=="week":
        n = 52
    elif freq=="daily":
        n = 365 
    elif freq=='hour':
        n = 365*24
    elif freq=="minute":
        n = 365*24*60 
    elif freq=="30minute":
        n = 365*24*2
    
    symbols = output_final.columns.tolist()
    output = pd.DataFrame(index=symbols,
                      columns=['Sharpe ratio', 'Max draw down', 
                               'Annual return', 'Annual voltility',
                               'Alpha', 'Beta',
                               'Info_ratio', 'Relative max draw down'])
    return_df = df.pct_change().fillna(0)
    ###计算回撤和相对最大回撤
    for symbol in symbols:
        output.loc[symbol,'Max draw down'] = Max_drawdown(df[symbol].tolist())
        if symbol==benchmark:
            continue
        else:
            output.loc[symbol,'Relative max draw down'] = Max_drawdown(
                (df[symbol].sub(df[benchmark],axis=0).add(1)).tolist())
            
            Y = return_df[symbol].to_numpy().reshape(-1,1)            
            X = return_df[benchmark].to_numpy().reshape(-1,1)
            X = sm.add_constant(X)
            results = sm.OLS(Y, X).fit()
            output.loc[symbol,['Alpha', 'Beta']] = results.params

    ###计算夏普比率
    output['Sharpe ratio'] = Sharpe_ratio(return_df, freq=freq)
    ###计算信息比率
    output['Info_ratio'] = np.sqrt(n) * return_df.sub(
    return_df[benchmark],axis=0).mean() / return_df.sub(return_df[benchmark],axis=0).std()  
    
    ###计算年化收益率和年化波动率 
    output['Annual return'] = (df.iloc[-1] ** (n / len(df))) -1
    output['Annual voltility'] = return_df.std() * np.sqrt(n)
    

     
    return output.T
#%%
###计算feature
def Get_feature(df):
    df['Pre_high'] = df['high'].shift(1)
    df['Pre_low'] = df['low'].shift(1)
    df['Pre_close'] = df['close'].shift(1)
    df['Pre_volume'] = df['volume'].shift(1)
    
    feature_names = [
        'Volume_Change', 
        'Market_info', 
        'Market_info_2', 
        'Market_info_3',
        'Market_info_4',
        'MA5', 'MA120', 'MA20', 'RSI', 'Corr', 'SAR', 'ADX',
        'ATR', 'OBV'
    ]
    df['Market_info'] = np.where(df.Pre_close > df.Pre_close.rolling(120).median(),1,0)
    df['Market_info_2'] = np.where(df.Pre_close < df.Pre_close.rolling(120).max() * 0.9,1,0)
    df['Market_info_3'] = np.where(df.Pre_close > df.Pre_close.rolling(120).min() * 1.1,1,0)
    df['Market_info_4'] = np.where(df.Pre_close.rolling(120).max() / df.Pre_close.rolling(120).min() > 1.2,1,0)
    df['Volume_Change'] = (df.volume / df.Pre_volume) - 1
    df['MA20'] = df.Pre_close.rolling(window=20).mean()
    df['MA5'] = ta.MA(df.Pre_close, 5)
    df['MA120'] = ta.MA(df.Pre_close, 120)
    df['RSI'] = ta.RSI(df.Pre_close, timeperiod=14)
    df['Corr'] = df['MA20'].rolling(window=20).corr(df['Pre_close'])
    df['SAR'] = ta.SAR(np.array(df['Pre_high']), np.array(df['Pre_low']), 0.2,
                       0.2)
    df['ADX'] = ta.ADX(np.array(df['Pre_high']),
                       np.array(df['Pre_low']),
                       np.array(df['Pre_close']),
                       timeperiod=14)
    df['ATR'] = ta.ATR(np.array(df['Pre_high']),
                       np.array(df['Pre_low']),
                       np.array(df['Pre_close']),
                       timeperiod=14)
    df['OBV'] = ta.OBV(df.Pre_close, df.Pre_volume)
    
    
    df = df.loc[:, feature_names + ['symbol', 'Return','True_return']].replace(
        np.inf, 10000).replace(
            -np.inf,-10000)
    df = df.dropna()
    return df


#%%
# #计算组合收益率分析:年化收益率、收益波动率、夏普比率、最大回撤
# def strategy_performance (nav_df):
    
#     ##part1:根据回测净值计算相关指标的数据准备（日度数据）
#     nav_next = nav_df.shift(1)
#     return_df = nav_df/nav_next - 1  #计算净值变化率，即为日收益率,包含组合与基准
#     return_df = return_df.dropna()  #在计算净值变化率时，首日得到的是缺失值，需将其删除
    
#     analyze=pd.DataFrame()  #用于存储计算的指标
    
#     ##part2:计算年化收益率
#     cum_return = np.exp(np.log1p(return_df).cumsum())-1   #计算整个回测期内的复利收益率
#     annual_return_df = (1+cum_return)**(365/len(return_df))-1  #计算年化收益率
#     analyze['annual_return'] = annual_return_df.iloc[-1]  #将年化收益率的Series赋值给数据框
    
#     #part3:计算收益波动率（以年为基准）
#     analyze['return_volatility']=return_df.std() * np.sqrt(365) #return中的收益率为日收益率，所以计算波动率转化为年时，需要乘上np.sqrt(252)
    
#     #part4:计算夏普比率
#     risk_free = 0.03
#     return_risk_adj = return_df - risk_free
#     analyze['sharpe_ratio'] = return_risk_adj.mean()/np.std(return_risk_adj, ddof=1)
    
#     #prat5:计算最大回撤
#     cumulative = np.exp(np.log1p(return_df).cumsum())*100  #计算累计收益率
#     max_return = cumulative.cummax()  #计算累计收益率的在各个时间段的最大值
#     analyze['max_drawdown'] = cumulative.sub(max_return).div(max_return).min()  #最大回撤一般小于0，越小，说明离1越远，各时间点与最大收益的差距越大
    
#     #part6:计算相对指标
#     analyze['relative_return'] = analyze['annual_return']-analyze.loc['benchmark','annual_return'] #计算相对年化波动率
#     analyze['relative_volatility'] = analyze['return_volatility']-analyze.loc['benchmark','return_volatility'] #计算相对波动
#     analyze['relative_drawdown'] = analyze['max_drawdown']-analyze.loc['benchmark','max_drawdown'] #计算相对最大回撤
    
#     #part6:计算信息比率
#     return_diff = return_df.sub(return_df['benchmark'],axis=0).std()*np.sqrt(365)  #计算策略与基准日收益差值的年化标准差
#     analyze['info_ratio'] = analyze['relative_return'].div(return_diff)

#     return analyze.T 

#%%
os.chdir('C:/Users/wang/Documents/GitHub/ML_project')
digital_coins = ['bch','btc','eos','eth','ltc']
list_store = []
for digital_coin in digital_coins:
    df = pd.read_csv('csv/{}Spot.csv'.format(digital_coin),usecols=[3,4,5,6,7,8,11])
    df['symbol'] =  digital_coin
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    # data['datetime'] = data['datetime'].apply(lambda x: 
    #     datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
    df.set_index('datetime',inplace=True)
    df = df.dropna()
    #df.index = pd.DatetimeIndex(df.index)
    df = df.resample('15min').aggregate({'close':'last',
                                      'high':'max',
                                      'low':'min',
                                      'open':'first',
                                      'volume':'sum',
                                      'symbol':'last'
                                      })
    df = df.dropna()
    df['Return'] = df['close'].pct_change()
    df['True_return'] = df['close'] / df['open'] - 1
    
    #分别计算feature
    df = Get_feature(df)
    list_store.append(df)





df = pd.concat(list_store)
del list_store
datelist = np.unique(df.index.date).tolist()

df = df.reset_index().set_index(['datetime','symbol']).sort_index()
df.loc[:pd.to_datetime('2019-7-31')]
df.loc[:datelist[100]]
cv = TimeSeriesSplit(n_splits=5,max_train_size=100)


#%%
feature_names = [
        'Volume_Change', 
        'Market_info', 
        'Market_info_2', 
        'Market_info_3',
        'Market_info_4',
        'MA5', 'MA120', 'MA20', 'RSI', 'Corr', 'SAR', 'ADX',
        'ATR', 'OBV'
    ]
df_temp = df.loc[:,feature_names + ['Return', 'True_return']].copy()
features = df_temp.loc[:, feature_names]
split = int(0.6 * len(datelist))
# step = 15
# split += step
bench = df_temp['True_return'].iloc[:split].quantile(q=0.75)


df_temp['Class'] = np.where(df_temp['True_return'] -0.0005 > 
                            min(0.002,round(bench,4)), 1, 0)
targets = df_temp.loc[:, 'Class']

feature_num = len(feature_names)
x_train, x_test, y_train, y_test = np.array(features[:datelist[split]]).reshape(
    -1, feature_num), np.array(features[datelist[split]:]).reshape(
        -1, feature_num),np.array(targets[:datelist[split]]), np.array(targets[datelist[split]:])
        
ss = StandardScaler()
ss.fit(x_train)
x_train = pd.DataFrame(ss.transform(x_train))
x_test = pd.DataFrame(ss.transform(x_test))


###GridSearch效果不好，容易产生calss imbalance问题
# =============================================================================
# model = RandomForestClassifier(random_state=10)
# tuned_parameter ={
#     "n_estimators": [10, 20, 30, 50],
#     "max_depth": [4, 6, 8, 10],
#     "criterion": ["gini", "entropy"]
# }
# cur_cv = TimeSeriesSplit(n_splits=10).split(x_train)
# score = 'f1_micro'
# 
# clf = GridSearchCV(model,
#                    tuned_parameter,
#                    cv=cur_cv,
#                    scoring='%s' % score)
# =============================================================================

#%%
clf = RandomForestClassifier(n_estimators=100,criterion='gini',random_state=10,n_jobs=-1)

#clf = XGBClassifier(random_state=10)

# =============================================================================
# ##catboost想要有效果,保证depth至少12，itration越大越好，learning_rate保持0.1
# ##运行时长大约1min以上才能有效果，因此放弃
# clf = cb.CatBoostClassifier(iterations=100, depth=16, learning_rate=0.1, loss_function='Logloss',
#                               logging_level='Verbose')
# weight_array = np.where(y_train == 1, 1 - y_train.sum() / len(y_train),
#                                 y_train.sum() / len(y_train))
# 
# clf.fit(x_train, y_train,plot=True,
#         sample_weight=weight_array
#         )
# =============================================================================


clf.fit(x_train, y_train)

y_pred = np.where(clf.predict(x_test) > 0, 1, 0)
print(classification_report(y_pred,y_test))
y_pred_proba = clf.predict_proba(x_test).reshape(-1,2)[:,1]



#np.concatenate()

#%%
output = df[datelist[split]:].loc[:,['Return', 'True_return']].copy()
hold = np.where((y_pred_proba<0.55), 0, y_pred_proba)
hold = np.where((hold>=0.5) & (hold<0.55), 0.1, hold)
hold = np.where((hold>=0.55) & (hold<0.75), 0.75,hold)
hold = np.where(hold>0.75, 1, hold)
output['Hold'] = hold
#权重,持仓进行组合的平均化，非持仓的为0
weight_real = output.unstack().loc[:, ['Hold']].replace(
    0,np.nan).count(axis=1).rdiv(1).replace(np.inf,0)
output['Hold'] = output['Hold'] * weight_real
###交易费用
fee = output.unstack().loc[:, ['Hold']].diff().fillna(output.unstack().loc[:, ['Hold']])
fee = fee.stack(1)
fee = (abs(fee) * 0.0005 ).rename(columns={"Hold":"Fee"})

###滑点
###买时滑点
slip_buy = output.unstack().loc[:, ['Hold']].diff().fillna(output.unstack().loc[:, ['Hold']]).clip(0,1)
slip_buy = slip_buy.stack(1)
slip_buy = pd.concat([slip_buy,
                  (df['Return'][datelist[split]:] - 
                   df['True_return'][datelist[split]:] ).rename("Slip")],
                 axis=1)
slip_buy = (slip_buy['Hold'] * slip_buy['Slip']).rename("Slip")
###卖时滑点
slip_sell = output.unstack().loc[:, ['Hold']].diff().fillna(output.unstack().loc[:, ['Hold']]).clip(-1,0).abs()
slip_sell = slip_sell.stack(1)
slip_sell = pd.concat([slip_sell,
                  ((df['Return'][datelist[split]:]+1) / 
                   (df['True_return'][datelist[split]:]+1) -1).rename("Slip")],
                 axis=1)#开盘价除以前一日收盘价
slip_sell = (slip_sell['Hold'] * slip_sell['Slip'] *(-1)).rename("Slip")

slip = slip_buy + slip_sell
del slip_buy, slip_sell

output = pd.concat([output,fee,slip], axis=1)

#%%

#output.reset_index().groupby('datetime').aggregate({'Return':'sum'})

output_final = pd.DataFrame(index=output.reset_index()['datetime'].unique())
output['Strategy_return'] = (output['Hold'] * (output['Return']) - output['Slip'] - output['Fee'])

output_final['Strategy'] = output.reset_index().groupby('datetime').aggregate(
    {'Strategy_return':'sum'})
output_final['Strategy'] = (output_final['Strategy'] + 1).cumprod()
output_final['equal_weight'] = output.reset_index().groupby('datetime').aggregate(
    {'Return':'sum'})/len(digital_coins)
output_final['equal_weight'] = (output_final['equal_weight'] + 1).cumprod()

fig = output_final.plot(y=['Strategy','equal_weight'],
                legend=True,colormap='gist_rainbow',
                title='{}_of_{}'.format(str(clf)[0:2],' '.join(digital_coins))
                )
fig = fig.get_figure()
fig.savefig('netinv.jpg')

print(Strategy_performance(output_final))

# print(Sharpe_ratio((hold * (df['True_return'].iloc[datelist[split]:]-
#                               0.0005)),
#                     '30minute'))
# print(Max_drawdown(output['Strategy'].tolist()))
# print("Max draw down: {:.2f}".format(Max_drawdown(output_final['Strategy'].tolist())))
# print("Sharpe ratio: {:.2f}".format(Sharpe_ratio(output_final['Strategy'].pct_change(),'30minute')))
#strategy_performance (output.rename(columns={'Buy & hold':'benchmark'}))
