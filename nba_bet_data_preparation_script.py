import pandas as pd

df = pd.read_csv('nba_bet_stat.csv')
df["net_value"] = df.apply(lambda x: x['prize'] - x['invest'] if x['outcome'] == 1 else x['invest']*-1, axis=1)
df['balance'] = df['net_value'].rolling(window=100,min_periods=0).sum()
df['balance'] = df['balance'].map(lambda x: int(x))
df.to_csv('nba_bet_stat_prepared.csv')