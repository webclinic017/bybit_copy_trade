from COPYTRADE import *

pd.set_option('display.max_columns', None)
# get all trader name and leaderMark
df = get_all_leader_mark()
print(df)
# get report of historical trades
df_filter = df[df['cumHistoryTransactionsCount'] > 50]
df_hist = HistoricalTrades.get_historical_trades_report(df_filter['leaderMark'])
df_open = OpenTrades.get_open_trades_report(df_filter['leaderMark'])
df = df_filter.merge(df_open, on="leaderMark")
df = df.merge(df_hist, on="leaderMark")
df = df.set_index("leaderUserName")

op = df.loc["Banpodolphin", "open_trade_instance"]
print(op.df_trades)

bt = df.loc["Banpodolphin", "historical_trade_instance"]
print(bt)
bt.capital.plot()
bt.report.print()
tryy = bt.capital
print(type(tryy))
tryy.to_csv("trade_record.csv")