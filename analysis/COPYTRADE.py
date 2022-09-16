import datetime as dt
import time


import requests
from fake_useragent import UserAgent
from pybit import usdt_perpetual
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn")

# # suppress FutureWarning: Behavior when concatenating bool-dtype and numeric-type
# # I cannot figure out why it happen,
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def get_all_leader_mark() -> pd.DataFrame:
    """get info of leaders

    Returns:
        pd.DataFrame: DataFrame of all traders info
    """
    current_time = dt.datetime.utcnow()
    current_time_unix = time.mktime(current_time.timetuple())

    url = "https://api2.bybit.com/fapi/beehive/public/v1/common/ranking"
    params = {
        "timeStamp": current_time_unix,
        "page": 1,
        "pageSize": 10000,
        "sortType": "LEADER_SORT_TYPE_SORT_DAY_FOLLOWER_PROFIT",
    }
    ua = UserAgent().safari
    headers = {"user-agent": ua}
    response = requests.get(url=url, params=params, headers=headers)
    data = response.json()["result"]["data"]

    df_all_traders = pd.DataFrame(data)
    df_all_traders = df_all_traders[
        [
            "leaderUserName",
            "leaderMark",
            "locateDays",
            "cumHistoryTransactionsCount",
            "cumFollowerCount",
            "currentFollowerCount",
            "maxFollowerCount",
        ]
    ]

    df_all_traders[
        [
            "cumHistoryTransactionsCount",
            "cumFollowerCount",
            "locateDays",
            "currentFollowerCount",
            "maxFollowerCount",
        ]
    ] = df_all_traders[
        [
            "cumHistoryTransactionsCount",
            "cumFollowerCount",
            "locateDays",
            "currentFollowerCount",
            "maxFollowerCount",
        ]
    ].apply(
        pd.to_numeric
    )
    df_all_traders["available"] = np.where(
        df_all_traders["currentFollowerCount"] / df_all_traders["maxFollowerCount"]
        == 1,
        False,
        True,
    )

    return df_all_traders


class HistoricalTrades:

    __list_of_report = []

    def __init__(self) -> None:
        pass

    @classmethod
    def get_historical_trades_report(
        cls, list_of_leaders: pd.Series, time_interval="5min", transaction_fee=0.0008
    ) -> pd.DataFrame:
        """query and analysis historical trades of traders

        Args:
            list_of_leaders (pd.Series): a list/series of leaderMark
            time_interval (str, optional): _description_. Defaults to "5min". e.g. 1min,5min,1d,3d

        Returns:
            pd.DataFrame: DataFrame of statisical report of all traders
        """
        cls.__list_of_report = []
        for i, leadermark in enumerate(list_of_leaders):
            trade_records = cls.__query_trader_trade_record(
                leaderMark=leadermark, time_interval=time_interval
            )
            bt = AnalysisTradeRecord(
                trade_records, leaderMark=leadermark, transaction_fee=transaction_fee
            )
            cls.__list_of_report.append(bt)
            print(
                f"Historical Trades Scrapping Process: {i+1}/{len(list_of_leaders)}",
                end="\r",
            )
        reports = list(map(lambda x: x.report.__dict__, cls.__list_of_report))
        reports = pd.DataFrame(reports)
        reports.loc[:, "historical_trade_instance"] = cls.__list_of_report

        return reports

    @staticmethod
    def __query_trader_trade_record(leaderMark, time_interval="5min") -> pd.DataFrame:
        current_time = dt.datetime.utcnow()
        current_time_unix = time.mktime(current_time.timetuple())

        url = "https://api2.bybit.com/fapi/beehive/public/v1/common/leader-history"
        params = {
            "timeStamp": current_time_unix,
            "page": 1,
            "pageSize": 10000,
            "leaderMark": leaderMark,
        }
        ua = UserAgent().safari
        headers = {"user-agent": ua}
        response = requests.get(url=url, params=params, headers=headers)
        data = response.json()["result"]["data"]

        df_trades = pd.DataFrame(data)
        df_trades = df_trades[
            [
                "symbol",
                "side",
                "leverageE2",
                "size",
                "entryPrice",
                "closedPrice",
                "startedTimeE3",
                "closedTimeE3",
            ]
        ]
        df_trades.columns = [
            "symbol",
            "side",
            "leverage",
            "size",
            "entryPrice",
            "closedPrice",
            "startedTime",
            "closedTime",
        ]
        df_trades["startedTime"] = pd.to_datetime(df_trades["startedTime"], unit="ms")
        df_trades["closedTime"] = pd.to_datetime(df_trades["closedTime"], unit="ms")
        df_trades["leverage"] = df_trades["leverage"].astype(float)
        df_trades["entryPrice"] = df_trades["entryPrice"].astype(float)
        df_trades["closedPrice"] = df_trades["closedPrice"].astype(float)
        df_trades["leverage"] = df_trades["leverage"].div(100)
        df_trades["side"] = np.where(df_trades["side"] == "Buy", 1, -1)

        df_filter = HistoricalTrades.__filter_cheating_data(
            dataframe=df_trades,
            symbol_list=df_trades["symbol"].unique(),
            time_interval=time_interval,
        )

        return df_filter

    @staticmethod
    def __filter_cheating_data(
        dataframe: pd.DataFrame, symbol_list: list, time_interval="5min"
    ) -> pd.DataFrame:
        """filter our trade record which are repeated in a short time interval

        Args:
            dataframe (pd.DataFrame): DataFrame of trade record
            symbol_list (list): list of symbol which are traded in trade record
            interval (str, optional): _description_. Defaults to "1min. (it can be any interval: 3min/5min/10min)

        Returns:
            pd.DataFrame: trade record with unique trade
        """
        df_result = pd.DataFrame()
        for symbol in symbol_list:
            df = dataframe[dataframe["symbol"] == symbol]
            df = df.groupby(pd.Grouper(key="startedTime", freq=time_interval)).first()
            df = df.dropna()
            df = df.reset_index()
            df = df.groupby(pd.Grouper(key="closedTime", freq=time_interval)).first()
            df = df.dropna()
            df = df.reset_index()
            df = df[
                [
                    "symbol",
                    "side",
                    "leverage",
                    "size",
                    "entryPrice",
                    "closedPrice",
                    "startedTime",
                    "closedTime",
                ]
            ]
            df_result = pd.concat([df_result, df])
        df_result = df_result.sort_values("closedTime", ascending=True)
        df_result = df_result.reset_index(drop=True)
        return df_result


class AnalysisTradeRecord:
    """Analysis Historical trade record to generate equity curve and stat. report

    Returns:
        _type_: Backtest Instance
    """

    def __init__(
        self,
        df_trade: pd.DataFrame,
        leaderMark: str,
        no_of_tradable_days_a_year=365,
        transaction_fee=0.0008,
    ):
        self.leaderMark = leaderMark
        self.transaction_fee = transaction_fee
        self.trades_record = df_trade
        self.no_tradable_days = no_of_tradable_days_a_year
        self.capital = self.__get_capital()
        self.report = self.__get_statisical_report()

    def __get_capital(self):
        df = self.trades_record.copy()
        fee = (df["entryPrice"] + df["closedPrice"]) * self.transaction_fee
        df["pnl"] = (df["side"] * (df["closedPrice"] - df["entryPrice"]) - fee) / df[
            "entryPrice"
        ]
        df["capital"] = (df["pnl"] + 1).cumprod()
        df = self.__Capital(df)
        return df

    class __Capital(pd.DataFrame):
        """Generate Equity series

        Args:
            pd (_type_): None
        """

        def plot(self, log_scale=None):
            plt.figure(figsize=(20, 8))
            sns.lineplot(x=self.closedTime, y=self.capital, label="Strategy")
            if log_scale:
                plt.yscale("log")
            plt.title("Equity Curve")
            plt.legend()
            plt.show()
            plt.close()
            return None

    def __get_statisical_report(self):
        df_trade = self.capital
        report = self.__Statistics_report()
        report.leaderMark = self.leaderMark
        report.n_trades = len(df_trade)
        report.trading_period = np.ceil(
            (
                self.capital["closedTime"].iloc[-1].date()
                - self.capital["startedTime"].iloc[0].date()
            )
            / pd.Timedelta(days=1)
        )
        report.median_historical_trade_duration = (
            self.capital["closedTime"] - self.capital["startedTime"]
        ).median() / dt.timedelta(days=1)

        report.oldest_historical_trade_duration = (
            self.capital["closedTime"] - self.capital["startedTime"]
        ).max() / dt.timedelta(days=1)

        if (report.n_trades >= 2) & (report.trading_period != 0):
            report.win_rate = len(df_trade[df_trade["pnl"] > 0]) / report.n_trades
            report.median_return_per_trade = df_trade["pnl"].median()
            report.mean_return_per_trade = df_trade["pnl"].mean()
            if len(df_trade[df_trade["pnl"] > 0]) >= 1:
                report.max_gain_per_trade = df_trade[df_trade["pnl"] > 0]["pnl"].max()
            if len(df_trade[df_trade["pnl"] < 0]) >= 1:
                report.max_loss_per_trade = df_trade[df_trade["pnl"] < 0]["pnl"].min()
            if len(df_trade[df_trade["side"] == -1]) >= 1:
                report.long_short_ratio = len(df_trade[df_trade["side"] == 1]) / (
                    len(df_trade[df_trade["side"] == -1])
                )
            if len(df_trade[df_trade["side"] == 1]) >= 1:
                report.long_expected_return_per_trade = df_trade[df_trade["side"] == 1][
                    "pnl"
                ].mean()
            if len(df_trade[df_trade["side"] == -1]) >= 1:
                report.short_expected_return_per_trade = df_trade[
                    df_trade["side"] == -1
                ]["pnl"].mean()

            report.total_return = (
                df_trade.capital.iloc[-1] / df_trade.capital.iloc[0] - 1
            )
            report.mean_daily_return = (df_trade["pnl"].mean() + 1) ** (
                report.n_trades / report.trading_period
            ) - 1
            report.mean_annual_return = (
                1 + report.mean_daily_return
            ) ** self.no_tradable_days - 1
            report.max_drawdown_rate = -(
                (df_trade.capital - df_trade.capital.cummax())
                / df_trade.capital.cummax()
            ).min()
            if report.max_drawdown_rate != 0:
                report.calmar_ratio = report.total_return / report.max_drawdown_rate
            if df_trade["pnl"].std() != 0:
                report.sharpe_ratio = (
                    df_trade["pnl"].mean()
                    / df_trade["pnl"].std()
                    * (
                        (
                            report.n_trades
                            / report.trading_period
                            * self.no_tradable_days
                        )
                        ** 0.5
                    )
                )
        return report

    class __Statistics_report:
        """Generate report of strategy"""

        def __init__(self):
            self.leaderMark = np.nan
            self.n_trades = np.nan
            self.trading_period = np.nan
            self.median_historical_trade_duration = np.nan
            self.oldest_historical_trade_duration = np.nan
            self.win_rate = np.nan
            self.mean_return_per_trade = np.nan
            self.median_return_per_trade = np.nan
            self.max_gain_per_trade = np.nan
            self.max_loss_per_trade = np.nan
            self.long_short_ratio = np.nan
            self.long_expected_return_per_trade = np.nan
            self.short_expected_return_per_trade = np.nan
            self.total_return = np.nan
            self.mean_daily_return = np.nan
            self.mean_annual_return = np.nan
            self.max_drawdown_rate = np.nan
            self.calmar_ratio = np.nan
            self.sharpe_ratio = np.nan

        def print(self):

            if self.n_trades == 0:
                print("No trade is executed.")
            else:
                print(f"{'***************** Strategy *****************':<45}")
                print(f"{'No. of Trades:':<35}{self.n_trades:<10}")
                print(f"{'Trading_period (days)':<35}{self.trading_period:<10}")
                print(
                    f"{'median_historical_trade_duration':<35}{self.median_historical_trade_duration:<10}"
                )
                print(
                    f"{'oldest_historical_trade_duration':<35}{self.oldest_historical_trade_duration:<10}"
                )
                print(f"{'Win rate:':<35}{self.win_rate:<10.2%}")
                print(
                    f"{'Expected Return per Trade:':<35}{self.mean_return_per_trade:<10.2%}"
                )
                print(f"{'Max. Gain per Trade:':<35}{self.max_gain_per_trade:<10.2%}")
                print(f"{'Max Loss per Trade:':<35}{self.max_loss_per_trade:<10.2%}")
                print(f"{'Long Short Ratio:':<35}{self.long_short_ratio:<10.2}")
                print(
                    f"{'Long Expected Return per Trade:':<35}{self.long_expected_return_per_trade:<10.2%}"
                )
                print(
                    f"{'Short Expected Return per Trade:':<35}{self.short_expected_return_per_trade:<10.2%}"
                )
                print(f"{'Total Return:':<35}{self.total_return:<10.2%}{'':<5}")
                print(
                    f"{'Mean Daily Return:':<35}{self.mean_daily_return:<10.2%}{'':<5}"
                )
                print(
                    f"{'Mean Annual Return:':<35}{self.mean_annual_return:<10.2%}{'':<5}"
                )
                print(
                    f"{'Max. Drawdown Rate:':<35}{self.max_drawdown_rate:<10.2%}{'':<5}"
                )
                print(f"{'Calmar Ratio:':<35}{self.calmar_ratio:<10.4}{'':<5}")
                print(f"{'Sharpe Ratio:':<35}{self.sharpe_ratio:<10.4}{'':<5}")
                return None


class OpenTrades:
    __list_of_instance = []
    __session_unauth = usdt_perpetual.HTTP(endpoint="https://api.bybit.com")

    def __init__(self, leadermark, df_trades) -> None:
        self.df_trades = df_trades
        self.leadermark = leadermark
        self.report = self.__analysis_and_reporting()

    @classmethod
    def get_open_trades_report(
        cls, list_of_leaders: pd.Series, time_interval="5min", transaction_fee=0.0008
    ) -> pd.DataFrame:
        """return report of open trades of list of traders

        Args:
            list_of_leaders (pd.Series): it can be list/pd.Series of leadermark : ['vkbt1akj/7S0Xe3MB3bTgQ==','R+nmHxrzC38tQ8hKhZAqZA==',]
            time_interval (str, optional): for filtering intensive order in a short time interval : 1min, 5min, 1d, 5d

        Returns:
            pd.DataFrame: DataFrame of report (row with all nan mean there is no open trade)
        """

        cls.__list_of_instance = []
        for i, leadermark in enumerate(list_of_leaders):
            df_trades = cls.__get_open_trades(
                leaderMark=leadermark,
                time_interval=time_interval,
                transaction_fee=transaction_fee,
            )

            instance = OpenTrades(leadermark=leadermark, df_trades=df_trades)
            cls.__list_of_instance.append(instance)

            print(
                f"Open Trades Scraping Process: {i+1}/{len(list_of_leaders)}", end="\r"
            )

        report = map(lambda x: x.report.__dict__, cls.__list_of_instance)
        df = pd.DataFrame(report)
        df["open_trade_instance"] = cls.__list_of_instance

        return df

    @staticmethod
    def __get_current_price(symbol: str) -> float:
        """
        Query symbol current price.
        :param symbol: name of the query symbol
        :return: The price from Bybit
        """

        current_price = OpenTrades.__session_unauth.public_trading_records(
            symbol=symbol, limit=1
        )["result"][0]["price"]
        return current_price

    @staticmethod
    def __get_open_trades(leaderMark, time_interval, transaction_fee=0.0008):
        current_time = dt.datetime.utcnow()
        current_time_unix = time.mktime(current_time.timetuple())
        url = "https://api2.bybit.com/fapi/beehive/public/v1/common/order/list-detail"
        params = {
            "timeStamp": current_time_unix,
            "page": 1,
            "pageSize": 10000,
            "leaderMark": leaderMark,
        }

        ua = UserAgent().safari
        header = {"user-agent": ua}
        response = requests.get(url=url, params=params, headers=header)

        data = response.json()["result"]["data"]
        df = pd.DataFrame(data)

        if len(data) > 0:

            df["transactTimeE3"] = pd.to_datetime(df["transactTimeE3"], unit="ms")
            df["createdAtE3"] = pd.to_datetime(df["createdAtE3"], unit="ms")
            df["scrape_time"] = current_time
            df["side"] = df["side"].astype(str)
            df["symbol"] = df["symbol"].astype(str)
            df["sizeX"] = df["sizeX"].astype(float)

            df["entryPrice"] = df["entryPrice"].astype(float)
            df["leverageE2"] = df["leverageE2"].astype(float)
            df_filter = OpenTrades.__filter_cheating_data(
                df,
                df["symbol"].unique(),
                time_interval=time_interval,
                transaction_fee=transaction_fee,
            )
            return df_filter
        else:
            return pd.DataFrame()

    @staticmethod
    def __filter_cheating_data(
        dataframe: pd.DataFrame,
        symbol_list: list,
        time_interval="5min",
        transaction_fee=0.0008,
    ) -> pd.DataFrame:
        """filter our trade record which are repeated in a short time interval

        Args:
            dataframe (pd.DataFrame): DataFrame of trade record
            symbol_list (list): list of symbol which are traded in trade record
            interval (str, optional): _description_. Defaults to "1min. (it can be any interval: 3min/5min/10min)

        Returns:
            pd.DataFrame: trade record with unique trade
        """
        df_result = pd.DataFrame()
        for symbol in symbol_list:
            df = dataframe[dataframe["symbol"] == symbol]
            df = df.groupby(
                pd.Grouper(key="transactTimeE3", freq=time_interval)
            ).first()
            df.loc[:, "current_price"] = OpenTrades.__get_current_price(symbol=symbol)
            df = df.dropna()
            df = df.reset_index()

            df_result = pd.concat([df, df_result], axis=0, ignore_index=True)
        # df_result = df_result.sort_values("transactTimeE3", ascending=True)
        df_result = df_result.reset_index(drop=True)

        df_result = df_result[
            [
                # "leadermark",
                "symbol",
                "transactTimeE3",
                "sizeX",
                "leverageE2",
                "side",
                "entryPrice",
                "current_price",
            ]
        ]

        df_result["side"] = np.where(df_result["side"] == "Buy", 1, -1)
        fee = (df_result["current_price"] + df_result["entryPrice"]) * transaction_fee
        df_result["unrealized_pnl"] = (
            df_result["side"] * (df_result["current_price"] - df_result["entryPrice"])
            - fee
        ) / df_result["entryPrice"]
        return df_result

    def __analysis_and_reporting(self):
        report = self.__Report()
        report.leaderMark = self.leadermark
        if len(self.df_trades) >= 1:
            report.cum_unrealized_pnl = (
                self.df_trades["unrealized_pnl"] + 1
            ).prod() - 1
            report.n_open_trades = len(self.df_trades)
            report.mean_open_trade_pnl = self.df_trades["unrealized_pnl"].mean()
            report.worst_open_trade_pnl = self.df_trades["unrealized_pnl"].min()
            report.median_open_trade_pnl = self.df_trades["unrealized_pnl"].median()
            report.oldest_open_trade_duration = (
                dt.datetime.utcnow() - self.df_trades["transactTimeE3"].min()
            ) / dt.timedelta(days=1)

            report.median_open_trade_duration = (
                dt.datetime.utcnow() - self.df_trades["transactTimeE3"].median()
            ) / dt.timedelta(days=1)

        return report

    class __Report:
        def __init__(self) -> None:
            self.leaderMark = np.nan
            self.cum_unrealized_pnl = np.nan
            self.n_open_trades = np.nan
            self.mean_open_trade_pnl = np.nan
            self.median_open_trade_pnl = np.nan
            self.worst_open_trade_pnl = np.nan
            self.oldest_open_trade_duration = np.nan
            self.median_open_trade_duration = np.nan


if __name__ == "__main__":
    # get all trader name and leaderMark
    df_all_traders = get_all_leader_mark()

    # manually filter out no trader account
    df_filter = df_all_traders[df_all_traders["cumHistoryTransactionsCount"] > 20]

    # get open trade report
    df_open = OpenTrades.get_open_trades_report(df_filter["leaderMark"].iloc[:5])

    # get historical report
    df_hist = HistoricalTrades.get_historical_trades_report(
        df_filter["leaderMark"].iloc[:5]
    )

    df = df_filter.merge(df_open, on="leaderMark")
    df = df.merge(df_hist, on="leaderMark")
    df = df.set_index("leaderUserName")
    print(df)
