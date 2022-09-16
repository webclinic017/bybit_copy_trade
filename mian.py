import time
from time import sleep
from random import uniform
from datetime import datetime
import requests
import logging
from fake_useragent import UserAgent
import pandas as pd
from pybit import usdt_perpetual
from lib.utils import Bybit_instruments_info, Config
from logs.logger import logger
from pprint import pformat


pd.set_option('display.max_columns', None)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("pybit").setLevel(logging.ERROR)


class Bybit_copy_trade(Bybit_instruments_info):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.session_auth = usdt_perpetual.HTTP(
            endpoint=self.base_url,
            api_key=self.config.get_config_value(param="api_key"),
            api_secret=self.config.get_config_value(param="api_secret")
        )
        self.__logger = logger()

    def get_current_trades(self, leaderMark: str) -> list:
        current_time = datetime.utcnow()
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
        return data

    def place_market_order(self, symbol: str, side: str, reduce_only: bool) -> bool:
        mapping_dict = {
            "Buy": "Sell",
            "Sell": "Buy"
        }
        side = mapping_dict[side] if reduce_only else side
        min_qty = self.get_min_qty(symbol=symbol)
        try:
            response = self.session_auth.place_active_order(symbol=symbol, side=side, order_type="Market", qty=min_qty,
                                                            time_in_force="GoodTillCancel", reduce_only=reduce_only,
                                                            close_on_trigger=False)
        except BaseException as e:
            print(e)
        else:
            if response["ret_msg"] == "OK":
                self.__logger.info(response)
                return True

    def main(self) -> None:
        self.__logger.info("Program start running")
        leaderMark = "vxw+oqlRCHKEC5wSeUXM5Q=="
        followed_trades_created_timestamp_list = []
        opened_trades_created_timestamp_list = ["_".join([current_trade["createdAtE3"], current_trade["symbol"],
                                                          current_trade["side"]]) for current_trade in
                                                self.get_current_trades(leaderMark=leaderMark)]
        # opened_trades_created_timestamp_list = [current_trade["createdAtE3"] for current_trade in
        #                                         get_current_trades(leaderMark=leaderMark)]
        self.__logger.info(pformat(self.get_current_trades(leaderMark=leaderMark)))
        self.__logger.info(f"\n{pformat(opened_trades_created_timestamp_list)}")
        self.__logger.info("*" * 100)
        while True:
            # try:
            # reformat the list to (createdAtE3 + symbol + side) string
            current_trade_list = [
                "_".join([current_trade["createdAtE3"], current_trade["symbol"], current_trade["side"]])
                for current_trade in self.get_current_trades(leaderMark=leaderMark)]
            # serach for the trades which are new in current_trade_list and not in opened_trades_created_timestamp_list
            new_trade_list = set(opened_trades_created_timestamp_list).difference(current_trade_list)
            if new_trade_list:
                # if new trades open, replace the original opened_trades_created_timestamp_list to opened trade list
                opened_trades_created_timestamp_list = current_trade_list
                for new_trade in new_trade_list:
                    symbol = new_trade.split("_")[1]
                    side = new_trade.split("_")[2]
                    if self.place_market_order(symbol=symbol, side=side, reduce_only=False):
                        # Add new transaction key to followed_trades_created_timestamp_list
                        followed_trades_created_timestamp_list.append(new_trade)
                        self.__logger.info(f"{new_trade = }")
            # search for the trades which opened in followed_trades_created_timestamp_list and disappear in current_list
            # means the trades are closed by the PT
            closed_trade_list = set(followed_trades_created_timestamp_list).difference(current_trade_list)
            if closed_trade_list:
                for closed_trade in closed_trade_list:
                    symbol = closed_trade.split("_")[1]
                    side = closed_trade.split("_")[2]
                    if self.place_market_order(symbol=symbol, side=side, reduce_only=True):
                        # Remove the transaction key to followed_trades_created_timestamp_list
                        followed_trades_created_timestamp_list.remove(closed_trade)
                        self.__logger.info(f"{closed_trade = }")

            sleep(uniform(1, 10))
            # except BaseException as e:
            #     print(datetime.now(), e)


if __name__ == "__main__":
    bybit_copy_trade = Bybit_copy_trade()
    bybit_copy_trade.main()
