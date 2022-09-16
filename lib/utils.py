import requests
import typing
import inspect
import os
import json


class Bybit_instruments_info(object):

    def __init__(self):
        self.base_url = "https://api.bybit.com"
        self.min_qty_dict = self.get_min_qty_dict()

    def _get_all_instruments_info(self) -> list:
        url = f"{self.base_url}/derivatives/v3/public/instruments-info"
        response = requests.get(url, params={"category": "linear"})
        if response.ok:
            all_instruments_info = response.json().get("result", {}).get("list")
            if all_instruments_info:
                return all_instruments_info

    def get_min_qty_dict(self) -> dict:
        all_instruments_info = self._get_all_instruments_info()
        min_qty_dict = {instruments_info.get("symbol"): instruments_info.get("lotSizeFilter").get("minTradingQty") for
                        instruments_info in all_instruments_info}
        return min_qty_dict

    def get_min_qty(self, symbol: str) -> float:
        min_qty = float(self.min_qty_dict.get(symbol, 0))
        if min_qty == 0:
            self.min_qty_dict = self.get_min_qty_dict()
            if symbol in self.min_qty_dict:
                min_qty = float(self.min_qty_dict.get(symbol, 0))
            else:
                print("get min lot size err")
        return min_qty


class Config(object):
    _CONFIG_FILE: typing.Optional[str] = None
    _CONFIG: typing.Optional[dict] = None

    def __init__(self):
        config_file = Config.get_config_path(frame=inspect.stack()[1])
        with open(config_file, 'r') as f:
            Config._CONFIG = json.load(f)

    @staticmethod
    def get_config_path(frame: inspect.FrameInfo, config_file_name: str = "config.json"):
        print(type(frame))
        caller_file_name = frame[0].f_code.co_filename
        caller_folder_name = os.path.dirname(caller_file_name)
        config_file_name = os.path.join(caller_folder_name, config_file_name)
        return config_file_name

    @staticmethod
    def get_config_value(param: str):
        for key in Config._CONFIG.keys():
            value = Config._CONFIG.get(key).get(param, {})
            if value:
                return value
        raise ValueError(f"{param} doesn't exist")
