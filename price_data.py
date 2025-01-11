from datetime import datetime, timedelta
import os
import pickle
from typing import List
import pandas as pd
import yfinance as yf
import logging


class PriceData():
    
    def __init__(self):
        self.__temp_folder="temp/"
        self.__temp_file=self.__temp_folder + "price_data.pkl"
        self.__temp_tickers_file=self.__temp_folder + "tickers.txt"
        self.__temp_log_file=self.__temp_folder + "app_.log"
        
        logging.basicConfig(
            format='%(asctime)s - %(message)s',
            level=logging.INFO,
            handlers=[
                logging.FileHandler(self.__temp_log_file),
                logging.StreamHandler()
            ]
        )
        
        self.__data=None
        self.__load_temp()
        
        # Check and Create temp folder
        if not os.path.exists(self.__temp_folder):
            os.makedirs(self.__temp_folder)
    
            
    def __save_temp(self):
        with open(self.__temp_file, 'wb') as f:
            pickle.dump(self.__data, f)
            
    
    def __load_data(self):
        logging.info("Loading data from yahoo finance ... ")
        y_data = yf.download(list(self.__data["universe"].keys()), start="2000-01-01", end=datetime.now().strftime('%Y-%m-%d'))
        y_data.index = pd.to_datetime(y_data.index)  # S'assurer que les dates sont au format datetime64
        self.__data["last_refresh_date"] = datetime.now()
        self.__data["data"] = y_data
        self.__save_temp()
        
    def data(self):
        return self.__data["data"]
    
    def prices(self, tickers: List[str]):
        return self.__data["data"][[col for col in self.__data["data"].columns if col[1] in tickers]]
    
    def __load_temp(self):
        if os.path.exists(self.__temp_file):
            logging.info("Loading data from temp ... ")
            try:
                with open(self.__temp_file, 'rb') as f:
                    temp_data = pickle.load(f)
                    last_refresh_date = temp_data.get("last_refresh_date")
                    if last_refresh_date and datetime.now() - last_refresh_date < timedelta(hours=10):
                        self.__data = temp_data
                    else:
                        logging.info("Data is outdated or missing, reloading data...")
                        self.__data = {
                            "last_refresh_date": None,
                            "data": pd.DataFrame(),
                            "universe": self.__load_universe()
                        }
                        self.__load_data()
            except Exception as e:
                logging.error(f"Failed to load temp data: {e}")
                self.__data = {
                    "last_refresh_date": None,
                    "data": pd.DataFrame(),
                    "universe": self.__load_universe()
                }
                self.__load_data()
        else:
            logging.info("Temp file not found, initializing new data...")
            self.__data = {
                "last_refresh_date": None,
                "data": pd.DataFrame(),
                "universe": self.__load_universe()
            }
            self.__load_data()

    def __load_universe(self):
        tickers_dict = {}
        if os.path.exists(self.__temp_tickers_file):
            with open(self.__temp_tickers_file, 'r') as file:
                for line in file:
                    if ',' in line:
                        ticker, company = line.strip().split(',', 1)
                        tickers_dict[ticker] = company
        else:
            logging.warning(f"Tickers file not found: {self.__temp_tickers_file}")
        return tickers_dict
    def universe(self):
        return self.__data["universe"]