from datetime import datetime, timedelta
import os
import pickle
from typing import List
import pandas as pd
import yfinance as yf
import logging


class PriceData():
    
    def __init__(self, temp_folder="temp/"):
        self.__temp_folder = temp_folder
        self.__temp_file = self.__temp_folder + "price_data.pkl"
        self.__temp_tickers_file = self.__temp_folder + "tickers.txt"
        self.__temp_log_file = self.__temp_folder + "app_.log"
        
        logging.basicConfig(
            format='%(asctime)s - %(message)s',
            level=logging.INFO,
            handlers=[
                logging.FileHandler(self.__temp_log_file),
                logging.StreamHandler()
            ]
        )
        
        self.__data = None
        
        # Check and Create temp folder
        if not os.path.exists(self.__temp_folder):
            os.makedirs(self.__temp_folder)
            
        self.__load_temp()
    
            
    def __save_temp(self):
        """Sauvegarde les données dans le fichier temporaire"""
        if not os.path.exists(self.__temp_folder):
            os.makedirs(self.__temp_folder)
            
        with open(self.__temp_file, 'wb') as f:
            pickle.dump(self.__data, f)
            
    
    def __load_data(self):
        """Charge les données depuis Yahoo Finance"""
        logging.info("Loading data from yahoo finance ... ")
        tickers_list = list(self.__data["universe"].keys())
        
        if not tickers_list:
            logging.warning("Empty tickers list, no data to download")
            return
            
        try:
            y_data = yf.download(tickers_list, start="2000-01-01", end=datetime.now().strftime('%Y-%m-%d'))
            y_data.index = pd.to_datetime(y_data.index)  # S'assurer que les dates sont au format datetime64
            self.__data["last_refresh_date"] = datetime.now()
            self.__data["data"] = y_data
            self.__save_temp()
        except Exception as e:
            logging.error(f"Error downloading data from Yahoo Finance: {e}")
    
    def data(self):
        """Renvoie toutes les données stockées"""
        return self.__data["data"]
    
    def prices(self, tickers: List[str]):
        """
        Renvoie les données de prix pour les tickers spécifiés.
        Format compatible avec DataDownloader.
        
        :param tickers: Liste des tickers à récupérer
        :return: DataFrame avec les données des tickers demandés
        """
        if self.__data["data"].empty:
            return pd.DataFrame()
            
        # Vérifier si les tickers demandés sont dans notre univers
        available_tickers = [ticker for ticker in tickers if ticker in self.__data["data"].columns.get_level_values(1)]
        
        if not available_tickers:
            logging.warning(f"None of the requested tickers {tickers} found in data")
            return pd.DataFrame()
            
        # Sélectionner uniquement les colonnes pour les tickers demandés
        ticker_data = self.__data["data"][[col for col in self.__data["data"].columns if col[1] in available_tickers]]
        
        # Conserver la structure multi-index pour compatibilité avec DataDownloader
        # Le DataDownloader gère lui-même la transformation des colonnes
        return ticker_data
    
    def __load_temp(self):
        """Charge les données depuis le fichier temporaire ou initialise de nouvelles données"""
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
        """Charge l'univers des tickers depuis le fichier"""
        tickers_dict = {}
        if os.path.exists(self.__temp_tickers_file):
            with open(self.__temp_tickers_file, 'r') as file:
                for line in file:
                    if ',' in line:
                        ticker, company = line.strip().split(',', 1)
                        tickers_dict[ticker.strip()] = company.strip()
        else:
            logging.warning(f"Tickers file not found: {self.__temp_tickers_file}")
            # Créer un fichier vide de tickers
            with open(self.__temp_tickers_file, 'w') as file:
                pass
        return tickers_dict
    
    def universe(self):
        """Renvoie l'univers des tickers"""
        return self.__data["universe"]
    
    def add_ticker(self, ticker, company_name=""):
        """
        Ajoute un ticker à l'univers et met à jour le fichier des tickers
        
        :param ticker: Le symbole du ticker à ajouter
        :param company_name: Le nom de la compagnie (optionnel)
        """
        if ticker not in self.__data["universe"]:
            self.__data["universe"][ticker] = company_name
            
            # Mettre à jour le fichier des tickers
            with open(self.__temp_tickers_file, 'a') as file:
                file.write(f"{ticker},{company_name}\n")
            
            # Télécharger les données pour ce nouveau ticker
            try:
                new_data = yf.download(ticker, start="2000-01-01", end=datetime.now().strftime('%Y-%m-%d'))
                
                if not new_data.empty:
                    # Si c'est le premier ticker, initialiser le DataFrame
                    if self.__data["data"].empty:
                        # Conversion en MultiIndex pour compatibilité
                        columns = pd.MultiIndex.from_product([['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], [ticker]])
                        new_multi_data = pd.DataFrame(index=new_data.index, columns=columns)
                        
                        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
                            new_multi_data[(col, ticker)] = new_data[col]
                            
                        self.__data["data"] = new_multi_data
                    else:
                        # Ajouter les nouvelles données au DataFrame existant
                        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
                            self.__data["data"][(col, ticker)] = new_data[col]
                    
                    self.__data["last_refresh_date"] = datetime.now()
                    self.__save_temp()
                    logging.info(f"Added ticker {ticker} to universe and downloaded data")
                else:
                    logging.warning(f"No data available for ticker {ticker}")
            except Exception as e:
                logging.error(f"Error downloading data for new ticker {ticker}: {e}")
        else:
            logging.info(f"Ticker {ticker} already in universe")
