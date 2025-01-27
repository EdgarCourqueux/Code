import yfinance as yf
import pandas as pd
from datetime import datetime

class DataDownloader:
    def __init__(self):
        """
        Initialise le gestionnaire de téléchargement des données financières.
        Un cache interne peut être utilisé pour éviter des téléchargements répétitifs.
        """
        self.data_cache = {}

    def download_data(self, ticker, start_date, end_date):
        """
        Télécharge les données financières pour un ticker donné à partir de Yahoo Finance.
        
        :param ticker: str, le symbole boursier du ticker
        :param start_date: datetime, la date de début pour les données
        :param end_date: datetime, la date de fin pour les données
        :return: pd.DataFrame, les données téléchargées ou un DataFrame vide en cas d'erreur
        """
        # Crée une clé unique pour le cache
        cache_key = (ticker, start_date, end_date)

        # Vérifie si les données sont déjà en cache
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]

        try:
            # Télécharge les données via yfinance
            data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

            if data.empty:
                print(f"Aucune donnée disponible pour {ticker} entre {start_date} et {end_date}.")
            else:
                # Stocke dans le cache
                self.data_cache[cache_key] = data

            return data

        except Exception as e:
            print(f"Erreur lors du téléchargement des données pour {ticker}: {e}")
            return pd.DataFrame()

    def clear_cache(self):
        """
        Vide le cache interne des données téléchargées.
        """
        self.data_cache.clear()
