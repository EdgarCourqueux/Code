import yfinance as yf
import pandas as pd
from datetime import datetime
from price_data import PriceData
import os
import pickle

class DataDownloader:
    def __init__(self, cache_folder="data_cache"):
        """
        Initialise le gestionnaire de téléchargement des données financières.
        Utilise la classe PriceData pour gérer le téléchargement et le cache des tickers.
        """
        self.price_data = PriceData()  # Instanciation de PriceData
        self.data_cache = {}
        self.cache_folder = cache_folder

        # Vérifie et crée le dossier de cache si nécessaire
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)

    def _get_cache_path(self, ticker):
        """
        Génère le chemin du fichier cache pour un ticker donné.
        
        :param ticker: str, le symbole boursier du ticker
        :return: str, chemin complet du fichier cache
        """
        return os.path.join(self.cache_folder, f"{ticker}.pkl")

    def _load_from_cache(self, ticker):
        """
        Charge les données à partir du fichier cache si disponible.
        
        :param ticker: str, le symbole boursier du ticker
        :return: pd.DataFrame ou None si le fichier cache n'existe pas
        """
        cache_path = self._get_cache_path(ticker)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    data = pickle.load(f)
                    # Transformer les données pour correspondre au format des données téléchargées via yfinance
                    data = self._transform_cached_data(data)
                    return data
            except Exception as e:
                print(f"Erreur lors du chargement du cache pour {ticker}: {e}")
        return None

    def _save_to_cache(self, ticker, data):
        """
        Sauvegarde les données dans un fichier cache.
        
        :param ticker: str, le symbole boursier du ticker
        :param data: pd.DataFrame, les données à sauvegarder
        """
        cache_path = self._get_cache_path(ticker)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du cache pour {ticker}: {e}")

    def _transform_cached_data(self, data):
        """
        Transforme les données chargées depuis le cache pour qu'elles correspondent au format des données téléchargées via yfinance.
        
        :param data: pd.DataFrame, données brutes depuis le cache
        :return: pd.DataFrame, données transformées
        """
        if isinstance(data, pd.DataFrame):
            # Supprimer les niveaux inutiles du MultiIndex, si présents
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)

            # Renommer les colonnes pour correspondre au format yfinance
            expected_columns = {
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Adj Close": "Adj Close",
                "Volume": "Volume"
            }
            data = data.rename(columns=expected_columns)

            # Vérifier la présence de toutes les colonnes nécessaires et ajouter les colonnes manquantes
            for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
                if col not in data.columns:
                    data[col] = pd.NA  # Ajouter la colonne manquante avec des valeurs NaN

            # S'assurer que l'index est bien au format datetime
            if "Date" in data.columns:
                data["Date"] = pd.to_datetime(data["Date"], errors="coerce")  # Convertir en datetime
                data = data.set_index("Date")  # Définir la colonne "Date" comme index
            elif not pd.api.types.is_datetime64_any_dtype(data.index):
                data.index = pd.to_datetime(data.index, errors="coerce")

            # Supprimer les lignes où l'index ou les colonnes critiques sont NaN
            data = data.dropna(subset=["Adj Close", "Open", "High", "Low", "Close"], how="all")

        return data


    def download_data(self, ticker, start_date, end_date):
        """
        Télécharge les données financières pour un ticker donné à partir de PriceData, du cache, ou Yahoo Finance.
        
        :param ticker: str, le symbole boursier du ticker
        :param start_date: datetime, la date de début pour les données
        :param end_date: datetime, la date de fin pour les données
        :return: pd.DataFrame, les données téléchargées ou un DataFrame vide en cas d'erreur
        """
        # Convertir les dates en tz-naive pour éviter les comparaisons tz-naive/tz-aware
        if isinstance(start_date, pd.Timestamp):
            start_date = start_date.tz_localize(None)
        if isinstance(end_date, pd.Timestamp):
            end_date = end_date.tz_localize(None)

        # Étape 1 : Vérifier si les données sont dans le cache local
        cached_data = self._load_from_cache(ticker)
        if cached_data is not None:
            try:
                cached_data.index = cached_data.index.tz_localize(None)
                filtered_data = cached_data.loc[start_date:end_date]
                if not filtered_data.empty:
                    return filtered_data
            except Exception as e:
                print(f"Erreur lors de l'utilisation des données du cache pour {ticker}: {e}")

        # Étape 2 : Vérifier si les données sont disponibles via PriceData
        try:
            all_data = self.price_data.prices([ticker])
            all_data.index = all_data.index.tz_localize(None)  # Assurez-vous que l'index est tz-naive
            filtered_data = all_data.loc[start_date:end_date]

            if not filtered_data.empty:
                self._save_to_cache(ticker, all_data)
                return filtered_data
            else:
                print(f"Aucune donnée disponible pour {ticker} dans PriceData entre {start_date} et {end_date}.")
        except Exception as e:
            print(f"Erreur lors de la récupération des données depuis PriceData pour {ticker}: {e}")

        # Étape 3 : Télécharger les données via yfinance si non disponibles ailleurs
        try:
            data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

            if data.empty:
                print(f"Aucune donnée disponible pour {ticker} entre {start_date} et {end_date} via Yahoo Finance.")
            else:
                # Sauvegarder dans le cache pour réutilisation future
                self._save_to_cache(ticker, data)
                return data

        except Exception as e:
            print(f"Erreur lors du téléchargement des données pour {ticker} via Yahoo Finance: {e}")

        # Retourner un DataFrame vide en cas d'échec complet
        return pd.DataFrame()


    def clear_cache(self):
        """
        Vide le cache interne des données téléchargées et supprime les fichiers cache locaux.
        """
        self.data_cache.clear()
        for file in os.listdir(self.cache_folder):
            file_path = os.path.join(self.cache_folder, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Erreur lors de la suppression du fichier cache {file_path}: {e}")
