import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from strategies.data import DataDownloader
from Indicateurs import Indicateurs
from sklearn.preprocessing import StandardScaler
from datetime import timedelta
from sklearn.model_selection import TimeSeriesSplit
import multiprocessing
import os
from functools import partial

class DLInvestmentStrategy(Indicateurs):
    def __init__(self, tickers, start_date, end_date, initial_capital=1000, lookback_period=6):
        super().__init__()
        self.tickers = tickers if isinstance(tickers, list) else [tickers]
        self.lookback_period = lookback_period
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        self.allocation = initial_capital / len(self.tickers)
        self.data_downloader = DataDownloader()
        self.models = {}
        self.scalers = {}
        self.selected_features = {}
        # Pour stocker les vraies valeurs et prédictions
        self.y_true_dict = {}
        self.y_pred_dict = {}
        self.y_proba_dict = {}
        # Pour stocker les données de prix complètes pour chaque ticker
        self.price_data = {}

    def add_technical_features(self, data):
        """Ajoute des indicateurs techniques pour l'entraînement."""
        if data.empty:
            return pd.DataFrame()
            
        df = data.copy()
        
        # Vérifions que toutes les colonnes nécessaires existent
        if 'Close' not in df.columns or 'Volume' not in df.columns:
            print("⚠️ Colonnes manquantes dans les données. Vérifiez le téléchargement.")
            return pd.DataFrame()
        
        # Calculer toutes les colonnes en une seule opération
        try:
            features = pd.DataFrame({
                'Returns': df['Close'].pct_change(),
                'SMA_20': df['Close'].rolling(window=20).mean(),
                'RSI': self.calculate_rsi(df['Close']),
                'MACD': self.calculate_macd(df['Close'])[0],  # Ajout du MACD
                'MACD_Signal': self.calculate_macd(df['Close'])[1],  # Signal du MACD
                'BB_Upper': self.calculate_bollinger_bands(df['Close'])[0],  # Bollinger supérieure
                'BB_Lower': self.calculate_bollinger_bands(df['Close'])[1],  # Bollinger inférieure
                'ATR': self.calculate_atr(df),  # Average True Range
                'Volume_SMA': df['Volume'].rolling(window=20).mean(),
                'Volume_Ratio': df['Volume'] / df['Volume'].rolling(window=20).mean()
            })
            
            # Fusionner toutes les nouvelles colonnes en une seule fois
            df = pd.concat([df, features], axis=1)
            
        except Exception as e:
            print(f"⚠️ Erreur lors du calcul des indicateurs techniques: {e}")
            return pd.DataFrame()

        return df.dropna().copy()  # Supprimer les lignes avec NaN pour éviter les problèmes

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calcule le MACD et sa ligne de signal."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line
        
    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calcule les bandes de Bollinger."""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, lower_band
        
    def calculate_atr(self, data, window=14):
        """Calcule l'Average True Range."""
        high = data['High']
        low = data['Low']
        close = data['Close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        
        return atr

    def calculate_rsi(self, prices, periods=14):
        """Calcule l'indicateur RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def preprocess_data(self, data):
        """
        Prépare les données en ajoutant des indicateurs techniques et en générant les features de lag.
        Version optimisée avec mise en cache des résultats.
        """
        if data.empty:
            return pd.DataFrame()
        
        # Système de cache basé sur les données d'entrée
        cache_key = hash(tuple(data.index.astype(str).tolist()) + tuple(data['Close'].tolist()[:10]))
        if hasattr(self, '_preprocess_cache') and cache_key in self._preprocess_cache:
            return self._preprocess_cache[cache_key]
        
        # Ajouter les indicateurs techniques
        data = self.add_technical_features(data)
        
        if data.empty:
            return pd.DataFrame()

        # Définition des colonnes de features avec moins d'indicateurs pour accélérer
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 
                    'Returns', 'SMA_20', 'RSI', 'MACD', 'Volume_Ratio']

        try:
            # Créer un DataFrame séparé pour toutes les features de lag
            lag_features = pd.DataFrame(index=data.index)
            
            # Générer moins de lag features (optimisation)
            for i in range(1, min(self.lookback_period, 4) + 1):  # Limiter à 4 lags maximum
                for col in feature_cols:
                    if col in data.columns:
                        lag_features[f'{col}_lag_{i}'] = data[col].shift(i)
            
            # Fusionner les lag features avec le DataFrame original en une seule opération
            data = pd.concat([data, lag_features], axis=1)
            
            # Création de la variable cible : 1 si le prix monte à J+1, 0 sinon
            data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
            
            # Suppression des valeurs infinies et des NaN
            data = data.replace([np.inf, -np.inf], np.nan).dropna().copy()
        except Exception as e:
            print(f"⚠️ Erreur lors du prétraitement des données: {e}")
            return pd.DataFrame()
        
        # Stocker dans le cache
        if not hasattr(self, '_preprocess_cache'):
            self._preprocess_cache = {}
        self._preprocess_cache[cache_key] = data
        
        return data

    
    # Également, ajoutez cette méthode pour analyser l'overfitting
    def analyze_overfitting(self, model, X_train, y_train, X_val, y_val):
        """Analyse l'écart entre performance d'entraînement et validation."""
        train_accuracy = model.score(X_train, y_train)
        val_accuracy = model.score(X_val, y_val)
        
        print(f"Accuracy entraînement: {train_accuracy:.4f}, Accuracy validation: {val_accuracy:.4f}")
        print(f"Écart (entraînement - validation): {train_accuracy - val_accuracy:.4f}")
        
        if train_accuracy - val_accuracy > 0.1:
            print("⚠️ Possible overfitting détecté. Considérez plus de régularisation.")
        
        return train_accuracy, val_accuracy
    def train_and_evaluate(self, data, ticker):
        """
        Entraîne un réseau de neurones MLP sur les données d'entraînement
        et sélectionne le meilleur modèle via GridSearch.
        Version optimisée pour des performances améliorées.
        """
        if data.empty:
            print(f"⚠️ Données vides pour {ticker}, impossible d'entraîner un modèle.")
            return None, None, None, None, None, None
            
        # Vérifier si un modèle existe déjà pour ce ticker et décider si on le réutilise
        model_path = f'models/{ticker}_model.pkl'
        scaler_path = f'models/{ticker}_scaler.pkl'
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                import joblib
                print(f"🔄 Chargement du modèle existant pour {ticker}...")
                best_model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                
                # Trouver les features utilisées à partir du modèle
                # Pour MLP, la forme de coefs_[0] donne le nombre de features d'entrée
                n_features = best_model.coefs_[0].shape[0]
                
                # Utiliser une période d'entraînement plus courte
                train_cutoff = self.start_date - timedelta(days=30)  # 30 jours au lieu de plus
                validation_data = data[(data.index >= train_cutoff) & (data.index < self.start_date)]
                
                # Identifier les colonnes de lag
                feature_cols = [col for col in validation_data.columns if "_lag_" in col][:n_features]
                
                # Évaluer rapidement le modèle existant
                if not validation_data.empty and len(feature_cols) >= n_features:
                    X_val = validation_data[feature_cols[:n_features]]
                    y_val = validation_data['Target']
                    
                    if not X_val.empty and not y_val.empty:
                        X_val_scaled = scaler.transform(X_val)
                        y_pred_val = best_model.predict(X_val_scaled)
                        y_proba_val = best_model.predict_proba(X_val_scaled)[:, 1]
                        
                        print(f"✅ {ticker} - Modèle existant chargé et évalué")
                        return best_model, scaler, feature_cols[:n_features], y_val, y_pred_val, y_proba_val
            
            except Exception as e:
                print(f"⚠️ Erreur lors du chargement du modèle pour {ticker}: {e}")
                # Continuer avec l'entraînement d'un nouveau modèle
        
        # Utiliser une période d'entraînement plus courte
        train_cutoff = self.start_date - timedelta(days=90)  # 90 jours au lieu de plus
        train_data = data[data.index < train_cutoff]
        validation_data = data[(data.index >= train_cutoff) & (data.index < self.start_date)]
        
        if train_data.empty:
            print(f"⚠️ Pas assez de données pour l'entraînement de {ticker}.")
            return None, None, None, None, None, None

        # Identifier toutes les colonnes de lag - mais limiter le nombre pour l'efficacité
        feature_cols = [col for col in train_data.columns if "_lag_" in col][:20]  # Limiter à 20 max
        if not feature_cols:
            print(f"⚠️ Pas de features de lag disponibles pour {ticker}.")
            return None, None, None, None, None, None
            
        X = train_data[feature_cols]
        y = train_data['Target']

        # Vérifier si X contient des NaN ou des valeurs infinies
        if X.isnull().values.any() or np.isinf(X).values.any():
            print(f"⚠️ X contient des NaN ou des valeurs infinies pour {ticker}.")
            return None, None, None, None, None, None

        # Normalisation des données avec StandardScaler
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Données de validation
            X_val = validation_data[feature_cols] if not validation_data.empty else None
            y_val = validation_data['Target'] if not validation_data.empty else None
            
            if X_val is not None and not X_val.empty:
                X_val_scaled = scaler.transform(X_val)
        except Exception as e:
            print(f"⚠️ Erreur lors de la normalisation des données pour {ticker}: {e}")
            return None, None, None, None, None, None

        # Définition d'un modèle plus simple et plus rapide
        mlp = MLPClassifier(
            hidden_layer_sizes=(64,),  # Une seule couche cachée
            max_iter=500,              # Moins d'itérations
            solver="adam", 
            random_state=42, 
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=5,        # Arrêt plus précoce
            alpha=0.01,
            batch_size=64              # Taille de batch fixe
        )

        # GridSearch simplifiée avec beaucoup moins de combinaisons
        param_grid = {
            'hidden_layer_sizes': [(64,)],
            'activation': ['relu'],
            'alpha': [0.01],
            'learning_rate': ['adaptive']
        }

        # Si très peu de données, simplifier encore plus
        if len(X) < 500:
            mlp.set_params(hidden_layer_sizes=(32,))
            param_grid = {'activation': ['relu']}  # Un seul paramètre à ajuster

        try:
            # Utiliser moins de splits dans la validation croisée
            tscv = TimeSeriesSplit(n_splits=2, test_size=len(X) // 10)
            
            grid_search = GridSearchCV(
                estimator=mlp,
                param_grid=param_grid,
                scoring='accuracy',
                cv=tscv,
                n_jobs=-1,
                verbose=0  # Réduire la verbosité
            )

            # Entraînement du modèle
            grid_search.fit(X_scaled, y)
            
            # Récupération du meilleur modèle
            best_model = grid_search.best_estimator_
            
            # Sauvegarder le modèle pour une utilisation future
            if not os.path.exists('models'):
                os.makedirs('models')
            import joblib
            joblib.dump(best_model, f'models/{ticker}_model.pkl')
            joblib.dump(scaler, f'models/{ticker}_scaler.pkl')
            
            # Évaluation sur l'ensemble de validation
            y_pred_val = None
            y_proba_val = None
            
            if X_val is not None and not X_val.empty and y_val is not None and not y_val.empty:
                y_pred_val = best_model.predict(X_val_scaled)
                y_proba_val = best_model.predict_proba(X_val_scaled)[:, 1]
                best_accuracy = accuracy_score(y_val, y_pred_val)
                print(f"✅ {ticker} - Nouveau modèle entraîné, accuracy: {best_accuracy:.4f}")
            else:
                best_accuracy = grid_search.best_score_
                print(f"✅ {ticker} - Nouveau modèle entraîné, CV score: {best_accuracy:.4f}")
                    
        except Exception as e:
            print(f"❌ Erreur pendant l'entraînement pour {ticker}: {e}")
            return None, None, None, None, None, None

        # Analyse rapide de l'overfitting
        train_accuracy = best_model.score(X_scaled, y)
        if X_val is not None and not X_val.empty and y_val is not None and not y_val.empty:
            val_accuracy = best_model.score(X_val_scaled, y_val)
            if train_accuracy - val_accuracy > 0.1:
                print(f"⚠️ Possible overfitting pour {ticker}: train={train_accuracy:.4f}, val={val_accuracy:.4f}")

        return best_model, scaler, feature_cols, y_val, y_pred_val, y_proba_val

    def execute_trades(self):
        """
        Exécute la stratégie sur la période [start_date → end_date] pour chaque ticker.
        Retourne un dict de résultats par ticker.
        """
        results = {}

        # Télécharger et prétraiter les données une seule fois par ticker
        print("\n🔄 Préparation des données et entraînement des modèles...")
        for ticker in self.tickers:
            try:
                # 1) Télécharger les données historiques complètes (pour avoir assez de données pour l'entraînement)
                start_date_extended = pd.to_datetime("2010-01-01")  # Date de départ étendue
                data = self.data_downloader.download_data(ticker, start_date_extended, self.end_date)
                
                if data.empty:
                    print(f"⚠️ Les données pour {ticker} sont vides.")
                    continue
                    
                # Stocker les données de prix pour chaque ticker
                self.price_data[ticker] = data.copy()
                
                # 2) Prétraitement des données
                processed_data = self.preprocess_data(data)
                
                if processed_data.empty:
                    print(f"⚠️ Erreur lors du prétraitement des données pour {ticker}.")
                    continue
                
                # 3) Entraîner le modèle
                model_results = self.train_and_evaluate(processed_data, ticker)
                
                if model_results[0] is None:
                    print(f"❌ Échec de l'entraînement du modèle pour {ticker}.")
                    continue
                    
                best_model, scaler, features, y_val, y_pred_val, y_proba_val = model_results
                
                # Stocker le modèle, le scaler et les features pour ce ticker
                self.models[ticker] = best_model
                self.scalers[ticker] = scaler
                self.selected_features[ticker] = features
                
                # Stocker les évaluations de modèle pour les graphiques
                if y_val is not None and y_pred_val is not None:
                    self.y_true_dict[ticker] = y_val
                    self.y_pred_dict[ticker] = y_pred_val
                    self.y_proba_dict[ticker] = y_proba_val
                
            except Exception as e:
                print(f"❌ Erreur générale pour {ticker}: {e}")
                continue

        # Exécuter le trading avec les modèles entraînés
        print("\n🚀 Exécution du trading...")
        
        for ticker in self.tickers:
            if ticker not in self.models:
                print(f"⚠️ Pas de modèle disponible pour {ticker}, trading impossible.")
                continue
                
            print(f"\n📈 Trading sur {ticker}...")
            try:
                # Récupérer les données pour la période de trading
                test_data = self.data_downloader.download_data(ticker, self.start_date, self.end_date)
                
                if test_data.empty:
                    print(f"⚠️ Pas de données pour la période de trading pour {ticker}.")
                    continue
                    
                # Prétraiter les données de test
                processed_test = self.preprocess_data(test_data)
                
                if processed_test.empty or len(processed_test) <= self.lookback_period:
                    print(f"⚠️ Pas assez de données de trading pour {ticker}.")
                    continue
                
                # Récupérer modèle, scaler et features pour ce ticker
                model = self.models[ticker]
                scaler = self.scalers[ticker]
                features = self.selected_features[ticker]
                
                # Initialiser le suivi du capital et des positions
                capital = self.allocation
                capital_history = []
                buy_trades = 0
                sell_trades = 0
                daily_returns = []
                
                # Stocker toutes les prédictions du test
                all_test_dates = []
                all_test_true = []
                all_test_pred = []
                all_test_proba = []
                
                # Utilisons processed_test au lieu de test_data pour le trading
                # car les indicateurs sont déjà calculés dans processed_test
                for i in range(len(processed_test) - 1):
                    current_date = processed_test.index[i]
                    next_date = processed_test.index[i + 1]
                    
                    # S'assurer que les features existent dans processed_test
                    if not all(feat in processed_test.columns for feat in features):
                        missing = [f for f in features if f not in processed_test.columns]
                        print(f"⚠️ Features manquantes: {missing}")
                        continue
                    
                    # Extraire les features pour la prédiction
                    X = processed_test.loc[current_date, features]
                    
                    # Reformater en DataFrame si nécessaire
                    if isinstance(X, pd.Series):
                        X = X.to_frame().T
                    
                    # Vérifier que nous avons toutes les valeurs
                    if X.isnull().values.any():
                        continue
                    
                    # Normaliser avec le même scaler utilisé pendant l'entraînement
                    X_scaled = scaler.transform(X)
                    
                    # Prédiction pour le jour suivant
                    prediction = model.predict(X_scaled)[0]
                    probability = model.predict_proba(X_scaled)[0, 1]  # Probabilité de la classe positive
                    
                    # Stocker la valeur réelle pour l'évaluation
                    actual_value = processed_test.loc[next_date, 'Target']
                    
                    # Stocker pour l'évaluation du modèle
                    all_test_dates.append(next_date)
                    all_test_true.append(actual_value)
                    all_test_pred.append(prediction)
                    all_test_proba.append(probability)
                    
                    # Prix du jour suivant
                    next_open = processed_test.loc[next_date, 'Open']
                    next_close = processed_test.loc[next_date, 'Close']
                    
                    # Vérifier les valeurs NaN ou zéro
                    if pd.isna(next_open) or pd.isna(next_close) or next_open == 0:
                        continue
                    
                    # Calculer le rendement journalier
                    daily_return = (next_close - next_open) / next_open
                    
                    # Appliquer la stratégie selon la prédiction
                    if prediction == 1:  # Prévision de hausse → Achat
                        capital *= (1 + daily_return)
                        buy_trades += 1
                        action = "Buy"
                    else:  # Prévision de baisse → Vente à découvert
                        capital *= (1 - daily_return)  # Profit si le prix baisse
                        sell_trades += 1
                        action = "Sell"
                    
                    # Enregistrer les données
                    capital_history.append((next_date, capital))
                    daily_returns.append(daily_return)
                    
                    print(f"📅 {next_date}, {ticker} | Action: {action}, Daily Return: {daily_return*100:.2f}%, Capital: {capital:.2f}")
                
                # Stocker les prédictions et valeurs réelles pour les graphiques
                if all_test_dates:
                    # Mettre à jour ou compléter les données d'évaluation
                    if ticker not in self.y_true_dict:
                        self.y_true_dict[ticker] = np.array(all_test_true)
                        self.y_pred_dict[ticker] = np.array(all_test_pred)
                        self.y_proba_dict[ticker] = np.array(all_test_proba)
                    else:
                        # Compléter ou mettre à jour si nécessaire
                        test_results_df = pd.DataFrame({
                            'Date': all_test_dates,
                            'y_true': all_test_true,
                            'y_pred': all_test_pred,
                            'y_proba': all_test_proba
                        })
                        test_results_df.set_index('Date', inplace=True)
                        
                        # Ajouter aux dictionnaires
                        self.y_true_dict[ticker] = test_results_df['y_true'].values
                        self.y_pred_dict[ticker] = test_results_df['y_pred'].values 
                        self.y_proba_dict[ticker] = test_results_df['y_proba'].values
                
                # Préparer l'historique du capital au format DataFrame pour les graphiques
                capital_df = pd.DataFrame(capital_history, columns=['Date', 'Capital'])
                
                # Stocker les résultats du trading
                results[ticker] = {
                    "final_capital": capital,
                    "capital_history": capital_history,
                    "capital_history_df": capital_df,  # Pour les graphiques
                    "best_model": model,
                    "buy_trades": buy_trades,
                    "sell_trades": sell_trades,
                    "daily_returns": daily_returns,
                    "test_data": test_data  # Stocker les données de prix pour les graphiques
                }
                
            except Exception as e:
                print(f"❌ Erreur pendant le trading pour {ticker}: {e}")
                import traceback
                print(traceback.format_exc())
                continue

        return results

    def extract_feature_importances(self, model):
        """
        Extraire les importances des features pour les modèles de réseau de neurones.
        Pour les MLP, nous utilisons les poids des connexions comme proxy.
        """
        if not hasattr(model, 'coefs_'):
            return None
        
        # Pour les réseaux de neurones, utiliser la somme des valeurs absolues des poids de la première couche
        if len(model.coefs_) > 0:
            # Les poids de la première couche sont model.coefs_[0]
            feature_importance = np.abs(model.coefs_[0]).sum(axis=1)
            # Normaliser les importances
            if feature_importance.sum() > 0:
                feature_importance = feature_importance / feature_importance.sum()
            return feature_importance
        return None

    def execute(self):
        """
        Exécute la stratégie d'investissement sur l'ensemble des actifs (tickers) 
        et calcule les performances finales avec des métriques de risque fiables.
        """
        portfolio_results = {}
        total_final_capital = 0
        capital_evolution = []
        trade_summary = []
        
        # Pour stocker les données combinées pour les graphiques
        all_model_evaluations = {}
        all_feature_importances = {}
        all_price_data = {}
        all_capital_history = {}

        # Exécuter la stratégie pour chaque ticker
        trade_results = self.execute_trades()

        # Variables pour stocker les moyennes des métriques de risque
        all_volatility = []
        all_var_95 = []
        all_cvar_95 = []
        all_sharpe_ratio = []
        all_max_drawdown = []

        for ticker in self.tickers:
            print(f"\n📊 Analyse des résultats pour {ticker}...")  
            if ticker not in trade_results:
                print(f"⚠️ Aucune donnée de trading pour {ticker}.")
                continue

            ticker_results = trade_results[ticker]
            final_capital = ticker_results["final_capital"]
            capital_history = ticker_results["capital_history"]
            capital_history_df = ticker_results["capital_history_df"]
            buy_trades = ticker_results["buy_trades"]
            sell_trades = ticker_results["sell_trades"]
            best_model = ticker_results["best_model"]
            daily_returns = ticker_results.get("daily_returns", [])
            test_data = ticker_results.get("test_data", pd.DataFrame())

            total_final_capital += final_capital

            # Stocker l'historique du capital pour les graphiques
            all_capital_history[ticker] = capital_history_df
            
            # Stocker les données de prix pour les graphiques
            if not test_data.empty:
                all_price_data[ticker] = test_data
            
            # Extraire et stocker les importances de features, si disponibles
            features = self.selected_features.get(ticker, [])
            feature_importances = self.extract_feature_importances(best_model)
            
            if feature_importances is not None and len(features) > 0:
                # Créer un dictionnaire des importances par nom de feature
                importance_dict = dict(zip(features, feature_importances))
                all_feature_importances[ticker] = importance_dict

            # Construire l'historique d'évolution du capital
            for date, capital in capital_history:
                capital_evolution.append({"Date": date, "Capital": capital, "Ticker": ticker})

            # Calcul des jours investis en utilisant l'historique des positions
            if capital_history:
                start_date = capital_history[0][0]
                end_date = capital_history[-1][0]
                days_invested = (end_date - start_date).days + 1
            else:
                days_invested = 0

            # Calcul du gain total et pourcentage
            gain_total = final_capital - self.allocation
            pourcentage_gain_total = ((final_capital / self.allocation) - 1) * 100

            # Calculer les métriques de risque si on a des données de rendement
            volatility = float('nan')
            var_95 = float('nan')
            cvar_95 = float('nan')
            sharpe_ratio = float('nan')
            max_drawdown = float('nan')

            if daily_returns and len(daily_returns) > 5:  # Assurons-nous d'avoir suffisamment de données
                # Convertir en numpy array pour faciliter les opérations
                returns_array = np.array(daily_returns)
                
                # Filtrer les valeurs nan et inf
                returns_array = returns_array[~np.isnan(returns_array)]
                returns_array = returns_array[~np.isinf(returns_array)]
                
                if len(returns_array) > 0:
                    # Créer une série pandas pour les calculs statistiques
                    returns_series = pd.Series(returns_array)
                    
                    # Volatilité (écart-type annualisé des rendements)
                    volatility = returns_series.std() * np.sqrt(252)   # Annualisé et en %
                    
                    # Value at Risk (VaR) - perte maximale avec 95% de confiance
                    var_95 = np.percentile(returns_array, 5)   # En %
                    
                    # Conditional Value at Risk (CVaR) - perte moyenne au-delà de la VaR
                    cvar_threshold = np.percentile(returns_array, 5)
                    cvar_values = returns_array[returns_array <= cvar_threshold]
                    
                    if len(cvar_values) > 0:
                        cvar_95 = np.mean(cvar_values)  # En %
                    else:
                        cvar_95 = var_95  # Utilisez VaR comme approximation
                    
                    # Ratio de Sharpe (avec taux sans risque de 0% pour simplifier)
                    avg_daily_return = returns_series.mean()
                    std_daily_return = returns_series.std()
                    
                    # Annualiser pour le ratio de Sharpe
                    avg_annual_return = avg_daily_return * 252
                    std_annual_return = std_daily_return * np.sqrt(252)
                    
                    sharpe_ratio = avg_annual_return / std_annual_return if std_annual_return > 0 else 0
                    
                    # Drawdown maximal (perte maximale depuis un pic)
                    cum_returns = (1 + returns_series).cumprod()
                    running_max = cum_returns.cummax()
                    drawdown = (cum_returns / running_max - 1) * 100  # En %
                    max_drawdown = drawdown.min()
                    
                    # Collecter les métriques valides pour calculer les moyennes
                    if not np.isnan(volatility):
                        all_volatility.append(volatility)
                    if not np.isnan(var_95):
                        all_var_95.append(var_95)
                    if not np.isnan(cvar_95):
                        all_cvar_95.append(cvar_95)
                    if not np.isnan(sharpe_ratio):
                        all_sharpe_ratio.append(sharpe_ratio)
                    if not np.isnan(max_drawdown):
                        all_max_drawdown.append(max_drawdown)
                    
                    # Afficher les métriques pour vérification
                    print(f"📊 Métriques de risque pour {ticker}:")
                    print(f"   - Volatilité: {volatility:.2f}%")
                    print(f"   - VaR (95%): {var_95:.2f}%")
                    print(f"   - CVaR (95%): {cvar_95:.2f}%")
                    print(f"   - Ratio de Sharpe: {sharpe_ratio:.2f}")
                    print(f"   - Drawdown maximal: {max_drawdown:.2f}%")

            # Stocker tous les résultats pour ce ticker
            portfolio_results[ticker] = {
                "gain_total": gain_total,
                "pourcentage_gain_total": pourcentage_gain_total,
                "days_invested": days_invested,
                "volatilite_historique": volatility,
                "VaR Paramétrique": var_95,  # Renommé pour correspondre au ML
                "CVaR Paramétrique": cvar_95,  # Renommé pour correspondre au ML
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown
            }

            # Préparation des données d'évaluation du modèle pour les graphiques
            if ticker in self.y_true_dict and ticker in self.y_pred_dict:
                model_eval_data = {
                    'y_true': self.y_true_dict[ticker],
                    'y_pred': self.y_pred_dict[ticker]
                }
                if ticker in self.y_proba_dict:
                    model_eval_data['y_proba'] = self.y_proba_dict[ticker]
                
                all_model_evaluations[ticker] = model_eval_data
                
            # Ajout du résumé des trades pour ce ticker
            trade_summary.append({
                "Ticker": ticker,
                "Trades d'achat": buy_trades,
                "Trades de vente": sell_trades,
                "Total trades": buy_trades + sell_trades,
                "Gain total": gain_total,
                "Pourcentage de gain": pourcentage_gain_total,
                "Jours investis": days_invested
            })

        # Calcul des moyennes des métriques de risque pour le portefeuille global
        avg_volatility = np.mean(all_volatility) if all_volatility else float('nan')
        avg_var = np.mean(all_var_95) if all_var_95 else float('nan')
        avg_cvar = np.mean(all_cvar_95) if all_cvar_95 else float('nan')
        avg_sharpe = np.mean(all_sharpe_ratio) if all_sharpe_ratio else float('nan')
        avg_max_drawdown = np.mean(all_max_drawdown) if all_max_drawdown else float('nan')

        # Création d'un DataFrame pour l'évolution du capital global
        capital_evolution_df = pd.DataFrame(capital_evolution)
        
        # Convertir en DataFrame pour le tableau récapitulatif
        trade_summary_df = pd.DataFrame(trade_summary)

        # Préparer les résultats selon le format demandé
        results = {
            "gain_total": total_final_capital - self.initial_capital,
            "pourcentage_gain_total": ((total_final_capital / self.initial_capital) - 1) * 100,
            "capital_final_total": total_final_capital,
            "capital_evolution": capital_evolution_df,
            "trade_summary": trade_summary_df,
            "volatilite_historique": avg_volatility,
            "VaR Paramétrique": avg_var,
            "VaR Historique": avg_var,  # Utilise la même valeur pour rester compatible avec ML
            "VaR Cornish-Fisher": avg_var,  # Utilise la même valeur pour rester compatible avec ML
            "CVaR Paramétrique": avg_cvar,
            "CVaR Historique": avg_cvar,  # Utilise la même valeur pour rester compatible avec ML
            "CVaR Cornish-Fisher": avg_cvar,  # Utilise la même valeur pour rester compatible avec ML
            "sharpe_ratio": avg_sharpe,
            "max_drawdown": avg_max_drawdown,
            
            # Données supplémentaires pour les graphiques
            "portfolio_results": portfolio_results,
            "all_price_data": all_price_data,
            "all_capital_history": all_capital_history,
            "all_model_evaluations": all_model_evaluations,
            "all_feature_importances": all_feature_importances,
            
            # Modèles et leurs évaluations
            "models": self.models,
            "scalers": self.scalers,
            "selected_features": self.selected_features,
            "y_true_dict": self.y_true_dict,
            "y_pred_dict": self.y_pred_dict,
            "y_proba_dict": self.y_proba_dict
        }
        # Afficher un résumé final
        print("\n📈 Résumé de la performance du portefeuille:")
        print(f"Capital initial: ${self.initial_capital:.2f}")
        print(f"Capital final: ${total_final_capital:.2f}")
        print(f"Gain total: ${total_final_capital - self.initial_capital:.2f}")
        print(f"Rendement total: {((total_final_capital / self.initial_capital) - 1) * 100:.2f}%")
        
        if not np.isnan(avg_volatility):
            print(f"Volatilité moyenne: {avg_volatility:.2f}%")
        if not np.isnan(avg_var):
            print(f"VaR moyenne (95%): {avg_var:.2f}%")
        if not np.isnan(avg_cvar):
            print(f"CVaR moyenne (95%): {avg_cvar:.2f}%")
        if not np.isnan(avg_sharpe):
            print(f"Ratio de Sharpe moyen: {avg_sharpe:.2f}")
        if not np.isnan(avg_max_drawdown):
            print(f"Drawdown maximal moyen: {avg_max_drawdown:.2f}%")
        print(self.y_pred_dict)
        return results