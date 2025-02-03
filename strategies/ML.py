import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from strategies.data import DataDownloader
from Indicateurs import Indicateurs

class MLInvestmentStrategy(Indicateurs):
    def __init__(self, tickers, start_date, end_date, initial_capital=1000, lookback_period=6):
        super().__init__()
        self.tickers = tickers
        self.lookback_period = lookback_period
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        self.models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True),
            'KNN': KNeighborsClassifier(),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'AdaBoost': AdaBoostClassifier(algorithm='SAMME', random_state=42),
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
        }
        self.data_downloader = DataDownloader()

    def add_technical_features(self, data):
        """Ajoute des indicateurs techniques pour l'entraînement."""
        df = data.copy()
        
        # Calculer toutes les colonnes en une seule opération
        features = pd.DataFrame({
            'Returns': df['Close'].pct_change(),
            'SMA_20': df['Close'].rolling(window=20).mean(),
            'RSI': self.calculate_rsi(df['Close']),
            'Volume_SMA': df['Volume'].rolling(window=20).mean(),
            'Volume_Ratio': df['Volume'] / df['Volume'].rolling(window=20).mean()
        })

        # Fusionner toutes les nouvelles colonnes en une seule fois
        df = pd.concat([df, features], axis=1)

        return df.copy()  # Copie pour défragmenter


    def calculate_rsi(self, prices, periods=14):
        """Calcule l'indicateur RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def preprocess_data(self, data):
        """Crée les features avec `lookback_period` et génère la variable cible."""
        data = self.add_technical_features(data)
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Returns', 'SMA_20', 'RSI', 'Volume_Ratio']

        # Génération des colonnes en une seule opération
        lagged_features = {f'{col}_lag_{i}': data[col].shift(i) for i in range(1, self.lookback_period + 1) for col in feature_cols}
        
        # Fusionner en une seule opération avec pd.concat()
        data = pd.concat([data, pd.DataFrame(lagged_features, index=data.index)], axis=1)

        # Variable cible : hausse du prix à J+1
        data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

        return data.dropna().copy()  # Suppression des NaN et défragmentation


    def train_and_evaluate(self, data):
        """Entraîne le modèle en utilisant les `lookback_period` derniers jours comme features."""
        feature_cols = [col for col in data.columns if "lag" in col]  # Colonnes avec historique
        X = data[feature_cols]
        y = data['Target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        best_accuracy = 0
        best_model = None

        for name, model in self.models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model

            except Exception as e:
                print(f"⚠️ Erreur d'entraînement sur {name} : {e}")
                continue

        return best_model, feature_cols
    
    def execute_trades(self):
        """Exécute la stratégie et retourne les résultats."""
        combined_data = {}
        total_buy_trades = 0
        total_sell_trades = 0
        capital_history = []
        capital = self.initial_capital

        # Télécharger et prétraiter les données pour chaque ticker
        for ticker in self.tickers:
            data = self.data_downloader.download_data(ticker, self.start_date, self.end_date)
            if data.empty:
                print(f"⚠️ Les données pour {ticker} sont vides.")
                continue
            combined_data[ticker] = self.preprocess_data(data)
        
        if not combined_data:
            return "Aucune donnée disponible", [], 0, 0
        
        # Trouver les dates communes
        common_dates = sorted(set.intersection(*(set(data.index) for data in combined_data.values())))
        if not common_dates:
            return "Aucune date commune trouvée", [], 0, 0
        
        common_data = {ticker: data.loc[common_dates] for ticker, data in combined_data.items()}
        merged_data = pd.concat(common_data.values())
        
        # Entraîner et évaluer le modèle
        best_model, feature_cols = self.train_and_evaluate(merged_data)
        if best_model is None:
            return "Aucun modèle sélectionné", [], 0, 0
        
        # Simulation des transactions
        for date in common_dates:
            if date not in merged_data.index:
                continue

            # Récupérer les features du jour
            feature_values = merged_data.loc[date, feature_cols].values.reshape(1, -1)
            feature_df = pd.DataFrame(feature_values, columns=feature_cols)
            
            # Vérification des features
            missing_features = set(best_model.feature_names_in_) - set(feature_df.columns)
            if missing_features:
                raise ValueError(f"⚠️ Certaines features sont absentes : {missing_features}")

            # Prédiction du modèle
            prediction = best_model.predict(feature_df)[0]
            daily_return = (merged_data.loc[date, 'Close'] - merged_data.loc[date, 'Open']) / merged_data.loc[date, 'Open']
            daily_return_percentage = daily_return * 100
            
            action = "Hold"
            if prediction == 1:
                capital *= (1 + daily_return)
                action = "Buy"
                total_buy_trades += 1
            elif prediction == 0:
                capital *= (1 - daily_return)
                action = "Sell"
                total_sell_trades += 1
            
            capital_history.append((date, capital))
            print(f"Date: {date}, Action: {action}, Daily Return: {daily_return_percentage:.2f}%, Capital: {capital:.2f}")
        
        return capital, capital_history, total_buy_trades, total_sell_trades

    def execute(self):
        """Exécute la stratégie d'investissement sur l'ensemble des actifs et calcule les performances finales."""
        
        portfolio_results = {}
        total_initial_capital = self.initial_capital  # Capital initial global
        total_final_capital = 0  # Capital final global
        total_buy_trades = 0
        total_sell_trades = 0
        capital_evolution = []
        total_days = 0  # Nombre total de jours d'investissement
        
        for ticker in self.tickers:
            print(f"📈 Exécution de la stratégie pour {ticker}...")
            
            final_capital,capital_history,total_buy_trades,total_sell_trades = self.execute_trades()
            total_final_capital += final_capital

            # Télécharger les données pour le calcul des métriques
            data = self.data_downloader.download_data(ticker, self.start_date, self.end_date)
            if data.empty:
                print(f"⚠️ Les données pour {ticker} sont vides. Vérifiez le ticker ou la période de téléchargement.")
                continue

            data.reset_index(inplace=True)
            data['Date'] = pd.to_datetime(data['Date'])
            capital_evolution.extend(capital_history)
            try:
                date_investissement_proche = data.loc[data['Date'] >= self.start_date, 'Date'].iloc[0]
                date_du_jour_proche = data.loc[data['Date'] <= self.end_date, 'Date'].iloc[-1]
                days_invested = (date_du_jour_proche - date_investissement_proche).days  # Calcul du nombre de jours investis
                total_days += days_invested  # Ajout au total des jours
            except IndexError:
                print(f"⚠️ La date d'investissement ou de fin est hors de la plage des données disponibles pour {ticker}.")
                continue

            # 📊 Calcul des métriques de risque
            try:
                volatilite_historique = self.volatilite_historique(data).get("volatilite_historique", 0)
                var_parametric = self.calculate_var(data, alpha=0.05, method="parametric")
                var_historical = self.calculate_var(data, alpha=0.05, method="historical")
                var_cornish_fisher = self.calculate_var(data, alpha=0.05, method="cornish-fisher")
                cvar_parametric = self.calculate_cvar(data, alpha=0.05, method="parametric")
                cvar_historical = self.calculate_cvar(data, alpha=0.05, method="historical")
                cvar_cornish_fisher = self.calculate_cvar(data, alpha=0.05, method="cornish-fisher")
            except Exception as e:
                print(f"⚠️ Erreur lors du calcul des indicateurs pour {ticker} : {e}")
                continue

            print(f"🔹 Résultats pour {ticker}: Capital Final = {final_capital:.2f} €")

            portfolio_results[ticker] = {
                "gain_total": final_capital - self.initial_capital,
                "pourcentage_gain_total": ((final_capital / self.initial_capital) - 1) * 100,
                "volatilite_historique": volatilite_historique,
                "VaR Paramétrique": var_parametric,
                "VaR Historique": var_historical,
                "VaR Cornish-Fisher": var_cornish_fisher,
                "CVaR Paramétrique": cvar_parametric,
                "CVaR Historique": cvar_historical,
                "CVaR Cornish-Fisher": cvar_cornish_fisher,
                "buy_count": total_buy_trades,
                "sell_count": total_sell_trades,
                "days_invested": days_invested
            } 
        capital_evolution_df = pd.DataFrame(capital_evolution, columns=['Date', 'Capital'])
        capital_evolution_df.drop_duplicates(inplace=True)
        capital_evolution_df.sort_values(by='Date', inplace=True)
        # 📊 Agrégation des résultats
        total_gain = sum(result["gain_total"] for result in portfolio_results.values())
        total_percentage_gain = np.mean([result["pourcentage_gain_total"] for result in portfolio_results.values()])
        
        # **Correction de la performance annualisée**
        if total_days > 0:
            performance_annualisee = ((total_final_capital / total_initial_capital) ** (365 / total_days) - 1) * 100
        else:
            performance_annualisee = 0  # Éviter la division par zéro si aucun jour n'a été comptabilisé
        return {
            "gain_total": total_gain,
            "pourcentage_gain_total": total_percentage_gain,
            "performance_annualisee": performance_annualisee,
            "volatilite_historique": np.mean([result["volatilite_historique"] for result in portfolio_results.values()]),
            "VaR Paramétrique": np.mean([result["VaR Paramétrique"] for result in portfolio_results.values()]),
            "VaR Historique": np.mean([result["VaR Historique"] for result in portfolio_results.values()]),
            "VaR Cornish-Fisher": np.mean([result["VaR Cornish-Fisher"] for result in portfolio_results.values()]),
            "CVaR Paramétrique": np.mean([result["CVaR Paramétrique"] for result in portfolio_results.values()]),
            "CVaR Historique": np.mean([result["CVaR Historique"] for result in portfolio_results.values()]),
            "CVaR Cornish-Fisher": np.mean([result["CVaR Cornish-Fisher"] for result in portfolio_results.values()]),
            "total_buy_trades": total_buy_trades,
            "total_sell_trades": total_sell_trades,
            "capital_final_total": total_final_capital,
            "capital_evolution":capital_evolution_df
        }
