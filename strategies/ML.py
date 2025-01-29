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
    def __init__(self, tickers, start_date, end_date, initial_capital=1000, lookback_period=5):
        """
        Initialise la stratégie avec un nombre de jours d'historique configurable.
        """
        self.tickers = tickers
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        self.lookback_period = lookback_period  # Nombre de jours d'historique à utiliser
        self.models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(),
            'KNN': KNeighborsClassifier(),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'AdaBoost': AdaBoostClassifier(algorithm='SAMME', random_state=42),
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
        }
        self.data = None
        self.results = {}
        self.portfolio_performance = {}
        self.allocation = initial_capital / len(tickers)
        self.data_downloader = DataDownloader()

    def fetch_data(self):
        """Télécharge les données historiques pour les tickers sélectionnés."""
        self.data = {}
        for ticker in self.tickers:
            try:
                ticker_data = self.data_downloader.download_data(ticker, self.start_date, self.end_date)
                if not ticker_data.empty:
                    self.data[ticker] = ticker_data
                else:
                    print(f"Aucune donnée disponible pour {ticker}.")
            except Exception as e:
                print(f"Erreur lors du téléchargement des données pour {ticker} : {e}")

    def add_historical_features(self, data):
        """
        Ajoute des caractéristiques historiques basées sur le nombre de jours (lookback_period).
        """
        for i in range(1, self.lookback_period + 1):
            data[f'Return_{i}D'] = data['Close'].pct_change(i)  # Variation sur i jours
        data[f'SMA_{self.lookback_period}'] = data['Close'].rolling(window=self.lookback_period).mean()  # Moyenne mobile
        data[f'Volatility_{self.lookback_period}'] = data['Close'].pct_change().rolling(window=self.lookback_period).std()  # Volatilité
        return data

    def preprocess_data(self, ticker_data):
        """Prépare les données pour l'entraînement du modèle."""
        data = ticker_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy().dropna()

        # Ajouter les caractéristiques historiques
        data = self.add_historical_features(data)

        # Supprimer les lignes avec des valeurs manquantes dues au calcul d'indicateurs
        data = data.dropna()

        # Créer la cible (J+1)
        data['Target'] = (data['Close'].shift(-1) > data['Open'].shift(-1)).astype(int)

        # Supprimer la dernière ligne sans cible valide
        data = data[:-1]

        return data

    def train_and_evaluate(self, data):
        """Entraîne plusieurs modèles et sélectionne le meilleur basé sur la précision."""
        X = data.drop(columns=['Target', 'Close', 'Adj Close'])
        y = data['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        best_model = None
        best_model_name = None  # Ajout du nom du modèle
        best_accuracy = 0
        best_results = None

        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if y_pred.dtype != int and y_pred.dtype != bool:
                y_pred = (y_pred > 0.5).astype(int)

            accuracy = accuracy_score(y_test, y_pred)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_model_name = name  # Sauvegarde du nom du meilleur modèle
                best_results = pd.DataFrame({
                    'Open': X_test['Open'],
                    'Close': X_test.index.map(lambda idx: data.loc[idx, 'Close']),
                    'Real_Change': y_test,
                    'Predicted_Change': y_pred
                })

        if best_model is None:
            print("❌ Aucun modèle n'a été entraîné avec succès !")
            return None, None, None, 0  # Ajout de `None` pour `best_model_name`

        return best_model, best_model_name, best_results, best_accuracy  # Retourne aussi le nom du modèle


    def calculate_gains(self, results, allocation):
        """Calcule les gains basés sur les prédictions avec une allocation spécifique."""
        capital = allocation  # Utilise seulement l'allocation pour ce ticker
        capital_over_time = [capital]

        results['Percentage_Change'] = (results['Close'] - results['Open']) / results['Open']

        for _, row in results.iterrows():
            if row['Real_Change'] == row['Predicted_Change']:
                capital *= (1 + abs(row['Percentage_Change']))
            else:
                capital *= (1 - abs(row['Percentage_Change']))
            capital_over_time.append(capital)

        return capital_over_time

    def execute(self):
        """Exécute la stratégie sur tous les tickers."""
        portfolio_results = {}
        data_dict = {}

        for ticker in self.tickers:
            data = self.data_downloader.download_data(ticker, self.start_date, self.end_date)
            if data.empty:
                print(f"Les données pour {ticker} sont vides. Vérifiez le ticker ou la période de téléchargement.")
                continue

            data.reset_index(inplace=True)
            data['Date'] = pd.to_datetime(data['Date'])
            data_dict[ticker] = data

            try:
                date_investissement_proche = data.loc[data['Date'] >= self.start_date, 'Date'].iloc[0]
                date_du_jour_proche = data.loc[data['Date'] <= self.end_date, 'Date'].iloc[-1]
            except IndexError:
                print(f"La date d'investissement ou de fin est hors de la plage des données disponibles pour {ticker}.")
                continue

            processed_data = self.preprocess_data(data)
            best_model, best_model_name, best_results, best_accuracy = self.train_and_evaluate(processed_data)

            if best_model is None or best_results is None:
                print(f"⚠️ Aucune prédiction valide pour {ticker}, ce ticker sera ignoré.")
                continue

            performance_results = self.performance(data, self.allocation, date_investissement_proche, date_du_jour_proche)
            if performance_results is None:
                print(f"Erreur dans le calcul de la performance pour {ticker}.")
                continue

            try:
                performance_results["volatilite_historique"] = self.volatilite_historique(data).get("volatilite_historique", 0)
                performance_results["var_parametric"] = self.calculate_var(data, alpha=0.05, method="parametric")
                performance_results["var_historical"] = self.calculate_var(data, alpha=0.05, method="historical")
                performance_results["var_cornish_fisher"] = self.calculate_var(data, alpha=0.05, method="cornish-fisher")
                performance_results["cvar_parametric"] = self.calculate_cvar(data, alpha=0.05, method="parametric")
                performance_results["cvar_historical"] = self.calculate_cvar(data, alpha=0.05, method="historical")
                performance_results["cvar_cornish_fisher"] = self.calculate_cvar(data, alpha=0.05, method="cornish-fisher")
            except Exception as e:
                print(f"Erreur lors du calcul des indicateurs pour {ticker} : {e}")
                continue

            # Ajoute le modèle utilisé
            performance_results["model"] = best_model_name if best_model_name else "Aucun Modèle"

            # DEBUG - Vérifie les clés avant de les stocker
            print(f"\nDEBUG - Résultats pour {ticker}:")
            print(performance_results)

            for key in [
                "gain_total", "pourcentage_gain_total", "performance_annualisee", 
                "volatilite_historique", "var_parametric", "var_historical", "var_cornish_fisher", 
                "cvar_parametric", "cvar_historical", "cvar_cornish_fisher"
            ]:
                performance_results.setdefault(key, 0)

            portfolio_results[ticker] = performance_results

        self.results = portfolio_results
        self.portfolio_performance = self.aggregate_portfolio_results(portfolio_results)
        return self.portfolio_performance



    def aggregate_portfolio_results(self, portfolio_results):
        """Agrège les résultats pour le portefeuille global."""
        if not portfolio_results:
            return {
                "gain_total": 0,
                "pourcentage_gain_total": 0,
                "performance_annualisee": 0,
                "volatilite_historique": 0,
                "VaR Paramétrique": 0,
                "VaR Historique": 0,
                "VaR Cornish-Fisher": 0,
                "CVaR Paramétrique": 0,
                "CVaR Historique": 0,
                "CVaR Cornish-Fisher": 0,
                "model": "Aucun Modèle"
            }

        # Filtrer les valeurs valides et éviter les erreurs sur None
        def safe_mean(values):
            valid_values = [float(v) for v in values if v is not None]
            return np.mean(valid_values) if valid_values else 0

        # Calculer les agrégations
        total_gain = sum(float(result.get('gain_total', 0)) for result in portfolio_results.values())
        total_percentage_gain = safe_mean([result.get('pourcentage_gain_total', 0) for result in portfolio_results.values()])
        performance_annualisee = safe_mean([result.get('performance_annualisee', 0) for result in portfolio_results.values()])
        volatilite_historique = safe_mean([result.get('volatilite_historique', 0) for result in portfolio_results.values()])
        var_parametric = safe_mean([result.get('var_parametric', 0) for result in portfolio_results.values()])
        var_historical = safe_mean([result.get('var_historical', 0) for result in portfolio_results.values()])
        var_cornish_fisher = safe_mean([result.get('var_cornish_fisher', 0) for result in portfolio_results.values()])
        cvar_parametric = safe_mean([result.get('cvar_parametric', 0) for result in portfolio_results.values()])
        cvar_historical = safe_mean([result.get('cvar_historical', 0) for result in portfolio_results.values()])
        cvar_cornish_fisher = safe_mean([result.get('cvar_cornish_fisher', 0) for result in portfolio_results.values()])

        # Gérer les modèles utilisés
        models_used = list(set(result.get("model", "Aucun Modèle") for result in portfolio_results.values()))
        model_summary = ", ".join(models_used) if models_used else "Aucun Modèle"

        return {
            "gain_total": total_gain,
            "pourcentage_gain_total": total_percentage_gain,
            "performance_annualisee": performance_annualisee,
            "volatilite_historique": volatilite_historique,
            "VaR Paramétrique": var_parametric,
            "VaR Historique": var_historical,
            "VaR Cornish-Fisher": var_cornish_fisher,
            "CVaR Paramétrique": cvar_parametric,
            "CVaR Historique": cvar_historical,
            "CVaR Cornish-Fisher": cvar_cornish_fisher,
            "model": model_summary
        }




    def get_summary(self):
        """
        Retourne un résumé des performances de la stratégie sous forme de dictionnaire structuré.
        """
        if not self.results:
            return {"error": "Aucun résultat disponible. Assurez-vous d'avoir exécuté la stratégie."}

        summary_dict = {"detailed_results": {}, "portfolio_performance": self.portfolio_performance}

        # Récupérer les performances par ticker
        for ticker, result in self.results.items():
            summary_dict["detailed_results"][ticker] = {
                "Meilleur Modèle": result.get('model', "Aucun Modèle"),
                "Précision": round(result.get('accuracy', 0), 4),
                "Gain Total (€)": round(result.get('gain_total', 0), 2),
                "Pourcentage Gain Total (%)": round(result.get('pourcentage_gain_total', 0), 2),
                "Performance Annualisée (%)": round(result.get('performance_annualisee', 0), 2),
                "Volatilité Historique": round(result.get('volatilite_historique', 0), 4),
                "VaR Paramétrique": round(result.get('var_parametric', 0), 4),
                "VaR Historique": round(result.get('var_historical', 0), 4),
                "VaR Cornish-Fisher": round(result.get('var_cornish_fisher', 0), 4),
                "CVaR Paramétrique": round(result.get('cvar_parametric', 0), 4),
                "CVaR Historique": round(result.get('cvar_historical', 0), 4),
                "CVaR Cornish-Fisher": round(result.get('cvar_cornish_fisher', 0), 4)
            }
        return summary_dict
