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
        self.initial_capital = initial_capital/len(tickers)
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
        """Entraîne le modèle en utilisant uniquement les données jusqu'à start_date."""
        best_model = None
        # Garder uniquement les données avant start_date pour l'entraînement
        train_data = data[data.index < self.start_date]
        if train_data.empty:
            print("⚠️ Pas assez de données pour l'entraînement.")
            return None, None
        feature_cols = [col for col in train_data.columns if "lag" in col]  # Features historiques
        X = train_data[feature_cols]
        y = train_data['Target']
        # Séparer les données pour l'entraînement et le test (validation)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        best_accuracy = 0
        best_model = None
        for name, model in self.models.items():
            try:
                model.fit(X_train, y_train)  # Entraînement uniquement sur les données avant start_date
                y_pred = model.predict(X_val)
                accuracy = accuracy_score(y_val, y_pred)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
            except Exception as e:
                print(f"⚠️ Erreur d'entraînement sur {name} : {e}")
                continue
        return best_model, feature_cols  # On retourne le modèle entraîné

    
    def execute_trades(self):
        """Exécute la stratégie sur la période après l'entraînement (start_date → end_date) pour chaque ticker indépendamment."""

        results = {}  # Stocke les résultats par ticker

        for ticker in self.tickers:
            print(f"\n🚀 Exécution de la stratégie pour {ticker}...")

            # Télécharger les données pour le ticker
            data = self.data_downloader.download_data(ticker, "2010-01-01", self.end_date)

            if data.empty:
                print(f"⚠️ Les données pour {ticker} sont vides.")
                continue

            print(f"📊 Données téléchargées pour {ticker}: {data.shape}")

            # Prétraitement des données
            processed_data = self.preprocess_data(data)

            # Séparer les données pour l'entraînement et la prédiction
            train_data = processed_data.loc[processed_data.index < self.start_date]
            test_data = processed_data.loc[(processed_data.index >= self.start_date) & (processed_data.index <= self.end_date)]

            # Vérification finale pour éviter la fuite de données
            assert train_data.index.max() < test_data.index.min(), "⚠️ Fuite de données détectée ! Vérifiez self.start_date."

            if train_data.empty or test_data.empty:
                print(f"⚠️ Pas assez de données pour entraîner et tester {ticker}.")
                continue

            # Entraîner le modèle sur les données du ticker
            best_model, selected_features = self.train_and_evaluate(train_data)

            if best_model is None:
                print(f"❌ Aucun modèle sélectionné pour {ticker}.")
                continue

            # Initialiser les compteurs spécifiques au ticker
            buy_trades = 0
            sell_trades = 0
            capital = self.initial_capital
            capital_history = []

            # Exécution des trades sur la période de test
            for i in range(len(test_data) - 1):  # On ne peut pas prédire le dernier jour (pas de J+1 disponible)
                date_j = test_data.index[i]  # Date actuelle (J)
                date_j1 = test_data.index[i + 1]  # Date de J+1 (où l'on applique la prédiction)

                # Utiliser les données de J-lookback_period pour faire la prédiction
                if i - self.lookback_period < 0:
                    continue  # Ignorer les premiers jours où l'on n'a pas assez d'historique

                date_lookback = test_data.index[i - self.lookback_period]  # Date de référence pour les features
                feature_values = test_data.loc[date_lookback, selected_features]

                # Vérifier si on a bien une série, sinon transformer en DataFrame
                if isinstance(feature_values, pd.Series):
                    feature_values = feature_values.to_frame().T  

                feature_df = feature_values.reindex(columns=selected_features, fill_value=0)

                # Prédiction basée sur les données de J-lookback_period pour J+1
                prediction = best_model.predict(feature_df)[0]

                # Appliquer l'action sur la journée actuelle (J+1)
                open_price = test_data.loc[date_j1, 'Open']
                close_price = test_data.loc[date_j1, 'Close']

                # Vérification contre NaN ou division par zéro
                if pd.isna(open_price) or pd.isna(close_price) or open_price == 0:
                    print(f"⚠️ Données invalides pour {ticker} à la date {date_j1}.")
                    continue

                daily_return = (close_price - open_price) / open_price
                daily_return_percentage = daily_return * 100

                # Exécuter le trade en fonction de la prédiction
                action = "Hold"
                if prediction == 1:
                    capital *= (1 + daily_return)
                    action = "Buy"
                    buy_trades += 1
                elif prediction == 0:
                    capital *= (1 - daily_return)
                    action = "Sell"
                    sell_trades += 1

                capital_history.append((date_j1, capital))  # Stocker le capital à J+1

                print(f"📅 {date_j1}, {ticker} | Action: {action}, Daily Return: {daily_return_percentage:.2f}%, Capital: {capital:.2f}")

            # Stocker les résultats par ticker
            results[ticker] = {
                "final_capital": capital,
                "capital_history": capital_history,
                "best_model": best_model,
                "buy_trades": buy_trades,
                "sell_trades": sell_trades
            }

        return results



    def execute(self):
        """Exécute la stratégie d'investissement sur l'ensemble des actifs et calcule les performances finales."""

        portfolio_results = {}
        total_initial_capital = self.initial_capital  # Capital initial global
        total_final_capital = 0  # Capital final global
        capital_evolution = []  # Liste pour stocker l'évolution du capital avec les tickers
        days_invested_per_ticker = []  # Stocke les jours investis pour chaque ticker
        performance_annualisee_per_ticker = []  # Stocke la performance annualisée par ticker
        trade_summary = []  # Stocke le résumé des transactions et modèles pour chaque ticker

        # ✅ Exécuter la stratégie UNE SEULE FOIS pour tous les tickers
        trade_results = self.execute_trades()

        for ticker in self.tickers:
            print(f"\n📈 Analyse des résultats pour {ticker}...")  

            if ticker not in trade_results:
                print(f"⚠️ Aucune donnée de trading pour {ticker}.")
                continue

            # Extraire les résultats spécifiques à ce ticker
            final_capital = trade_results[ticker]["final_capital"]
            capital_history = trade_results[ticker]["capital_history"]
            buy_trades = trade_results[ticker]["buy_trades"]
            sell_trades = trade_results[ticker]["sell_trades"]
            best_model = trade_results[ticker]["best_model"]  # Ajouter le modèle utilisé

            total_final_capital += final_capital

            # ✅ Ajout du ticker à l'évolution du capital
            for date, capital in capital_history:
                capital_evolution.append({"Date": date, "Capital": capital, "Ticker": ticker})

            # Télécharger les données pour le calcul des métriques de risque
            data = self.data_downloader.download_data(ticker, self.start_date, self.end_date)
            if data.empty:
                print(f"⚠️ Les données pour {ticker} sont vides. Vérifiez le ticker ou la période de téléchargement.")
                continue

            data.reset_index(inplace=True)
            data['Date'] = pd.to_datetime(data['Date'])

            # Calcul du nombre de jours investis
            try:
                date_investissement_proche = data.loc[data['Date'] >= self.start_date, 'Date'].iloc[0]
                date_du_jour_proche = data.loc[data['Date'] <= self.end_date, 'Date'].iloc[-1]
                days_invested = (date_du_jour_proche - date_investissement_proche).days
                days_invested_per_ticker.append(days_invested)  # Ajout à la liste des jours investis
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

            # ✅ Calcul de la performance annualisée par ticker
            if days_invested > 0:
                performance_ticker = ((final_capital / self.initial_capital) ** (365 / max(days_invested, 1)) - 1) * 100
                performance_annualisee_per_ticker.append(performance_ticker)

            # ✅ Calcul du gain total et pourcentage de gain
            gain_total = float(final_capital - self.initial_capital)
            pourcentage_gain_total = float(((final_capital / self.initial_capital) - 1) * 100)

            # ✅ Ajout des résultats dans `portfolio_results`
            portfolio_results[ticker] = {
                "gain_total": gain_total,
                "pourcentage_gain_total": pourcentage_gain_total,
                "volatilite_historique": float(volatilite_historique) if isinstance(volatilite_historique, pd.Series) else volatilite_historique,
                "VaR Paramétrique": float(var_parametric) if isinstance(var_parametric, pd.Series) else var_parametric,
                "VaR Historique": float(var_historical) if isinstance(var_historical, pd.Series) else var_historical,
                "VaR Cornish-Fisher": float(var_cornish_fisher) if isinstance(var_cornish_fisher, pd.Series) else var_cornish_fisher,
                "CVaR Paramétrique": float(cvar_parametric) if isinstance(cvar_parametric, pd.Series) else cvar_parametric,
                "CVaR Historique": float(cvar_historical) if isinstance(cvar_historical, pd.Series) else cvar_historical,
                "CVaR Cornish-Fisher": float(cvar_cornish_fisher) if isinstance(cvar_cornish_fisher, pd.Series) else cvar_cornish_fisher,
                "days_invested": int(days_invested)
            }

            trade_summary.append({
                "Ticker": ticker,
                "Buy_Trades": buy_trades,
                "Sell_Trades": sell_trades,
                "Model": type(best_model).__name__,  # ✅ Extrait uniquement le nom du modèle
                "Capital Généré (€)": gain_total,
                "Gain (%)": pourcentage_gain_total
            })


        # ✅ Moyenne pondérée des performances annualisées par ticker
        if performance_annualisee_per_ticker:
            performance_annualisee = np.mean(performance_annualisee_per_ticker)
        else:
            performance_annualisee = 0  # Cas où aucun ticker n'a de performance

        # ✅ Convertir la liste capital_evolution en DataFrame
        capital_evolution_df = pd.DataFrame(capital_evolution)
        capital_evolution_df.sort_values(by=['Date', 'Ticker'], inplace=True)

        # ✅ Convertir `trade_summary` en DataFrame
        trade_summary_df = pd.DataFrame(trade_summary)

        return {
            "gain_total": sum(result["gain_total"] for result in portfolio_results.values()),
            "pourcentage_gain_total": np.mean([result["pourcentage_gain_total"] for result in portfolio_results.values()]),
            "performance_annualisee": performance_annualisee,
            "volatilite_historique": np.mean([result["volatilite_historique"] for result in portfolio_results.values()]).item(),
            "VaR Paramétrique": np.mean([result["VaR Paramétrique"] for result in portfolio_results.values()]).item(),
            "VaR Historique": np.mean([result["VaR Historique"] for result in portfolio_results.values()]),
            "VaR Cornish-Fisher": np.mean([result["VaR Cornish-Fisher"] for result in portfolio_results.values()]),
            "CVaR Paramétrique": np.mean([result["CVaR Paramétrique"] for result in portfolio_results.values()]),
            "CVaR Historique": np.mean([result["CVaR Historique"] for result in portfolio_results.values()]),
            "CVaR Cornish-Fisher": np.mean([result["CVaR Cornish-Fisher"] for result in portfolio_results.values()]),
            "capital_final_total": total_final_capital,
            "capital_evolution": capital_evolution_df,
            "trade_summary": trade_summary_df  # ✅ Contient maintenant le capital et le pourcentage de gain
        }
