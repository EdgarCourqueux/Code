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
        super().__init__()
        self.tickers = tickers
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        self.lookback_period = lookback_period
        self.models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True),
            'KNN': KNeighborsClassifier(),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'AdaBoost': AdaBoostClassifier(algorithm='SAMME', random_state=42),
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
        }
        self.allocation = initial_capital / len(tickers)
        self.data_downloader = DataDownloader()
        self.positions = {ticker: {'status': 'cash', 'shares': 0} for ticker in tickers}

    def add_technical_features(self, data):
        """Add technical indicators for prediction."""
        df = data.copy()
        df['Returns'] = df['Close'].pct_change()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        return df
    
    def count_trades(self, trades):
        buy_count = sum(1 for trade in trades if trade['action'] == 'buy')
        sell_count = sum(1 for trade in trades if trade['action'] == 'sell')
        return buy_count, sell_count

    def calculate_rsi(self, prices, periods=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def preprocess_data(self, data):
        """Prepare data for model training."""
        processed_data = self.add_technical_features(data)
        processed_data['Target'] = np.where(processed_data['Returns'].shift(-1) > 0, 1, 0)
        return processed_data.dropna()

    def train_and_evaluate(self, data):
        """Train models and select the best one."""
        feature_cols = ['Returns', 'SMA_20', 'RSI', 'Volume_Ratio']
        X = data[feature_cols]
        y = data['Target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        best_model = None
        best_model_name = None
        best_accuracy = 0
        
        for name, model in self.models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    best_model_name = name
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        return best_model, best_model_name, best_accuracy

    def execute_trades(self, model, data, ticker):
        """Execute trades based on predictions and track performance."""
        feature_cols = ['Returns', 'SMA_20', 'RSI', 'Volume_Ratio']
        capital = self.allocation
        position = None
        entry_price = None
        trades_history = []
        
        for i in range(len(data)-1):
            features = data.iloc[i][feature_cols]
            features_df = pd.DataFrame([features], columns=feature_cols)
            prediction = model.predict(features_df)[0]
            current_price = data.iloc[i]['Close']
            
            if prediction == 1 and position is None:  # Buy signal
                shares = capital // current_price
                capital -= shares * current_price
                position = shares
                entry_price = current_price
                trades_history.append({
                    'date': data.index[i],
                    'action': 'buy',
                    'price': current_price,
                    'shares': shares
                })
            
            elif prediction == 0 and position is not None:  # Sell signal
                capital += position * current_price
                trades_history.append({
                    'date': data.index[i],
                    'action': 'sell',
                    'price': current_price,
                    'shares': position
                })
                position = None
        
        # Final liquidation if still holding
        if position is not None:
            final_price = data.iloc[-1]['Close']
            capital += position * final_price
            trades_history.append({
                'date': data.index[-1],
                'action': 'sell',
                'price': final_price,
                'shares': position
            })
        
        return capital, trades_history

    def execute(self):
        print(1)
        """Execute the strategy and calculate performance metrics."""
        portfolio_results = {}
        
        for ticker in self.tickers:
            data = self.data_downloader.download_data(ticker, self.start_date, self.end_date)
            if data.empty:
                print(f"Les données pour {ticker} sont vides. Vérifiez le ticker ou la période de téléchargement.")
                continue

            data.reset_index(inplace=True)
            data['Date'] = pd.to_datetime(data['Date'])

            try:
                date_investissement_proche = data.loc[data['Date'] >= self.start_date, 'Date'].iloc[0]
                date_du_jour_proche = data.loc[data['Date'] <= self.end_date, 'Date'].iloc[-1]
            except IndexError:
                print(f"La date d'investissement ou de fin est hors de la plage des données disponibles pour {ticker}.")
                continue

            processed_data = self.preprocess_data(data)
            best_model, model_name, accuracy = self.train_and_evaluate(processed_data)
            
            if best_model is None:
                continue

            final_capital, trades = self.execute_trades(best_model, processed_data, ticker)
            
            # Calculate performance metrics
            performance_results = self.performance(data, self.allocation, date_investissement_proche, date_du_jour_proche)
            if performance_results is None:
                continue

            try:
                # Add risk metrics
                performance_results["volatilite_historique"] = self.volatilite_historique(data).get("volatilite_historique", 0)
                performance_results["var_parametric"] = self.calculate_var(data, alpha=0.05, method="parametric")
                performance_results["var_historical"] = self.calculate_var(data, alpha=0.05, method="historical")
                performance_results["var_cornish_fisher"] = self.calculate_var(data, alpha=0.05, method="cornish-fisher")
                performance_results["cvar_parametric"] = self.calculate_cvar(data, alpha=0.05, method="parametric")
                performance_results["cvar_historical"] = self.calculate_cvar(data, alpha=0.05, method="historical")
                performance_results["cvar_cornish_fisher"] = self.calculate_cvar(data, alpha=0.05, method="cornish-fisher")
                performance_results["model"] = model_name
                performance_results["model_accuracy"] = accuracy
                performance_results["trades"] = trades
            except Exception as e:
                print(f"Erreur lors du calcul des indicateurs pour {ticker} : {e}")
                continue

            portfolio_results[ticker] = performance_results

        return self.aggregate_portfolio_results(portfolio_results)

    def aggregate_portfolio_results(self, portfolio_results):
        """Aggregate results across all tickers."""
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

        # Calculate means for portfolio metrics
        total_gain = sum(float(result['gain_total']) for result in portfolio_results.values() if result['gain_total'] is not None)
        total_percentage_gain = np.mean([float(result['pourcentage_gain_total']) for result in portfolio_results.values() if result['pourcentage_gain_total'] is not None])
        performance_annualisee = np.mean([float(result['performance_annualisee']) for result in portfolio_results.values() if result['performance_annualisee'] is not None])
        
        # Calculate risk metrics
        volatilite_historique = np.mean([float(result['volatilite_historique']) for result in portfolio_results.values() if result['volatilite_historique'] is not None])
        var_parametric = np.mean([float(result['var_parametric']) for result in portfolio_results.values() if result['var_parametric'] is not None])
        var_historical = np.mean([float(result['var_historical']) for result in portfolio_results.values() if result['var_historical'] is not None])
        var_cornish_fisher = np.mean([float(result['var_cornish_fisher']) for result in portfolio_results.values() if result['var_cornish_fisher'] is not None])
        cvar_parametric = np.mean([float(result['cvar_parametric']) for result in portfolio_results.values() if result['cvar_parametric'] is not None])
        cvar_historical = np.mean([float(result['cvar_historical']) for result in portfolio_results.values() if result['cvar_historical'] is not None])
        cvar_cornish_fisher = np.mean([float(result['cvar_cornish_fisher']) for result in portfolio_results.values() if result['cvar_cornish_fisher'] is not None])

        # Get used models
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
