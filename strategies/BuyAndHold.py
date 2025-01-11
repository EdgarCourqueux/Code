import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Indicateurs import Indicateurs

class BuyAndHold(Indicateurs):
    def __init__(self, montant_initial, date_investissement, tickers, date_fin_investissement):
        self.montant_initial = montant_initial
        self.date_investissement = pd.to_datetime(date_investissement)
        self.date_fin_investissement = pd.to_datetime(date_fin_investissement)
        self.tickers = tickers
        self.allocation = montant_initial / len(tickers)

    def execute(self):
        portfolio_results = {}
        data_dict = {}

        for ticker in self.tickers:
            data = yf.download(ticker, start="2000-01-01", end=self.date_fin_investissement.strftime('%Y-%m-%d'))
            if data.empty:
                print(f"Les données pour {ticker} sont vides. Vérifiez le ticker ou la période de téléchargement.")
                continue

            data.reset_index(inplace=True)
            data['Date'] = pd.to_datetime(data['Date'])
            data_dict[ticker] = data

            try:
                date_investissement_proche = data.loc[data['Date'] >= self.date_investissement, 'Date'].iloc[0]
                date_du_jour_proche = data.loc[data['Date'] <= self.date_fin_investissement, 'Date'].iloc[-1]
            except IndexError:
                print(f"La date d'investissement ou de fin est hors de la plage des données disponibles pour {ticker}.")
                continue

            # Calculer la performance pour chaque ticker
            performance_results = self.performance(data, self.allocation, date_investissement_proche, date_du_jour_proche)
            if performance_results is None:
                print(f"Erreur dans le calcul de la performance pour {ticker}.")
                continue

            # Calcul des indicateurs de volatilité, VaR, et CVaR
            try:
                performance_results["volatilite_historique"] = self.volatilite_historique(data).get("volatilite_historique", 0)
                performance_results["ewma_volatility"] = self.calculate_ewma_volatility(data['Adj Close']).iloc[-1] if not data['Adj Close'].empty else 0
                performance_results["var_parametric"] = self.calculate_var(data, alpha=0.05, method="parametric")
                performance_results["var_historical"] = self.calculate_var(data, alpha=0.05, method="historical")
                performance_results["var_cornish_fisher"] = self.calculate_var(data, alpha=0.05, method="cornish-fisher")
                performance_results["cvar_parametric"] = self.calculate_cvar(data, alpha=0.05, method="parametric")
                performance_results["cvar_historical"] = self.calculate_cvar(data, alpha=0.05, method="historical")
                performance_results["cvar_cornish_fisher"] = self.calculate_cvar(data, alpha=0.05, method="cornish-fisher")
            except Exception as e:
                print(f"Erreur lors du calcul des indicateurs pour {ticker} : {e}")
                continue

            # Assurez-vous que toutes les valeurs sont présentes
            for key in ["gain_total", "pourcentage_gain_total", "performance_annualisee", 
                        "volatilite_historique", "ewma_volatility", 
                        "var_parametric", "var_historical", "var_cornish_fisher", 
                        "cvar_parametric", "cvar_historical", "cvar_cornish_fisher"]:
                performance_results.setdefault(key, 0)
            portfolio_results[ticker] = performance_results

        # Agréger les résultats du portefeuille
        portfolio_performance = self.aggregate_portfolio_results(portfolio_results)
        return portfolio_performance

    def aggregate_portfolio_results(self, portfolio_results):
        # Calculer la performance globale du portefeuille
        total_gain = sum(float(result['gain_total']) for result in portfolio_results.values() if result['gain_total'] is not None)
        total_percentage_gain = sum(float(result['pourcentage_gain_total']) for result in portfolio_results.values() if result['pourcentage_gain_total'] is not None) / len(portfolio_results)
        performance_annualisee = sum(float(result['performance_annualisee']) for result in portfolio_results.values() if result['performance_annualisee'] is not None) / len(portfolio_results)

        # Agrégation des indicateurs de volatilité, VaR, CVaR
        volatilite_historique = np.mean([float(result['volatilite_historique']) for result in portfolio_results.values() if result['volatilite_historique'] is not None])
        ewma_volatility = np.mean([float(result['ewma_volatility']) for result in portfolio_results.values() if result['ewma_volatility'] is not None])
        
        # Calcul de la moyenne pour chaque type de VaR
        var_parametric = np.mean([float(result['var_parametric']) for result in portfolio_results.values() if result['var_parametric'] is not None])
        var_historical = np.mean([float(result['var_historical']) for result in portfolio_results.values() if result['var_historical'] is not None])
        var_cornish_fisher = np.mean([float(result['var_cornish_fisher']) for result in portfolio_results.values() if result['var_cornish_fisher'] is not None])

        # Calcul de la moyenne pour chaque type de CVaR
        cvar_parametric = np.mean([float(result['cvar_parametric']) for result in portfolio_results.values() if result['cvar_parametric'] is not None])
        cvar_historical = np.mean([float(result['cvar_historical']) for result in portfolio_results.values() if result['cvar_historical'] is not None])
        cvar_cornish_fisher = np.mean([float(result['cvar_cornish_fisher']) for result in portfolio_results.values() if result['cvar_cornish_fisher'] is not None])

        return {
            "gain_total": total_gain,
            "pourcentage_gain_total": total_percentage_gain,
            "performance_annualisee": performance_annualisee,
            "volatilite_historique": volatilite_historique,
            "ewma_volatility": ewma_volatility,
            "VaR Paramétrique": var_parametric,
            "VaR Historique": var_historical,
            "VaR Cornish-Fisher": var_cornish_fisher,
            "CVaR Paramétrique": cvar_parametric,
            "CVaR Historique": cvar_historical,
            "CVaR Cornish-Fisher": cvar_cornish_fisher
        }