import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from strategies.BuyAndHold import BuyAndHold

class Momentum:
    def __init__(self, montant_initial, tickers, nombre_actifs, periode_reroll, fenetre_retrospective, date_debut, date_fin):
        self.montant_initial = montant_initial
        self.tickers = tickers
        self.nombre_actifs = nombre_actifs
        self.periode_reroll = periode_reroll
        self.fenetre_retrospective = fenetre_retrospective
        self.date_debut = pd.to_datetime(date_debut)
        self.date_fin = pd.to_datetime(date_fin)

    def get_data(self, ticker, start_date, end_date):
        try:
            data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
            if not data.empty:
                data.reset_index(inplace=True)
                data['Date'] = pd.to_datetime(data['Date'])
            return data
        except Exception as e:
            print(f"Erreur lors du téléchargement des données pour {ticker}: {e}")
            return pd.DataFrame()

    def calculate_momentum(self, data, start_date, end_date):
        data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
        if len(data) < 2:
            return None
        return data['Adj Close'].iloc[-1] / data['Adj Close'].iloc[0] - 1

    def select_top_assets(self, current_date):
        start_date = current_date - timedelta(days=self.fenetre_retrospective)
        momentum_scores = {}

        for ticker in self.tickers:
            data = self.get_data(ticker, start_date, current_date)
            if data.empty:
                continue
            momentum = self.calculate_momentum(data, start_date, current_date)
            if momentum is not None:
                momentum_scores[ticker] = momentum

        sorted_assets = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        return [asset[0] for asset in sorted_assets[:self.nombre_actifs]]

    def execute(self):
        portfolio_value = self.montant_initial  # Valeur initiale correcte
        current_date = self.date_debut

        # Résultats pour les graphiques
        dates = []
        valeurs_portefeuille = [self.montant_initial]  # Ajoutez la valeur initiale ici
        repartition = pd.DataFrame()
        buy_and_hold_indicators = {}

        # Suivi des occurrences des actifs dans le portefeuille
        asset_occurrences = {ticker: 0 for ticker in self.tickers}

        final_date = min(self.date_fin, datetime.now())

        while current_date <= final_date:
            dates.append(current_date)

            # Sélectionner les meilleurs actifs
            top_assets = self.select_top_assets(current_date)
            if not top_assets:
                print("Aucun actif sélectionné.")
                break

            try:
                # Définir la date de fin de la période actuelle
                date_fin_periode = min(current_date + timedelta(days=self.periode_reroll), final_date)

                # Exécuter BuyAndHold pour cette période
                buy_and_hold = BuyAndHold(portfolio_value, current_date, top_assets, date_fin_periode)
                performance_results = buy_and_hold.execute()

                # Mise à jour de la valeur du portefeuille
                portfolio_value += performance_results.get('gain_total', 0)
                valeurs_portefeuille.append(portfolio_value)

                # Mise à jour de la répartition
                allocation = {asset: 1 / len(top_assets) for asset in top_assets}
                repartition_current = pd.DataFrame([allocation], index=[current_date])
                repartition = pd.concat([repartition, repartition_current])

                # Mettre à jour les occurrences des actifs
                for asset in top_assets:
                    asset_occurrences[asset] += 1

                # Ajouter les indicateurs supplémentaires de BuyAndHold, sauf ceux déjà présents
                for key, value in performance_results.items():
                    if key not in ["gain_total", "pourcentage_gain_total", "dates", "valeurs_portefeuille", "repartition", "performance_annualisee"]:
                        buy_and_hold_indicators[key] = value

            except Exception as e:
                print(f"Erreur lors de l'évaluation : {e}")
                break

            current_date += timedelta(days=self.periode_reroll)

        # Calcul de la performance annualisée
        years_elapsed = (self.date_fin - self.date_debut).days / 365.25
        performance_annualisee = (portfolio_value / self.montant_initial) ** (1 / years_elapsed) - 1 if years_elapsed > 0 else 0

        # Calcul du taux d'apparition
        total_months = (self.date_fin - self.date_debut).days / 30.44
        asset_appearance_rate = {ticker: occurrences / total_months for ticker, occurrences in asset_occurrences.items()}

        return {
            "dates": dates,
            "valeurs_portefeuille": valeurs_portefeuille,
            "repartition": repartition,
            "gain_total": portfolio_value - self.montant_initial,
            "pourcentage_gain_total": (portfolio_value / self.montant_initial - 1) * 100,
            "performance_annualisee": performance_annualisee * 100,
            "taux_apparition": asset_appearance_rate,  # Ajouter le taux d'apparition
            **buy_and_hold_indicators,
        }
