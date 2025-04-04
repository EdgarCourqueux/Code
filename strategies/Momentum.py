from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from strategies.BuyAndHold import BuyAndHold
from strategies.data import DataDownloader

class Momentum:
    def __init__(self, montant_initial, tickers, nombre_actifs, periode_reroll, fenetre_retrospective, date_debut, date_fin):
        self.montant_initial = montant_initial
        self.tickers = tickers
        self.nombre_actifs = nombre_actifs
        self.periode_reroll = periode_reroll
        self.fenetre_retrospective = fenetre_retrospective
        self.date_debut = pd.to_datetime(date_debut)
        self.date_fin = pd.to_datetime(date_fin)
        self.data_downloader = DataDownloader()  # Instanciation de DataDownloader

    def validate_data(self, data):
        """
        Vérifie et transforme les données pour s'assurer qu'elles sont au bon format.
        """
        if data.empty:
            return data

        # S'assurer que l'index est une colonne temporelle
        if not pd.api.types.is_datetime64_any_dtype(data.index):
            data.index = pd.to_datetime(data.index, errors="coerce")

        # Supprimer les colonnes inutiles si MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(0)

        # Vérifier les colonnes critiques
        required_columns = ["Adj Close"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Colonne manquante dans les données : {col}")

        # Remplir les valeurs manquantes avec ffill/bfill
        data = data.ffill().bfill()
        return data

    def calculate_momentum(self, data):
        if len(data) < 2:
            return None
        return data['Adj Close'].iloc[-1] / data['Adj Close'].iloc[0] - 1

    def select_top_assets(self, current_date):
        """
        Sélectionne les actifs ayant le mieux performé pendant la période fenetre_retrospective.
        """
        start_date = max(pd.to_datetime("2010-01-01"), current_date - timedelta(days=self.fenetre_retrospective))
        momentum_scores = {}

        for ticker in self.tickers:
            # Utilisation de DataDownloader pour télécharger les données
            data = self.data_downloader.download_data(ticker, start_date, current_date)
            try:
                data = self.validate_data(data)
            except ValueError as e:
                print(f"Erreur dans les données pour {ticker} : {e}")
                continue

            if data.empty:
                print(f"Pas de données pour {ticker} entre {start_date} et {current_date}")
                continue

            momentum = self.calculate_momentum(data)
            if momentum is not None:
                momentum_scores[ticker] = momentum

        # Trier les actifs par momentum décroissant et sélectionner les meilleurs
        sorted_assets = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        selected_assets = [asset[0] for asset in sorted_assets[:self.nombre_actifs]]

        print(f"Actifs sélectionnés à la date {current_date} : {selected_assets}")
        return selected_assets

    def execute(self):
        portfolio_value = self.montant_initial
        current_date = self.date_debut

        # Résultats pour les graphiques
        dates = []
        valeurs_portefeuille = [self.montant_initial]
        actifs_selectionnes = []
        valeurs_momentum = []
        performances_buy_and_hold = []
        repartition = pd.DataFrame()
        
        # Initialiser les accumulateurs pour les indicateurs de risque
        risk_indicators = {
            "volatilite_historique": 0,
            "ewma_volatility": 0,
            "VaR Paramétrique": 0,
            "VaR Historique": 0,
            "VaR Cornish-Fisher": 0,
            "CVaR Paramétrique": 0,
            "CVaR Historique": 0,
            "CVaR Cornish-Fisher": 0
        }
        
        # Suivi des occurrences des actifs dans le portefeuille
        asset_occurrences = {ticker: 0 for ticker in self.tickers}
        
        # Compteur pour le nombre de périodes avec données valides
        valid_periods = 0

        final_date = min(self.date_fin, datetime.now())
        
        all_results = []  # Pour stocker tous les résultats des périodes

        while current_date <= final_date:
            dates.append(current_date)

            # Sélectionner les meilleurs actifs
            top_assets = self.select_top_assets(current_date)
            actifs_selectionnes.append({"Actifs sélectionnés": top_assets})

            if not top_assets:
                print("Aucun actif sélectionné.")
                break

            try:
                # Définir la date de fin de la période actuelle
                date_fin_periode = min(current_date + timedelta(days=self.periode_reroll), final_date)

                # Exécuter BuyAndHold pour cette période
                buy_and_hold = BuyAndHold(portfolio_value, current_date, top_assets, date_fin_periode)
                performance_results = buy_and_hold.execute()
                
                # Stocker les résultats de cette période
                all_results.append(performance_results)

                # Vérifier si nous avons des données valides pour cette période
                has_valid_data = performance_results and any(
                    key in performance_results and performance_results[key] != 0 
                    for key in ["VaR Paramétrique", "VaR Historique", "VaR Cornish-Fisher"]
                )
                
                if has_valid_data:
                    valid_periods += 1
                    # Accumuler les indicateurs de risque
                    for key in risk_indicators.keys():
                        if key in performance_results and performance_results[key] is not None:
                            # Gérer à la fois les séries et les valeurs simples
                            value = float(performance_results[key].iloc[0]) if isinstance(performance_results[key], pd.Series) else float(performance_results[key])
                            risk_indicators[key] += value

                # Mise à jour de la valeur du portefeuille
                portfolio_value += performance_results.get('gain_total', 0)
                valeurs_portefeuille.append(portfolio_value)

                # Ajouter les performances buy and hold
                performances_buy_and_hold.append(performance_results.get('pourcentage_gain_total', None))

                # Ajouter la valeur momentum pour cette période
                valeurs_momentum.append(performance_results.get('gain_total', 0))

                # Mise à jour de la répartition
                allocation = {asset: 1 / len(top_assets) for asset in top_assets}
                repartition_current = pd.DataFrame([allocation], index=[current_date])
                repartition = pd.concat([repartition, repartition_current])

                # Mettre à jour les occurrences des actifs
                for asset in top_assets:
                    asset_occurrences[asset] += 1

            except Exception as e:
                print(f"Erreur lors de l'évaluation : {e}")
                # Ne pas sortir de la boucle, essayons la période suivante

            current_date += timedelta(days=self.periode_reroll)

        # Calcul de la performance annualisée
        years_elapsed = (self.date_fin - self.date_debut).days / 365.25
        performance_annualisee = (portfolio_value / self.montant_initial) ** (1 / years_elapsed) - 1 if years_elapsed > 0 else 0

        # Calcul du taux d'apparition
        total_months = (self.date_fin - self.date_debut).days / 30.44
        asset_appearance_rate = {ticker: occurrences / total_months for ticker, occurrences in asset_occurrences.items()}
        
        # Calculer la moyenne des indicateurs de risque basée uniquement sur les périodes valides
        final_risk_indicators = risk_indicators.copy()
        if valid_periods > 0:
            for key in risk_indicators:
                final_risk_indicators[key] = risk_indicators[key] / valid_periods
        else:
            # Si aucune période valide, utiliser les données de la dernière période qui avait des indicateurs non nuls
            for result in reversed(all_results):
                if result and any(key in result and result[key] != 0 for key in risk_indicators):
                    for key in risk_indicators:
                        if key in result:
                            final_risk_indicators[key] = float(result[key].iloc[0]) if isinstance(result[key], pd.Series) else float(result[key])
                    break
        
        return {
            "dates": dates,
            "valeurs_portefeuille": valeurs_portefeuille,
            "actifs_selectionnes": actifs_selectionnes,
            "valeurs_momentum": valeurs_momentum,
            "performances_buy_and_hold": performances_buy_and_hold,
            "repartition": repartition,
            "gain_total": portfolio_value - self.montant_initial,
            "pourcentage_gain_total": (portfolio_value / self.montant_initial - 1) * 100,
            "performance_annualisee": performance_annualisee * 100,
            "taux_apparition": asset_appearance_rate,
            **final_risk_indicators,  # Utiliser les indicateurs de risque moyennés ou de la dernière période valide
        }
