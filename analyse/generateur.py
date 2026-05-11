# analyse/generateur.py
from groq import Groq
import streamlit as st

def generer_analyse_arima(filtres: dict, metriques: dict) -> str:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    prompt = f"""
Tu es un expert en prévision de demande et gestion de stock.
Analyse les résultats du modèle ARIMA appliqués aux ventes.

Filtres appliqués :
{chr(10).join([f"- {k} : {v}" for k, v in filtres.items()])}

Métriques du modèle :
{chr(10).join([f"- {k} : {v}" for k, v in metriques.items()])}

Fournis une analyse structurée avec :
1. **Résumé** : Ce que montrent ces résultats
2. **Qualité du modèle** : Interprétation MAE, RMSE, MAPE, R²
3. **Analyse des prévisions** : Que prévoir pour les prochains mois ?
4. **Recommandations** : 2-3 actions concrètes
5. **Conclusion** : Synthèse pour le décideur
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # modèle gratuit et très puissant
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024
    )
    return response.choices[0].message.content
def generer_analyse_xgboost(filtres: dict, metriques: dict) -> str:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    prompt = f"""
Tu es un expert en prévision de demande et gestion de stock.
Analyse les résultats du modèle XGBoost appliqué aux ventes.

Filtres appliqués :
{chr(10).join([f"- {k} : {v}" for k, v in filtres.items()])}

Métriques du modèle :
{chr(10).join([f"- {k} : {v}" for k, v in metriques.items()])}

Fournis une analyse structurée avec :
1. **Résumé** : Ce que montrent ces résultats
2. **Qualité du modèle** : Interprétation MAE, RMSE, MAPE, R²
3. **Analyse des prévisions** : Que prévoir pour les prochains mois ?
4. **Recommandations** : 2-3 actions concrètes
5. **Conclusion** : Synthèse pour le décideur
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024
    )
    return response.choices[0].message.content
def generer_analyse_lstm(filtres: dict, metriques: dict) -> str:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    prompt = f"""
Tu es un expert en prévision de demande et gestion de stock.
Analyse les résultats du modèle LSTM (réseau de neurones récurrent) appliqué aux ventes.

Filtres appliqués :
{chr(10).join([f"- {k} : {v}" for k, v in filtres.items()])}

Métriques du modèle :
{chr(10).join([f"- {k} : {v}" for k, v in metriques.items()])}

Fournis une analyse structurée avec :
1. **Résumé** : Ce que montrent ces résultats
2. **Qualité du modèle** : Interprétation MAE, RMSE, MAPE, R²
3. **Analyse des prévisions** : Que prévoir pour les prochains mois ?
4. **Recommandations** : 2-3 actions concrètes
5. **Conclusion** : Synthèse pour le décideur
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024
    )
    return response.choices[0].message.content
def generer_analyse_comparaison(filtres: dict, metriques: dict) -> str:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    prompt = f"""
Tu es un expert en prévision de demande et gestion de stock.
Analyse la comparaison entre les modèles ARIMA, XGBoost et LSTM.

Filtres appliqués :
{chr(10).join([f"- {k} : {v}" for k, v in filtres.items()])}

Résultats des modèles :
{chr(10).join([f"- {k} : {v}" for k, v in metriques.items()])}

Fournis une analyse structurée avec :
1. **Résumé** : Synthèse globale de la comparaison
2. **Analyse par modèle** : Points forts et faibles de chaque modèle
3. **Modèle recommandé** : Lequel choisir et pourquoi
4. **Recommandations** : 2-3 actions concrètes pour le gestionnaire
5. **Conclusion** : Synthèse pour le décideur
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024
    )
    return response.choices[0].message.content