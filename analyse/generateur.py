# analyse/generateur.py
from groq import Groq
import streamlit as st


# ══════════════════════════════════════════════
# 1. PAGE SARIMA
# ══════════════════════════════════════════════
def generer_analyse_sarima(filtres: dict, metriques: dict) -> str:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    prompt = f"""
Tu es un expert en prévision de demande et gestion de stock.
Analyse les résultats du modèle SARIMA (ARIMA saisonnier avec jours fériés comme variable exogène) appliqué aux ventes.

Filtres appliqués :
{chr(10).join([f"- {k} : {v}" for k, v in filtres.items()])}

Métriques du modèle :
{chr(10).join([f"- {k} : {v}" for k, v in metriques.items()])}

Fournis une analyse structurée avec :
1. **Résumé** : Ce que montre cette page et ce qu'est le modèle SARIMA
2. **Qualité du modèle** : Interprétation MAE, RMSE, MAPE, R² et ce qu'ils signifient concrètement
3. **Ordre SARIMA retenu** : Explique ce que signifient les paramètres (p,d,q)(P,D,Q)[s] affichés et pourquoi ils ont été sélectionnés automatiquement
4. **Impact des jours fériés** : Explique comment la variable exogène jours fériés influence les prévisions
5. **Analyse des prévisions** : Que prévoir pour les prochains mois selon les KPIs affichés ?
6. **Recommandations** : 2-3 actions concrètes pour le gestionnaire
7. **Conclusion** : Synthèse claire pour le décideur
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200
    )
    return response.choices[0].message.content


# ══════════════════════════════════════════════
# 2. PAGE XGBOOST
# ══════════════════════════════════════════════
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
1. **Résumé** : Ce que montre cette page et ce qu'est le modèle XGBoost
2. **Qualité du modèle** : Interprétation MAE, RMSE, MAPE, R² et ce qu'ils signifient concrètement
3. **Importance des features** : Si disponible, quelles variables influencent le plus les prévisions ?
4. **Analyse des prévisions** : Que prévoir pour les prochains mois selon les KPIs affichés ?
5. **Recommandations** : 2-3 actions concrètes pour le gestionnaire
6. **Conclusion** : Synthèse claire pour le décideur
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200
    )
    return response.choices[0].message.content


# ══════════════════════════════════════════════
# 3. PAGE LSTM
# ══════════════════════════════════════════════
def generer_analyse_lstm(filtres: dict, metriques: dict) -> str:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    prompt = f"""
Tu es un expert en prévision de demande et gestion de stock.
Analyse les résultats du modèle LSTM (réseau de neurones récurrent à mémoire longue) appliqué aux ventes.

Filtres appliqués :
{chr(10).join([f"- {k} : {v}" for k, v in filtres.items()])}

Métriques du modèle :
{chr(10).join([f"- {k} : {v}" for k, v in metriques.items()])}

Fournis une analyse structurée avec :
1. **Résumé** : Ce que montre cette page et ce qu'est le modèle LSTM
2. **Qualité du modèle** : Interprétation MAE, RMSE, MAPE, R² et ce qu'ils signifient concrètement
3. **Architecture du modèle** : Explique ce que signifient les paramètres affichés (couches, unités, fenêtre de temps, epochs)
4. **Analyse des prévisions** : Que prévoir pour les prochains mois selon les KPIs affichés ?
5. **Recommandations** : 2-3 actions concrètes pour le gestionnaire
6. **Conclusion** : Synthèse claire pour le décideur
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200
    )
    return response.choices[0].message.content


# ══════════════════════════════════════════════
# 4. PAGE COMPARAISON DES MODÈLES
# ══════════════════════════════════════════════
def generer_analyse_comparaison(filtres: dict, metriques: dict) -> str:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    prompt = f"""
Tu es un expert en prévision de demande et gestion de stock.
Analyse la comparaison entre les modèles SARIMA, XGBoost et LSTM pour ce produit et ce dépôt.

Filtres appliqués :
{chr(10).join([f"- {k} : {v}" for k, v in filtres.items()])}

Résultats des modèles :
{chr(10).join([f"- {k} : {v}" for k, v in metriques.items()])}

Fournis une analyse structurée avec :
1. **Résumé** : Ce que montre cette page de comparaison et son objectif
2. **Analyse par modèle** : Pour chaque modèle (SARIMA, XGBoost, LSTM), explique ses points forts et ses limites selon les métriques affichées (MAE, RMSE, MAPE, R²)
3. **Modèle recommandé** : Lequel choisir pour ce produit/dépôt et pourquoi, en t'appuyant sur les chiffres
4. **Prévisions retenues** : Ce que prédit le meilleur modèle pour les prochains mois
5. **Recommandations** : 2-3 actions concrètes pour le gestionnaire
6. **Conclusion** : Synthèse claire pour le décideur
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200
    )
    return response.choices[0].message.content


# ══════════════════════════════════════════════
# 5. PAGE STOCK MANAGEMENT
# ══════════════════════════════════════════════
def generer_analyse_stock(filtres: dict, metriques: dict) -> str:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
 
    prompt = f"""
Tu es un expert en gestion de stock et supply chain. Un gestionnaire consulte sa page "Stock Management" et tu dois lui expliquer précisément et en détail ce qu'il voit, chiffres à l'appui.
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXTE DE LA SESSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{chr(10).join([f"- {k} : {v}" for k, v in filtres.items()])}
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DONNÉES AFFICHÉES SUR LA PAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{chr(10).join([f"- {k} : {v}" for k, v in metriques.items()])}
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TA MISSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rédige une analyse complète, précise et pédagogique de la page en utilisant UNIQUEMENT les chiffres fournis ci-dessus.
Ne parle pas de modèles ARIMA, XGBoost ou LSTM de façon générique — concentre-toi sur ce que la page affiche concrètement.
Réponds en français avec exactement cette structure :
 
---
 
##  Objectif de la page
Explique en 2-3 phrases ce que fait concrètement la page Stock Management : elle simule l'évolution du stock jour par jour en combinant les prévisions du meilleur modèle sélectionné automatiquement, le stock actuel, et le délai de réapprovisionnement. Mentionne le produit, le dépôt et l'horizon d'analyse affichés.
 
---
 
##  Explication des KPIs affichés
Pour chacun des indicateurs suivants, explique sa valeur concrète et ce qu'elle signifie pour ce produit :
 
- **Stock actuel** : quelle quantité est disponible maintenant et pourquoi c'est le point de départ de la simulation
- **Demande prévue totale** : combien d'unités seront vendues sur l'horizon, et ce que ça représente par jour en moyenne
- **Stock fin horizon simulé** : ce qui restera à la fin de la période APRÈS les réapprovisionnements automatiques simulés
- **Seuil de sécurité (SS)** : à partir de quel niveau le stock devient dangereux et pourquoi (formule Q1 × délai)
- **Point de réapprovisionnement (ROP)** : le niveau déclencheur d'une commande (formule Q3 × délai) et ce que ça signifie ici
- **Stock optimal** : la cible à atteindre lors d'une commande (ROP + Q2 × délai)
- **Stock maximum** : le plafond au-delà duquel on est en surstock
 
---
 
##  Lecture du graphique "Évolution stock simulée"
Explique précisément ce que le gestionnaire voit dans ce graphique avec les vraies valeurs :
- Les **barres violettes** (stock simulé) : comment elles évoluent sur l'horizon, si elles descendent, remontent, franchissent des seuils
- Les **barres colorées** (demande prévue par le modèle) : le rythme de consommation quotidien ou hebdomadaire
- Les **lignes horizontales pointillées** : à quoi correspondent chaque seuil tracé (SS, ROP, Optimal, Max) avec leurs valeurs exactes
- Les **triangles bleus** : quand une commande est déclenchée automatiquement, combien d'unités sont commandées, et quand la livraison arrive (J + délai réappro)
 
---
 
##  Lecture du tableau journalier
Explique colonne par colonne ce que le gestionnaire doit regarder :
- **Date** : chaque ligne = un jour de simulation
- **Demande prévue** : quantité que le modèle anticipe pour ce jour
- **Livraison reçue** : apparaît uniquement les jours où une commande passée arrive (après le délai réappro)
- **Stock début de jour** : stock disponible avant la vente du jour
- **Stock fin de jour** : stock restant après la vente, base du calcul du lendemain
- **Commande passée** : si ce jour le stock passe sous le ROP, une commande est automatiquement générée — montant et date de livraison prévue
- **État** : code couleur du stock ce jour-là — explique chaque couleur avec son seuil : 🔴 Rupture (≤ 0), 🟠 Risque rupture (< SS), 🟡 Proche ROP (≤ ROP), 🟢 Optimal (≤ stock optimal), 🟡 Surstock léger (≤ stock max), 🟡 Surstock (> stock max)
 
---
 
##  Diagnostic de la situation actuelle
En t'appuyant sur les chiffres exacts fournis, donne un diagnostic clair :
- Le stock actuel est-il au-dessus ou en dessous du ROP ? Du seuil de sécurité ?
- La demande prévue sur l'horizon dépasse-t-elle le stock disponible ?
- Le stock simulé en fin d'horizon est-il positif ? À quel niveau par rapport aux seuils ?
- Y a-t-il des commandes automatiques déclenchées ? Combien ? Sont-elles suffisantes ?
- Quel est le risque principal : rupture, surstock, ou situation stable ?
 
---
 
## ✅ Actions recommandées
Donne 3 actions concrètes, prioritaires et chiffrées que le gestionnaire doit faire maintenant, basées uniquement sur les données affichées.
"""
 
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000
    )
    return response.choices[0].message.content
