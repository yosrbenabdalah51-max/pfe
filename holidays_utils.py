"""
holidays_utils.py
=================
Fournit les jours fériés par dépôt pour les modèles ARIMA, XGBoost et LSTM.

Utilisation :
    from holidays_utils import get_holidays_for_depot, get_holiday_features

Pays couverts : France (FR), Allemagne (DE), Espagne (ES), Italie (IT), Grèce (GR)

Note : Dépôt 6 (Agence Ouled Fayet) → Allemagne (DE) — correction appliquée.
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# MAPPING  depot_id → country_code
# ─────────────────────────────────────────────
DEPOT_COUNTRY = {
    # France
    1:  "FR",   # Agence Paris
    2:  "FR",   # Siège Lyon
    3:  "FR",   # Agence Marseille
    8:  "FR",   # Depot Transporteur France
    67: "FR",   # Agence Toulouse
    # Allemagne (incl. correction dépôt 6)
    6:  "DE",   # Agence Ouled Fayet — corrigé (était DZ → DE)
    20: "DE",   # Agence Munich
    21: "DE",   # Agence Hamburg
    51: "DE",   # Agence Cologne
    52: "DE",   # Agence Frankfurt
    57: "DE",   # Dépôt Berlin
    65: "DE",   # Agence Stuttgart
    # Espagne
    24: "ES",   # Siège Madrid
    # Italie
    12: "IT",   # Siège Rome
    # Grèce
    41: "GR",   # Dépôt Export Athens
}


def _year_range(start_date, end_date):
    y1 = pd.Timestamp(start_date).year
    y2 = pd.Timestamp(end_date).year
    return list(range(y1, y2 + 1))


# ─────────────────────────────────────────────
# CALCUL PÂQUES (algorithme de Butcher)
# ─────────────────────────────────────────────

def _easter(year: int) -> pd.Timestamp:
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day   = ((h + l - 7 * m + 114) % 31) + 1
    return pd.Timestamp(year=year, month=month, day=day)


# ─────────────────────────────────────────────
# JOURS FÉRIÉS PAR PAYS
# ─────────────────────────────────────────────

def _france_holidays(years):
    dates = []
    for y in years:
        e = _easter(y)
        dates += [
            pd.Timestamp(y,  1,  1),        # Jour de l'An
            e + pd.Timedelta(days=1),        # Lundi de Pâques
            pd.Timestamp(y,  5,  1),        # Fête du Travail
            pd.Timestamp(y,  5,  8),        # Victoire 1945
            e + pd.Timedelta(days=39),       # Ascension
            e + pd.Timedelta(days=50),       # Lundi de Pentecôte
            pd.Timestamp(y,  7, 14),        # Fête Nationale
            pd.Timestamp(y,  8, 15),        # Assomption
            pd.Timestamp(y, 11,  1),        # Toussaint
            pd.Timestamp(y, 11, 11),        # Armistice
            pd.Timestamp(y, 12, 25),        # Noël
        ]
    return pd.DatetimeIndex(sorted(set(dates)))


def _germany_holidays(years):
    dates = []
    for y in years:
        e = _easter(y)
        dates += [
            pd.Timestamp(y,  1,  1),        # Neujahr
            e - pd.Timedelta(days=2),        # Karfreitag
            e + pd.Timedelta(days=1),        # Ostermontag
            pd.Timestamp(y,  5,  1),        # Tag der Arbeit
            e + pd.Timedelta(days=39),       # Christi Himmelfahrt
            e + pd.Timedelta(days=50),       # Pfingstmontag
            pd.Timestamp(y, 10,  3),        # Tag der Deutschen Einheit
            pd.Timestamp(y, 12, 25),        # 1. Weihnachtstag
            pd.Timestamp(y, 12, 26),        # 2. Weihnachtstag
        ]
    return pd.DatetimeIndex(sorted(set(dates)))


def _spain_holidays(years):
    dates = []
    for y in years:
        e = _easter(y)
        dates += [
            pd.Timestamp(y,  1,  1),        # Año Nuevo
            pd.Timestamp(y,  1,  6),        # Reyes Magos
            e - pd.Timedelta(days=2),        # Viernes Santo
            pd.Timestamp(y,  5,  1),        # Día del Trabajo
            pd.Timestamp(y,  8, 15),        # Asunción
            pd.Timestamp(y, 10, 12),        # Fiesta Nacional
            pd.Timestamp(y, 11,  1),        # Todos los Santos
            pd.Timestamp(y, 12,  6),        # Día de la Constitución
            pd.Timestamp(y, 12,  8),        # Inmaculada Concepción
            pd.Timestamp(y, 12, 25),        # Navidad
        ]
    return pd.DatetimeIndex(sorted(set(dates)))


def _italy_holidays(years):
    dates = []
    for y in years:
        e = _easter(y)
        dates += [
            pd.Timestamp(y,  1,  1),        # Capodanno
            pd.Timestamp(y,  1,  6),        # Epifania
            e + pd.Timedelta(days=1),        # Lunedì dell'Angelo
            pd.Timestamp(y,  4, 25),        # Festa della Liberazione
            pd.Timestamp(y,  5,  1),        # Festa dei Lavoratori
            pd.Timestamp(y,  6,  2),        # Festa della Repubblica
            pd.Timestamp(y,  8, 15),        # Ferragosto
            pd.Timestamp(y, 11,  1),        # Ognissanti
            pd.Timestamp(y, 12,  8),        # Immacolata Concezione
            pd.Timestamp(y, 12, 25),        # Natale
            pd.Timestamp(y, 12, 26),        # Santo Stefano
        ]
    return pd.DatetimeIndex(sorted(set(dates)))


def _greece_holidays(years):
    dates = []
    for y in years:
        # Pâques orthodoxe ≈ Pâques grégorien + 13 jours
        e = _easter(y) + pd.Timedelta(days=13)
        dates += [
            pd.Timestamp(y,  1,  1),        # Πρωτοχρονιά
            pd.Timestamp(y,  1,  6),        # Θεοφάνεια
            pd.Timestamp(y,  3, 25),        # Εθνική Εορτή
            e - pd.Timedelta(days=2),        # Μεγάλη Παρασκευή
            e + pd.Timedelta(days=1),        # Δευτέρα του Πάσχα
            pd.Timestamp(y,  5,  1),        # Εργατική Πρωτομαγιά
            e + pd.Timedelta(days=50),       # Αγίου Πνεύματος
            pd.Timestamp(y,  8, 15),        # Κοίμηση Θεοτόκου
            pd.Timestamp(y, 10, 28),        # Εθνική Εορτή (Όχι)
            pd.Timestamp(y, 12, 25),        # Χριστούγεννα
            pd.Timestamp(y, 12, 26),        # Σύναξη Θεοτόκου
        ]
    return pd.DatetimeIndex(sorted(set(dates)))


_HOLIDAY_FUNCS = {
    "FR": _france_holidays,
    "DE": _germany_holidays,
    "ES": _spain_holidays,
    "IT": _italy_holidays,
    "GR": _greece_holidays,
}


# ─────────────────────────────────────────────
# API PUBLIQUE
# ─────────────────────────────────────────────

def get_country_for_depot(depot_id) -> str:
    """Retourne le code pays ISO2 pour un depot_id. 'FR' par défaut."""
    if depot_id is None or depot_id == "all":
        return "FR"
    try:
        return DEPOT_COUNTRY.get(int(depot_id), "FR")
    except (ValueError, TypeError):
        return "FR"


def get_holidays_for_depot(depot_id, start_date, end_date) -> pd.DatetimeIndex:
    """
    Retourne les jours fériés pour le dépôt donné entre start_date et end_date.

    Paramètres
    ----------
    depot_id   : int ou str  (ex: 1, "6", "all")
    start_date : str ou Timestamp
    end_date   : str ou Timestamp
    """
    country  = get_country_for_depot(depot_id)
    years    = _year_range(start_date, end_date)
    func     = _HOLIDAY_FUNCS.get(country, _france_holidays)
    holidays = func(years)
    mask     = (holidays >= pd.Timestamp(start_date)) & (holidays <= pd.Timestamp(end_date))
    return holidays[mask]


def get_holiday_features(df_dates: pd.Series, depot_id) -> pd.DataFrame:
    """
    Génère 3 features booléennes de jours fériés pour XGBoost / LSTM.

    Retour : DataFrame avec colonnes
        - is_holiday      : 1 si jour férié
        - pre_holiday     : 1 si veille d'un jour férié
        - post_holiday    : 1 si lendemain d'un jour férié
        - holiday_country : code pays (pour debug)
    """
    dates    = pd.DatetimeIndex(df_dates)
    start    = dates.min() - pd.Timedelta(days=2)
    end      = dates.max() + pd.Timedelta(days=2)
    holidays = get_holidays_for_depot(depot_id, start, end)
    country  = get_country_for_depot(depot_id)

    is_hol   = dates.isin(holidays).astype(int)
    pre_hol  = (dates + pd.Timedelta(days=1)).isin(holidays).astype(int)
    post_hol = (dates - pd.Timedelta(days=1)).isin(holidays).astype(int)

    return pd.DataFrame({
        "is_holiday":      is_hol,
        "pre_holiday":     pre_hol,
        "post_holiday":    post_hol,
        "holiday_country": country,
    }, index=df_dates.index)


def get_prophet_holidays(depot_id, start_date="2020-01-01", end_date="2026-12-31") -> pd.DataFrame:
    """
    Retourne un DataFrame au format Prophet holidays :
        colonnes : ['holiday', 'ds', 'lower_window', 'upper_window']
    Compatible avec prophet.Prophet(holidays=df)
    """
    country  = get_country_for_depot(depot_id)
    years    = _year_range(start_date, end_date)
    func     = _HOLIDAY_FUNCS.get(country, _france_holidays)
    holidays = func(years)
    mask     = (holidays >= pd.Timestamp(start_date)) & (holidays <= pd.Timestamp(end_date))
    filtered = holidays[mask]

    return pd.DataFrame({
        "holiday":      f"public_holiday_{country}",
        "ds":           filtered,
        "lower_window": -1,
        "upper_window":  1,
    })