# analyse/holidays_utils.py
"""
Utilitaire de jours fériés pour les pays des dépôts :
  France (FR), Allemagne (DE) — remplace Algérie (DZ),
  Espagne (ES), Italie (IT), Grèce (GR)

Usage :
    from holidays_utils import get_holidays_df, add_holiday_feature
"""

import pandas as pd

try:
    import holidays as hol_lib
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False

# ──────────────────────────────────────────────
# Mapping pays → code ISO
# NB : Algérie (DZ) remplacée par Allemagne (DE)
# ──────────────────────────────────────────────
COUNTRY_CODE_MAP = {
    # Noms français / anglais tolérés
    "france":      "FR",
    "FR":          "FR",
    "allemagne":   "DE",
    "germany":     "DE",
    "DE":          "DE",
    "algerie":     "DE",   # ← remplacé par DE
    "algérie":     "DE",
    "DZ":          "DE",
    "espagne":     "ES",
    "spain":       "ES",
    "ES":          "ES",
    "italie":      "IT",
    "italy":       "IT",
    "IT":          "IT",
    "grece":       "GR",
    "grèce":       "GR",
    "greece":      "GR",
    "GR":          "GR",
}

# Pays présents dans les dépôts (après remplacement DZ→DE)
ALL_COUNTRIES = ["FR", "DE", "ES", "IT", "GR"]


def resolve_country_code(country_str: str) -> str:
    """Convertit un nom ou code pays en code ISO 2 lettres."""
    if not country_str:
        return "FR"
    key = country_str.strip().lower()
    # Essai direct
    for k, v in COUNTRY_CODE_MAP.items():
        if k.lower() == key:
            return v
    # Essai partiel (ex: "France (FR)")
    for k, v in COUNTRY_CODE_MAP.items():
        if k.lower() in key:
            return v
    return "FR"   # fallback


def get_holidays_df(country_code: str, years=None) -> pd.DataFrame:
    """
    Retourne un DataFrame des jours fériés avec colonnes :
        ds (datetime), holiday (str), country (str)
    """
    if years is None:
        years = list(range(2020, 2027))
    if not HOLIDAYS_AVAILABLE:
        return pd.DataFrame(columns=["ds", "holiday", "country"])

    code = resolve_country_code(country_code)
    try:
        h = hol_lib.country_holidays(code, years=years)
        rows = [{"ds": pd.Timestamp(d), "holiday": name, "country": code}
                for d, name in h.items()]
        return pd.DataFrame(rows).sort_values("ds").reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["ds", "holiday", "country"])


def add_holiday_feature(df: pd.DataFrame, country_code: str,
                        date_col: str = "ds") -> pd.DataFrame:
    """
    Ajoute une colonne binaire `is_holiday` (1/0) à un DataFrame
    ayant une colonne de dates `date_col`.
    """
    df = df.copy()
    hdf = get_holidays_df(country_code)
    if hdf.empty:
        df["is_holiday"] = 0
        return df
    holiday_dates = set(hdf["ds"].dt.normalize())
    df["is_holiday"] = df[date_col].dt.normalize().isin(holiday_dates).astype(int)
    return df


def get_holiday_info_text(country_code: str) -> str:
    """Retourne un résumé lisible des jours fériés du pays."""
    code = resolve_country_code(country_code)
    hdf  = get_holidays_df(code)
    if hdf.empty:
        return f"Aucun jour férié chargé pour {code}."
    n = len(hdf)
    years = sorted(hdf["ds"].dt.year.unique())
    return (f"📅 **{n} jours fériés** chargés pour **{code}** "
            f"({years[0]}–{years[-1]})")