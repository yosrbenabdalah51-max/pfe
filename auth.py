import streamlit as st
import mysql.connector
import hashlib
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

def get_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_user(email: str, password: str):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT * FROM authentification WHERE email = %s AND password = %s",
        (email, hash_password(password))
    )
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    return user

ROLE_PAGES = {
    # ✅ "ceo" remplace "pdg"
    "ceo":     ["App", "Comparaison Modèles"],
    "manager": ["App", "Comparaison Modèles", "SARIMA", "LSTM", "XGBoost", "Gestion de Stock"],
    "stock":   ["App", "Gestion de Stock"],
}

# ✅ Labels affichés pour chaque rôle
ROLE_LABELS = {
    "ceo":     "CEO",
    "manager": "Manager",
    "stock":   "Stock",
}

def login_page():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    .login-wrapper {
        display: flex; flex-direction: column; align-items: center;
        justify-content: center; min-height: 70vh; padding-top: 40px;
    }
    .login-box {
        background: #ffffff; border-radius: 20px; padding: 36px 40px;
        box-shadow: 0 8px 40px rgba(108,99,255,0.13); width: 100%;
        max-width: 380px; border-top: 5px solid #6c63ff; margin: 0 auto;
    }
    .login-title {
        font-size: 24px; font-weight: 800; color: #1a1a2e;
        text-align: center; margin-bottom: 6px;
    }
    .login-sub {
        font-size: 13px; color: #9ca3af;
        text-align: center; margin-bottom: 24px;
    }
    section[data-testid="stMain"] .stTextInput > div > div > input {
        font-size: 14px !important; padding: 8px 12px !important;
        border-radius: 10px !important;
    }
    section[data-testid="stMain"] .stTextInput {
        max-width: 300px !important; margin: 0 auto 12px auto !important;
    }
    section[data-testid="stMain"] .stTextInput label {
        font-size: 13px !important; font-weight: 600 !important; color: #374151 !important;
    }
    div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #6c63ff, #a78bfa) !important;
        border: none !important; border-radius: 10px !important;
        font-weight: 700 !important; font-size: 14px !important;
        padding: 10px 0 !important; width: 300px !important;
        margin: 8px auto 0 auto !important; display: block !important;
    }
    </style>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown('<div class="login-title">🔐 Connexion</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-sub">Bienvenue — identifiez-vous pour continuer</div>', unsafe_allow_html=True)

        email    = st.text_input("Email", placeholder="votre@email.com")
        password = st.text_input("Mot de passe", type="password", placeholder="••••••••")

        if st.button("Se connecter", type="primary", use_container_width=True):
            if not email or not password:
                st.warning("Veuillez remplir tous les champs.")
                return
            user = verify_user(email, password)
            if user:
                st.session_state["authenticated"] = True
                st.session_state["user"]           = user
                st.session_state["role"]           = user["role"]
                st.session_state["name"]           = f"{user['prenom']} {user['nom']}"
                st.rerun()
            else:
                st.error("Email ou mot de passe incorrect.")


def logout():
    for key in ["authenticated", "user", "role", "name"]:
        st.session_state.pop(key, None)
    st.rerun()


def user_topbar():
    """
    Affiche le nom + rôle + bouton déconnexion en haut de chaque page.
    À appeler juste après require_auth() dans CHAQUE page.
    """
    name = st.session_state.get("name", "Utilisateur")
    role = st.session_state.get("role", "")

    # ✅ Libellé affiché (CEO, Manager, Stock)
    role_label = ROLE_LABELS.get(role, role.upper())

    role_colors = {
        "ceo":     ("#fbbf24", "#78350f"),
        "manager": ("#6c63ff", "#ede9fe"),
        "stock":   ("#10b981", "#d1fae5"),
    }
    bg, fg = role_colors.get(role, ("#6c63ff", "#ede9fe"))

    st.markdown(f"""
    <style>
    .topbar-user {{
        display: flex; align-items: center; gap: 12px;
        justify-content: flex-end; margin-bottom: 12px;
    }}
    .topbar-avatar {{
        width: 36px; height: 36px; border-radius: 50%;
        background: linear-gradient(135deg, {bg}, {fg});
        display: flex; align-items: center; justify-content: center;
        font-size: 15px; font-weight: 800; color: #1a1a2e;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }}
    .topbar-name {{ font-size: 14px; font-weight: 700; color: #1a1a2e; }}
    .topbar-role {{
        font-size: 11px; color: #9ca3af; font-weight: 500;
        text-transform: uppercase; letter-spacing: 1px;
    }}
    </style>
    <div class="topbar-user">
        <div>
            <div class="topbar-name">{name}</div>
            <div class="topbar-role">{role_label}</div>
        </div>
        <div class="topbar-avatar">{name[0].upper()}</div>
    </div>
    """, unsafe_allow_html=True)

    col_spacer, col_btn = st.columns([6, 1])
    with col_btn:
        if st.button(" Déconnexion", use_container_width=True):
            logout()


def require_auth(page_name: str):
    """
    À appeler en haut de chaque page.
    Redirige vers login si non connecté ou si le rôle n'a pas accès.
    """
    if not st.session_state.get("authenticated"):
        login_page()
        st.stop()

    role          = st.session_state.get("role")
    allowed_pages = ROLE_PAGES.get(role, [])

    if page_name not in allowed_pages:
        role_label = ROLE_LABELS.get(role, role)
        st.error(f"⛔ Accès refusé. Votre rôle **{role_label}** n'a pas accès à cette page.")
        st.stop()