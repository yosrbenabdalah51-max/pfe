import streamlit as st

USERS = {
    "ceo": {
        "password": "ceo123",
        "role":     "CEO",
        "icon":     "👔",
        "pages":    ["Comparaison modèles", "Stock Management"],
    },
    "manager": {
        "password": "manager123",
        "role":     "Manager / Data Analyst",
        "icon":     "📊",
        "pages":    ["ARIMA", "XGBoost", "LSTM", "Comparaison modèles", "Stock Management"],
    },
    "stock": {
        "password": "stock123",
        "role":     "Responsable Stock",
        "icon":     "🏭",
        "pages":    ["Stock Management"],
    },
}

def login_page():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    .block-container { max-width: 420px !important; margin: auto; padding-top: 80px; }
    .login-title {
        font-size: 28px; font-weight: 800;
        color: #6c63ff; text-align: center; margin-bottom: 6px;
    }
    .login-sub {
        font-size: 13px; color: #9ca3af;
        text-align: center; margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="login-title">🔐 Connexion</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-sub">Prévision de Demande & Gestion de Stock</div>', unsafe_allow_html=True)

    with st.form("login_form"):
        username = st.text_input("👤 Nom d'utilisateur")
        password = st.text_input("🔑 Mot de passe", type="password")
        submit   = st.form_submit_button("Se connecter", use_container_width=True)

    if submit:
        username = username.strip().lower()
        if username in USERS and USERS[username]["password"] == password:
            st.session_state["authenticated"] = True
            st.session_state["username"]      = username
            st.session_state["role"]          = USERS[username]["role"]
            st.session_state["icon"]          = USERS[username]["icon"]
            st.session_state["pages"]         = USERS[username]["pages"]
            st.rerun()
        else:
            st.error("❌ Identifiant ou mot de passe incorrect.")

def logout():
    for key in ["authenticated", "username", "role", "icon", "pages"]:
        st.session_state.pop(key, None)
    st.rerun()

def require_auth(page_name: str):
    if not st.session_state.get("authenticated"):
        login_page()
        st.stop()

    allowed_pages = st.session_state.get("pages", [])
    if page_name not in allowed_pages:
        st.error(f"🚫 Accès refusé — votre rôle **{st.session_state['role']}** n'a pas accès à cette page.")
        st.stop()