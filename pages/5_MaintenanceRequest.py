import streamlit as st

from utils import require_login, render_role_navigation

st.set_page_config(page_title="Maintenance Request", page_icon="🛠️", layout="wide")

auth = require_login(min_level_rank=1)
render_role_navigation(auth)

st.title("🛠️ Maintenance Request")
st.info("Coming soon.")
