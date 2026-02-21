import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
import re
from utils import load_existing_data, filter_dataframe

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Asset Catalogue",
    page_icon="📘",
    layout="wide"
)

st.title("📘 Asset List")

# --------------------------------------------------
# PATHS
# --------------------------------------------------
DATA_FILE = Path("data/DataBase_ME_Asset.csv")
IMAGE_FOLDER = Path("images")
DEFAULT_IMAGE_NAME = "No Image Found"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
if not DATA_FILE.exists():
    st.error("Database file not found.")
    st.stop()

df = pd.read_csv(DATA_FILE)

# Required columns
required_cols = {
    "Description of Asset",
    "Department ID",
    "Status",
}

missing = required_cols - set(df.columns)
if missing:
    st.error(f"Missing columns in CSV: {', '.join(missing)}")
    st.stop()

# Remove empty equipment names
df = df.dropna(subset=["Description of Asset"])

if df.empty:
    st.warning("No equipment records found.")
    st.stop()

# --------------------------------------------------
# SEARCH BAR
# --------------------------------------------------
st.markdown("### 🔍 Search Asset")
search_col1, search_col2 = st.columns([4, 1])
with search_col1:
    search_term = st.text_input(
        "Search",
        placeholder="Search by equipment name, Asset Number, Type, Manufacturer, Model, Location, Project, or Status...",
        label_visibility="collapsed"
    )
with search_col2:
    if st.button("🔍 Search", use_container_width=True, type="primary"):
        st.rerun()

st.markdown("---")

# --------------------------------------------------
# FILTER DATA BASED ON SEARCH
# --------------------------------------------------
existing_df = load_existing_data()

if existing_df is not None and not existing_df.empty:
    filtered_df = filter_dataframe(existing_df, search_term) if search_term else existing_df.copy()
    
    # Update equipment list based on search results
    if search_term and not filtered_df.empty:
        filtered_equipment_list = sorted(filtered_df["Description of Asset"].unique())
    else:
        filtered_equipment_list = sorted(df["Description of Asset"].unique())
else:
    filtered_equipment_list = sorted(df["Description of Asset"].unique())
    filtered_df = df.copy()

# --------------------------------------------------
# DISPLAY DATABASE METRICS & SEARCH RESULTS
# --------------------------------------------------
st.markdown("### 📋 Asset Database")

if existing_df is not None and not existing_df.empty:
    # ===== METRICS =====
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(existing_df))
    with col2:
        st.metric("Search Results", len(filtered_df) if search_term else len(existing_df))
    with col3:
        st.metric("Good Status", len(existing_df[existing_df["Status"] == "Good"]) if "Status" in existing_df.columns else 0)
    with col4:
        st.metric("Expired Soon", len(existing_df[existing_df["Status"] == "Expired Soon"]) if "Status" in existing_df.columns else 0)

    if not filtered_df.empty:
        st.markdown("#### Records")
        display_cols = ["Description of Asset","Department ID", "Asset Number", "Type", "Functional Location", "Status", "Day Left"]
        available_cols = [c for c in display_cols if c in filtered_df.columns]
        st.dataframe(filtered_df[available_cols], use_container_width=True)
        
        with st.expander("📊 View Full Details"):
            st.dataframe(filtered_df, use_container_width=True)
        
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            "📥 Download Filtered Data as CSV",
            csv,
            file_name=f"equipment_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        if search_term:
            st.info("🔍 No records found matching your search.")
        else:
            st.info("No records to display.")
else:
    st.info("📝 No equipment registered yet.")

st.markdown("---")

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "equipment_index" not in st.session_state:
    st.session_state.equipment_index = 0

# Keep index valid for filtered list
st.session_state.equipment_index = max(
    0,
    min(st.session_state.equipment_index, len(filtered_equipment_list) - 1)
)

# --------------------------------------------------
# NAVIGATION BAR
# --------------------------------------------------
st.markdown("### 🔄 Asset Navigator")

if filtered_equipment_list:
    nav_left, nav_mid, nav_right = st.columns([1, 6, 1])

    with nav_left:
        if st.button("⬅ Previous", disabled=st.session_state.equipment_index == 0):
            st.session_state.equipment_index -= 1
            st.rerun()

    with nav_mid:
        selected_equipment = st.selectbox(
            "Jump to equipment",
            filtered_equipment_list,
            index=st.session_state.equipment_index
        )

        new_index = filtered_equipment_list.index(selected_equipment)
        if new_index != st.session_state.equipment_index:
            st.session_state.equipment_index = new_index
            st.rerun()

    with nav_right:
        if st.button(
            "Next ➡",
            disabled=st.session_state.equipment_index == len(filtered_equipment_list) - 1
        ):
            st.session_state.equipment_index += 1
            st.rerun()

    # --------------------------------------------------
    # CURRENT EQUIPMENT
    # --------------------------------------------------
    current_equipment = filtered_equipment_list[st.session_state.equipment_index]
    group = df[df["Description of Asset"] == current_equipment]

    # --------------------------------------------------
    # HEADER
    # --------------------------------------------------
    st.caption(
        f"Equipment {st.session_state.equipment_index + 1} "
        f"of {len(filtered_equipment_list)}"
    )

    st.subheader(current_equipment)
    st.divider()

    # --------------------------------------------------
    # IMAGE SEARCH FUNCTIONS
    # --------------------------------------------------
    def normalize_name(name: str) -> str:
        name = name.lower()
        name = re.sub(r"[^\w\s-]", "", name)
        return name.replace(" ", "_")


    def find_equipment_image(equipment_name: str) -> Path | None:
        if not IMAGE_FOLDER.exists():
            return None

        # 1️⃣ Try equipment image
        safe_name = normalize_name(equipment_name)
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            path = IMAGE_FOLDER / f"{safe_name}{ext}"
            if path.exists():
                return path

        # 2️⃣ Fallback: default image
        fallback_name = normalize_name(DEFAULT_IMAGE_NAME)
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            fallback = IMAGE_FOLDER / f"{fallback_name}{ext}"
            if fallback.exists():
                return fallback

        return None

    # --------------------------------------------------
    # TWO-COLUMN LAYOUT
    # --------------------------------------------------
    left_col, right_col = st.columns([2, 1])

    # LEFT COLUMN: Department IDs
    with left_col:
        st.markdown("### Department ID List")

        display_df = (
            group[['Department ID','Asset Number', 'SAP No.', 'Type', 'Manufacturer/Supplier', 'Model',
           'Mfg SN', 'Mfg Year', 'Est Value','Status']]
            .drop_duplicates()
            .sort_values("Department ID")
            .reset_index(drop=True)
        )

        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True
        )

    # RIGHT COLUMN: Image
    with right_col:
        st.markdown("### Image")

        image_path = find_equipment_image(current_equipment)

        if image_path:
            st.image(image_path, use_container_width=True)
        else:
            st.error("Default image not found in image folder.")

    st.markdown("---")
else:
    st.warning("No equipment found matching your search criteria.")