import streamlit as st
import pandas as pd
from pathlib import Path
import re
from utils import (
    load_existing_data,
    filter_dataframe,
    require_login,
    decode_qr_payload_from_image,
    uploaded_file_sha256,
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Asset Catalogue",
    page_icon="📘",
    layout="wide"
)

require_login()

st.title("📘 Asset List")

# --------------------------------------------------
# PATHS
# --------------------------------------------------
# Use an absolute images path anchored to the project root.
# This avoids issues when Streamlit's working directory isn't the repo root.
APP_ROOT = Path(__file__).resolve().parents[1]
IMAGE_FOLDER = APP_ROOT / "images"
DEFAULT_IMAGE_NAME = "No Image Found"

# --------------------------------------------------
# IMAGE HELPERS (match Asset Editor naming)
# --------------------------------------------------
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def _sanitize_filename_base(value: str, max_len: int = 80) -> str:
    """Match Asset Editor's image filename sanitation (Windows-safe)."""
    s = str(value or "").strip().upper()
    s = re.sub(r'[<>:"/\\|?*]', "_", s)
    s = re.sub(r"\s+", "_", s)
    s = "".join(ch for ch in s if ch.isprintable())
    s = s.strip("._ ")
    if not s:
        s = "ASSET"
    return s[: int(max_len)]


def _asset_key_prefix(department_id: str, asset_number: str) -> str:
    raw = f"{str(department_id or '').strip()}_{str(asset_number or '').strip()}"
    return _sanitize_filename_base(raw, max_len=80)


def _asset_image_prefix(department_id: str, asset_number: str, description: str) -> str:
    key = _asset_key_prefix(department_id, asset_number)
    desc = _sanitize_filename_base(description, max_len=80)
    return f"{key}_{desc}" if desc else key


def _find_images_for_group(image_folder: Path, group_df: pd.DataFrame, equipment_name: str) -> list[Path]:
    if not image_folder.exists():
        return []

    matches: list[Path] = []

    # 1) Preferred (NEW): files saved by Asset Editor directly under /images
    #    {DEPTID}_{ASSETNO}_{DESCRIPTION}_01.jpg
    if group_df is not None and not group_df.empty:
        for _, r in group_df.iterrows():
            dept_id = str(r.get("Department ID", "") or "").strip()
            asset_no = str(r.get("Asset Number", "") or "").strip()
            desc = str(r.get("Description of Asset", equipment_name) or equipment_name).strip()

            if dept_id or asset_no:
                pref = _asset_image_prefix(dept_id, asset_no, desc)
                for ext in _IMAGE_EXTS:
                    matches.extend(sorted(image_folder.glob(f"{pref}_*{ext}")))

                # Backward-compat: some earlier versions might save without description
                key_pref = _asset_key_prefix(dept_id, asset_no)
                for ext in _IMAGE_EXTS:
                    matches.extend(sorted(image_folder.glob(f"{key_pref}_*{ext}")))

    # 2) Backward-compat (OLD): normalized equipment name as a single file
    legacy_safe = re.sub(r"[^\w\s-]", "", str(equipment_name or "").lower()).replace(" ", "_")
    for ext in _IMAGE_EXTS:
        p = image_folder / f"{legacy_safe}{ext}"
        if p.exists():
            matches.append(p)

    # Deduplicate while preserving order
    seen = set()
    out: list[Path] = []
    for p in matches:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def find_equipment_image(image_folder: Path, group_df: pd.DataFrame, equipment_name: str, default_image_name: str) -> Path | None:
    images = _find_images_for_group(image_folder, group_df, equipment_name)
    if images:
        return images[0]

    # Fallback: default image
    fallback_name = re.sub(r"[^\w\s-]", "", str(default_image_name or "").lower()).replace(" ", "_")
    for ext in _IMAGE_EXTS:
        fallback = image_folder / f"{fallback_name}{ext}"
        if fallback.exists():
            return fallback

    return None

# --------------------------------------------------
# LOAD DATA (from main_data.db)
# --------------------------------------------------
df = load_existing_data()
if df is None or df.empty:
    st.error("No asset data found in main_data.db.")
    st.stop()

# Required columns
required_cols = {
    "Description of Asset",
    "Department ID",
    "Status",
}

missing = required_cols - set(df.columns)
if missing:
    st.error(f"Missing columns in asset database: {', '.join(missing)}")
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
        label_visibility="collapsed",
        key="catalog_search_term",
    )
with search_col2:
    if st.button("🔍 Search", use_container_width=True, type="primary"):
        st.rerun()

st.caption("Scan via camera or a handheld QR scanner.")
qr_mode = st.radio(
    "Scan method",
    options=["QR scanner", "Camera"],
    horizontal=True,
    key="catalog_qr_mode",
    label_visibility="collapsed",
)

if qr_mode == "QR scanner":
    scanned = st.text_input(
        "QR scanner input",
        placeholder="Click here then scan (scanner types + Enter)",
        key="catalog_qr_scanner",
        label_visibility="collapsed",
    )
    scanned = str(scanned or "").strip()
    if scanned and st.session_state.get("catalog_qr_scanner_last") != scanned:
        st.session_state["catalog_qr_scanner_last"] = scanned
        st.session_state["catalog_search_term"] = scanned
        st.rerun()
else:
    cam = st.camera_input("Scan QR and search", key="catalog_qr_cam")
    digest = uploaded_file_sha256(cam)
    if cam is not None and digest and st.session_state.get("catalog_qr_cam_digest") != digest:
        st.session_state["catalog_qr_cam_digest"] = digest
        payload = decode_qr_payload_from_image(cam)
        if payload:
            st.session_state["catalog_search_term"] = payload
            st.rerun()
        else:
            st.warning("No QR detected. Try again with a clearer shot.")

st.markdown("---")

# --------------------------------------------------
# FILTER DATA BASED ON SEARCH
# --------------------------------------------------
existing_df = df

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

        image_path = find_equipment_image(IMAGE_FOLDER, group, current_equipment, DEFAULT_IMAGE_NAME)

        if image_path:
            st.image(image_path, use_container_width=True)
        else:
            st.error("Default image not found in image folder.")

    st.markdown("---")
else:
    st.warning("No equipment found matching your search criteria.")