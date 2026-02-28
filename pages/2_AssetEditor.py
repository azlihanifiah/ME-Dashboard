import streamlit as st
import pandas as pd
from datetime import datetime, date
from pathlib import Path
import sqlite3
import re
from utils import (
    ensure_data_directory, load_existing_data, save_data,
    check_duplicate, calculate_due_date, calculate_days_left,
    calculate_status, validate_equipment_details,
    generate_acronym, log_asset_operation, get_asset_logs, initialize_log_database,
    delete_asset_by_dept_id, require_login,
    decode_qr_payload_from_image, uploaded_file_sha256,
)

auth = require_login()


def _performed_by_label() -> str:
    name = str(auth.get("name", "") or "").strip()
    user_id = str(auth.get("user_id", "") or "").strip()
    return name or user_id or "System"


def _current_user_level_rank() -> int:
    try:
        return int(auth.get("level_rank") or 0)
    except Exception:
        return 0

st.title("📝 Asset Editor")
st.markdown("---")

ensure_data_directory()
initialize_log_database()  # <-- ADD: make sure logging DB/table exists before any log write

# Use an absolute images path anchored to the project root.
# This avoids issues when Streamlit's working directory isn't the repo root.
APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = APP_ROOT / "data"
IMG_DIR = APP_ROOT / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _sanitize_filename_base(value: str, max_len: int = 80) -> str:
    """Create a Windows-safe filename base (no extension)."""
    s = str(value or "").strip().upper()
    # Replace invalid Windows filename characters
    s = re.sub(r'[<>:"/\\|?*]', "_", s)
    # Replace whitespace runs with single underscore
    s = re.sub(r"\s+", "_", s)
    # Remove non-printable characters
    s = "".join(ch for ch in s if ch.isprintable())
    s = s.strip("._ ")
    if not s:
        s = "ASSET"
    return s[: int(max_len)]


def _asset_key_prefix(department_id: str, asset_number: str) -> str:
    """Stable prefix for identifying an asset's images (used for replace/delete)."""
    raw = f"{str(department_id or '').strip()}_{str(asset_number or '').strip()}"
    return _sanitize_filename_base(raw, max_len=80)


def _asset_image_prefix(department_id: str, asset_number: str, description: str) -> str:
    """Prefix used for naming images (includes description as requested)."""
    key = _asset_key_prefix(department_id, asset_number)
    desc = _sanitize_filename_base(description, max_len=80)
    # Ensure we always have a reasonable base, even if description is empty.
    return f"{key}_{desc}" if desc else key


def _save_uploaded_images_replace(target_dir: Path, delete_key_prefix: str, save_prefix: str, images) -> None:
    """Replace existing images for this asset (by delete_key_prefix) with newly uploaded ones."""
    target_dir.mkdir(parents=True, exist_ok=True)

    # Clear old images for this asset only
    for old in target_dir.iterdir():
        if not old.is_file() or old.suffix.lower() not in _IMAGE_EXTS:
            continue
        if old.name.upper().startswith(f"{str(delete_key_prefix or '').upper()}_"):
            old.unlink(missing_ok=True)

    base = str(save_prefix or "").strip() or "ASSET"
    for i, f in enumerate(images, start=1):
        ext = Path(getattr(f, "name", "") or "").suffix.lower()
        if ext not in _IMAGE_EXTS:
            # Fallback (shouldn't happen due to uploader restriction)
            ext = ".png"
        out_name = f"{base}_{i:02d}{ext}"
        out_path = target_dir / out_name
        with open(out_path, "wb") as out:
            out.write(f.getbuffer())

existing_df = load_existing_data()

# initialize session state keys
for k, v in {
    "show_add_form": False,
    "description": "",
    "prefix": "",
    "delete_confirm_dept_id": None,   # <-- ADD: used later
    "delete_confirm_asset": None,     # <-- ADD: used later
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ================= HELPER FUNCTIONS =================

FUNCTIONAL_LOCATION_OPTIONS = [
    "Obsolete",
    "1006-10PE",
    "1006-10PE-P1F0",
    "1006-10PE-P1F0-Z001",
    "1006-10PE-P1F1",
    "1006-10PE-P1F1-Z001",
    "1006-10PE-P1F2",
    "1006-10PE-P1F2-Z001",
    "1006-10PE-P1F2-ZP01",
    "1006-10PE-P4F0",
    "1006-10PE-P4F0-Z001",
    "1006-10PE-P6F0",
    "1006-10PE-P6F0-Z001",
]

FUNCTIONAL_LOCATION_DESCRIPTION = {
    "Obsolete": "Obsolete / Discarded",
    "1006-10PE": "Assembly (Production)",
    "1006-10PE-P1F0": "Plo 62 Ground Floor",
    "1006-10PE-P1F0-Z001": "ME General Area",
    "1006-10PE-P1F1": "Plo 62 Floor 01",
    "1006-10PE-P1F1-Z001": "ME General Area",
    "1006-10PE-P1F2": "Plo 62 Floor 02",
    "1006-10PE-P1F2-Z001": "ME General Area",
    "1006-10PE-P1F2-ZP01": "PE General Area",
    "1006-10PE-P4F0": "Plo 65 Ground Floor",
    "1006-10PE-P4F0-Z001": "ME General Area",
    "1006-10PE-P6F0": "Plo 67 Ground Floor",
    "1006-10PE-P6F0-Z001": "ME General Area",
}


def _normalize_dept_code(value: str) -> str:
    v = str(value or "").strip().upper()
    return v


def _normalize_item_prefix(value: str) -> str:
    return str(value or "").strip().upper()


def generate_department_id_add(dept_code: str, item_prefix: str, df: pd.DataFrame | None) -> str:
    """Generate Department ID for Add flow:
    88-{15ME/15PE}-{PREFIX}-{NNN}, where NNN runs per dept+prefix.
    """
    dept_code = _normalize_dept_code(dept_code)
    item_prefix = _normalize_item_prefix(item_prefix)

    if dept_code not in {"15ME", "15PE"}:
        return ""
    if not item_prefix:
        return ""

    pattern = re.compile(rf"^88-{re.escape(dept_code)}-{re.escape(item_prefix)}-(\\d{{3}})$", re.IGNORECASE)
    max_n = 0

    if df is not None and not df.empty and "Department ID" in df.columns:
        for raw in df["Department ID"].dropna().astype(str).tolist():
            m = pattern.match(raw.strip())
            if not m:
                continue
            try:
                max_n = max(max_n, int(m.group(1)))
            except Exception:
                continue

    return f"88-{dept_code}-{item_prefix}-{(max_n + 1):03d}"

def safe_index(options, value, default: int = 0) -> int:
    """Return index of value in options; otherwise default."""
    try:
        if value in options:
            return int(options.index(value))
    except Exception:
        pass
    return int(default)

def save_row_to_df(row: dict) -> dict:
    """
    Normalizes row values before saving to CSV (dates -> YYYY-MM-DD strings, NaN -> '').
    Prevents mixed types that can break search/filter later.
    """
    out = dict(row or {})
    for k, v in list(out.items()):
        # Uppercase all text values except Status (requirement)
        if isinstance(v, str) and k != "Status":
            out[k] = v.strip().upper()
            continue

        # Normalize pandas NaN
        if isinstance(v, float) and pd.isna(v):
            out[k] = ""
            continue

        # Normalize dates
        if isinstance(v, (datetime, date)):
            out[k] = v.strftime("%Y-%m-%d")
            continue

        # Keep as-is otherwise
        out[k] = v

    return out

def _safe_parse_date(value, fallback: date | None = None) -> date | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return fallback
    if isinstance(value, date):
        return value
    try:
        return pd.to_datetime(value, errors="coerce").date()
    except Exception:
        return fallback

def _safe_calc_due_date(start_date_val, freq_val):
    try:
        return calculate_due_date(start_date_val, freq_val)
    except Exception:
        return None

def _safe_calc_days_left(due_date_val):
    try:
        return calculate_days_left(due_date_val)
    except Exception:
        return ""

def _safe_calc_status(days_left_val):
    try:
        return calculate_status(days_left_val)
    except Exception:
        return ""

def render_equipment_form(prefix: str, record: dict | None = None, is_update: bool = False) -> dict:
    """
    Renders the Asset form and returns a dict with all fields used by save/update logic.
    Fixes NameError in Update Asset flow.
    """
    record = record or {}

    # Options (kept flexible; users can still type values in text inputs where needed)
    type_options = ["Machine", "Equipment", "Jig", "Fixture", "Tester", "Tool", "Other"]
    freq_options = ["Weekly", "Monthly", "Quarterly", "Half-Yearly", "Yearly", "None"]
    floor_options = ["", "1", "2", "3", "4", "5"]
    status_options = ["Good", "Idle", "NG", "Expired", "Expired Soon"]

    # ---- Basic details ----
    col1, col2, col3 = st.columns(3)
    with col1:
        if is_update:
            department = st.text_input(
                "Department",
                value=str(record.get("Department", "") or ""),
                key=f"{prefix}_department",
            )
        else:
            department = st.text_input(
                "Department (15ME / 15PE)",
                value=str(record.get("Department", "") or ""),
                key=f"{prefix}_department",
                placeholder="15ME or 15PE",
            )
        department = _normalize_dept_code(department)
    with col2:
        desc = st.text_input(
            "Description of Asset *",
            value=str(record.get("Description of Asset", "") or ""),
            key=f"{prefix}_description",
        )
        desc_norm = str(desc or "").strip().upper()
    with col3:
        # If empty, try auto acronym from description (but keep editable)
        default_prefix = str(record.get("Prefix", "") or "")
        if not default_prefix and str(desc).strip():
            try:
                default_prefix = generate_acronym(str(desc).strip())
            except Exception:
                default_prefix = ""
        asset_prefix = st.text_input(
            "Prefix",
            value=default_prefix,
            key=f"{prefix}_prefix",
        )
        asset_prefix = _normalize_item_prefix(asset_prefix)

    col4, col5, col6 = st.columns(3)
    with col4:
        asset_number = st.text_input(
            "Asset Number *",
            value=str(record.get("Asset Number", "") or ""),
            key=f"{prefix}_asset_number",
        )
    with col5:
        sap_no = st.text_input(
            "SAP No.",
            value=str(record.get("SAP No.", "") or ""),
            key=f"{prefix}_sap_no",
        )
    with col6:
        # Type as selectbox for consistency; fallback to "Other" if unknown
        rec_type = str(record.get("Type", "") or "").strip()
        type_idx = safe_index(type_options, rec_type, default=safe_index(type_options, "Other", 0))
        asset_type = st.selectbox("Type *", options=type_options, index=type_idx, key=f"{prefix}_type")

    col7, col8, col9 = st.columns(3)
    with col7:
        manufacturer = st.text_input(
            "Manufacturer/Supplier *",
            value=str(record.get("Manufacturer/Supplier", "") or ""),
            key=f"{prefix}_manufacturer",
        )
    with col8:
        model = st.text_input(
            "Model *",
            value=str(record.get("Model", "") or ""),
            key=f"{prefix}_model",
        )
    with col9:
        est_value = st.text_input(
            "Est Value",
            value=str(record.get("Est Value", "") or ""),
            key=f"{prefix}_est_value",
        )

    col10, col11, col12 = st.columns(3)
    with col10:
        mfg_sn = st.text_input(
            "Mfg SN *",
            value=str(record.get("Mfg SN", "") or ""),
            key=f"{prefix}_mfg_sn",
        )
    with col11:
        mfg_year = st.text_input(
            "Mfg Year *",
            value=str(record.get("Mfg Year", "") or ""),
            key=f"{prefix}_mfg_year",
        )
    with col12:
        rec_freq = str(record.get("Maintenance Frequency", "") or "").strip()
        freq_idx = safe_index(freq_options, rec_freq, default=safe_index(freq_options, "None", 0))
        maint_freq = st.selectbox(
            "Maintenance Frequency",
            options=freq_options,
            index=freq_idx,
            key=f"{prefix}_maint_freq",
        )

    # ---- Location / assignment ----
    col13, col14 = st.columns(2)
    with col13:
        existing_loc = str(record.get("Functional Location", "") or "").strip()
        loc_options = [""] + FUNCTIONAL_LOCATION_OPTIONS
        if existing_loc and existing_loc not in loc_options:
            loc_options.insert(1, existing_loc)

        func_loc = st.selectbox(
            "Functional Location",
            options=loc_options,
            index=safe_index(loc_options, existing_loc, default=0),
            key=f"{prefix}_func_loc",
        )
    with col14:
        mapped_desc = FUNCTIONAL_LOCATION_DESCRIPTION.get(str(func_loc or "").strip(), "")
        fallback_desc = str(record.get("Functional Loc. Description", "") or record.get("Functional Location Description", "") or "").strip()
        func_loc_desc_val = mapped_desc if mapped_desc else (fallback_desc if str(func_loc or "").strip() == existing_loc else "")
        func_loc_desc = st.text_input(
            "Functional Loc. Description",
            value=func_loc_desc_val,
            disabled=True,
            key=f"{prefix}_func_loc_desc",
        )

    col15, col16, col17 = st.columns(3)
    with col15:
        assign_project = st.text_input(
            "Assign Project",
            value=str(record.get("Assign Project", "") or ""),
            key=f"{prefix}_assign_project",
        )
    with col16:
        floor = st.selectbox(
            "Floor",
            options=floor_options,
            index=safe_index(floor_options, str(record.get("Floor", "") or ""), default=0),
            key=f"{prefix}_floor",
        )
    with col17:
        prod_line = st.text_input(
            "Prod. Line",
            value=str(record.get("Prod. Line", "") or record.get("Production Line", "") or ""),
            key=f"{prefix}_prod_line",
        )

    # ---- Dates + auto status ----
    col18, col19, col20 = st.columns(3)
    with col18:
        start_date_val = st.date_input(
            "Start Date",
            value=_safe_parse_date(record.get("Start Date"), fallback=date.today()) or date.today(),
            key=f"{prefix}_start_date",
        )

    due_date_val = _safe_calc_due_date(start_date_val, maint_freq) or _safe_parse_date(record.get("Due Date"), fallback=None)
    days_left_val = _safe_calc_days_left(due_date_val) if due_date_val else (record.get("Day Left", "") or "")

    # ---- Status Rules (priority order) ----
    # 1) Functional Location == Obsolete -> Status = Obsolete
    # 2) Day Left <= 0 -> Expired
    # 3) Day Left < 7 -> Expired Soon
    # 4) Functional Location == 1006-10PE -> Good
    # 5) Functional Location other than 1006-10PE -> Idle
    func_loc_norm = str(func_loc or "").strip()
    record_status = str(record.get("Status", "") or "").strip()

    days_left_int = None
    try:
        if days_left_val is not None and str(days_left_val).strip() != "":
            days_left_int = int(float(str(days_left_val).strip()))
    except Exception:
        days_left_int = None

    if func_loc_norm == "Obsolete":
        status_val = "Obsolete"
    elif days_left_int is not None and days_left_int <= 0:
        status_val = "Expired"
    elif days_left_int is not None and days_left_int < 7:
        status_val = "Expired Soon"
    elif func_loc_norm == "1006-10PE":
        status_val = "Good"
    elif func_loc_norm:
        status_val = "Idle"
    else:
        status_val = record_status

    with col19:
        st.date_input(
            "Due Date (auto)",
            value=due_date_val if isinstance(due_date_val, date) else date.today(),
            disabled=True,
            key=f"{prefix}_due_date_display",
        )
    with col20:
        st.text_input(
            "Day Left (auto)",
            value=str(days_left_val),
            disabled=True,
            key=f"{prefix}_day_left_display",
        )

    col21, col22 = st.columns(2)
    with col21:
        # Status is auto; show as disabled text (prevents manual inconsistency)
        st.text_input(
            "Status (auto)",
            value=str(status_val) if status_val else "",
            disabled=True,
            key=f"{prefix}_status_display",
        )
    with col22:
        remark = st.text_input(
            "Remark",
            value=str(record.get("Remark", "") or ""),
            key=f"{prefix}_remark",
        )

    # ---- Department ID (auto / locked on update) ----
    if is_update:
        dept_id = str(record.get("Department ID", "") or "")
    else:
        dept_id = generate_department_id_add(department, asset_prefix, load_existing_data())

    st.text_input(
        "Department ID (auto)" if not is_update else "Department ID",
        value=dept_id,
        disabled=True,
        key=f"{prefix}_dept_id_display",
    )

    # ---- Images (optional) ----
    # Keep behavior simple: Add/Update can upload additional images; existing images are not removed here.
    images = st.file_uploader(
        "Upload Images (optional)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key=f"{prefix}_images",
    ) or []

    return {
        "Department ID": dept_id,
        "Department": department,
        "Description of Asset": desc_norm,
        "Prefix": asset_prefix,
        "Asset Number": asset_number,
        "SAP No.": sap_no,
        "Type": asset_type,
        "Manufacturer/Supplier": manufacturer,
        "Model": model,
        "Mfg SN": mfg_sn,
        "Mfg Year": mfg_year,
        "Est Value": est_value,
        "Maintenance Frequency": maint_freq,
        "Functional Location": func_loc,
        "Functional Loc. Description": func_loc_desc_val,
        "Assign Project": assign_project,
        "Floor": floor,
        "Prod. Line": prod_line,
        "Start Date": start_date_val,
        "Due Date": due_date_val if isinstance(due_date_val, date) else None,
        "Day Left": days_left_val,
        "Status": status_val if status_val else "",
        "Remark": remark,
        "Images": images,
    }

def windows_confirm_delete(message: str, title: str = "Confirm delete") -> bool:
    """Windows system message box confirmation. Returns True if user clicks Yes."""
    try:
        import ctypes
        MB_YESNO = 0x00000004
        MB_ICONWARNING = 0x00000030
        IDYES = 6
        result = ctypes.windll.user32.MessageBoxW(0, message, title, MB_YESNO | MB_ICONWARNING)
        return result == IDYES
    except Exception as e:
        st.error(f"Cannot open Windows confirmation dialog. ({e})")
        return False

# ================= ROW 1: ADD NEW EQUIPMENT =================
st.markdown("### ➕ Add New Asset")
add_button_col1, add_button_col2 = st.columns([1, 5])
with add_button_col1:
    if st.button("➕ Add New Asset" if not st.session_state.show_add_form else "➖ Hide Form",
                 use_container_width=True,
                 type="primary" if not st.session_state.show_add_form else "secondary"):
        st.session_state.show_add_form = not st.session_state.show_add_form
        st.rerun()

if st.session_state.show_add_form:
    st.markdown("#### 🏭 Asset Details")
    form_vals = render_equipment_form(prefix="add", is_update=False)

    submit = st.button("📝 Register Asset")
    if submit:
        # basic validation
        is_valid, error_msg = validate_equipment_details(
            form_vals["Description of Asset"],
            form_vals["Type"],
            form_vals["Manufacturer/Supplier"],
            form_vals["Model"],
            form_vals["Mfg SN"],
            form_vals["Mfg Year"]
        )
        if not is_valid:
            st.error(error_msg)
        elif check_duplicate(form_vals["Asset Number"], existing_df):
            st.warning(f"⚠️ Equipment with Asset Number '{form_vals['Asset Number']}' already exists.")
        else:
            verified_username = _performed_by_label()
            row = {
                "Department ID": form_vals["Department ID"],
                "Department": form_vals["Department"],
                "Description of Asset": form_vals["Description of Asset"],
                "Prefix": form_vals["Prefix"],
                "Asset Number": form_vals["Asset Number"],
                "SAP No.": form_vals["SAP No."],
                "Type": form_vals["Type"],
                "Manufacturer/Supplier": form_vals["Manufacturer/Supplier"],
                "Model": form_vals["Model"],
                "Mfg SN": form_vals["Mfg SN"],
                "Mfg Year": form_vals["Mfg Year"],
                "Est Value": form_vals["Est Value"],
                "Maintenance Frequency": form_vals["Maintenance Frequency"],
                "Functional Location": form_vals["Functional Location"],
                "Functional Loc. Description": form_vals.get("Functional Loc. Description", ""),
                "Assign Project": form_vals["Assign Project"],
                "Floor": form_vals["Floor"],
                "Prod. Line": form_vals.get("Prod. Line", ""),
                "Start Date": form_vals["Start Date"],
                "Due Date": form_vals["Due Date"],
                "Day Left": form_vals["Day Left"],
                "Status": form_vals["Status"],
                "Remark": form_vals["Remark"],
            }
            row = save_row_to_df(row)
            new_entry = pd.DataFrame([row])
            updated_df = load_existing_data()
            updated_df = pd.concat([updated_df, new_entry], ignore_index=True) if updated_df is not None else new_entry

            if save_data(updated_df):
                log_asset_operation(
                    action="ADD",
                    department_id=row["Department ID"],
                    asset_number=row["Asset Number"],
                    description=row["Description of Asset"],
                    details=f"Type: {row['Type']}, Manufacturer: {row['Manufacturer/Supplier']}, Model: {row['Model']}",
                    user_name=verified_username,
                )

                images = form_vals.get("Images")
                if images:
                    try:
                        delete_key = _asset_key_prefix(
                            department_id=row.get("Department ID", ""),
                            asset_number=row.get("Asset Number", ""),
                        )
                        save_prefix = _asset_image_prefix(
                            department_id=row.get("Department ID", ""),
                            asset_number=row.get("Asset Number", ""),
                            description=row.get("Description of Asset", ""),
                        )
                        _save_uploaded_images_replace(
                            target_dir=IMG_DIR,
                            delete_key_prefix=delete_key,
                            save_prefix=save_prefix,
                            images=images,
                        )
                    except Exception as e:
                        st.warning(f"Image save failed: {e}")

                st.success(f"✅ Registered: {row['Description of Asset']} (Asset Number: {row['Asset Number']})")
                st.session_state.description = ""
                st.session_state.show_add_form = False
                st.rerun()

# ================= DIVIDER =================
st.markdown("---")

# ================= ROW 2: UPDATE ASSET DATABASE =================
st.markdown("### ✏️ Update Asset Database")

# Row 1: Search bar (Department ID / Asset Number / SAP No.)
search_col1, search_col2 = st.columns([4, 1])
with search_col1:
    search_text = st.text_input(
        "Search (Department ID / Asset Number / SAP No.)",
        placeholder="Type Department ID or Asset Number or SAP No. (e.g. 88-15ME-ABC-001 / A-0001 / 5100001234)",
        label_visibility="collapsed",
        key="search_asset_text",
    )
with search_col2:
    if st.button("🔍 Find", use_container_width=True):
        st.rerun()

st.caption("Scan via camera or a handheld QR scanner.")
qr_mode = st.radio(
    "Scan method",
    options=["QR scanner", "Camera"],
    horizontal=True,
    key="editor_qr_mode",
    label_visibility="collapsed",
)

if qr_mode == "QR scanner":
    scanned = st.text_input(
        "QR scanner input",
        placeholder="Click here then scan (scanner types + Enter)",
        key="editor_qr_scanner",
        label_visibility="collapsed",
    )
    scanned = str(scanned or "").strip()
    if scanned and st.session_state.get("editor_qr_scanner_last") != scanned:
        st.session_state["editor_qr_scanner_last"] = scanned
        st.session_state["search_asset_text"] = scanned
        st.rerun()
else:
    cam = st.camera_input("Scan QR and fill search", key="editor_qr_cam")
    digest = uploaded_file_sha256(cam)
    if cam is not None and digest and st.session_state.get("editor_qr_cam_digest") != digest:
        st.session_state["editor_qr_cam_digest"] = digest
        payload = decode_qr_payload_from_image(cam)
        if payload:
            st.session_state["search_asset_text"] = payload
            st.rerun()
        else:
            st.warning("No QR detected. Try again with a clearer shot.")

# Load fresh data
existing_df = load_existing_data()

if existing_df is None or existing_df.empty:
    st.info("📝 No assets registered yet.")
else:
    # Columns to search (only use those that exist)
    candidate_cols = ["Department ID", "Asset Number", "SAP No."]
    search_cols = [c for c in candidate_cols if c in existing_df.columns]

    if not search_cols:
        st.error("Missing required columns for search. Need at least one of: Department ID, Asset Number, SAP No.")
        st.stop()

    q = str(search_text or "").strip()

    if not q:
        st.info("Type in the search box to find an asset.")
        matches = pd.DataFrame()
    else:
        mask = None
        for c in search_cols:
            m = existing_df[c].astype(str).str.contains(q, case=False, na=False)
            mask = m if mask is None else (mask | m)
        matches = existing_df[mask].copy() if mask is not None else pd.DataFrame()

    if q and matches.empty:
        st.info("🔍 No records found.")
    elif not matches.empty:
        st.caption(f"Found {len(matches)} record(s). Select one to edit.")

        def _label_for_row(row: pd.Series) -> str:
            dept = str(row.get("Department ID", "") or "")
            asset_no = str(row.get("Asset Number", "") or "")
            sap = str(row.get("SAP No.", "") or "")
            desc = str(row.get("Description of Asset", "") or "")
            return f"{dept} | {asset_no} | SAP:{sap} | {desc}".strip()

        options = {}
        for idx, row in matches.iterrows():
            options[_label_for_row(row)] = idx

        # Optional: clear pending update when switching record
        def _on_select_record_change():
            st.session_state.pending_update = None

        selected_label = st.selectbox(
            "Select record",
            options=list(options.keys()),
            key="selected_asset_record",
            on_change=_on_select_record_change,
        )
        record_index = options[selected_label]
        record = existing_df.loc[record_index]

        # IMPORTANT FIX: unique prefix per record so the form refreshes when selection changes
        upd_prefix = f"upd_{record_index}"

        st.markdown("#### ✏️ Edit Asset Details")
        form_vals = render_equipment_form(prefix=upd_prefix, record=record.to_dict(), is_update=True)

        # Update and Delete buttons
        col_update, col_delete = st.columns(2)

        with col_update:
            update_submit = st.button(
                "💾 Update Asset",
                type="primary",
                use_container_width=True,
                key=f"{upd_prefix}_update_submit",
            )

        with col_delete:
            delete_btn = st.button(
                "🗑️ Delete Asset",
                type="secondary",
                use_container_width=True,
                key=f"{upd_prefix}_delete_btn",
            )
            if delete_btn:
                st.session_state.delete_confirm_dept_id = record.get("Department ID", "")
                st.session_state.delete_confirm_asset = (
                    f"{record.get('Department ID', '')} - {record.get('Asset Number', '')} - {record.get('Description of Asset', '')}"
                )

        if update_submit:
            is_valid, error_msg = validate_equipment_details(
                form_vals["Description of Asset"],
                form_vals["Type"],
                form_vals["Manufacturer/Supplier"],
                form_vals["Model"],
                form_vals["Mfg SN"],
                form_vals["Mfg Year"]
            )
            if not is_valid:
                st.error(error_msg)
            else:
                verified_username = _performed_by_label()
                updated_row = {
                    "Department ID": record.get("Department ID", ""),
                    "Department": form_vals["Department"],
                    "Description of Asset": form_vals["Description of Asset"],
                    "Prefix": form_vals["Prefix"],
                    "Asset Number": form_vals["Asset Number"],
                    "SAP No.": form_vals["SAP No."],
                    "Type": form_vals["Type"],
                    "Manufacturer/Supplier": form_vals["Manufacturer/Supplier"],
                    "Model": form_vals["Model"],
                    "Mfg SN": form_vals["Mfg SN"],
                    "Mfg Year": form_vals["Mfg Year"],
                    "Est Value": form_vals["Est Value"],
                    "Maintenance Frequency": form_vals["Maintenance Frequency"],
                    "Functional Location": form_vals["Functional Location"],
                    "Functional Loc. Description": form_vals.get("Functional Loc. Description", ""),
                    "Assign Project": form_vals["Assign Project"],
                    "Floor": form_vals["Floor"],
                    "Prod. Line": form_vals.get("Prod. Line", ""),
                    "Start Date": form_vals["Start Date"],
                    "Due Date": form_vals["Due Date"],
                    "Day Left": form_vals["Day Left"],
                    "Status": form_vals["Status"],
                    "Remark": form_vals["Remark"]
                }
                updated_row = save_row_to_df(updated_row)
                existing_df = load_existing_data()
                for k, v in updated_row.items():
                    existing_df.at[record_index, k] = v

                if save_data(existing_df):
                    log_asset_operation(
                        action="UPDATE",
                        department_id=updated_row.get("Department ID", ""),
                        asset_number=updated_row.get("Asset Number", ""),
                        description=updated_row.get("Description of Asset", ""),
                        details=f"Type: {updated_row.get('Type', '')}, Manufacturer: {updated_row.get('Manufacturer/Supplier', '')}, Model: {updated_row.get('Model', '')}",
                        user_name=verified_username,
                    )

                    images = form_vals.get("Images") or []
                    if images:
                        try:
                            delete_key = _asset_key_prefix(
                                department_id=str(record.get("Department ID", "") or ""),
                                asset_number=str(record.get("Asset Number", "") or ""),
                            )
                            save_prefix = _asset_image_prefix(
                                department_id=updated_row.get("Department ID", ""),
                                asset_number=updated_row.get("Asset Number", ""),
                                description=updated_row.get("Description of Asset", ""),
                            )
                            _save_uploaded_images_replace(
                                target_dir=IMG_DIR,
                                delete_key_prefix=delete_key,
                                save_prefix=save_prefix,
                                images=images,
                            )
                        except Exception as e:
                            st.warning(f"Image save failed: {e}")

                    st.success("✅ Asset record updated.")
                    st.rerun()
        
        # Delete confirmation dialog
        if st.session_state.get("delete_confirm_dept_id"):
            st.markdown("---")
            with st.container(border=True):
                st.error("⚠️ DELETE CONFIRMATION")
                st.markdown(f"""
                You are about to **permanently delete** this asset:

                **{st.session_state.get('delete_confirm_asset', '')}**

                ⛔ This action **CANNOT BE UNDONE** ⛔
                """)

                col_confirm, col_cancel = st.columns(2)
                with col_confirm:
                    if st.button("🔴 DELETE PERMANENTLY", type="primary", use_container_width=True):
                        # Only rely on current login clearance (no second verification)
                        if _current_user_level_rank() < 2:
                            st.error("Access denied: requires SuperUser/Admin clearance.")
                            st.stop()

                        dept_id_to_delete = st.session_state.get("delete_confirm_dept_id", "")
                        confirm = windows_confirm_delete(
                            message=(
                                "Permanently delete this asset?\n\n"
                                f"{st.session_state.get('delete_confirm_asset', '')}\n\n"
                                "This action cannot be undone."
                            ),
                            title="Asset Editor - Delete confirmation",
                        )
                        if not confirm:
                            st.stop()

                        verified_username = _performed_by_label()
                        if delete_asset_by_dept_id(dept_id_to_delete):
                            log_asset_operation(
                                action="DELETE",
                                department_id=record.get("Department ID", ""),
                                asset_number=record.get("Asset Number", ""),
                                description=record.get("Description of Asset", ""),
                                details=f"Type: {record.get('Type', '')}, Manufacturer: {record.get('Manufacturer/Supplier', '')}, Model: {record.get('Model', '')}",
                                user_name=verified_username,
                            )
                            st.session_state.delete_confirm_dept_id = None
                            st.session_state.delete_confirm_asset = None
                            st.success("✅ Asset permanently deleted from database!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to delete asset.")

                with col_cancel:
                    if st.button("❌ CANCEL DELETION", use_container_width=True):
                        st.session_state.delete_confirm_dept_id = None
                        st.session_state.delete_confirm_asset = None
                        st.rerun()
    else:
        st.info("📝 No assets available to update.")

# ================= DOWNLOAD CSV BUTTON =================
st.markdown("---")
existing_df = load_existing_data()

col1, col2 = st.columns(2)
with col1:
    if existing_df is not None and not existing_df.empty:
        st.caption("Data export removed (CSV-free mode).")
    else:
        st.info("📝 No assets registered yet. Click 'Add New Asset' to register assets.")

with col2:
    if st.button("📋 View Asset Log History"):
        st.session_state.show_log = not st.session_state.get("show_log", False)
        st.rerun()

# Display asset log history if toggled
if st.session_state.get("show_log", False):
    st.markdown("---")
    st.markdown("### 📊 Asset Operation Log")
    logs_df = get_asset_logs(limit=200)
    if logs_df is not None and not logs_df.empty:
        # Format the dataframe for display
        display_logs = logs_df.copy()
        display_logs = display_logs.rename(columns={
            'timestamp': '📅 Timestamp',
            'action': '🔄 Action',
            'department_id': '🏢 Department ID',
            'asset_number': '📦 Asset Number',
            'description': '📝 Description',
            'details': '📄 Details',
            'user_name': '👤 User'
        })
        st.dataframe(display_logs, use_container_width=True)
    else:
        st.info("📝 No operation logs available yet.")