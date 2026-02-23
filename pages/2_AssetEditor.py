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
    delete_asset_by_dept_id, verify_user_qr_id, require_login
)

require_login()

st.title("📝 Asset Editor")
st.markdown("---")

ensure_data_directory()
initialize_log_database()  # <-- ADD: make sure logging DB/table exists before any log write

DATA_DIR = Path("data")
IMG_DIR = Path("images")
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

# regdata.db (access level source)
REGDATA_DB_PATH = DATA_DIR / "regdata.db"

existing_df = load_existing_data()

# initialize session state keys
for k, v in {
    "show_add_form": False,
    "description": "",
    "prefix": "",
    "pending_add": None,
    "pending_update": None,
    "delete_pending_qr_verify": None,
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

# --- NEW: SuperUser (admin) verification via regdata.db (QR scan ONLY) ---

def _is_superuser_level(level_value: str) -> bool:
    v = str(level_value or "").strip().lower()
    return ("super" in v) or ("admin" in v)

@st.cache_data(show_spinner=False)
def _discover_regdata_layout_qr_only(db_path_str: str):
    """
    Finds a table in regdata.db that contains:
      - QRID column
      - level column
    Also tries to find an optional display/name column.
    """
    db_path = Path(db_path_str)
    if not db_path.exists():
        return None

    conn = sqlite3.connect(db_path)
    try:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

        def norm(s: str) -> str:
            return str(s).strip().lower()

        qr_candidates = {"qrid", "qr_id", "qr id", "qrcode", "qr_code", "badge", "badgeid"}
        level_candidates = {"level", "userlevel", "user_level", "role", "access_level", "access level"}
        name_candidates = {"username", "user_name", "name", "fullname", "full_name", "staffname", "staff_name"}

        for table in tables:
            cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
            col_names = [c[1] for c in cols]
            col_norm = {norm(c): c for c in col_names}

            qr_col = next((col_norm[n] for n in qr_candidates if n in col_norm), None)
            level_col = next((col_norm[n] for n in level_candidates if n in col_norm), None)
            name_col = next((col_norm[n] for n in name_candidates if n in col_norm), None)

            if qr_col and level_col:
                return {"table": table, "qr_col": qr_col, "level_col": level_col, "name_col": name_col}

        return None
    finally:
        conn.close()

def verify_superuser_qr_scan(scanned_qr: str):
    """
    Returns (ok: bool, user_name_or_message: str)
    QR scan ONLY: matches regdata.db QRID column (manual userID won't match).
    """
    scanned_qr = str(scanned_qr or "").strip()
    if not scanned_qr:
        return False, "Please scan SuperUser QR (QRID)."

    layout = _discover_regdata_layout_qr_only(str(REGDATA_DB_PATH))
    if not layout:
        return False, "regdata.db layout not found (requires QRID + level columns)."

    conn = sqlite3.connect(REGDATA_DB_PATH)
    try:
        cur = conn.cursor()
        cols = [layout["level_col"]]
        if layout.get("name_col"):
            cols.append(layout["name_col"])

        query = f"SELECT {', '.join(cols)} FROM {layout['table']} WHERE {layout['qr_col']} = ? LIMIT 1"
        cur.execute(query, (scanned_qr,))
        row = cur.fetchone()
        if not row:
            return False, "QRID not found in regdata.db."

        level_value = row[0]
        if not _is_superuser_level(level_value):
            return False, "Access denied: SuperUser only."

        user_name = ""
        if layout.get("name_col") and len(row) >= 2 and row[1] is not None:
            user_name = str(row[1]).strip()

        return True, (user_name or "SuperUser")
    finally:
        conn.close()

def verify_superuser_qr_scan_or_show_error(scanned_qr: str):
    ok, msg = verify_superuser_qr_scan(scanned_qr)
    if not ok:
        st.error(msg)
        return False
    return True

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
            # Require QR verification before saving
            st.session_state.pending_add = form_vals
            st.rerun()

    # QR verification step for Add
    if st.session_state.pending_add:
        form_vals = st.session_state.pending_add
        st.markdown("#### 🔐 Verification required")

        # NEW: choose input method (fixes wrong entry errors)
        add_method = st.radio(
            "Verification method",
            options=["QR Scan", "Manual UserID"],
            horizontal=True,
            key="add_verify_method",
        )

        qr_col1, qr_col2 = st.columns([2, 1])
        with qr_col1:
            add_qr_id = st.text_input(
                "Scan QR / Enter UserID",
                key="add_verify_qr",
                placeholder="Scan QR (QRID) or type UserID...",
            )
        with qr_col2:
            st.write("")
            st.write("")
            confirm_add = st.button("✅ Confirm & Register", type="primary", use_container_width=True)

        if confirm_add:
            if not add_qr_id or not add_qr_id.strip():
                st.error("Please scan QR or enter UserID.")
                st.stop()

            ok, msg = verify_user_qr_id(
                add_qr_id.strip(),
                is_qr_scan=(add_method == "QR Scan"),
            )
            if not ok:
                st.error(msg)  # shows QR not found / UserID not found
                st.stop()

            verified_username = msg
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
                "Remark": form_vals["Remark"]
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
                    user_name=verified_username
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
                st.session_state.pending_add = None
                st.session_state.description = ""
                st.session_state.show_add_form = False
                st.rerun()
        if st.button("❌ Cancel", key="cancel_add_verify"):
            st.session_state.pending_add = None
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
                st.session_state.pending_update = {
                    "record_index": record_index,
                    "record": record,
                    "updated_row": updated_row,
                    "images": form_vals.get("Images") or [],
                }
                st.rerun()

        # QR verification step for Update
        if st.session_state.pending_update and st.session_state.pending_update.get("record_index") == record_index:
            pend = st.session_state.pending_update
            st.markdown("---")
            st.markdown("#### 🔐 Verification required")

            upd_method = st.radio(
                "Verification method",
                options=["QR Scan", "Manual UserID"],
                horizontal=True,
                key=f"{upd_prefix}_verify_method",
            )

            uqr_col1, uqr_col2 = st.columns([2, 1])
            with uqr_col1:
                upd_qr_id = st.text_input(
                    "Scan QR / Enter UserID",
                    key=f"{upd_prefix}_verify_qr",
                    placeholder="Scan QR (QRID) or type UserID...",
                )
            with uqr_col2:
                st.write("")
                st.write("")
                confirm_upd = st.button(
                    "✅ Confirm & Update",
                    type="primary",
                    use_container_width=True,
                    key=f"{upd_prefix}_confirm_upd_btn",
                )

            if confirm_upd:
                if not upd_qr_id or not upd_qr_id.strip():
                    st.error("Please scan QR or enter UserID.")
                    st.stop()

                ok, msg = verify_user_qr_id(
                    upd_qr_id.strip(),
                    is_qr_scan=(upd_method == "QR Scan"),
                )
                if not ok:
                    st.error(msg)
                    st.stop()

                verified_username = msg
                existing_df = load_existing_data()
                for k, v in pend["updated_row"].items():
                    existing_df.at[pend["record_index"], k] = v

                if save_data(existing_df):
                    log_asset_operation(
                        action="UPDATE",
                        department_id=pend["updated_row"].get("Department ID", ""),
                        asset_number=pend["updated_row"].get("Asset Number", ""),
                        description=pend["updated_row"].get("Description of Asset", ""),
                        details=f"Type: {pend['updated_row'].get('Type', '')}, Manufacturer: {pend['updated_row'].get('Manufacturer/Supplier', '')}, Model: {pend['updated_row'].get('Model', '')}",
                        user_name=verified_username
                    )

                    # Save uploaded images (if any)
                    images = pend.get("images") or []
                    if images:
                        try:
                            # Replace existing images for the ORIGINAL asset key,
                            # then save under the UPDATED description/name.
                            delete_key = _asset_key_prefix(
                                department_id=pend.get("record", {}).get("Department ID", ""),
                                asset_number=pend.get("record", {}).get("Asset Number", ""),
                            )
                            save_prefix = _asset_image_prefix(
                                department_id=pend["updated_row"].get("Department ID", ""),
                                asset_number=pend["updated_row"].get("Asset Number", ""),
                                description=pend["updated_row"].get("Description of Asset", ""),
                            )
                            _save_uploaded_images_replace(
                                target_dir=IMG_DIR,
                                delete_key_prefix=delete_key,
                                save_prefix=save_prefix,
                                images=images,
                            )
                        except Exception as e:
                            st.warning(f"Image save failed: {e}")

                    st.session_state.pending_update = None
                    st.success("✅ Asset record updated.")
                    st.rerun()

            if st.button("❌ Cancel", key=f"{upd_prefix}_cancel_upd_verify"):
                st.session_state.pending_update = None
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

                # If not yet in QR verify step, show YES DELETE to go to QR step
                if not st.session_state.get("delete_pending_qr_verify"):
                    col_confirm, col_cancel = st.columns(2)
                    with col_confirm:
                        if st.button("🔴 YES, DELETE PERMANENTLY", type="primary", use_container_width=True):
                            st.session_state.delete_pending_qr_verify = st.session_state.get("delete_confirm_dept_id", "")
                            st.rerun()
                    with col_cancel:
                        if st.button("❌ CANCEL DELETION", use_container_width=True):
                            st.session_state.delete_confirm_dept_id = None
                            st.session_state.delete_confirm_asset = None
                            st.rerun()
                else:
                    # NEW: SuperUser QR scan ONLY before delete
                    st.markdown("#### 🔐 Admin verification required (QR scan only)")
                    st.caption("Scan QR badge. Manual Key entry will NOT work.")

                    dqr_col1, dqr_col2 = st.columns([2, 1])
                    with dqr_col1:
                        del_qr_scan = st.text_input(
                            "Scan SuperUser QR (QRID)",
                            key="del_verify_qr_scan_only",
                            placeholder="Scan QR here...",
                        )
                    with dqr_col2:
                        st.write("")
                        st.write("")
                        verify_del_btn = st.button("✅ Verify & Delete", type="primary", use_container_width=True)

                    if verify_del_btn:
                        # 1) Verify SuperUser via regdata.db QRID+level
                        if not verify_superuser_qr_scan_or_show_error(del_qr_scan):
                            st.stop()

                        # 2) Windows system confirm dialog
                        dept_id_to_delete = st.session_state.get("delete_pending_qr_verify", "")
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

                        # 3) Delete
                        ok_name, verified_username = verify_superuser_qr_scan(del_qr_scan)
                        verified_username = verified_username if ok_name else "SuperUser"

                        if delete_asset_by_dept_id(dept_id_to_delete):
                            log_asset_operation(
                                action="DELETE",
                                department_id=record.get("Department ID", ""),
                                asset_number=record.get("Asset Number", ""),
                                description=record.get("Description of Asset", ""),
                                details=f"Type: {record.get('Type', '')}, Manufacturer: {record.get('Manufacturer/Supplier', '')}, Model: {record.get('Model', '')}",
                                user_name=verified_username
                            )
                            st.session_state.delete_confirm_dept_id = None
                            st.session_state.delete_confirm_asset = None
                            st.session_state.delete_pending_qr_verify = None
                            st.success("✅ Asset permanently deleted from database!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to delete asset.")

                    if st.button("❌ Cancel delete", key="cancel_del_verify"):
                        st.session_state.delete_pending_qr_verify = None
                        st.rerun()
    else:
        st.info("📝 No assets available to update.")

# ================= DOWNLOAD CSV BUTTON =================
st.markdown("---")
existing_df = load_existing_data()

col1, col2 = st.columns(2)
with col1:
    if existing_df is not None and not existing_df.empty:
        csv = existing_df.to_csv(index=False)
        st.download_button(
            "📥 Download Data as CSV", 
            csv, 
            file_name=f"asset_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 
            mime="text/csv"
        )
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
        
        # Download logs button
        st.download_button(
            "📥 Download Log as CSV",
            logs_df.to_csv(index=False),
            file_name=f"asset_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("📝 No operation logs available yet.")