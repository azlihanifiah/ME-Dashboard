import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Optional

# ======================================
# CONSTANTS
# ======================================
DATA_FILE = Path("data/DataBase_ME_Asset.csv")
LOG_DB_FILE = Path("data/asset_log.db")
REGDATA_DB = Path("data/regdata.db")
REQUIRED_COLUMNS = [
    "Prefix",
    "Department ID",
    "Department",
    "Description of Asset",
    "Asset Number",
    "SAP No.",
    "Type",
    "Manufacturer/Supplier",
    "Model",
    "Mfg SN",
    "Mfg Year",
    "Est Value",
    "Maintenance Frequency",
    "Functional Location",
    "Functional Loc. Description",
    "Assign Project",
    "Floor",
    "Prod. Line",
    "Start Date",
    "Due Date",
    "Day Left",
    "Status",
    "Remark",
]

ASSET_COLUMN_RENAMES = {
    "Functional Location Description": "Functional Loc. Description",
    "Production Line": "Prod. Line",
}

ASSET_COLUMNS_REMOVE = {
    "Functional Location Description",
    "Production Line",
    "Description of Equipment",
    "Acronym",
}

# ======================================
# HELPER FUNCTIONS
# ======================================
def ensure_data_directory() -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)


def clean_asset_database_schema() -> bool:
    """One-time / idempotent schema cleanup for data/DataBase_ME_Asset.csv.

    - Renames old columns into new ones while preserving data:
      - Functional Location Description -> Functional Loc. Description
      - Production Line -> Prod. Line
    - Removes deprecated columns:
      - Functional Location Description, Production Line, Description of Equipment, Acronym
    - Ensures REQUIRED_COLUMNS exist
    """
    ensure_data_directory()
    if not DATA_FILE.exists():
        return False

    try:
        df = pd.read_csv(DATA_FILE, encoding="utf-8", index_col=0)
    except Exception:
        return False

    changed = False

    # Migrate Description of Equipment -> Description of Asset (only if asset desc empty)
    if "Description of Equipment" in df.columns:
        if "Description of Asset" not in df.columns:
            df["Description of Asset"] = None
            changed = True

        asset_desc = df["Description of Asset"]
        equip_desc = df["Description of Equipment"]
        mask = (
            asset_desc.isna()
            | asset_desc.astype(str).str.strip().eq("")
        ) & (
            equip_desc.notna()
            & ~equip_desc.astype(str).str.strip().eq("")
        )
        if bool(mask.any()):
            df.loc[mask, "Description of Asset"] = equip_desc.loc[mask]
            changed = True

    # Migrate old column values into new columns
    for old_col, new_col in ASSET_COLUMN_RENAMES.items():
        if old_col not in df.columns:
            continue

        if new_col not in df.columns:
            df[new_col] = df[old_col]
            changed = True
            continue

        new_series = df[new_col]
        old_series = df[old_col]
        fill_mask = (
            new_series.isna() | new_series.astype(str).str.strip().eq("")
        ) & (
            old_series.notna() & ~old_series.astype(str).str.strip().eq("")
        )
        if bool(fill_mask.any()):
            df.loc[fill_mask, new_col] = old_series.loc[fill_mask]
            changed = True

    # Normalize Description of Asset to uppercase
    if "Description of Asset" in df.columns:
        new_desc = df["Description of Asset"].fillna("").astype(str).str.strip().str.upper()
        old_desc = df["Description of Asset"].fillna("").astype(str)
        if not new_desc.equals(old_desc):
            df["Description of Asset"] = new_desc
            changed = True

    # Uppercase all text cells except Status
    # (Keep Status values as-is, per requirement)
    for col in list(df.columns):
        if col == "Status":
            continue
        try:
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                new_series = df[col].fillna("").astype(str).str.strip().str.upper()
                old_series = df[col].fillna("").astype(str)
                if not new_series.equals(old_series):
                    df[col] = new_series
                    changed = True
        except Exception:
            continue

    # Drop deprecated columns
    drop_cols = [c for c in ASSET_COLUMNS_REMOVE if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        changed = True

    # Ensure required columns exist
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = None
            changed = True

    # Keep a consistent column order (required first, then any extras)
    ordered = [c for c in REQUIRED_COLUMNS if c in df.columns] + [c for c in df.columns if c not in REQUIRED_COLUMNS]
    if ordered != list(df.columns):
        df = df[ordered]
        changed = True

    if changed:
        df.to_csv(DATA_FILE, encoding="utf-8")
        try:
            load_existing_data.clear()
        except Exception:
            pass

    return changed

# ======================================
# LOGGING FUNCTIONS (SQLite)
# ======================================
def initialize_log_database() -> None:
    """Initialize SQLite database for logging asset operations"""
    try:
        conn = sqlite3.connect(LOG_DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS asset_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,
                department_id TEXT,
                asset_number TEXT,
                description TEXT,
                details TEXT,
                user_name TEXT DEFAULT 'System'
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"❌ Error initializing log database: {str(e)}")

def log_asset_operation(action: str, department_id: str, asset_number: str, 
                        description: str, details: str = "", user_name: str = "System") -> bool:
    """Log asset add/update operations to SQLite database"""
    try:
        initialize_log_database()
        conn = sqlite3.connect(LOG_DB_FILE)
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        cursor.execute("""
            INSERT INTO asset_logs 
            (timestamp, action, department_id, asset_number, description, details, user_name)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, action, department_id, asset_number, description, details, user_name))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"❌ Error logging operation: {str(e)}")
        return False

def get_asset_logs(limit: int = 100) -> Optional[pd.DataFrame]:
    """Retrieve asset operation logs from SQLite database"""
    try:
        if not LOG_DB_FILE.exists():
            return None
        
        conn = sqlite3.connect(LOG_DB_FILE)
        df = pd.read_sql_query(
            "SELECT * FROM asset_logs ORDER BY timestamp DESC LIMIT ?",
            conn,
            params=(limit,)
        )
        conn.close()
        return df if not df.empty else None
    except Exception as e:
        st.error(f"❌ Error retrieving logs: {str(e)}")
        return None

def export_logs_to_csv() -> Optional[str]:
    """Export logs to CSV file"""
    try:
        logs_df = get_asset_logs(limit=10000)
        if logs_df is not None and not logs_df.empty:
            csv_file = Path("data") / f"asset_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            logs_df.to_csv(csv_file, index=False)
            return str(csv_file)
        return None
    except Exception as e:
        st.error(f"❌ Error exporting logs: {str(e)}")
        return None

@st.cache_data
def load_existing_data() -> Optional[pd.DataFrame]:
    try:
        if DATA_FILE.exists():
            # Clean schema on load (idempotent)
            try:
                clean_asset_database_schema()
            except Exception:
                pass
            df = pd.read_csv(DATA_FILE, encoding="utf-8", index_col=0)
            for col in REQUIRED_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            return df
        return None
    except Exception as e:
        st.error(f"❌ Error reading data file: {str(e)}")
        return None

def save_data(df: pd.DataFrame) -> bool:
    try:
        df = df.dropna(how="all")
        df = df.reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = "Index"
        df.to_csv(DATA_FILE, encoding="utf-8")
        load_existing_data.clear()
        return True
    except Exception as e:
        st.error(f"❌ Error saving data: {str(e)}")
        return False

def delete_asset_by_dept_id(department_id: str) -> bool:
    """Delete an asset by Department ID"""
    try:
        # Force reload without cache
        if DATA_FILE.exists():
            df = pd.read_csv(DATA_FILE, encoding="utf-8", index_col=0)
            # Filter out the asset with matching Department ID
            df_filtered = df[df["Department ID"].astype(str) != str(department_id)]
            
            # If rows were actually deleted
            if len(df_filtered) < len(df):
                df_filtered = df_filtered.reset_index(drop=True)
                df_filtered.index = df_filtered.index + 1
                df_filtered.index.name = "Index"
                df_filtered.to_csv(DATA_FILE, encoding="utf-8")
                # Clear cache to force reload
                load_existing_data.clear()
                return True
        return False
    except Exception as e:
        st.error(f"❌ Error deleting asset: {str(e)}")
        return False

def check_duplicate(asset_number: str, existing_df: Optional[pd.DataFrame]) -> bool:
    if existing_df is None or existing_df.empty or not asset_number:
        return False
    asset_match = existing_df["Asset Number"].astype(str).str.strip() == str(asset_number).strip()
    return not existing_df[asset_match].empty

def calculate_due_date(start_date: date, maintenance_frequency: str) -> Optional[date]:
    frequency_days = {"Weekly": 7, "Biweekly": 14, "Monthly": 30, "Quarterly": 90, "Yearly": 365}
    days_to_add = frequency_days.get(maintenance_frequency)
    if days_to_add:
        return start_date + timedelta(days=days_to_add)
    return None

def calculate_days_left(due_date) -> Optional[int]:
    if not due_date:
        return None
    today = date.today()
    if isinstance(due_date, datetime):
        due = due_date.date()
    elif isinstance(due_date, date):
        due = due_date
    else:
        return None
    return (due - today).days

def calculate_status(days_left: Optional[int]) -> str:
    if days_left is None:
        return ""
    if days_left < 0:
        return "Expired"
    elif days_left <= 7:
        return "Expired Soon"
    else:
        return "Good"

def validate_equipment_details(description, equipment_type, manufacturer, model, mfg_sn, mfg_year) -> tuple[bool, str | None]:
    if not description.strip():
        return False, "Description of Asset is required."
    if not equipment_type.strip():
        return False, "Equipment Type is required."
    if not manufacturer.strip():
        return False, "Manufacturer/Supplier is required."
    if not model.strip():
        return False, "Model is required."
    if not mfg_sn.strip():
        return False, "Mfg S/N is required."
    if not mfg_year:
        return False, "Mfg Year is required."
    return True, None

def generate_department_id(acronym: str, df: Optional[pd.DataFrame], prefix: str = "88-15ME") -> str:
    if not acronym:
        return ""
    acronym = acronym.upper()
    if df is None or df.empty:
        next_num = 1
    else:
        existing_ids = df["Department ID"].dropna().astype(str)
        matching = existing_ids[existing_ids.str.contains(f"-{acronym}-")]
        next_num = len(matching) + 1
    return f"{prefix}-{acronym}-{next_num:03d}"

def generate_acronym(description: str, max_length: int = 5) -> str:
    if not description:
        return ""
    STOP_WORDS = {"of", "and", "the", "for", "to"}
    words = [w for w in description.strip().split() if w.lower() not in STOP_WORDS]
    if not words:
        return ""
    acronym = ""
    if len(words) < 3:
        for word in words:
            acronym += word[:2].upper()
    else:
        for word in words:
            acronym += word[0].upper()
    return acronym[:max_length]


# ======================================
# REGDATA (USER) VERIFICATION – read-only
# Manual enter: match RegData.userID  |  QR scan: match RegData.QRID
# ======================================
def verify_user_qr_id(value: str, is_qr_scan: bool = False) -> tuple[bool, str]:
    """
    Verify user from RegData table.
    - Manual entry: match UserID
    - QR scan: match QRID
    Returns (True, UserID) if found, else (False, error_message)
    """
    value = str(value or "").strip()
    if not value:
        return False, "User ID / QR is required."

    if not REGDATA_DB.exists():
        return False, "regdata.db not found (data/regdata.db)."

    conn = None
    try:
        conn = sqlite3.connect(REGDATA_DB)
        cursor = conn.cursor()

        if is_qr_scan:
            # QR scanner → match QRID
            cursor.execute(
                "SELECT UserID FROM RegData WHERE QRID = ? LIMIT 1",
                (value,)
            )
        else:
            # Manual entry → match UserID
            cursor.execute(
                "SELECT UserID FROM RegData WHERE UserID = ? LIMIT 1",
                (value,)
            )

        row = cursor.fetchone()
        if not row:
            # More helpful error depending on mode
            return False, ("QR not found in RegData (QRID)." if is_qr_scan else "UserID not found in RegData.")

        return True, str(row[0])

    except sqlite3.OperationalError as e:
        return False, f"Verification database error: {str(e)}"
    except Exception as e:
        return False, f"Verification error: {str(e)}"
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass


# ======================================
# REGDATA (LOGIN) HELPERS – flexible schema discovery
# ======================================
_LEVEL_RANK = {
    "user": 1,
    "operator": 1,
    "staff": 1,
    "tech": 1,
    "technician": 1,
    "superuser": 2,
    "super_user": 2,
    "super": 2,
    "admin": 3,
    "administrator": 3,
}


def _rank_from_level(value) -> int:
    if value is None:
        return 0
    v = str(value).strip().lower()
    # be forgiving on many schemas
    if "admin" in v:
        return 3
    if "super" in v:
        return 2
    if v in _LEVEL_RANK:
        return int(_LEVEL_RANK[v])
    if "user" in v or "staff" in v or "operator" in v or "tech" in v:
        return 1
    return 0


def _quote_ident(name: str) -> str:
    # SQLite identifier quoting
    return '"' + str(name).replace('"', '""') + '"'


@st.cache_data(show_spinner=False)
def _discover_regdata_layout(db_path_str: str):
    """Best-effort discovery of regdata table/columns.

    Supports common variations (RegData/regdata, UserID/user_id, QRID/qr_id, etc.).
    Returns a dict with table + column names, or None.
    """
    db_path = Path(db_path_str)
    if not db_path.exists():
        return None

    conn = sqlite3.connect(db_path)
    try:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

        user_candidates = {"userid", "user_id", "user id", "employeeid", "empid"}
        qr_candidates = {"qrid", "qr_id", "qr id", "qrcode", "qr_code", "badge", "badgeid"}
        level_candidates = {"userlevel", "user_level", "user level", "level", "role", "access_level", "access level"}
        is_super_candidates = {"issuperuser", "is_superuser", "superuser", "super_user"}
        name_candidates = {"name", "username", "user_name", "full_name", "fullname", "staff_name", "employee_name"}

        def norm(s: str) -> str:
            return str(s).strip().lower()

        for table in tables:
            cols = conn.execute(f"PRAGMA table_info({_quote_ident(table)})").fetchall()
            col_names = [c[1] for c in cols]
            col_norm = {norm(c): c for c in col_names}

            user_col = next((col_norm[n] for n in user_candidates if n in col_norm), None)
            qr_col = next((col_norm[n] for n in qr_candidates if n in col_norm), None)
            if not user_col and not qr_col:
                continue

            level_col = next((col_norm[n] for n in level_candidates if n in col_norm), None)
            is_super_col = next((col_norm[n] for n in is_super_candidates if n in col_norm), None)
            name_col = next((col_norm[n] for n in name_candidates if n in col_norm), None)

            return {
                "table": table,
                "user_col": user_col,
                "qr_col": qr_col,
                "level_col": level_col,
                "is_super_col": is_super_col,
                "name_col": name_col,
            }

        return None
    finally:
        conn.close()


def lookup_regdata_user(identifier: str, *, allow_userid: bool = True, allow_qr: bool = True) -> dict:
    """Lookup a user record in regdata.db using best-effort schema discovery.

    Returns dict: {ok, user_id, display_name, level_name, level_rank, error}
    """
    identifier = str(identifier or "").strip()
    if not identifier:
        return {"ok": False, "error": "User ID / QR is required.", "user_id": "", "display_name": "", "level_name": "", "level_rank": 0}

    if not REGDATA_DB.exists():
        return {"ok": False, "error": "regdata.db not found (data/regdata.db).", "user_id": "", "display_name": "", "level_name": "", "level_rank": 0}

    layout = _discover_regdata_layout(str(REGDATA_DB))
    if not layout:
        # Fall back to the legacy hard-coded schema (RegData: UserID, QRID)
        ok, msg = verify_user_qr_id(identifier, is_qr_scan=bool(allow_qr and not allow_userid))
        if not ok:
            return {"ok": False, "error": msg, "user_id": "", "display_name": "", "level_name": "", "level_rank": 0}
        return {"ok": True, "error": "", "user_id": msg, "display_name": msg, "level_name": "", "level_rank": 1}

    table = layout["table"]
    user_col = layout.get("user_col")
    qr_col = layout.get("qr_col")
    level_col = layout.get("level_col")
    is_super_col = layout.get("is_super_col")
    name_col = layout.get("name_col")

    where_parts: list[str] = []
    params: list[str] = []
    if allow_userid and user_col:
        where_parts.append(f"{_quote_ident(user_col)} = ?")
        params.append(identifier)
    if allow_qr and qr_col:
        where_parts.append(f"{_quote_ident(qr_col)} = ?")
        params.append(identifier)

    if not where_parts:
        return {"ok": False, "error": "regdata.db layout found but no usable UserID/QRID columns.", "user_id": "", "display_name": "", "level_name": "", "level_rank": 0}

    conn = sqlite3.connect(REGDATA_DB)
    try:
        query = f"SELECT * FROM {_quote_ident(table)} WHERE " + " OR ".join(where_parts) + " LIMIT 1"
        cur = conn.cursor()
        cur.execute(query, tuple(params))
        row = cur.fetchone()
        if not row:
            return {"ok": False, "error": "User not found in regdata.db.", "user_id": "", "display_name": "", "level_name": "", "level_rank": 0}

        colnames = [d[0] for d in cur.description]
        rec = {colnames[i]: row[i] for i in range(len(colnames))}

        user_id = ""
        if user_col and user_col in rec and rec[user_col] is not None:
            user_id = str(rec[user_col]).strip()

        display_name = user_id
        if name_col and name_col in rec and rec[name_col] is not None:
            n = str(rec[name_col]).strip()
            if n:
                display_name = n

        level_name = ""
        level_rank = 1
        if level_col and level_col in rec and rec[level_col] is not None:
            level_name = str(rec[level_col]).strip()
            level_rank = max(level_rank, _rank_from_level(level_name))

        if is_super_col and is_super_col in rec and rec[is_super_col] is not None:
            v = str(rec[is_super_col]).strip().lower()
            is_super = v in {"1", "true", "yes", "y", "t"} or rec[is_super_col] is True
            if is_super:
                level_name = level_name or "SuperUser"
                level_rank = max(level_rank, 2)

        return {
            "ok": True,
            "error": "",
            "user_id": user_id or identifier,
            "display_name": display_name or (user_id or identifier),
            "level_name": level_name,
            "level_rank": int(level_rank),
        }
    except sqlite3.OperationalError as e:
        return {"ok": False, "error": f"Verification database error: {str(e)}", "user_id": "", "display_name": "", "level_name": "", "level_rank": 0}
    except Exception as e:
        return {"ok": False, "error": f"Verification error: {str(e)}", "user_id": "", "display_name": "", "level_name": "", "level_rank": 0}
    finally:
        try:
            conn.close()
        except Exception:
            pass


def require_login(*, min_level_rank: int = 1) -> dict:
    """Require a user to be authenticated (based on regdata.db) before continuing.

    Renders a sidebar login form and stops the app if not authenticated.
    On success, stores identity in st.session_state:
      auth_ok, auth_user_id, auth_name, auth_level_name, auth_level_rank
    """
    # Initialize
    defaults = {
        "auth_ok": False,
        "auth_user_id": "",
        "auth_name": "",
        "auth_level_name": "",
        "auth_level_rank": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    with st.sidebar:
        st.markdown("### 🔐 Login")

        if st.session_state.get("auth_ok"):
            name = st.session_state.get("auth_name") or st.session_state.get("auth_user_id")
            lvl = st.session_state.get("auth_level_name")
            st.success(f"Signed in: {name}" + (f" ({lvl})" if lvl else ""))
            if st.button("Logout", use_container_width=True):
                for k in list(defaults.keys()):
                    st.session_state[k] = defaults[k]
                st.rerun()
        else:
            method = st.radio(
                "Login method",
                options=["QR Scan", "Manual UserID"],
                horizontal=True,
                key="login_method",
            )
            identifier = st.text_input(
                "Scan QR / Enter UserID",
                key="login_identifier",
                placeholder="Scan QR (QRID) or type UserID...",
            )
            if st.button("Login", type="primary", use_container_width=True):
                res = lookup_regdata_user(
                    identifier,
                    allow_userid=(method == "Manual UserID"),
                    allow_qr=(method == "QR Scan"),
                )
                if not res.get("ok"):
                    st.error(res.get("error") or "Login failed.")
                elif int(res.get("level_rank") or 0) < int(min_level_rank):
                    st.error("Access denied for this app.")
                else:
                    st.session_state.auth_ok = True
                    st.session_state.auth_user_id = res.get("user_id", "")
                    st.session_state.auth_name = res.get("display_name", "")
                    st.session_state.auth_level_name = res.get("level_name", "")
                    st.session_state.auth_level_rank = int(res.get("level_rank") or 0)
                    st.rerun()

    if not st.session_state.get("auth_ok"):
        st.info("Please login from the sidebar to use this dashboard.")
        st.stop()

    return {
        "ok": True,
        "user_id": st.session_state.get("auth_user_id", ""),
        "name": st.session_state.get("auth_name", ""),
        "level_name": st.session_state.get("auth_level_name", ""),
        "level_rank": int(st.session_state.get("auth_level_rank") or 0),
    }

def filter_dataframe(df: pd.DataFrame, search_term: str) -> pd.DataFrame:
    """Filter dataframe by search term across multiple columns"""
    try:
        if not search_term or df is None or df.empty:
            return df
        
        # Reset index to avoid alignment issues
        df = df.reset_index(drop=True)
        
        search_term = search_term.lower()
        
        # Create mask with proper index
        mask = pd.Series([False] * len(df), index=df.index)
        
        search_columns = [
            "Description of Asset", 
            "Asset Number", 
            "SAP No.", 
            "Type",
            "Manufacturer/Supplier", 
            "Model", 
            "Functional Location",
            "Functional Loc. Description",
            "Assign Project", 
            "Prod. Line", 
            "Status",
            "Department ID"
        ]
        
        # Search in columns that exist
        for col in search_columns:
            if col in df.columns:
                try:
                    col_mask = df[col].astype(str).str.lower().str.contains(search_term, na=False)
                    mask |= col_mask
                except:
                    # Skip column if error occurs
                    continue
        
        return df[mask]
    
    except:
        # Return empty dataframe if any error occurs
        return pd.DataFrame()

def initialize_stock_log_database() -> None:
    """
    Creates stock_log table inside data/asset_log.db
    (separate from existing asset logs).
    """
    LOG_DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(LOG_DB_FILE)
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS stock_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,              -- IN_ADD / OUT_ADJUST / OUT_TASK / EDIT
                part_number TEXT NOT NULL,
                qty INTEGER NOT NULL,

                before_total_add INTEGER,
                after_total_add INTEGER,
                before_total_used INTEGER,
                after_total_used INTEGER,
                before_total_quantity INTEGER,
                after_total_quantity INTEGER,

                performed_by TEXT,
                source TEXT,                       -- e.g. "Stock IN/OUT" / "Task Report"
                note TEXT
            )
        """)
        conn.commit()
    finally:
        conn.close()

def log_stock_operation(
    action: str,
    part_number: str,
    qty: int,
    before_total_add: int,
    after_total_add: int,
    before_total_used: int,
    after_total_used: int,
    performed_by: str = "",
    source: str = "",
    note: str = "",
) -> None:
    initialize_stock_log_database()

    before_total_quantity = int(before_total_add) + int(before_total_used)
    after_total_quantity = int(after_total_add) + int(after_total_used)

    conn = sqlite3.connect(LOG_DB_FILE)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO stock_log (
                timestamp, action, part_number, qty,
                before_total_add, after_total_add,
                before_total_used, after_total_used,
                before_total_quantity, after_total_quantity,
                performed_by, source, note
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                str(action),
                str(part_number),
                int(qty),
                int(before_total_add),
                int(after_total_add),
                int(before_total_used),
                int(after_total_used),
                int(before_total_quantity),
                int(after_total_quantity),
                str(performed_by or ""),
                str(source or ""),
                str(note or ""),
            ),
        )
        conn.commit()
    finally:
        conn.close()