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
    "Department ID", "Department", "Description of Asset", "Acronym",
    "Asset Number", "SAP No.", "Type", "Manufacturer/Supplier", "Model",
    "Mfg SN", "Mfg Year", "Est Value", "Maintenance Frequency",
    "Functional Location", "Functional Location Description",
    "Assign Project", "Floor", "Production Line", "Start Date",
    "Due Date", "Day Left", "Status", "Remark"
]

# ======================================
# HELPER FUNCTIONS
# ======================================
def ensure_data_directory() -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

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
            "Assign Project", 
            "Production Line", 
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