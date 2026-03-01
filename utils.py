import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Optional
import hashlib
import re
import json
import os
import subprocess

# ======================================
# CONSTANTS
# ======================================
# Single source of truth DB
APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "data"
MAIN_DB_FILE = DATA_DIR / "main_data.db"

# Legacy DBs (migration only)
LEGACY_LOG_DB_FILE = DATA_DIR / "asset_log.db"
LEGACY_WORKSHOP_DB_FILE = DATA_DIR / "workshop.db"

# Keep existing name used across the app
LOG_DB_FILE = MAIN_DB_FILE

REGDATA_DB = DATA_DIR / "regdata.db"

ASSET_TABLE = "database_me_asset"
BREAKDOWN_TABLE = "breakdown_report"

BREAKDOWN_COLUMNS = [
    "Date",
    "Job ID",
    "Job Type",
    "Severity",
    "Shift",
    "Location",
    "Machine/Equipment",
    "Machine ID",
    "Date/Time Start",
    "Date/Time End",
    "Duration",
    "JobStatus",
    "Problem Description",
    "Immediate Action",
    "Root Cause",
    "Preventive Action",
    "Spare Parts Used",
    "Reported By",
    "Created At",
]
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
    "Require Calibration",
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

def decode_qr_payload_from_image(uploaded_file) -> Optional[str]:
    """Decode a QR code payload from a Streamlit uploaded image.

    Designed for use with `st.camera_input()` or `st.file_uploader()`.
    Returns the decoded string, or None if no QR is detected.

    Requires: `opencv-python-headless` (or `opencv-python`) and `numpy`.
    """

    if uploaded_file is None:
        return None

    try:
        image_bytes = uploaded_file.getvalue()
    except Exception:
        try:
            image_bytes = uploaded_file.read()
        except Exception:
            return None

    if not image_bytes:
        return None

    try:
        import importlib
        np = importlib.import_module("numpy")
        cv2 = importlib.import_module("cv2")
    except Exception:
        return None

    try:
        data = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return None

        detector = cv2.QRCodeDetector()

        # Prefer multi-decode if available.
        if hasattr(detector, "detectAndDecodeMulti"):
            ok, decoded_info, _, _ = detector.detectAndDecodeMulti(img)
            if ok and decoded_info:
                for val in decoded_info:
                    val = str(val or "").strip()
                    if val:
                        return val

        val, _, _ = detector.detectAndDecode(img)
        val = str(val or "").strip()
        return val or None
    except Exception:
        return None


def uploaded_file_sha256(uploaded_file) -> Optional[str]:
    """Stable digest for Streamlit UploadedFile objects to avoid re-processing the same image."""
    if uploaded_file is None:
        return None
    try:
        b = uploaded_file.getvalue()
    except Exception:
        try:
            b = uploaded_file.read()
        except Exception:
            return None
    if not b:
        return None
    return hashlib.sha256(b).hexdigest()

def ensure_data_directory() -> None:
    MAIN_DB_FILE.parent.mkdir(parents=True, exist_ok=True)


def _get_secret_or_env(key: str, default: str = "") -> str:
    """Read a value from Streamlit secrets first, then environment variables."""
    try:
        # st.secrets behaves like a dict on Streamlit Cloud.
        v = st.secrets.get(key)  # type: ignore[attr-defined]
        if v is not None:
            return str(v)
    except Exception:
        pass
    return str(os.environ.get(key, default) or default)


def _parse_github_https_repo(url: str) -> str:
    """Return 'owner/repo' from common GitHub remote formats (https/ssh)."""
    s = str(url or "").strip()
    if not s:
        return ""

    # Common patterns:
    # - https://github.com/owner/repo.git
    # - http://github.com/owner/repo
    # - git@github.com:owner/repo.git
    # - ssh://git@github.com/owner/repo.git
    if s.startswith("git@github.com:"):
        tail = s[len("git@github.com:") :]
    elif s.startswith("ssh://git@github.com/"):
        tail = s[len("ssh://git@github.com/") :]
    else:
        # Normalize https/http (and also handle any string containing github.com)
        s2 = s
        if s2.startswith("https://"):
            s2 = s2[len("https://") :]
        elif s2.startswith("http://"):
            s2 = s2[len("http://") :]

        idx = s2.find("github.com")
        if idx < 0:
            return ""
        s2 = s2[idx + len("github.com") :]
        s2 = s2.lstrip(":/")
        tail = s2

    tail = tail.strip().strip("/")
    if tail.endswith(".git"):
        tail = tail[: -len(".git")]

    parts = [p for p in tail.split("/") if p]
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return ""


def persist_repo_changes(paths: list[str], *, reason: str = "Auto-save") -> bool:
    """Best-effort: git add/commit/push changed files so Streamlit Cloud sleep won't lose edits.

    Requires Streamlit secrets (or env vars):
    - GITHUB_TOKEN
    Optional:
    - GITHUB_REPO (owner/repo) or existing origin remote
    - GIT_BRANCH (default: main)
    - GIT_USER_NAME (default: streamlit-bot)
    - GIT_USER_EMAIL (default: streamlit-bot@users.noreply.github.com)
    """
    token = _get_secret_or_env("GITHUB_TOKEN", "").strip()
    if not token:
        return False

    repo_root = Path(__file__).resolve().parent

    def _run_git(args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )

    # Ensure we are in a git repo
    chk = _run_git(["rev-parse", "--is-inside-work-tree"])
    if chk.returncode != 0:
        return False

    # Stage provided paths
    rel_paths: list[str] = []
    for p in list(paths or []):
        if not p:
            continue
        try:
            pp = Path(p)
            if pp.is_absolute():
                try:
                    rel_paths.append(str(pp.relative_to(repo_root)).replace("\\", "/"))
                except Exception:
                    # If outside repo, skip.
                    continue
            else:
                rel_paths.append(str(pp).replace("\\", "/"))
        except Exception:
            continue

    if not rel_paths:
        return False

    add = _run_git(["add", "-A", "--", *rel_paths])
    if add.returncode != 0:
        return False

    # No changes -> no commit
    stt = _run_git(["status", "--porcelain"])
    if stt.returncode != 0:
        return False
    if not (stt.stdout or "").strip():
        return True

    # Configure author
    user_name = _get_secret_or_env("GIT_USER_NAME", "streamlit-bot").strip() or "streamlit-bot"
    user_email = _get_secret_or_env("GIT_USER_EMAIL", "streamlit-bot@users.noreply.github.com").strip() or "streamlit-bot@users.noreply.github.com"
    _run_git(["config", "user.name", user_name])
    _run_git(["config", "user.email", user_email])

    # Commit
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"Auto-save: {str(reason or 'Update').strip()} ({ts})"
    cmt = _run_git(["commit", "-m", msg])
    if cmt.returncode != 0:
        # If nothing to commit (race), treat as success.
        if "nothing to commit" in (cmt.stdout or "").lower() or "nothing to commit" in (cmt.stderr or "").lower():
            return True
        return False

    # Push
    branch = _get_secret_or_env("GIT_BRANCH", "main").strip() or "main"

    repo_slug = _get_secret_or_env("GITHUB_REPO", "").strip()
    if repo_slug and "github.com" in repo_slug:
        repo_slug = _parse_github_https_repo(repo_slug)
    if not repo_slug:
        origin = _run_git(["remote", "get-url", "origin"])
        if origin.returncode == 0:
            repo_slug = _parse_github_https_repo((origin.stdout or "").strip())

    if not repo_slug:
        return False

    # Push URL without persisting token in git config
    push_url = f"https://x-access-token:{token}@github.com/{repo_slug}.git"
    push = _run_git(["push", push_url, f"HEAD:{branch}"])
    return push.returncode == 0


def _connect_main_db() -> sqlite3.Connection:
    ensure_data_directory()
    conn = sqlite3.connect(MAIN_DB_FILE)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    except Exception:
        pass
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (str(table),),
    )
    return cur.fetchone() is not None


def _table_row_count(conn: sqlite3.Connection, table: str) -> int:
    try:
        cur = conn.cursor()
        cur.execute(f'SELECT COUNT(*) FROM "{table}"')
        return int(cur.fetchone()[0] or 0)
    except Exception:
        return 0


def _ensure_text_columns(conn: sqlite3.Connection, table: str, columns: list[str]) -> None:
    """Best-effort schema migration: add any missing TEXT columns.

    SQLite supports `ALTER TABLE ... ADD COLUMN` (but not drop/modify easily),
    which is enough for our flexible TEXT-based schema.
    """
    try:
        cur = conn.cursor()
        cur.execute(f'PRAGMA table_info("{table}")')
        existing = {str(r[1]) for r in (cur.fetchall() or [])}
        for c in list(columns or []):
            if c in existing:
                continue
            cur.execute(f'ALTER TABLE "{table}" ADD COLUMN "{c}" TEXT')
    except Exception:
        # Keep migration best-effort; app can still work via DataFrame padding.
        return


def _ensure_tables_in_main_db() -> None:
    conn = _connect_main_db()
    try:
        cur = conn.cursor()

        # Assets (keep schema flexible; store as TEXT like CSV import)
        if not _table_exists(conn, ASSET_TABLE):
            cols = ", ".join([f'"{c}" TEXT' for c in REQUIRED_COLUMNS])
            cur.execute(f'CREATE TABLE IF NOT EXISTS "{ASSET_TABLE}" ({cols})')
        else:
            _ensure_text_columns(conn, ASSET_TABLE, REQUIRED_COLUMNS)

        # Breakdown report
        if not _table_exists(conn, BREAKDOWN_TABLE):
            cols = ", ".join([f'"{c}" TEXT' for c in BREAKDOWN_COLUMNS])
            cur.execute(f'CREATE TABLE IF NOT EXISTS "{BREAKDOWN_TABLE}" ({cols})')
        else:
            _ensure_text_columns(conn, BREAKDOWN_TABLE, BREAKDOWN_COLUMNS)

        # Asset logs (formerly data/asset_log.db)
        cur.execute(
            """
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
            """
        )

        # Stock log (formerly data/asset_log.db)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS stock_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,
                part_number TEXT NOT NULL,
                qty INTEGER NOT NULL,
                before_total_add INTEGER,
                after_total_add INTEGER,
                before_total_used INTEGER,
                after_total_used INTEGER,
                before_total_quantity INTEGER,
                after_total_quantity INTEGER,
                performed_by TEXT,
                source TEXT,
                note TEXT
            )
            """
        )

        # Inventory history (workshop storage add/edit/delete)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS inventory_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,              -- ADD_PART / UPDATE_PART / DELETE_PART
                part_number TEXT NOT NULL,
                performed_by TEXT,
                note TEXT,
                before_state TEXT,                 -- JSON
                after_state TEXT                   -- JSON
            )
            """
        )

        # Workshop tables (formerly data/workshop.db)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS storage (
                part_number TEXT PRIMARY KEY,
                item_name TEXT,
                brand TEXT,
                model TEXT,
                specification TEXT,
                total_quantity INTEGER,
                total_used INTEGER,
                total_add INTEGER,
                part_type TEXT,
                usage TEXT
            )
            """
        )

        # Schema migration (best-effort): ensure new columns exist on older DBs.
        _ensure_text_columns(conn, "storage", ["brand", "model"])
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS task_reports (
                job_id TEXT PRIMARY KEY,
                date TEXT,
                time_start TEXT,
                time_end TEXT,
                task_type TEXT,
                problem TEXT,
                immediate_action TEXT,
                root_cause TEXT,
                preventive_action TEXT,
                spare_parts TEXT,
                reported_by TEXT,
                created_at TEXT
            )
            """
        )

        conn.commit()
    finally:
        conn.close()


def get_next_department_id(dept_code: str, item_prefix: str) -> str:
    """Return the next Department ID for a dept+prefix sequence.

    Format: 88-{15ME/15PE}-{PREFIX}-{NNN}
    Uses the DB as source of truth (more reliable than cached DataFrames).
    """
    dept_code = str(dept_code or "").strip().upper()
    item_prefix = str(item_prefix or "").strip().upper()
    if dept_code not in {"15ME", "15PE"}:
        return ""
    if not item_prefix:
        return ""

    if not ensure_main_database() or not MAIN_DB_FILE.exists():
        return f"88-{dept_code}-{item_prefix}-001"

    # LIKE prefix: escape % and _ so they don't act as wildcards.
    like_prefix = f"88-{dept_code}-{item_prefix}-"
    like_esc = like_prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    conn = _connect_main_db()
    try:
        if not _table_exists(conn, ASSET_TABLE):
            return f"88-{dept_code}-{item_prefix}-001"

        cur = conn.cursor()
        cur.execute(
            f'SELECT "Department ID" FROM "{ASSET_TABLE}" WHERE COALESCE("Department ID", "") LIKE ? ESCAPE "\\"',
            (like_esc + "%",),
        )
        rows = cur.fetchall() or []
    except Exception:
        rows = []
    finally:
        try:
            conn.close()
        except Exception:
            pass

    pattern = re.compile(
        rf"^88-{re.escape(dept_code)}-{re.escape(item_prefix)}-(\d{{3}})$",
        re.IGNORECASE,
    )
    max_n = 0
    for (val,) in rows:
        s = str(val or "").strip()
        m = pattern.match(s)
        if not m:
            continue
        try:
            max_n = max(max_n, int(m.group(1)))
        except Exception:
            continue

    return f"88-{dept_code}-{item_prefix}-{(max_n + 1):03d}"


def _migrate_legacy_into_main_db() -> None:
    """One-time best-effort migration into main_data.db.

    - If assets/breakdown tables are empty, import from CSV.
    - If log/workshop tables are empty, copy from legacy DBs (if present).
    """
    _ensure_tables_in_main_db()
    def _copy_table_if_dest_empty(dest_conn: sqlite3.Connection, *, src_db: Path, table: str) -> None:
        if not src_db.exists():
            return
        if not _table_exists(dest_conn, table):
            return
        if _table_row_count(dest_conn, table) > 0:
            return

        try:
            src_conn = sqlite3.connect(src_db)
        except Exception:
            return
        try:
            if not _table_exists(src_conn, table):
                return
            df_src = pd.read_sql_query(f'SELECT * FROM "{table}"', src_conn)
            if df_src is None or df_src.empty:
                return
            # Append into existing dest table
            df_src.to_sql(table, dest_conn, if_exists="append", index=False)
        except Exception:
            return
        finally:
            try:
                src_conn.close()
            except Exception:
                pass

    conn = _connect_main_db()
    try:
        # Legacy DBs -> main DB (best-effort, only if dest empty)
        _copy_table_if_dest_empty(conn, src_db=LEGACY_LOG_DB_FILE, table="asset_logs")
        _copy_table_if_dest_empty(conn, src_db=LEGACY_LOG_DB_FILE, table="stock_log")
        _copy_table_if_dest_empty(conn, src_db=LEGACY_WORKSHOP_DB_FILE, table="storage")
        _copy_table_if_dest_empty(conn, src_db=LEGACY_WORKSHOP_DB_FILE, table="task_reports")

        conn.commit()
    finally:
        conn.close()


def ensure_main_database() -> bool:
    try:
        _migrate_legacy_into_main_db()
        return True
    except Exception:
        return False


# ======================================
# LOGGING FUNCTIONS (SQLite)
# ======================================
def initialize_log_database() -> None:
    """Initialize SQLite database for logging asset operations"""
    try:
        ensure_main_database()
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

        try:
            persist_repo_changes([str(MAIN_DB_FILE)], reason=f"Asset log: {action}")
        except Exception:
            pass
        return True
    except Exception as e:
        st.error(f"❌ Error logging operation: {str(e)}")
        return False

def get_asset_logs(limit: int = 100) -> Optional[pd.DataFrame]:
    """Retrieve asset operation logs from SQLite database"""
    try:
        ensure_main_database()
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

@st.cache_data
def load_existing_data() -> Optional[pd.DataFrame]:
    try:
        if not ensure_main_database():
            return None
        if not MAIN_DB_FILE.exists():
            return None

        conn = _connect_main_db()
        try:
            if not _table_exists(conn, ASSET_TABLE):
                return None
            df = pd.read_sql_query(f'SELECT * FROM "{ASSET_TABLE}"', conn)
        finally:
            conn.close()

        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                df[col] = None

        # Keep derived fields consistent (Day Left / Status can become stale over time).
        # Avoid DataFrame truthiness ("truth value of a DataFrame is ambiguous").
        fixed_df = recompute_asset_derived_fields(df)
        if fixed_df is not None:
            df = fixed_df

        # Keep a consistent column order (required first, then any extras)
        ordered = [c for c in REQUIRED_COLUMNS if c in df.columns] + [c for c in df.columns if c not in REQUIRED_COLUMNS]
        df = df[ordered]

        df = df.reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = "Index"
        return df
    except Exception as e:
        st.error(f"❌ Error reading data file: {str(e)}")
        return None

def save_data(df: pd.DataFrame) -> bool:
    try:
        if not ensure_main_database():
            return False

        if df is None or not isinstance(df, pd.DataFrame):
            df = pd.DataFrame()
        df = df.dropna(how="all")
        df = df.reset_index(drop=True)

        # Ensure required columns exist before writing
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                df[col] = None

        conn = _connect_main_db()
        try:
            df.to_sql(ASSET_TABLE, conn, if_exists="replace", index=False)
            conn.commit()
        finally:
            conn.close()

        try:
            load_existing_data.clear()
        except Exception:
            pass

        try:
            persist_repo_changes([str(MAIN_DB_FILE)], reason="Update main_data.db")
        except Exception:
            pass
        return True
    except Exception as e:
        st.error(f"❌ Error saving data: {str(e)}")
        return False

def delete_asset_by_dept_id(department_id: str) -> bool:
    """Delete an asset by Department ID"""
    try:
        if not ensure_main_database():
            return False

        dep = str(department_id or "").strip()
        if not dep:
            return False

        conn = _connect_main_db()
        try:
            cur = conn.cursor()
            cur.execute(
                f'DELETE FROM "{ASSET_TABLE}" WHERE TRIM(COALESCE("Department ID", "")) = ?',
                (dep,),
            )
            deleted = int(cur.rowcount or 0)
            conn.commit()
        finally:
            conn.close()

        if deleted > 0:
            try:
                load_existing_data.clear()
            except Exception:
                pass

            try:
                persist_repo_changes([str(MAIN_DB_FILE)], reason=f"Delete asset {dep}")
            except Exception:
                pass
        return deleted > 0
    except Exception as e:
        st.error(f"❌ Error deleting asset: {str(e)}")
        return False


def load_breakdown_report() -> pd.DataFrame:
    """Load breakdown report from main_data.db (breakdown_report table)."""
    if not ensure_main_database() or not MAIN_DB_FILE.exists():
        return pd.DataFrame(columns=BREAKDOWN_COLUMNS)

    conn = _connect_main_db()
    try:
        if not _table_exists(conn, BREAKDOWN_TABLE):
            return pd.DataFrame(columns=BREAKDOWN_COLUMNS)
        df = pd.read_sql_query(f'SELECT * FROM "{BREAKDOWN_TABLE}"', conn)
    except Exception:
        return pd.DataFrame(columns=BREAKDOWN_COLUMNS)
    finally:
        conn.close()

    for col in BREAKDOWN_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[BREAKDOWN_COLUMNS]


def save_breakdown_report(df: pd.DataFrame) -> bool:
    """Replace the breakdown_report table with the provided dataframe."""
    try:
        if not ensure_main_database():
            return False
        if df is None or not isinstance(df, pd.DataFrame):
            df = pd.DataFrame()
        df = df.copy()
        for col in BREAKDOWN_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        df = df[BREAKDOWN_COLUMNS]

        conn = _connect_main_db()
        try:
            df.to_sql(BREAKDOWN_TABLE, conn, if_exists="replace", index=False)
            conn.commit()
        finally:
            conn.close()

        try:
            persist_repo_changes([str(MAIN_DB_FILE)], reason="Update breakdown report")
        except Exception:
            pass
        return True
    except Exception:
        return False

def check_duplicate(asset_number: str, existing_df: Optional[pd.DataFrame]) -> bool:
    if existing_df is None or existing_df.empty or not asset_number:
        return False
    asset_match = existing_df["Asset Number"].astype(str).str.strip() == str(asset_number).strip()
    return not existing_df[asset_match].empty

def calculate_due_date(start_date: date, maintenance_frequency: str) -> Optional[date]:
    """Calculate due date from Start Date + Maintenance Frequency.

    Notes:
    - Accepts mixed casing from DB (e.g. "YEARLY"), because rows are normalized to uppercase.
    - Returns None for unknown/blank/"None" frequencies.
    """
    if not start_date or not maintenance_frequency:
        return None

    # Best-effort normalize date input.
    if not isinstance(start_date, (date, datetime)):
        start_date = _safe_parse_date_any(start_date)  # type: ignore[assignment]
        if start_date is None:
            return None
    if isinstance(start_date, datetime):
        start_date = start_date.date()

    freq_key = str(maintenance_frequency or "").strip().casefold()
    if not freq_key or freq_key in {"none", "n/a", "na"}:
        return None

    frequency_days = {
        "weekly": 7,
        "biweekly": 14,
        "bi-weekly": 14,
        "monthly": 30,
        "quarterly": 90,
        "half-yearly": 182,
        "half yearly": 182,
        "halfyearly": 182,
        "yearly": 365,
        "annual": 365,
        "annually": 365,
    }
    days_to_add = frequency_days.get(freq_key)
    if days_to_add is None:
        return None

    return start_date + timedelta(days=int(days_to_add))

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
    # Keep rules consistent with Asset Editor:
    # - Day Left <= 0 -> Expired
    # - Day Left < 7  -> Expired Soon
    if days_left <= 0:
        return "Expired"
    elif days_left < 7:
        return "Expired Soon"
    else:
        return "Good"


def _safe_parse_date_any(value) -> Optional[date]:
    """Parse various DB / dataframe date representations into a `date`.

    Accepts `date`, `datetime`, strings (YYYY-MM-DD, etc). Returns None if invalid.
    """
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    s = str(value).strip()
    if not s:
        return None

    try:
        ts = pd.to_datetime(s, errors="coerce")
    except Exception:
        return None
    if ts is None or pd.isna(ts):
        return None
    try:
        return ts.date()
    except Exception:
        return None


def recompute_asset_derived_fields(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Recompute best-effort derived fields: Due Date (when missing), Day Left, Status.

    This fixes stale rows where Status/Day Left no longer match the rules.
    The rules mirror `pages/2_AssetEditor.py`:
    1) Functional Location == Obsolete -> Status = Obsolete
    2) Day Left <= 0 -> Expired
    3) Day Left < 7 -> Expired Soon
    4) Functional Location == 1006-10PE -> Good
    5) Functional Location other than 1006-10PE -> Idle

    Notes:
    - If Due Date is missing and Require Calibration != Yes, we derive Due Date from Start Date + Maintenance Frequency.
    - If Status has a custom value (e.g. NG), we won't overwrite it unless the computed status is Expired/Expired Soon/Obsolete.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df

    out = df.copy()

    # Ensure columns exist
    for col in ["Due Date", "Start Date", "Maintenance Frequency", "Day Left", "Status", "Functional Location", "Require Calibration"]:
        if col not in out.columns:
            out[col] = ""

    for idx, row in out.iterrows():
        func_loc = str(row.get("Functional Location", "") or "").strip()
        existing_status = str(row.get("Status", "") or "").strip()
        existing_status_norm = existing_status.casefold()

        # Parse / derive due date
        due_raw = row.get("Due Date", None)
        due_dt = _safe_parse_date_any(due_raw)

        req_cal_raw = str(row.get("Require Calibration", "") or "").strip().casefold()
        calib_required = req_cal_raw in {"yes", "y", "true", "1"}

        if due_dt is None and not calib_required:
            start_dt = _safe_parse_date_any(row.get("Start Date", None))
            freq = str(row.get("Maintenance Frequency", "") or "").strip()
            try:
                due_dt = calculate_due_date(start_dt, freq) if start_dt else None
            except Exception:
                due_dt = None

            # Only backfill Due Date if it was blank/missing
            if due_dt is not None and not str(due_raw or "").strip():
                try:
                    out.at[idx, "Due Date"] = due_dt.strftime("%Y-%m-%d")
                except Exception:
                    pass

        # Day Left
        days_left = None
        try:
            days_left = calculate_days_left(due_dt) if due_dt else None
        except Exception:
            days_left = None
        if days_left is not None:
            out.at[idx, "Day Left"] = days_left

        # Status (computed)
        # Apply Expired/Expired Soon when Day Left is known; otherwise fall back to location-based Good/Idle.
        computed_status = None
        if func_loc == "Obsolete":
            computed_status = "Obsolete"
        elif days_left is not None and days_left <= 0:
            computed_status = "Expired"
        elif days_left is not None and days_left < 7:
            computed_status = "Expired Soon"
        elif func_loc == "1006-10PE":
            computed_status = "Good"
        elif func_loc:
            computed_status = "Idle"

        # Only overwrite if the existing status is one of our auto statuses,
        # OR if the computed status is a high-priority state.
        auto_statuses = {"", "good", "idle", "expired", "expired soon", "obsolete"}
        high_priority = {"Expired", "Expired Soon", "Obsolete"}
        if computed_status:
            if existing_status_norm in auto_statuses or computed_status in high_priority:
                out.at[idx, "Status"] = computed_status

    return out

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
        return {
            "ok": False,
            "error": "Unsupported regdata.db schema (no matching table/columns found).",
            "user_id": "",
            "display_name": "",
            "level_name": "",
            "level_rank": 0,
        }

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
            identifier = st.text_input(
                "UserID / QRID",
                key="login_identifier",
                placeholder="Scan QR (QRID) or type UserID...",
            )
            if st.button("Login", type="primary", use_container_width=True):
                res = lookup_regdata_user(
                    identifier,
                    allow_userid=True,
                    allow_qr=True,
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
    Creates stock_log table inside data/main_data.db
    (shared with asset logs and other app tables).
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


def initialize_inventory_history_database() -> None:
    """Creates inventory_history table inside data/main_data.db."""
    ensure_data_directory()
    conn = sqlite3.connect(MAIN_DB_FILE)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS inventory_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,
                part_number TEXT NOT NULL,
                performed_by TEXT,
                note TEXT,
                before_state TEXT,
                after_state TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def log_inventory_history(
    *,
    action: str,
    part_number: str,
    performed_by: str = "",
    note: str = "",
    before_state: dict | None = None,
    after_state: dict | None = None,
) -> None:
    """Append a row to inventory_history (best-effort).

    `before_state` and `after_state` are stored as JSON strings.
    """
    try:
        initialize_inventory_history_database()

        def _to_json(value: dict | None) -> str:
            if not value:
                return ""
            return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)

        conn = sqlite3.connect(MAIN_DB_FILE)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO inventory_history (
                    timestamp, action, part_number,
                    performed_by, note,
                    before_state, after_state
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    str(action or ""),
                    str(part_number or ""),
                    str(performed_by or ""),
                    str(note or ""),
                    _to_json(before_state),
                    _to_json(after_state),
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        # Never break the app due to logging failures.
        return