import streamlit as st
import pandas as pd
from pathlib import Path
import sqlite3
from io import BytesIO
import re
from datetime import datetime, date, time, timedelta
from utils import (
    ensure_data_directory,
    initialize_stock_log_database,
    log_stock_operation,
    initialize_inventory_history_database,
    log_inventory_history,
    persist_repo_changes,
    now_sg,
    today_sg,
    format_ts_sg,
    require_login,
    load_existing_data,
    TASK_COLUMNS,
    load_task_table,
    save_task_table,
    list_regdata_display_names,
    show_system_error,
    show_user_error,
    render_role_navigation,
)

st.set_page_config(page_title="Task Update", page_icon="🔧", layout="wide")
auth = require_login(min_level_rank=2)
render_role_navigation(auth)


def _performed_by_label() -> str:
    name = str(auth.get("name", "") or "").strip()
    user_id = str(auth.get("user_id", "") or "").strip()
    return name or user_id or "System"


def _current_user_id() -> str:
    user_id = str(auth.get("user_id", "") or "").strip()
    return user_id or _performed_by_label()


def _current_user_display_name() -> str:
    """Display name from regdata (fallback to user_id)."""
    return _performed_by_label()


def _current_level_rank() -> int:
    try:
        return int(auth.get("level_rank") or 0)
    except Exception:
        return 0


def _normalize_job_type(value: str) -> str:
    v = str(value or "").strip()
    if not v:
        return ""
    key = v.casefold()
    if key in {"other", "others"}:
        return "General"
    if key == "general":
        return "General"
    if key == "breakdown":
        return "Breakdown"
    if key == "maintenance":
        return "Maintenance"
    return v


def _normalize_job_status(value: object) -> str:
    s = str(value or "").strip()
    if not s or s.casefold() in {"none", "nan"}:
        return "In Progress"
    key = s.casefold()
    if key in {"pending"}:
        return "Pending"
    if key in {"open", "in progress", "in_progress", "inprogress"}:
        return "In Progress"
    if key in {"completed", "complete"}:
        return "Completed"
    if key in {"close", "closed"}:
        # Legacy value; keep for backward-compat but UI should not create it.
        return "Completed"
    return s


def _normalize_approval_status(value: object) -> str:
    s = str(value or "").strip()
    if not s or s.casefold() in {"none", "nan"}:
        return "In Review"
    key = s.casefold()
    # Legacy mappings
    if key == "pending":
        return "In Review"
    if key == "approved":
        return "Approved"
    if key in {"close", "closed"}:
        return "Approved"
    if key in {"inprogress", "in progress", "in_progress"}:
        return "Not Submitted"
    if key == "completed":
        return "In Review"
    # Current states
    if key in {"not submitted", "not_submitted", "notsubmitted"}:
        return "Not Submitted"
    if key in {"waiting approval", "waiting_approval", "waitingapproval", "in review", "in_review", "inreview"}:
        return "In Review"
    if key == "rejected":
        return "Rejected"
    return s


def _unique_job_ids(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty or "Job ID" not in df.columns:
        return []
    ids = df["Job ID"].astype(str).fillna("").map(lambda x: str(x).strip())
    ids = [x for x in ids.tolist() if x]
    return sorted(set(ids))


def _parse_ts_sg(value: object) -> datetime | None:
    s = str(value or "").strip()
    if not s or s.casefold() in {"none", "nan"}:
        return None
    try:
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        # Treat stored timestamps as Singapore local time
        tz = now_sg().tzinfo
        return dt.replace(tzinfo=tz) if tz is not None else dt
    except Exception:
        return None


def _to_int_minutes(value: object, *, default: int = 0) -> int:
    if value is None:
        return default
    try:
        if isinstance(value, (int, float)):
            return int(value)
    except Exception:
        pass
    s = str(value or "").strip()
    if not s or s.casefold() in {"none", "nan"}:
        return default
    m = re.search(r"-?\d+", s)
    if not m:
        return default
    try:
        return int(m.group(0))
    except Exception:
        return default
APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = APP_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "main_data.db"

# Report images (single image per Job ID)
REPORT_IMAGES_DIR = DATA_DIR / "report_images"
REPORT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
_REPORT_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _save_report_image(job_id: str, uploaded_file) -> Path:
    """Save/replace the single report image for the given Job ID.

    Stores under data/report_images with filename exactly Job ID + original extension.
    """
    jid = str(job_id or "").strip()
    if not jid:
        raise ValueError("Missing Job ID")
    if uploaded_file is None:
        raise ValueError("No file uploaded")

    ext = Path(getattr(uploaded_file, "name", "") or "").suffix.lower()
    if ext not in _REPORT_IMG_EXTS:
        raise ValueError(f"Unsupported image type: {ext or '(no extension)'}")

    # Remove any existing image for this Job ID (regardless of extension)
    try:
        for old in REPORT_IMAGES_DIR.glob(f"{jid}.*"):
            if old.is_file():
                old.unlink(missing_ok=True)
    except Exception:
        pass

    out_path = REPORT_IMAGES_DIR / f"{jid}{ext}"
    with open(out_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        persist_repo_changes([str(REPORT_IMAGES_DIR)], reason=f"Update report image: {jid}")
    except Exception:
        pass
    return out_path

# Task structure (stored in main_data.db -> task)
COLUMNS = list(TASK_COLUMNS)

# Usage reference only (free text is allowed)
USAGE_OPTIONS = ["Equipment", "Machine", "Jig", "Fixture", "Tester"]

MAINTENANCE_FREQUENCY_OPTIONS = [
    "NONE",
    "Twice a Day",
    "Daily",
    "Every Two Days",
    "Weekly",
    "BiWeekly",
    "Fortnight",
    "Monthly",
    "Quaterly",
    "Biannual",
    "Yearly",
]


@st.cache_data(show_spinner=False)
def _get_machine_catalog() -> tuple[list[str], dict[str, str]]:
    df = load_existing_data()
    if df is None or df.empty:
        df = pd.DataFrame()

    id_col = "Department ID" if "Department ID" in df.columns else None
    name_col = "Description of Asset" if "Description of Asset" in df.columns else None
    machine_map: dict[str, str] = {}
    if id_col:
        ids = df[id_col].astype(str).fillna("").map(lambda s: str(s).strip())
        names = df[name_col].astype(str).fillna("").map(lambda s: str(s).strip()) if name_col else pd.Series([""] * len(df))

        for mid, mname in zip(ids.tolist(), names.tolist()):
            if not mid:
                continue
            if mid in machine_map:
                continue
            machine_map[mid] = mname

    # Also remember manual Machine IDs from past task reports.
    # This supports the "Machine ID not in list" workflow without touching the asset database.
    try:
        tdf = load_task_table()
        if tdf is not None and not tdf.empty:
            flag_col = "Machine ID not in list" if "Machine ID not in list" in tdf.columns else None
            mid_col = "Machine ID" if "Machine ID" in tdf.columns else None
            name_col = "Machine/Equipment" if "Machine/Equipment" in tdf.columns else None
            if mid_col and name_col and flag_col:
                tmp = tdf[[flag_col, mid_col, name_col]].copy()
                tmp[flag_col] = tmp[flag_col].astype(str).str.strip().str.casefold()
                tmp[mid_col] = tmp[mid_col].astype(str).str.strip()
                tmp[name_col] = tmp[name_col].astype(str).str.strip()
                tmp = tmp[(tmp[flag_col].isin({"yes", "y", "true", "1"})) & (tmp[mid_col] != "")]

                for mid, mname in zip(tmp[mid_col].tolist(), tmp[name_col].tolist()):
                    if not mid:
                        continue
                    if mid not in machine_map:
                        machine_map[mid] = mname
    except Exception:
        pass

    options = sorted(machine_map.keys(), key=lambda s: s.casefold())
    return (options, machine_map)


def _fmt_date_ddmmyy(value) -> str:
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        return str(value or "").strip()
    return dt.strftime("%d/%m/%y")


def _fmt_datetime_ddmmyy_hhmm(value) -> str:
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        return str(value or "").strip()
    return dt.strftime("%d/%m/%y %H:%M")


def build_task_report_pdf(row: dict) -> bytes:
    """Build a single Task Report PDF matching the requested layout.

    Notes:
    - Does not require user-filled Attend/Verify fields; renders empty boxes in the PDF.
    """
    try:
        import importlib
        canvas = importlib.import_module("reportlab.pdfgen.canvas")
        pagesizes = importlib.import_module("reportlab.lib.pagesizes")
        A4 = getattr(pagesizes, "A4")
    except Exception as e:
        raise RuntimeError(
            "PDF export is unavailable because ReportLab could not be imported. "
            f"({type(e).__name__}: {e})\n\n"
            "Fix: ensure `reportlab` is listed in requirements.txt and redeploy/reboot the Streamlit app. "
            "If it is already listed, the error above indicates the real root cause (e.g., build/import failure)."
        ) from e

    page_w, page_h = A4
    margin_x = 48
    margin_y = 48

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)

    def new_page():
        c.setFont("Helvetica-Bold", 18)
        c.drawCentredString(page_w / 2, page_h - margin_y, "Task Report")
        c.setFont("Helvetica", 11)

    def draw_wrapped(text: str, x: float, y: float, max_width: float, line_height: float = 14) -> float:
        """Draw wrapped text; returns new y."""
        text = str(text or "")
        words = text.replace("\r", "").split()
        if not words:
            c.drawString(x, y, "")
            return y - line_height

        line = ""
        for w in words:
            candidate = (line + " " + w).strip()
            if c.stringWidth(candidate, "Helvetica", 11) <= max_width:
                line = candidate
                continue
            # emit current line
            if y <= margin_y + 40:
                c.showPage()
                new_page()
                y = page_h - margin_y - 28
            c.drawString(x, y, line)
            y -= line_height
            line = w

        if line:
            if y <= margin_y + 40:
                c.showPage()
                new_page()
                y = page_h - margin_y - 28
            c.drawString(x, y, line)
            y -= line_height

        return y

    # Header
    new_page()
    y = page_h - margin_y - 36

    # Two columns
    col_gap = 24
    col_w = (page_w - (margin_x * 2) - col_gap) / 2
    x1 = margin_x
    x2 = margin_x + col_w + col_gap

    left_lines = [
        ("Date", _fmt_date_ddmmyy(row.get("Date"))),
        ("Job Type", _normalize_job_type(str(row.get("Job Type", "") or "").strip())),
        ("Shift", str(row.get("Shift", "") or "").strip()),
        ("Machine", str(row.get("Machine/Equipment", "") or "").strip()),
        ("Date/Time Start", _fmt_datetime_ddmmyy_hhmm(row.get("Date_Time Start"))),
    ]
    right_lines = [
        ("Job ID", str(row.get("Job ID", "") or "").strip()),
        ("Severity", str(row.get("Severity", "") or "").strip()),
        ("Location", str(row.get("Location", "") or "").strip()),
        ("Machine ID", str(row.get("Machine ID", "") or "").strip()),
        ("Date/Time End", _fmt_datetime_ddmmyy_hhmm(row.get("Date_Time End"))),
    ]

    line_h = 16
    y_left = y
    y_right = y
    for label, val in left_lines:
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x1, y_left, f"{label}:")
        c.setFont("Helvetica", 10)
        c.drawString(x1 + 110, y_left, str(val or ""))
        y_left -= line_h

    for label, val in right_lines:
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x2, y_right, f"{label}:")
        c.setFont("Helvetica", 10)
        c.drawString(x2 + 110, y_right, str(val or ""))
        y_right -= line_h

    y = min(y_left, y_right) - 8

    # Narrative blocks
    blocks = [
        ("Problem / Task Description", row.get("Problem_Task_Job Description", "")),
        ("Immediate Action / Action", row.get("Immediate Action_Action", "")),
        ("Root Cause", row.get("Root Cause", "")),
        ("Preventive Action", row.get("Preventive Action", "")),
        ("Spare Part Use & Quantity", row.get("Spare Parts Used", "")),
    ]

    for title, body in blocks:
        c.setFont("Helvetica-Bold", 10)
        if y <= margin_y + 40:
            c.showPage()
            new_page()
            y = page_h - margin_y - 28
        c.drawString(margin_x, y, title)
        y -= 14
        c.setFont("Helvetica", 10)
        y = draw_wrapped(body, margin_x, y, max_width=page_w - (margin_x * 2), line_height=14)
        y -= 6

    # Footer/signatures (align right):
    # Report by | Attend By | Verify
    # <name>   | <name or empty box> | <name or empty box>
    reported_by = str(row.get("Reported by", "") or "").strip()
    attend_by = str(
        row.get("Attend By", "")
        or row.get("Attend by", "")
        or row.get("Attend", "")
        or ""
    ).strip()
    verify_by = str(
        row.get("Verify By", "")
        or row.get("Verify by", "")
        or row.get("Verify", "")
        or ""
    ).strip()

    footer_base_y = margin_y
    # Ensure there is room for the footer; otherwise create a new page.
    if y <= footer_base_y + 70:
        c.showPage()
        new_page()
        y = page_h - margin_y - 28

    table_w = 450
    cell_w = table_w / 3
    header_y = footer_base_y + 28
    box_y = footer_base_y + 6
    box_h = 18
    x0 = page_w - margin_x - table_w

    c.setFont("Helvetica-Bold", 10)
    headers = ["Report by", "Attend By", "Verify"]
    for i, h in enumerate(headers):
        c.drawCentredString(x0 + (cell_w * i) + (cell_w / 2), header_y, h)

    # Draw boxes for values row
    c.setLineWidth(1)
    c.rect(x0, box_y, table_w, box_h, stroke=1, fill=0)
    c.line(x0 + cell_w, box_y, x0 + cell_w, box_y + box_h)
    c.line(x0 + (2 * cell_w), box_y, x0 + (2 * cell_w), box_y + box_h)

    c.setFont("Helvetica", 10)
    values = [reported_by, attend_by, verify_by]
    for i, v in enumerate(values):
        c.drawString(x0 + (cell_w * i) + 6, box_y + 5, str(v or ""))

    c.save()
    return buf.getvalue()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Storage Table (v2)
    # - total_quantity: current available stock
    # - total_in: cumulative stock-in quantity
    # - total_out: cumulative stock-out quantity
    # Rule (normalized): total_in = total_out + total_quantity
    c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='storage' LIMIT 1")
    storage_exists = c.fetchone() is not None

    def _create_storage_table(table_name: str) -> None:
        c.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                part_number TEXT PRIMARY KEY,
                part_type TEXT,
                item_name TEXT,
                brand TEXT,
                model TEXT,
                specification TEXT,
                preferred_supplier TEXT,
                item_cost_rm REAL,
                total_quantity INTEGER,
                usage_area TEXT,
                total_in INTEGER,
                total_out INTEGER
            )
            """
        )

    if not storage_exists:
        _create_storage_table("storage")
    else:
        # Decide whether a rebuild is needed (legacy/mismatched columns).
        c.execute("PRAGMA table_info(storage)")
        existing_cols = [r[1] for r in (c.fetchall() or [])]
        desired_names = [
            "part_number",
            "part_type",
            "item_name",
            "brand",
            "model",
            "specification",
            "preferred_supplier",
            "item_cost_rm",
            "total_quantity",
            "usage_area",
            "total_in",
            "total_out",
        ]
        needs_rebuild = set(existing_cols) != set(desired_names)

        if not needs_rebuild:
            c.execute(
                """
                UPDATE storage
                SET total_quantity = COALESCE(total_quantity, 0),
                    total_out = COALESCE(total_out, 0),
                    total_in = COALESCE(total_out, 0) + COALESCE(total_quantity, 0)
                WHERE total_quantity IS NULL
                   OR total_out IS NULL
                   OR total_in IS NULL
                   OR total_in != (COALESCE(total_out, 0) + COALESCE(total_quantity, 0))
                """
            )
            # Continue; init_db() also ensures other tables below.

        if not needs_rebuild:
            # No storage rebuild needed.
            pass
        else:
            try:
                df_old = pd.read_sql("SELECT * FROM storage", conn)
            except Exception:
                df_old = pd.DataFrame()

            def _to_int_series(s: pd.Series) -> pd.Series:
                return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)

            out = pd.DataFrame()
            out["part_number"] = df_old.get("part_number", "").astype(str)
            out["part_type"] = df_old.get("part_type", "").astype(str)
            out["item_name"] = df_old.get("item_name", "").astype(str)
            out["brand"] = df_old.get("brand", "").astype(str)
            out["model"] = df_old.get("model", "").astype(str)
            out["specification"] = df_old.get("specification", "").astype(str)
            out["preferred_supplier"] = df_old.get("preferred_supplier", "").astype(str)
            out["item_cost_rm"] = pd.to_numeric(df_old.get("item_cost_rm", 0), errors="coerce").fillna(0.0)
            if "usage_area" in df_old.columns:
                out["usage_area"] = df_old.get("usage_area", "").astype(str)
            else:
                out["usage_area"] = df_old.get("usage", "").astype(str)

            has_legacy_totals = ("total_add" in df_old.columns) or ("total_used" in df_old.columns)
            if has_legacy_totals:
                legacy_add = _to_int_series(df_old.get("total_add", 0)).clip(lower=0)
                legacy_used = _to_int_series(df_old.get("total_used", 0)).clip(lower=0)
                out["total_quantity"] = legacy_add
                out["total_out"] = legacy_used
                out["total_in"] = (legacy_add + legacy_used).astype(int)
            else:
                avail = _to_int_series(df_old.get("total_quantity", 0)).clip(lower=0)
                tout = _to_int_series(df_old.get("total_out", 0)).clip(lower=0)
                out["total_quantity"] = avail
                out["total_out"] = tout
                out["total_in"] = (avail + tout).astype(int)

            # Rebuild to the v2 schema (drops legacy columns like total_add/total_used/usage).
            c.execute("DROP TABLE IF EXISTS storage__new")
            _create_storage_table("storage__new")
            if not out.empty:
                out[
                    [
                        "part_number",
                        "part_type",
                        "item_name",
                        "brand",
                        "model",
                        "specification",
                        "preferred_supplier",
                        "item_cost_rm",
                        "total_quantity",
                        "usage_area",
                        "total_in",
                        "total_out",
                    ]
                ].to_sql("storage__new", conn, if_exists="append", index=False)
            c.execute("DROP TABLE IF EXISTS storage")
            c.execute("ALTER TABLE storage__new RENAME TO storage")

    # Task Report Table
    c.execute("""
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
    """)

    conn.commit()
    conn.close()


PART_TYPE_CONFIG = {
    "Electrical": {"type_code": "ELEC", "pn_prefix": "PN1"},
    "Mechanical": {"type_code": "MECH", "pn_prefix": "PN2"},
    "Pneumatic": {"type_code": "PNE", "pn_prefix": "PN3"},
    "Hydraulic": {"type_code": "HYD", "pn_prefix": "PN4"},
    "General Item": {"type_code": "GEN", "pn_prefix": "PN5"},
}

TYPE_CODE_TO_PN_PREFIX = {cfg["type_code"]: cfg["pn_prefix"] for cfg in PART_TYPE_CONFIG.values()}


def part_type_to_code(part_type_value: str) -> str:
    if part_type_value is None:
        return ""
    v = str(part_type_value).strip()
    if v in PART_TYPE_CONFIG:
        return PART_TYPE_CONFIG[v]["type_code"]
    codes = {cfg["type_code"] for cfg in PART_TYPE_CONFIG.values()}
    return v if v in codes else v


def get_storage():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM storage", conn)
    conn.close()

    # Ensure columns exist (for safety)
    for col in [
        "usage_area",
        "brand",
        "model",
        "preferred_supplier",
        "item_cost_rm",
        "total_quantity",
        "total_in",
        "total_out",
    ]:
        if col not in df.columns:
            df[col] = "" if col in {"usage_area", "brand", "model", "preferred_supplier"} else 0

    df["total_quantity"] = pd.to_numeric(df["total_quantity"], errors="coerce").fillna(0).astype(int).clip(lower=0)
    df["total_out"] = pd.to_numeric(df["total_out"], errors="coerce").fillna(0).astype(int).clip(lower=0)
    df["total_in"] = (df["total_out"] + df["total_quantity"]).astype(int)

    # Backward-compatibility aliases for older code paths in this page
    df["usage"] = df.get("usage_area", "")
    df["total_add"] = df.get("total_quantity", 0)
    df["total_used"] = df.get("total_out", 0)
    return df


def _fetch_storage_row(conn: sqlite3.Connection, part_number: str) -> dict | None:
    pn = str(part_number or "").strip()
    if not pn:
        return None
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM storage WHERE part_number = ? LIMIT 1", (pn,))
        row = cur.fetchone()
        if not row:
            return None
        cols = [d[0] for d in (cur.description or [])]
        return {cols[i]: row[i] for i in range(len(cols))}
    except Exception:
        return None


def _diff_state_for_log(
    before_state: dict | None,
    after_state: dict | None,
    *,
    include_keys: list[str] | None = None,
) -> tuple[dict | None, dict | None]:
    """Return (before, after) dicts containing only changed fields."""
    if not before_state and not after_state:
        return None, None

    before_state = dict(before_state or {})
    after_state = dict(after_state or {})
    keys = set(include_keys or (set(before_state.keys()) | set(after_state.keys())))

    def _norm(v: object) -> object:
        if v is None:
            return ""
        if isinstance(v, (int, float)):
            return v
        s = str(v).strip()
        try:
            if s != "" and s.replace(".", "", 1).isdigit():
                if "." in s:
                    return round(float(s), 2)
                return int(s)
        except Exception:
            pass
        return s

    changed: list[str] = []
    for k in keys:
        if _norm(before_state.get(k)) != _norm(after_state.get(k)):
            changed.append(k)

    if not changed:
        return None, None
    return ({k: before_state.get(k, "") for k in changed}, {k: after_state.get(k, "") for k in changed})


def save_part(
    part_number: str,
    item_name: str,
    specification: str,
    total_add: int,
    part_type: str,
    usage: str,
    brand: str = "",
    model: str = "",
    performed_by: str = "",
    note: str = "",
):
    """
    New part:
      total_used = 0
      total_quantity = total_used + total_add
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    total_quantity = int(total_add)
    total_out = 0
    total_in = total_out + max(total_quantity, 0)

    c.execute(
        """
        INSERT INTO storage (
            part_number,
            part_type,
            item_name,
            brand,
            model,
            specification,
            preferred_supplier,
            item_cost_rm,
            total_quantity,
            usage_area,
            total_in,
            total_out
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            part_number,
            str(part_type or "").strip(),
            item_name,
            str(brand or "").strip(),
            str(model or "").strip(),
            specification,
            "",  # preferred_supplier
            0.0,  # item_cost_rm
            int(max(total_quantity, 0)),
            str(usage or "").strip(),
            int(max(total_in, 0)),
            int(max(total_out, 0)),
        ),
    )
    conn.commit()

    # Inventory history (ADD)
    after_state = _fetch_storage_row(conn, part_number)
    # ADD: keep full after_state so audit shows what was created.
    log_inventory_history(
        action="ADD_PART",
        part_number=part_number,
        performed_by=performed_by,
        note=note,
        before_state=None,
        after_state=after_state,
    )

    try:
        persist_repo_changes([str(DB_PATH)], reason=f"Inventory ADD_PART {part_number}")
    except Exception:
        pass
    conn.close()


def update_storage_row(
    part_number: str,
    item_name: str,
    specification: str,
    total_add: int,
    total_used: int,
    part_type: str,
    usage: str,
    brand: str = "",
    model: str = "",
    performed_by: str = "",
    note: str = "",
):
    """
    Always re-calc total_quantity = total_used + total_add.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    before_state = _fetch_storage_row(conn, part_number)
    total_quantity = int(total_add)
    total_out = int(total_used)
    total_in = total_out + max(total_quantity, 0)

    c.execute(
        """
        UPDATE storage
        SET item_name = ?,
            brand = ?,
            model = ?,
            specification = ?,
            part_type = ?,
            preferred_supplier = ?,
            item_cost_rm = ?,
            total_quantity = ?,
            usage_area = ?,
            total_in = ?,
            total_out = ?
        WHERE part_number = ?
        """,
        (
            item_name,
            str(brand or "").strip(),
            str(model or "").strip(),
            specification,
            part_type,
            "",  # preferred_supplier
            0.0,  # item_cost_rm
            int(max(total_quantity, 0)),
            str(usage or "").strip(),
            int(max(total_in, 0)),
            int(max(total_out, 0)),
            part_number,
        ),
    )
    conn.commit()

    # Inventory history (UPDATE)
    after_state = _fetch_storage_row(conn, part_number)
    b_diff, a_diff = _diff_state_for_log(
        before_state,
        after_state,
        include_keys=[
            "part_number",
            "part_type",
            "item_name",
            "brand",
            "model",
            "specification",
            "preferred_supplier",
            "item_cost_rm",
            "usage_area",
            "total_quantity",
            "total_in",
            "total_out",
        ],
    )
    log_inventory_history(
        action="UPDATE_PART",
        part_number=part_number,
        performed_by=performed_by,
        note=note,
        before_state=b_diff,
        after_state=a_diff,
    )

    try:
        persist_repo_changes([str(DB_PATH)], reason=f"Inventory UPDATE_PART {part_number}")
    except Exception:
        pass
    conn.close()


def _get_storage_totals(conn: sqlite3.Connection, part_number: str) -> tuple[int, int, int]:
    """
    Helper required by stock_in_add / stock_out_adjust / stock_out_task
    Returns (total_in, total_out, total_quantity) for a part_number.
    """
    pn = str(part_number or "").strip()
    if not pn:
        raise ValueError("Part Number is required")

    cur = conn.cursor()
    cur.execute(
        """
        SELECT COALESCE(total_in, 0), COALESCE(total_out, 0), COALESCE(total_quantity, 0)
        FROM storage
        WHERE part_number = ?
        LIMIT 1
        """,
        (pn,),
    )
    row = cur.fetchone()
    if not row:
        raise ValueError(f"Part not found: {pn}")

    return int(row[0]), int(row[1]), int(row[2])


# --- FIX: keep old function names but make them consistent with the new rules/logging ---
def stock_in(part_number: str, qty_in: int):
    """
    Backward-compatible wrapper:
    IN increases total_add only (available stock).
    """
    stock_in_add(part_number, qty_in, performed_by="", note="")

def stock_out(part_number: str, qty_out: int):
    """
    Backward-compatible wrapper:
    OUT decreases total_add only (available stock).
    total_used must come from Task Report only.
    """
    stock_out_adjust(part_number, qty_out, performed_by="", note="")


def stock_in_add(part_number: str, qty_in: int, performed_by: str = "", note: str = ""):
    """
    IN (new quantity add): increases total_add only.
    total_quantity = total_used + total_add
    """
    qty_in = int(qty_in)
    if qty_in <= 0:
        raise ValueError("IN quantity must be > 0")

    conn = sqlite3.connect(DB_PATH)
    try:
        before_state = _fetch_storage_row(conn, part_number)
        before_in, before_out, before_qty = _get_storage_totals(conn, part_number)

        after_in = before_in + qty_in
        after_out = before_out
        after_qty = before_qty + qty_in

        cur = conn.cursor()
        cur.execute(
            "UPDATE storage SET total_in = ?, total_out = ?, total_quantity = ? WHERE part_number = ?",
            (after_in, after_out, after_qty, part_number),
        )
        conn.commit()

        after_state = _fetch_storage_row(conn, part_number)
        b_diff, a_diff = _diff_state_for_log(
            before_state,
            after_state,
            include_keys=["total_in", "total_out", "total_quantity"],
        )

        # Merge stock_log into inventory history
        log_inventory_history(
            action="IN_ADD",
            part_number=part_number,
            performed_by=performed_by,
            note=note,
            before_state=b_diff,
            after_state=a_diff,
        )

        log_stock_operation(
            action="IN_ADD",
            part_number=part_number,
            qty=qty_in,
            before_total_add=before_qty,
            after_total_add=after_qty,
            before_total_used=before_out,
            after_total_used=after_out,
            performed_by=performed_by,
            source="Stock IN/OUT",
            note=note,
        )

        try:
            persist_repo_changes([str(DB_PATH)], reason=f"Inventory IN_ADD {part_number}")
        except Exception:
            pass
    finally:
        conn.close()


def stock_out_adjust(part_number: str, qty_out: int, performed_by: str = "", note: str = ""):
    """
    OUT (available stock adjustment): decreases total_add only.
    total_used is NOT changed (total_used comes from task_report only)
    """
    qty_out = int(qty_out)
    if qty_out <= 0:
        raise ValueError("OUT quantity must be > 0")

    conn = sqlite3.connect(DB_PATH)
    try:
        before_state = _fetch_storage_row(conn, part_number)
        before_in, before_out, before_qty = _get_storage_totals(conn, part_number)
        if qty_out > before_qty:
            raise ValueError("Not enough available stock")

        after_qty = before_qty - qty_out
        after_out = before_out + qty_out
        after_in = before_in  # OUT does not change total_in

        cur = conn.cursor()
        cur.execute(
            "UPDATE storage SET total_in = ?, total_out = ?, total_quantity = ? WHERE part_number = ?",
            (after_in, after_out, after_qty, part_number),
        )
        conn.commit()

        after_state = _fetch_storage_row(conn, part_number)
        b_diff, a_diff = _diff_state_for_log(
            before_state,
            after_state,
            include_keys=["total_in", "total_out", "total_quantity"],
        )

        # Merge stock_log into inventory history
        log_inventory_history(
            action="OUT_ADJUST",
            part_number=part_number,
            performed_by=performed_by,
            note=note,
            before_state=b_diff,
            after_state=a_diff,
        )

        log_stock_operation(
            action="OUT_ADJUST",
            part_number=part_number,
            qty=qty_out,
            before_total_add=before_qty,
            after_total_add=after_qty,
            before_total_used=before_out,
            after_total_used=after_out,
            performed_by=performed_by,
            source="Stock IN/OUT",
            note=note,
        )

        try:
            persist_repo_changes([str(DB_PATH)], reason=f"Inventory OUT_ADJUST {part_number}")
        except Exception:
            pass
    finally:
        conn.close()


def stock_out_task(part_number: str, qty_used: int, performed_by: str = "", note: str = ""):
    """
    OUT from Task Report:
    - decreases total_add (available)
    - increases total_used (usage record)
    - keeps formula consistent
    """
    qty_used = int(qty_used)
    if qty_used <= 0:
        raise ValueError("Qty used must be > 0")

    conn = sqlite3.connect(DB_PATH)
    try:
        before_state = _fetch_storage_row(conn, part_number)
        before_in, before_out, before_qty = _get_storage_totals(conn, part_number)
        if qty_used > before_qty:
            raise ValueError("Not enough available stock")

        after_qty = before_qty - qty_used
        after_out = before_out + qty_used
        after_in = before_in  # OUT does not change total_in

        cur = conn.cursor()
        cur.execute(
            "UPDATE storage SET total_in = ?, total_out = ?, total_quantity = ? WHERE part_number = ?",
            (after_in, after_out, after_qty, part_number),
        )
        conn.commit()

        after_state = _fetch_storage_row(conn, part_number)
        b_diff, a_diff = _diff_state_for_log(
            before_state,
            after_state,
            include_keys=["total_in", "total_out", "total_quantity"],
        )

        # Merge stock_log into inventory history
        log_inventory_history(
            action="OUT_TASK",
            part_number=part_number,
            performed_by=performed_by,
            note=note,
            before_state=b_diff,
            after_state=a_diff,
        )

        log_stock_operation(
            action="OUT_TASK",
            part_number=part_number,
            qty=qty_used,
            before_total_add=before_qty,
            after_total_add=after_qty,
            before_total_used=before_out,
            after_total_used=after_out,
            performed_by=performed_by,
            source="Task Report",
            note=note,
        )

        try:
            persist_repo_changes([str(DB_PATH)], reason=f"Inventory OUT_TASK {part_number}")
        except Exception:
            pass
    finally:
        conn.close()


def delete_part(part_number: str, performed_by: str = "", note: str = ""):
    pn = str(part_number or "").strip()
    if not pn:
        raise ValueError("Part Number is required")

    conn = sqlite3.connect(DB_PATH)
    try:
        before_state = _fetch_storage_row(conn, pn)
        c = conn.cursor()
        c.execute(
            """
            SELECT COALESCE(total_in, 0), COALESCE(total_out, 0), COALESCE(total_quantity, 0)
            FROM storage
            WHERE part_number = ?
            LIMIT 1
            """,
            (pn,),
        )
        row = c.fetchone()
        if not row:
            raise ValueError(f"Part not found: {pn}")

        before_in = int(row[0])
        before_out = int(row[1])
        before_qty = int(row[2])

        c.execute("DELETE FROM storage WHERE part_number = ?", (pn,))
        if c.rowcount <= 0:
            raise ValueError(f"Part not found: {pn}")

        conn.commit()

        log_inventory_history(
            action="DELETE_PART",
            part_number=pn,
            performed_by=performed_by,
            note=note,
            before_state=(
                {
                    k: (before_state or {}).get(k, "")
                    for k in [
                        "part_number",
                        "part_type",
                        "item_name",
                        "brand",
                        "model",
                        "specification",
                        "preferred_supplier",
                        "item_cost_rm",
                        "usage_area",
                        "total_quantity",
                        "total_in",
                        "total_out",
                    ]
                }
                if before_state
                else None
            ),
            after_state=None,
        )

        log_stock_operation(
            action="DELETE",
            part_number=pn,
            qty=0,
            before_total_add=before_qty,
            after_total_add=0,
            before_total_used=before_out,
            after_total_used=0,
            performed_by=performed_by,
            source="Stock IN/OUT",
            note=note,
        )

        try:
            persist_repo_changes([str(DB_PATH)], reason=f"Inventory DELETE_PART {pn}")
        except Exception:
            pass
    finally:
        conn.close()


def load_breakdown_data() -> pd.DataFrame:
    try:
        df = load_task_table()
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[COLUMNS]
    except Exception as e:
        show_system_error("Failed to load task table from database.", e, context="TaskReport.load_breakdown_data")
        return pd.DataFrame(columns=COLUMNS)


def _save_breakdown_data(df: pd.DataFrame) -> bool:
    try:
        out = df.copy() if df is not None else pd.DataFrame(columns=COLUMNS)
        for col in COLUMNS:
            if col not in out.columns:
                out[col] = ""
        out = out[COLUMNS]
        ok = bool(save_task_table(out))
        if ok:
            try:
                _get_machine_catalog.clear()
            except Exception:
                pass
        return ok
    except Exception:
        return False


def generate_job_id(entry_date: date, job_type: str, existing_df: pd.DataFrame) -> str:
    jt = _normalize_job_type(job_type)
    if jt == "Maintenance":
        type_code = "M"
    elif jt == "Breakdown":
        type_code = "B"
    else:
        type_code = "G"

    date_str = entry_date.strftime("%y%m%d")
    prefix = f"{date_str}_{type_code}_"

    if existing_df is None or existing_df.empty or "Job ID" not in existing_df.columns:
        return f"{prefix}001"

    job_ids = existing_df["Job ID"].astype(str).fillna("").tolist()
    max_n = 0
    for jid in job_ids:
        jid = str(jid or "").strip()
        if not jid.startswith(prefix):
            continue
        tail = jid[len(prefix):].strip()
        # Support revision suffixes like "01-01"; first token is the base sequence.
        if "-" in tail:
            tail = tail.split("-", 1)[0].strip()
        try:
            max_n = max(max_n, int(tail))
        except Exception:
            continue

    return f"{prefix}{(max_n + 1):03d}"


def _next_resubmission_job_id(current_job_id: str, existing_df: pd.DataFrame) -> str:
    """Return the next revision Job ID for a rejected resubmission.

    Format:
    - Base: yymmdd_<Type>_<NNN>
    - Revisions: yymmdd_<Type>_<NNN>-01, -02, ...
    """
    jid = str(current_job_id or "").strip()
    if not jid:
        return jid
    base = jid.split("-", 1)[0].strip()
    if existing_df is None or existing_df.empty or "Job ID" not in existing_df.columns:
        return f"{base}-01"

    max_rev = 0
    for raw in existing_df["Job ID"].astype(str).fillna("").tolist():
        raw = str(raw or "").strip()
        if raw == base:
            continue
        if not raw.startswith(base + "-"):
            continue
        tail = raw[len(base) + 1 :].strip()
        # Only accept a numeric revision token.
        try:
            max_rev = max(max_rev, int(tail))
        except Exception:
            continue
    return f"{base}-{(max_rev + 1):02d}"


def generate_part_number(part_type_label: str, storage_df: pd.DataFrame) -> str:
    cfg = PART_TYPE_CONFIG.get(part_type_label)
    if not cfg:
        raise ValueError(f"Unknown part type: {part_type_label}")

    pn_prefix = cfg["pn_prefix"]
    if storage_df is None or storage_df.empty or "part_number" not in storage_df.columns:
        return f"{pn_prefix}001"

    existing = storage_df[storage_df["part_number"].astype(str).str.startswith(pn_prefix)].copy()
    if existing.empty:
        return f"{pn_prefix}001"

    suffix = existing["part_number"].astype(str).str.replace(pn_prefix, "", regex=False)
    suffix_num = pd.to_numeric(suffix, errors="coerce").dropna().astype(int)
    if suffix_num.empty:
        return f"{pn_prefix}001"

    next_number = int(suffix_num.max()) + 1
    return f"{pn_prefix}{next_number:03d}"


def generate_part_number_by_prefix(pn_prefix: str, storage_df: pd.DataFrame, reserved: set[str] | None = None) -> str:
    pn_prefix = str(pn_prefix or "").strip()
    if not pn_prefix:
        raise ValueError("pn_prefix is required")

    reserved = set(reserved or set())

    if storage_df is None or storage_df.empty or "part_number" not in storage_df.columns:
        candidate = f"{pn_prefix}001"
        if candidate in reserved:
            # Find next available
            i = 2
            while True:
                candidate = f"{pn_prefix}{i:03d}"
                if candidate not in reserved:
                    return candidate
                i += 1
        return candidate

    existing = storage_df[storage_df["part_number"].astype(str).str.startswith(pn_prefix)].copy()
    suffix = existing["part_number"].astype(str).str.replace(pn_prefix, "", regex=False)
    suffix_num = pd.to_numeric(suffix, errors="coerce").dropna().astype(int)
    max_existing = int(suffix_num.max()) if not suffix_num.empty else 0

    # Also consider reserved part numbers.
    for pn in reserved:
        if not str(pn).startswith(pn_prefix):
            continue
        tail = str(pn)[len(pn_prefix):]
        try:
            max_existing = max(max_existing, int(tail))
        except Exception:
            continue

    next_number = max_existing + 1
    return f"{pn_prefix}{next_number:03d}"


def update_storage_row_allow_renumber(
    old_part_number: str,
    new_part_number: str,
    item_name: str,
    specification: str,
    total_add: int,
    total_used: int,
    part_type: str,
    usage: str,
    *,
    brand: str = "",
    model: str = "",
    performed_by: str = "",
    note: str = "",
):
    """Update a storage row, optionally changing its primary key (part_number).

    Used for the Storage Editor: if part_type changes, we auto-generate a new part_number
    and persist it.
    """
    old_pn = str(old_part_number or "").strip()
    new_pn = str(new_part_number or "").strip()
    if not old_pn:
        raise ValueError("Old Part Number is required")
    if not new_pn:
        raise ValueError("New Part Number is required")

    total_quantity = int(total_add)
    total_out = int(total_used)
    if total_quantity < 0 or total_out < 0:
        raise ValueError("Quantities cannot be negative")
    total_in = total_out + total_quantity

    conn = sqlite3.connect(DB_PATH)
    try:
        before_state = _fetch_storage_row(conn, old_pn)

        cur = conn.cursor()
        cur.execute(
            """
            UPDATE storage
            SET part_number = ?,
                item_name = ?,
                brand = ?,
                model = ?,
                specification = ?,
                part_type = ?,
                preferred_supplier = ?,
                item_cost_rm = ?,
                total_quantity = ?,
                usage_area = ?,
                total_in = ?,
                total_out = ?
            WHERE part_number = ?
            """,
            (
                new_pn,
                item_name,
                str(brand or "").strip(),
                str(model or "").strip(),
                specification,
                str(part_type or "").strip(),
                "",  # preferred_supplier
                0.0,  # item_cost_rm
                int(total_quantity),
                str(usage or "").strip(),
                int(total_in),
                int(total_out),
                old_pn,
            ),
        )
        if cur.rowcount <= 0:
            raise ValueError(f"Part not found: {old_pn}")

        conn.commit()

        after_state = _fetch_storage_row(conn, new_pn)

        b_diff, a_diff = _diff_state_for_log(
            before_state,
            after_state,
            include_keys=[
                "part_number",
                "part_type",
                "item_name",
                "brand",
                "model",
                "specification",
                "preferred_supplier",
                "item_cost_rm",
                "usage_area",
                "total_quantity",
                "total_in",
                "total_out",
            ],
        )

        action = "RENUMBER_PART" if old_pn != new_pn else "UPDATE_PART"
        extra = f"old_pn={old_pn}" if old_pn != new_pn else ""
        combined_note = (str(note or "").strip() + (" | " + extra if extra else "")).strip(" |")

        log_inventory_history(
            action=action,
            part_number=new_pn,
            performed_by=performed_by,
            note=combined_note,
            before_state=b_diff,
            after_state=a_diff,
        )

        try:
            persist_repo_changes([str(DB_PATH)], reason=f"Inventory {action} {new_pn}")
        except Exception:
            pass
    finally:
        conn.close()


ensure_data_directory()
initialize_stock_log_database()
initialize_inventory_history_database()
init_db()

st.title("🔧 Task Update")
st.markdown("Technical team: report and update breakdown entries.")

with st.expander("Status guide", expanded=False):
    st.markdown("**Approval Status**")
    st.markdown(
        "- **Approved** — Report reviewed and approved by SuperUser\n"
        "- **In Review** — Waiting SuperUser to review before approved or rejected\n"
        "- **Rejected** — Form submitted has been rejected by SuperUser\n"
        "- **Not Submitted** — Task still in Pending/In Progress status"
    )
    st.markdown("**JobStatus**")
    st.markdown(
        "- **Pending** — No work progress started\n"
        "- **In Progress** — Work started and not completed\n"
        "- **Completed** — Work finished and submitted"
    )
st.markdown("---")

tab_generate, tab_review = st.tabs(["📝 Task Entry", "📋 Review Entries"])

# ================= TAB 1: TASK ENTRY =================
with tab_generate:
    st.markdown("### Task Entry")
    b1, b2 = st.columns(2)
    with b1:
        if st.button("📝 New Task", type="primary", use_container_width=True, key="br_mode_new_btn"):
            st.session_state.task_entry_mode = "new"
            # Reset New Task form so entries initialize empty.
            for k in [
                "br_job_type",
                "br_maint_freq",
                "br_date",
                "br_job_id_display",
                "br_shift",
                "br_time_start",
                "br_time_end",
                "br_severity",
                "br_priority",
                "br_location",
                "br_machine_not_in_list",
                "br_machine_id_sel",
                "br_machine_id_manual",
                "br_machine_name_manual",
                "br_machine_name_display",
                "br_job_status",
                "br_problem_desc",
                "br_root",
                "br_immediate",
                "br_preventive",
                "br_remark",
                "br_task_desc",
                "br_action",
                "br_assign_by",
                "br_verify_by",
                "br_associates",
                "br_start_date",
                "br_end_date",
                "br_spare_used",
                "br_sp_select_name",
                "br_sp_qty_used",
                "br_report_image",
                "spare_parts",
            ]:
                st.session_state.pop(k, None)
            st.rerun()
    with b2:
        if st.button("✏️ Update/Edit", use_container_width=True, key="br_mode_edit_btn"):
            st.session_state.task_entry_mode = "edit"
            st.rerun()

    mode = str(st.session_state.get("task_entry_mode", "") or "").strip().casefold()
    if mode not in {"new", "edit"}:
        st.info("Select **New Task** or **Update/Edit** to continue.")

    if mode == "new":
        st.markdown("### New Report")
    
        if "spare_parts" not in st.session_state:
            st.session_state.spare_parts = []

        existing_df = load_breakdown_data()

        # Job Type
        # Session-state migration: old value "Other(s)" -> "General"
        if str(st.session_state.get("br_job_type", "") or "").strip().casefold() in {"other", "others"}:
            st.session_state["br_job_type"] = "General"

        job_type = st.selectbox(
            "Job Type *",
            options=[""] + ["Maintenance", "Breakdown", "General"],
            key="br_job_type",
        )
        job_type_norm = _normalize_job_type(job_type)

        # Machine catalogue (Asset list)
        machine_id_options, machine_map = _get_machine_catalog()

        # Verify By options from regdata (free text allowed by spec, but we keep a guided list)
        regdata_names = list_regdata_display_names() or []
        _verify_cur = str(st.session_state.get("br_verify_by", "") or "").strip()
        verify_options = [str(x).strip() for x in regdata_names if str(x).strip()]
        if _verify_cur and _verify_cur not in verify_options:
            verify_options = [_verify_cur] + verify_options
        verify_options = [""] + verify_options

        # ================= Layout (requested arrangement) =================
        # Row 1: Date | Job ID | Severity | Priority | Maintenance Frequency (Maintenance only)
        if job_type_norm == "Maintenance":
            c_date, c_jid, c_sev, c_pri, c_mf = st.columns(5)
        else:
            c_date, c_jid, c_sev, c_pri = st.columns(4)
            c_mf = None

        with c_date:
            entry_date = st.date_input("Date *", value=today_sg(), key="br_date")
        with c_jid:
            job_id = generate_job_id(entry_date, job_type_norm, existing_df) if str(job_type_norm).strip() else ""
            st.text_input("Job ID (auto)", value=job_id, disabled=True, key="br_job_id_display")
        with c_sev:
            severity = st.selectbox("Severity *", options=[""] + ["Low", "Medium", "High", "Critical"], key="br_severity")
        with c_pri:
            priority = st.selectbox("Priority *", options=[""] + ["Low", "High"], key="br_priority")
        if c_mf is not None:
            with c_mf:
                maintenance_frequency = st.selectbox(
                    "Maintenance Frequency *",
                    options=[""] + list(MAINTENANCE_FREQUENCY_OPTIONS),
                    key="br_maint_freq",
                )
        else:
            maintenance_frequency = "NA"

        # Row 2: Shift | Location | Job Status | Assign by
        c_shift, c_loc, c_status, c_assign = st.columns(4)
        with c_shift:
            shift = st.selectbox("Shift *", options=[""] + ["Day", "Night"], key="br_shift")
        with c_loc:
            location = st.text_input("Location *", key="br_location")
        with c_status:
            job_status = st.selectbox("Job Status *", options=[""] + ["Pending", "In Progress", "Completed"], key="br_job_status")
        with c_assign:
            assign_by = st.text_input("Assign by", value=str(st.session_state.get("br_assign_by", "") or ""), key="br_assign_by")

        # Row 3: Date Start | Time Start | Machine ID
        c_sd, c_st, c_mid = st.columns(3)
        with c_sd:
            start_date = st.date_input("Date Start *", value=entry_date, key="br_start_date")
        with c_st:
            time_start = st.time_input("Time Start *", value=now_sg().time().replace(microsecond=0), key="br_time_start")
        with c_mid:
            machine_not_in_list = st.checkbox("Machine ID not in list", value=False, key="br_machine_not_in_list")
            if machine_not_in_list or not machine_id_options:
                machine_id = st.text_input("Machine ID", key="br_machine_id_manual")
            else:
                machine_id = st.selectbox("Machine ID", options=[""] + list(machine_id_options), key="br_machine_id_sel")

        # Row 4: Date End | Time End | Machine/Equipment
        c_ed, c_et, c_meq = st.columns(3)
        with c_ed:
            end_date = st.date_input("Date End *", value=entry_date, key="br_end_date")
        with c_et:
            time_end = st.time_input("Time End *", value=now_sg().time().replace(microsecond=0), key="br_time_end")
        with c_meq:
            if machine_not_in_list or not machine_id_options:
                machine_name = st.text_input("Machine/Equipment", key="br_machine_name_manual")
            else:
                machine_name = str(machine_map.get(str(machine_id).strip(), "") or "").strip()
                st.text_input("Machine/Equipment", value=machine_name, disabled=True, key="br_machine_name_display")

        start_dt = datetime.combine(start_date, time_start)
        end_dt = datetime.combine(end_date, time_end)
        duration_err = None
        duration_min = 0
        if end_dt < start_dt:
            # Allow overnight spans (common for Night shift):
            # If End time is earlier than Start time on the same date, treat End as next day.
            is_night_shift = str(shift or "").strip() == "Night"
            is_overnight_existing = end_date > start_date
            if is_overnight_existing or is_night_shift:
                end_dt = datetime.combine(start_date + timedelta(days=1), time_end)

        if end_dt < start_dt:
            duration_err = "Date/Time End must be after Date/Time Start."
        else:
            duration_min = int((end_dt - start_dt).total_seconds() // 60)

        # ================= Job-type specific narrative fields =================
        verify_by = ""
        remark = ""
        associates = ""
        problem_description = ""
        immediate_action = ""
        root_cause = ""
        preventive_action = ""

        if not str(job_type_norm).strip():
            st.info("Select Job Type to continue.")
        elif job_type_norm == "Breakdown":
            problem_description = st.text_area("Problem Description *", height=120, key="br_problem_desc")
            immediate_action = st.text_area("Immediate Action *", height=120, key="br_immediate")
            root_cause = st.text_area("Root Cause *", height=120, key="br_root")
            preventive_action = st.text_area("Preventive Action *", height=120, key="br_preventive")
            remark = st.text_area("Remark", height=90, key="br_remark")
            c_v, c_a = st.columns(2)
            with c_v:
                verify_by = st.selectbox("Verify By *", options=verify_options, key="br_verify_by")
            with c_a:
                associates = st.text_input("Associates", key="br_associates")

        elif job_type_norm == "Maintenance":
            task_description = st.text_area("Task Description *", height=120, key="br_task_desc")
            action = st.text_area("Action *", height=120, key="br_action")
            remark = st.text_area("Remark", height=90, key="br_remark")
            c_v, c_a = st.columns(2)
            with c_v:
                verify_by = st.selectbox("Verify By *", options=verify_options, key="br_verify_by")
            with c_a:
                associates = st.text_input("Associates", key="br_associates")

            problem_description = str(task_description or "").strip()
            immediate_action = str(action or "").strip()
            root_cause = ""
            preventive_action = ""

        else:
            job_description = st.text_area("Job Description *", height=120, key="br_task_desc")
            action = st.text_area("Action *", height=120, key="br_action")
            remark = st.text_area("Remark", height=90, key="br_remark")
            c_v, c_a = st.columns(2)
            with c_v:
                verify_by = st.selectbox("Verify By *", options=verify_options, key="br_verify_by")
            with c_a:
                associates = st.text_input("Associates", key="br_associates")

            problem_description = str(job_description or "").strip()
            immediate_action = str(action or "").strip()
            root_cause = ""
            preventive_action = ""

        # Spare Part Use
        st.markdown("### 🧰 Spare Parts Used")
        spare_used = st.checkbox("Spare parts used?", value=False, key="br_spare_used")

        if not spare_used:
            st.session_state.spare_parts = []
            st.info("No spare parts will be recorded for this job.")
            available_parts = pd.DataFrame()
        else:
            storage_df = get_storage()
            available_parts = storage_df[storage_df["total_add"].fillna(0).astype(int) > 0].copy()

            if available_parts.empty:
                st.warning("No spare parts available in inventory.")
            else:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    selected_part_name = st.selectbox("Select Spare Part", available_parts["item_name"], key="br_sp_select_name")

                selected_row = available_parts[available_parts["item_name"] == selected_part_name].iloc[0]
                max_qty = int(selected_row["total_add"])

                with col2:
                    use_qty = st.number_input("Qty Used", min_value=1, max_value=max_qty, step=1, key="br_sp_qty_used")
                with col3:
                    st.write("")
                    st.write("")
                    if st.button("➕ Add", key="br_sp_add_btn"):
                        st.session_state.spare_parts.append(
                            {"part_number": selected_row["part_number"], "name": selected_part_name, "qty": int(use_qty)}
                        )
                        st.rerun()

        if st.session_state.spare_parts:
            st.markdown("#### Parts Selected for This Job")
            for i, part in enumerate(st.session_state.spare_parts):
                col_a, col_b = st.columns([4, 1])
                col_a.write(f"• {part['name']} x{part['qty']}")
                if col_b.button("❌", key=f"remove_sp_{i}"):
                    st.session_state.spare_parts.pop(i)
                    st.rerun()

        st.markdown("### 📷 Report Image")
        report_image = st.file_uploader(
            "Upload 1 image (optional)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=False,
            key="br_report_image",
        )

        if st.button("✅ Submit Report", type="primary"):
            if duration_err:
                show_user_error(duration_err)
            elif not str(job_type_norm).strip():
                show_user_error("Job Type is required.")
            elif not str(shift).strip():
                show_user_error("Shift is required.")
            elif not str(location).strip():
                show_user_error("Location is required.")
            elif not str(severity).strip():
                show_user_error("Severity is required.")
            elif not str(priority).strip():
                show_user_error("Priority is required.")
            elif job_type_norm in {"Breakdown", "Maintenance"} and not str(machine_id).strip():
                show_user_error("Machine ID is required.")
            elif job_type_norm in {"Breakdown", "Maintenance"} and (machine_not_in_list or not machine_id_options) and not str(machine_name).strip():
                show_user_error("Machine/Equipment is required.")
            elif job_type_norm == "Maintenance" and not str(maintenance_frequency).strip():
                show_user_error("Maintenance Frequency is required.")
            elif job_type_norm == "Breakdown" and not str(problem_description).strip():
                show_user_error("Problem Description is required.")
            elif job_type_norm == "Breakdown" and not str(root_cause).strip():
                show_user_error("Root Cause is required.")
            elif job_type_norm == "Breakdown" and not str(immediate_action).strip():
                show_user_error("Immediate Action is required.")
            elif job_type_norm == "Breakdown" and not str(preventive_action).strip():
                show_user_error("Preventive Action is required.")
            elif job_type_norm in {"Maintenance", "General"} and not str(problem_description).strip():
                show_user_error("Task/Job Description is required.")
            elif job_type_norm in {"Maintenance", "General"} and not str(immediate_action).strip():
                show_user_error("Action is required.")
            elif not str(verify_by).strip():
                show_user_error("Verify By is required.")
            elif spare_used and not st.session_state.spare_parts:
                show_user_error("You ticked 'Spare parts used?' but did not add any parts.")
            else:
                df = load_breakdown_data()
                now_ts = format_ts_sg()
                user_display = _current_user_display_name()

                spares_text = (
                    " | ".join([f"{p['part_number']}:{p['name']} x{p['qty']}" for p in st.session_state.spare_parts])
                    if spare_used
                    else ""
                )

                next_approval = (
                    "Not Submitted" if _normalize_job_status(job_status) in {"Pending", "In Progress"} else "In Review"
                )

                status_norm = _normalize_job_status(job_status)
                is_completed_now = status_norm == "Completed"

                # Duration report: time from first report create until report status Completed.
                # If later Rejected, additional time after rejection is added on the next completion.
                report_started_at = now_ts
                report_cycle_start_at = "" if is_completed_now else now_ts
                report_accum_min = 0
                completed_at = now_ts if is_completed_now else ""
                completed_by = user_display if is_completed_now else ""
                duration_report_min = str(report_accum_min) if is_completed_now else ""

                new_entry = {
                    "Create by": user_display,
                    "Create at": now_ts,
                    "Reported by": user_display,
                    "Reported at": now_ts,
                    "Verify By": str(verify_by or "").strip(),
                    "Associates": str(associates or "").strip(),
                    "Assign by": str(assign_by or "").strip(),
                    "Date": entry_date.strftime("%Y-%m-%d"),
                    "Job ID": job_id,
                    "Job Type": job_type_norm,
                    "Maintenance Frequency": str(maintenance_frequency or "NA").strip() if job_type_norm == "Maintenance" else "NA",
                    "Severity": str(severity or "").strip(),
                    "Priority": str(priority or "").strip(),
                    "Shift": str(shift).strip(),
                    "Location": str(location).strip(),
                    "Machine/Equipment": str(machine_name or "").strip(),
                    "Machine ID": str(machine_id or "").strip(),
                    "Machine ID not in list": "Yes" if bool(machine_not_in_list or not machine_id_options) else "No",
                    "Date_Time Start": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "Date_Time End": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "Duration E": str(int(duration_min)),
                    "Duration report": str(duration_report_min),
                    "JobStatus": status_norm,
                    "Problem_Task_Job Description": str(problem_description or "").strip(),
                    "Immediate Action_Action": str(immediate_action or "").strip(),
                    "Root Cause": str(root_cause or "").strip(),
                    "Preventive Action": str(preventive_action or "").strip(),
                    "Remark": str(remark or "").strip(),
                    "Spare Parts Used": str(spares_text or "").strip(),
                    "Approval Status": next_approval,
                    "Approved By": "",
                    "Approved At": "",
                    "Rejected By": "",
                    "Rejected At": "",
                    "Rejection Justification": "",

                    # Duration report tracking
                    "Report Started At": report_started_at,
                    "Report Cycle Start At": report_cycle_start_at,
                    "Report Accumulated Min": str(report_accum_min),
                    "Completed At": completed_at,
                    "Completed By": completed_by,
                }

                df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
                for col in COLUMNS:
                    if col not in df.columns:
                        df[col] = ""
                df = df[COLUMNS]

                if not _save_breakdown_data(df):
                    st.error("Failed to save task report to main_data.db")
                    st.stop()

                for part in st.session_state.get("spare_parts", []):
                    try:
                        stock_out_task(
                            part_number=part["part_number"],
                            qty_used=part["qty"],
                            performed_by=_performed_by_label(),
                            note=f"JobID={job_id}",
                        )
                    except Exception as e:
                        st.error(f"Stock OUT (Task) failed for {part['part_number']}: {e}")
                        st.stop()

                # Save report image (optional)
                if report_image is not None:
                    try:
                        _save_report_image(job_id, report_image)
                    except Exception as e:
                        st.warning(f"Image upload failed (report still saved): {e}")

                for key in [
                    "br_job_type",
                    "br_maint_freq",
                    "br_date",
                    "br_job_id_display",
                    "br_shift",
                    "br_time_start",
                    "br_time_end",
                    "br_severity",
                    "br_priority",
                    "br_location",
                    "br_machine_not_in_list",
                    "br_machine_id_sel",
                    "br_machine_id_manual",
                    "br_machine_name_manual",
                    "br_machine_name_display",
                    "br_job_status",
                    "br_problem_desc",
                    "br_root",
                    "br_immediate",
                    "br_preventive",
                    "br_remark",
                    "br_task_desc",
                    "br_action",
                    "br_assign_by",
                    "br_assign_by_breakdown",
                    "br_assign_by_general",
                    "br_assign_by_display",
                    "br_verify_by",
                    "br_associates",
                    "br_start_date",
                    "br_end_date",
                    "br_spare_used",
                    "br_sp_select_name",
                    "br_sp_qty_used",
                    "br_report_image",
                    "spare_parts",
                ]:
                    if key in st.session_state:
                        del st.session_state[key]

                if _normalize_job_status(job_status) in {"Pending", "In Progress"}:
                    st.success("Task saved (Not Submitted) & stock updated!")
                else:
                    st.success("Task submitted (In Review) & stock updated!")
                st.rerun()

    elif mode == "edit":
        st.markdown("### Update/Edit Report")

        df_all = load_breakdown_data()
        if df_all is None or df_all.empty:
            st.info("No task entries yet. Use **New Task** to add one.")
        else:
            # Normalize types/status fields so filters work reliably
            if "Job Type" in df_all.columns:
                df_all["Job Type"] = df_all["Job Type"].astype(str).map(_normalize_job_type)
            if "JobStatus" not in df_all.columns:
                df_all["JobStatus"] = "In Progress"
            df_all["JobStatus"] = df_all["JobStatus"].map(_normalize_job_status)
            if "Approval Status" not in df_all.columns:
                df_all["Approval Status"] = "In Review"
            df_all["Approval Status"] = df_all["Approval Status"].map(_normalize_approval_status)
            for col in ["Rejected By", "Rejected At", "Rejection Justification", "Approved By", "Approved At"]:
                if col not in df_all.columns:
                    df_all[col] = ""

            # If a report is Rejected, it goes back to In Progress for revision.
            # Persist this normalization so rejected items don't remain stuck as Completed.
            try:
                rej_mask = df_all["Approval Status"].astype(str) == "Rejected"
                needs_fix = rej_mask & (df_all["JobStatus"].astype(str).map(_normalize_job_status) != "In Progress")
                if bool(needs_fix.any()):
                    df_all.loc[rej_mask, "JobStatus"] = "In Progress"
                    if not _save_breakdown_data(df_all):
                        st.error("Failed to normalize rejected report status.")
                        st.stop()
                    st.rerun()
            except Exception:
                pass

            editable_df = df_all.loc[
                (df_all["JobStatus"].astype(str).isin(["Pending", "In Progress"]))
                | (df_all["Approval Status"].astype(str) == "Rejected")
            ].copy()

            # Non-SuperUser: only see own editable/rejected submissions
            if _current_level_rank() < 3 and "Reported by" in editable_df.columns:
                cur_uid = str(_current_user_id() or "").strip()
                cur_name = str(_current_user_display_name() or "").strip()
                editable_df = editable_df[editable_df["Reported by"].astype(str).isin({cur_uid, cur_name})]

            editable_ids = _unique_job_ids(editable_df)
            if not editable_ids:
                st.info("No editable reports available (Pending/In Progress or Rejected).")
            else:
                # Ensure a valid selection exists so details render immediately.
                # (If session_state has an old/non-eligible Job ID, Streamlit may show only the selector.)
                cur_sel = str(st.session_state.get("br_edit_job_id", "") or "").strip()
                if (not cur_sel) or (cur_sel not in set(editable_ids)):
                    st.session_state["br_edit_job_id"] = str(editable_ids[0])
                    st.session_state.pop("br_edit_loaded_job_id", None)
                    # Force a one-time rerun so the widget binds to the new selection.
                    if not bool(st.session_state.get("br_edit_autoselected", False)):
                        st.session_state["br_edit_autoselected"] = True
                        st.rerun()

                edit_job_id = st.selectbox("Select Job ID", options=editable_ids, key="br_edit_job_id")
                base = load_breakdown_data()
                row_df = base[base["Job ID"].astype(str) == str(edit_job_id)].head(1)
                if row_df.empty:
                    st.info("Job not found.")
                    st.stop()

                idx = int(row_df.index[0])
                row = row_df.iloc[0].to_dict()
                row_job_type = _normalize_job_type(str(row.get("Job Type", "") or "").strip())
                row_status = _normalize_job_status(row.get("JobStatus", ""))
                row_approval = _normalize_approval_status(row.get("Approval Status", ""))
                row_reported_by = str(row.get("Reported by", "") or "").strip()
                editable_state = (row_status in {"Pending", "In Progress"}) or (row_approval == "Rejected")
                cur_uid = str(_current_user_id() or "").strip()
                cur_name = str(_current_user_display_name() or "").strip()
                can_edit = editable_state and (
                    _current_level_rank() >= 3 or row_reported_by in {cur_uid, cur_name}
                )

                # If a report is Rejected, it should return to In Progress for revision.
                # Auto-correct the stored value once to avoid confusing states like Completed+Rejected.
                if can_edit and row_approval == "Rejected" and row_status != "In Progress":
                    base.loc[idx, "JobStatus"] = "In Progress"
                    if not _save_breakdown_data(base):
                        st.error("Failed to update rejected report status.")
                        st.stop()
                    st.rerun()

                # Machine catalogue (used for edit defaults)
                machine_id_options, machine_map = _get_machine_catalog()
                existing_mid = str(row.get("Machine ID", "") or "").strip()
                existing_mname = str(row.get("Machine/Equipment", "") or "").strip()

                # Default date/time parsing
                def _safe_time(val: str, fallback: time) -> time:
                    try:
                        dt = pd.to_datetime(val, errors="coerce")
                        if pd.isna(dt):
                            return fallback
                        return dt.to_pydatetime().time()
                    except Exception:
                        return fallback

                _dt_tmp = pd.to_datetime(row.get("Date", ""), errors="coerce")
                entry_date = _dt_tmp.date() if not pd.isna(_dt_tmp) else today_sg()

                # Use full timestamps (not just times) so overnight jobs can be edited/saved.
                _dt_start_full = pd.to_datetime(row.get("Date_Time Start", ""), errors="coerce")
                _dt_end_full = pd.to_datetime(row.get("Date_Time End", ""), errors="coerce")

                ts_default = _safe_time(row.get("Date_Time Start", ""), now_sg().time().replace(microsecond=0))
                te_default = _safe_time(row.get("Date_Time End", ""), now_sg().time().replace(microsecond=0))

                start_date_base = _dt_start_full.date() if not pd.isna(_dt_start_full) else entry_date
                end_date_base = _dt_end_full.date() if not pd.isna(_dt_end_full) else start_date_base

                # Description defaults
                desc_default = str(row.get("Problem_Task_Job Description", "") or "").strip()
                if not desc_default:
                    desc_default = str(row.get("Job Description", "") or "").strip()

                remark_default = str(row.get("Remark", "") or "").strip()

                # IMPORTANT: When switching Job ID, Streamlit widget keys retain old values.
                # Re-initialize edit widget state from the DB row so the form always displays
                # the currently selected report.
                loaded_job = str(st.session_state.get("br_edit_loaded_job_id", "") or "")
                # Initialize defaults only when switching Job ID.
                # If we re-initialize on every rerun, user edits (e.g., Time End / Job Status) get overwritten.
                if loaded_job != str(edit_job_id):
                    st.session_state["br_edit_loaded_job_id"] = str(edit_job_id)

                    jt = row_job_type if row_job_type in {"Maintenance", "Breakdown", "General"} else "General"
                    st.session_state["br_edit_job_type"] = jt

                    # Simple fixed-option widgets
                    shift_val = str(row.get("Shift", "Day") or "Day").strip()
                    st.session_state["br_edit_shift"] = shift_val if shift_val in {"Day", "Night"} else "Day"

                    sev_val = str(row.get("Severity", "") or "").strip()
                    st.session_state["br_edit_severity"] = sev_val if sev_val in {"", "Low", "Medium", "High", "Critical"} else ""

                    pri_val = str(row.get("Priority", "") or "").strip()
                    st.session_state["br_edit_priority"] = pri_val if pri_val in {"", "Low", "High"} else ""

                    # Time widgets
                    st.session_state["br_edit_time_start"] = ts_default
                    st.session_state["br_edit_time_end"] = te_default

                    # Date widgets
                    st.session_state["br_edit_start_date"] = start_date_base
                    st.session_state["br_edit_end_date"] = end_date_base

                    # Free-text widgets
                    st.session_state["br_edit_location"] = str(row.get("Location", "") or "")
                    st.session_state["br_edit_assign_by"] = str(row.get("Assign by", "") or "").strip()

                    # Maintenance Frequency (Maintenance only)
                    mf_raw = str(row.get("Maintenance Frequency", "") or "").strip()
                    if mf_raw not in set(MAINTENANCE_FREQUENCY_OPTIONS):
                        mf_raw = "Daily" if "Daily" in set(MAINTENANCE_FREQUENCY_OPTIONS) else MAINTENANCE_FREQUENCY_OPTIONS[0]
                    st.session_state["br_edit_maint_freq"] = mf_raw

                    # Status (only if editable)
                    if not (row_approval == "Approved" or row_status == "Close"):
                        # If the report is Rejected, it is considered back "In Progress" until resubmitted.
                        if row_approval == "Rejected":
                            st.session_state["br_edit_job_status"] = "In Progress"
                        else:
                            st.session_state["br_edit_job_status"] = row_status if row_status in {"Pending", "In Progress", "Completed"} else "In Progress"
                    else:
                        st.session_state.pop("br_edit_job_status", None)

                    # Machine widgets: keep separate keys for manual vs selectbox
                    stored_manual = str(row.get("Machine ID not in list", "") or "").strip().casefold()
                    if stored_manual in {"yes", "y", "true", "1"}:
                        st.session_state["br_edit_machine_manual"] = True
                    elif stored_manual in {"no", "n", "false", "0"}:
                        st.session_state["br_edit_machine_manual"] = False
                    else:
                        st.session_state["br_edit_machine_manual"] = bool(existing_mid and existing_mid not in set(machine_id_options))
                    st.session_state["br_edit_machine_id_manual"] = existing_mid
                    st.session_state["br_edit_machine_name_manual"] = existing_mname
                    st.session_state.pop("br_edit_machine_id_sel", None)

                    # Description widgets
                    st.session_state["br_edit_problem"] = desc_default
                    st.session_state["br_edit_immediate"] = str(row.get("Immediate Action_Action", "") or "")
                    st.session_state["br_edit_root"] = str(row.get("Root Cause", "") or "")
                    st.session_state["br_edit_prev"] = str(row.get("Preventive Action", "") or "")
                    st.session_state["br_edit_remark"] = remark_default

                    # Verify By
                    st.session_state["br_edit_verify_by"] = str(row.get("Verify By", "") or "").strip()

                    # Associates
                    st.session_state["br_edit_associates"] = str(row.get("Associates", "") or "").strip()

                st.caption(f"Status: {row_status or '—'} | Approval: {row_approval or '—'}")
                if row_approval == "Rejected":
                    rej_msg = str(row.get("Rejection Justification", "") or "").strip()
                    if rej_msg:
                        st.warning(f"Rejected reason: {rej_msg}")

                # Row 0: Job Type | Create By | Create At (Create fields are view-only)
                can_change_job_type = can_edit and (row_status in {"Pending", "In Progress"})
                created_by = str(row.get("Create by", "") or row.get("Reported by", "") or "").strip()
                created_at = str(row.get("Create at", "") or row.get("Reported at", "") or "").strip()

                c_jt, c_cb, c_ca = st.columns(3)
                with c_jt:
                    edit_job_type = st.selectbox(
                        "Job Type",
                        options=["Maintenance", "Breakdown", "General"],
                        disabled=not can_change_job_type,
                        key="br_edit_job_type",
                    )
                    edit_job_type = _normalize_job_type(edit_job_type)
                with c_cb:
                    st.text_input("Create By", value=created_by, disabled=True)
                with c_ca:
                    st.text_input("Create At", value=created_at, disabled=True)

                preview_job_id = str(edit_job_id)
                if can_change_job_type and edit_job_type and edit_job_type != row_job_type:
                    preview_job_id = generate_job_id(entry_date, edit_job_type, base)

                # Row 1: Date | Job ID | Severity | Priority | Maintenance Frequency (Maintenance only)
                if edit_job_type == "Maintenance":
                    c_date, c_jid, c_sev, c_pri, c_mf = st.columns(5)
                else:
                    c_date, c_jid, c_sev, c_pri = st.columns(4)
                    c_mf = None

                with c_date:
                    st.date_input("Date", value=entry_date, disabled=True)
                with c_jid:
                    st.text_input("Job ID", value=str(preview_job_id), disabled=True)
                with c_sev:
                    sev_opts = ["", "Low", "Medium", "High", "Critical"]
                    sev_default = str(st.session_state.get("br_edit_severity", "") or "")
                    if sev_default not in set(sev_opts):
                        sev_default = ""
                    edit_sev = st.selectbox(
                        "Severity",
                        options=sev_opts,
                        index=sev_opts.index(sev_default),
                        disabled=not can_edit,
                        key="br_edit_severity",
                    )
                with c_pri:
                    pri_opts = ["", "Low", "High"]
                    pri_default = str(st.session_state.get("br_edit_priority", "") or "")
                    if pri_default not in set(pri_opts):
                        pri_default = ""
                    edit_pri = st.selectbox(
                        "Priority",
                        options=pri_opts,
                        index=pri_opts.index(pri_default),
                        disabled=not can_edit,
                        key="br_edit_priority",
                    )
                if c_mf is not None:
                    with c_mf:
                        st.selectbox(
                            "Maintenance Frequency",
                            options=MAINTENANCE_FREQUENCY_OPTIONS,
                            disabled=not can_edit,
                            key="br_edit_maint_freq",
                        )

                # Row 2: Shift | Location | Job Status | Assign by
                c_shift, c_loc, c_status, c_assign = st.columns(4)
                with c_shift:
                    shift_v = st.selectbox("Shift", options=["Day", "Night"], disabled=not can_edit, key="br_edit_shift")
                with c_loc:
                    edit_loc = st.text_input("Location", disabled=not can_edit, key="br_edit_location")
                with c_status:
                    if row_approval == "Approved" or row_status == "Close":
                        edit_status = row_status
                        st.text_input("Job Status", value=str(row_status), disabled=True)
                    else:
                        opts = ["Pending", "In Progress", "Completed"]
                        current = row_status if row_status in set(opts) else "In Progress"
                        edit_status = st.selectbox(
                            "Job Status",
                            options=opts,
                            index=opts.index(current),
                            disabled=not can_edit,
                            key="br_edit_job_status",
                        )
                with c_assign:
                    edit_assign = st.text_area("Assign by", height=68, disabled=not can_edit, key="br_edit_assign_by")

                # Row 3: Date Start | Time Start | Machine ID
                c_sd, c_st, c_mid = st.columns(3)
                with c_sd:
                    edit_start_date = st.date_input(
                        "Date Start",
                        value=st.session_state.get("br_edit_start_date", start_date_base),
                        disabled=not can_edit,
                        key="br_edit_start_date",
                    )
                with c_st:
                    edit_time_start = st.time_input(
                        "Time Start",
                        value=st.session_state.get("br_edit_time_start", ts_default),
                        disabled=not can_edit,
                        key="br_edit_time_start",
                    )
                with c_mid:
                    use_manual = st.checkbox(
                        "Machine ID not in list",
                        value=bool(st.session_state.get("br_edit_machine_manual", False)),
                        disabled=not can_edit,
                        key="br_edit_machine_manual",
                    )
                    if use_manual or not machine_id_options:
                        edit_mid = st.text_input(
                            "Machine ID",
                            value=str(st.session_state.get("br_edit_machine_id_manual", existing_mid) or ""),
                            disabled=not can_edit,
                            key="br_edit_machine_id_manual",
                        )
                    else:
                        if existing_mid and existing_mid in machine_id_options:
                            idx_mid = machine_id_options.index(existing_mid)
                        else:
                            idx_mid = 0
                        edit_mid = st.selectbox(
                            "Machine ID",
                            options=machine_id_options,
                            index=idx_mid,
                            disabled=not can_edit,
                            key="br_edit_machine_id_sel",
                        )

                # Row 4: Date End | Time End | Machine/Equipment
                c_ed, c_et, c_meq = st.columns(3)
                with c_ed:
                    edit_end_date = st.date_input(
                        "Date End",
                        value=st.session_state.get("br_edit_end_date", end_date_base),
                        disabled=not can_edit,
                        key="br_edit_end_date",
                    )
                with c_et:
                    edit_time_end = st.time_input(
                        "Time End",
                        value=st.session_state.get("br_edit_time_end", te_default),
                        disabled=not can_edit,
                        key="br_edit_time_end",
                    )
                with c_meq:
                    if use_manual or not machine_id_options:
                        edit_mname = st.text_input(
                            "Machine/Equipment",
                            value=str(st.session_state.get("br_edit_machine_name_manual", existing_mname) or ""),
                            disabled=not can_edit,
                            key="br_edit_machine_name_manual",
                        )
                    else:
                        edit_mname = str(machine_map.get(str(edit_mid).strip(), "") or "").strip()
                        st.text_input("Machine/Equipment", value=edit_mname, disabled=True)

                if edit_job_type == "Breakdown":
                    edit_prob = st.text_area("Problem Description", height=120, disabled=not can_edit, key="br_edit_problem")
                    edit_immediate = st.text_area("Immediate Action", height=120, disabled=not can_edit, key="br_edit_immediate")
                    edit_root = st.text_area("Root Cause", height=120, disabled=not can_edit, key="br_edit_root")
                    edit_prev = st.text_area("Preventive Action", height=120, disabled=not can_edit, key="br_edit_prev")
                    edit_remark = st.text_area("Remark", height=90, disabled=not can_edit, key="br_edit_remark")
                else:
                    desc_label = "Task Description" if edit_job_type == "Maintenance" else "Job Description"
                    edit_prob = st.text_area(desc_label, height=120, disabled=not can_edit, key="br_edit_problem")
                    edit_immediate = st.text_area("Action", height=120, disabled=not can_edit, key="br_edit_immediate")
                    edit_remark = st.text_area("Remark", height=90, disabled=not can_edit, key="br_edit_remark")
                    edit_root = ""
                    edit_prev = ""

                # Verify By (same placement as New Task: after narrative fields, before Spare Parts Used)
                regdata_names = list_regdata_display_names() or []
                verify_options = [str(x).strip() for x in regdata_names if str(x).strip()]
                cur_verify = str(st.session_state.get("br_edit_verify_by", "") or "").strip()
                if cur_verify and cur_verify not in verify_options:
                    verify_options = [cur_verify] + verify_options
                verify_options = [""] + verify_options
                c_v, c_a = st.columns(2)
                with c_v:
                    edit_verify_by = st.selectbox(
                        "Verify By *",
                        options=verify_options,
                        disabled=not can_edit,
                        key="br_edit_verify_by",
                    )
                with c_a:
                    edit_associates = st.text_input(
                        "Associates",
                        disabled=not can_edit,
                        key="br_edit_associates",
                    )

                # Spare parts edit (best-effort, uses stored part_number format)
                def _parse_spares(text: str) -> list[dict]:
                    out: list[dict] = []
                    s = str(text or "").strip()
                    if not s:
                        return out
                    for token in [t.strip() for t in s.split("|") if t.strip()]:
                        if "x" not in token:
                            continue
                        left, qty_s = token.rsplit("x", 1)
                        try:
                            qty = int(str(qty_s).strip())
                        except Exception:
                            continue
                        left = left.strip()
                        if ":" not in left:
                            continue
                        pn, name = left.split(":", 1)
                        pn = pn.strip()
                        name = name.strip()
                        if not pn:
                            continue
                        out.append({"part_number": pn, "name": name or pn, "qty": int(qty)})
                    return out

                def _spares_to_str(parts: list[dict]) -> str:
                    safe = []
                    for p in parts or []:
                        pn = str(p.get("part_number", "") or "").strip()
                        name = str(p.get("name", "") or "").strip()
                        try:
                            qty = int(p.get("qty", 0) or 0)
                        except Exception:
                            qty = 0
                        if pn and qty > 0:
                            safe.append(f"{pn}:{name or pn} x{qty}")
                    return " | ".join(safe)

                old_spares_text = str(row.get("Spare Parts Used", "") or "").strip()
                parsed_old = _parse_spares(old_spares_text)
                old_has_pn = bool(parsed_old) or (":" in old_spares_text and "x" in old_spares_text)

                st.markdown("##### 🧰 Spare Parts Used")
                if not can_edit:
                    st.info("Spare parts are view-only for this job.")
                    st.text_area("Spare Parts Used", value=old_spares_text, height=80, disabled=True)
                    edit_spares_list = parsed_old
                else:
                    if "br_edit_spares_job" not in st.session_state or st.session_state.get("br_edit_spares_job") != str(edit_job_id):
                        st.session_state["br_edit_spares_job"] = str(edit_job_id)
                        st.session_state["br_edit_spares_list"] = parsed_old

                    edit_spares_list = list(st.session_state.get("br_edit_spares_list") or [])
                    if not old_spares_text or old_has_pn:
                        spare_used = st.checkbox("Spare parts used?", value=bool(edit_spares_list), key="br_edit_spare_used")
                        if not spare_used:
                            edit_spares_list = []
                            st.session_state["br_edit_spares_list"] = []
                        else:
                            storage_df = get_storage()
                            avail = storage_df[storage_df["total_add"].fillna(0).astype(int) > 0].copy()
                            if avail.empty:
                                st.warning("No spare parts available in inventory.")
                            else:
                                c1, c2, c3 = st.columns([3, 1, 1])
                                with c1:
                                    sel_name = st.selectbox("Select Spare Part", avail["item_name"], key="br_edit_sp_select_name")
                                sel_row = avail[avail["item_name"] == sel_name].iloc[0]
                                max_qty = int(sel_row["total_add"])
                                with c2:
                                    qty = st.number_input("Qty Used", min_value=1, max_value=max_qty, step=1, key="br_edit_sp_qty")
                                with c3:
                                    st.write("")
                                    st.write("")
                                    if st.button("➕ Add", key="br_edit_sp_add"):
                                        edit_spares_list.append({"part_number": sel_row["part_number"], "name": sel_name, "qty": int(qty)})
                                        st.session_state["br_edit_spares_list"] = edit_spares_list
                                        st.rerun()

                        if edit_spares_list:
                            st.markdown("Parts Selected")
                            for i, p in enumerate(edit_spares_list):
                                ca, cb = st.columns([4, 1])
                                ca.write(f"• {p.get('name','')} x{p.get('qty','')}")
                                if cb.button("❌", key=f"br_edit_sp_rm_{i}"):
                                    edit_spares_list.pop(i)
                                    st.session_state["br_edit_spares_list"] = edit_spares_list
                                    st.rerun()
                    else:
                        st.warning("This job's spare parts format cannot be edited (missing part numbers).")
                        st.text_area("Spare Parts Used", value=old_spares_text, height=80, disabled=True)

                st.markdown("##### 📷 Report Image")
                if can_edit:
                    edit_report_image = st.file_uploader(
                        "Upload 1 image (optional)",
                        type=["png", "jpg", "jpeg", "webp"],
                        accept_multiple_files=False,
                        key="br_edit_report_image",
                    )
                else:
                    edit_report_image = None

                # Action buttons
                selected_status_for_btns = _normalize_job_status(edit_status)
                submit_disabled = (not can_edit) or (selected_status_for_btns != "Completed")

                col_btn_save, col_btn_submit = st.columns(2)
                save_clicked = col_btn_save.button(
                    "Save Changes",
                    type="primary",
                    disabled=not can_edit,
                    key="br_edit_save",
                    use_container_width=True,
                )
                submit_clicked = col_btn_submit.button(
                    "Submit",
                    disabled=submit_disabled,
                    key="br_edit_submit",
                    use_container_width=True,
                )

                if save_clicked or submit_clicked:
                    if not str(edit_loc).strip():
                        show_user_error("Location is required.")
                        st.stop()
                    if edit_job_type == "Maintenance":
                        mf_val = str(st.session_state.get("br_edit_maint_freq", "") or "").strip()
                        if not mf_val:
                            show_user_error("Maintenance Frequency is required.")
                            st.stop()
                    if edit_job_type in {"Breakdown", "Maintenance"}:
                        if not str(edit_mid).strip():
                            show_user_error("Machine ID is required.")
                            st.stop()
                        if st.session_state.get("br_edit_machine_manual") and not str(edit_mname).strip():
                            show_user_error("Machine/Equipment is required.")
                            st.stop()
                    if not str(edit_verify_by).strip():
                        show_user_error("Verify By is required.")
                        st.stop()
                    if not str(edit_prob).strip():
                        show_user_error("Description is required.")
                        st.stop()
                    if edit_job_type == "Breakdown":
                        if not str(edit_root).strip() or not str(edit_immediate).strip() or not str(edit_prev).strip():
                            show_user_error("Root Cause / Immediate Action / Preventive Action are required.")
                            st.stop()
                    else:
                        if not str(edit_immediate).strip():
                            show_user_error("Action is required.")
                            st.stop()

                    # Use edited dates; allow overnight spans (common for Night shift).
                    start_dt = datetime.combine(edit_start_date, edit_time_start)
                    end_dt = datetime.combine(edit_end_date, edit_time_end)

                    if end_dt < start_dt:
                        is_overnight_existing = edit_end_date > edit_start_date
                        is_night_shift = str(shift_v or "").strip() == "Night"
                        if is_overnight_existing or is_night_shift:
                            end_dt = datetime.combine(edit_start_date + timedelta(days=1), edit_time_end)

                    if end_dt < start_dt:
                        show_user_error("Time End must be after Time Start.")
                        st.stop()
                    duration_min = int((end_dt - start_dt).total_seconds() // 60)

                    now_ts = format_ts_sg()
                    now_dt = _parse_ts_sg(now_ts)

                    # Spare parts delta adjustment (best-effort)
                    old_map: dict[str, int] = {}
                    for p in _parse_spares(old_spares_text):
                        old_map[p["part_number"]] = old_map.get(p["part_number"], 0) + int(p.get("qty", 0) or 0)
                    new_map: dict[str, int] = {}
                    for p in (edit_spares_list or []):
                        pn = str(p.get("part_number", "") or "").strip()
                        if not pn:
                            continue
                        try:
                            q = int(p.get("qty", 0) or 0)
                        except Exception:
                            q = 0
                        new_map[pn] = new_map.get(pn, 0) + max(q, 0)

                    note_job_id = str(edit_job_id)

                    # Streamlit widget state is the source of truth.
                    selected_status = _normalize_job_status(edit_status)
                    cur_approval = _normalize_approval_status(row.get("Approval Status", ""))
                    next_approval = cur_approval
                    new_job_id = ""

                    # SAVE vs SUBMIT behavior
                    is_submit = bool(submit_clicked)
                    if is_submit:
                        # Only allow submission when the user explicitly sets Job Status to Completed.
                        if selected_status != "Completed":
                            show_user_error("Set Job Status to Completed before Submit.")
                            st.rerun()
                        next_status = "Completed"
                        next_approval = "In Review"
                    else:
                        # Save only: keep it editable; do not submit to approver.
                        if selected_status == "Completed":
                            # Still save the user's field edits, but keep the item editable.
                            st.session_state["br_edit_job_status"] = "In Progress"
                            next_status = "In Progress"
                        else:
                            next_status = selected_status

                    job_type_changed = bool(can_change_job_type and edit_job_type and edit_job_type != row_job_type)
                    job_type_new_id = str(preview_job_id).strip() if job_type_changed else ""
                    if job_type_new_id:
                        note_job_id = job_type_new_id

                    # Approval status transitions
                    if cur_approval == "Rejected":
                        if is_submit:
                            # Resubmission after rejection
                            if job_type_new_id:
                                new_job_id = job_type_new_id
                            else:
                                new_job_id = _next_resubmission_job_id(str(edit_job_id), base)
                            note_job_id = new_job_id
                            next_approval = "In Review"
                        else:
                            # Still revising; keep rejected state until resubmitted
                            next_status = "In Progress"
                            next_approval = "Rejected"
                    else:
                        next_approval = "In Review" if is_submit else "Not Submitted"

                    # Duration report computation on completion (Submit only)
                    duration_report_new: str | None = None
                    report_started_at_new: str | None = None
                    report_accum_new: str | None = None
                    report_cycle_start_new: str | None = None
                    completed_at_new: str | None = None
                    completed_by_new: str | None = None

                    if is_submit:
                        # Determine report start/cycle/accumulated values from the stored row.
                        row_started_raw = row.get("Report Started At", "") or row.get("Create at", "") or row.get("Reported at", "")
                        report_started_at_new = str(row_started_raw or "").strip() or now_ts

                        cycle_raw = row.get("Report Cycle Start At", "")
                        cycle_str = str(cycle_raw or "").strip()
                        if not cycle_str:
                            # If rejected, prefer Rejected At as the start for the remaining-time segment.
                            if cur_approval == "Rejected":
                                cycle_str = str(row.get("Rejected At", "") or "").strip()
                        if not cycle_str:
                            cycle_str = report_started_at_new

                        started_dt = _parse_ts_sg(report_started_at_new)
                        cycle_dt = _parse_ts_sg(cycle_str)
                        if cycle_dt is None:
                            cycle_dt = started_dt
                        if now_dt is None:
                            now_dt = now_sg()

                        acc_raw = row.get("Report Accumulated Min", "")
                        # If the new tracking fields are still empty (legacy rows), treat accumulated as 0.
                        has_new_tracking = any(
                            str(row.get(k, "") or "").strip()
                            for k in [
                                "Report Started At",
                                "Report Cycle Start At",
                                "Report Accumulated Min",
                                "Completed At",
                            ]
                        )
                        if not has_new_tracking and not str(acc_raw or "").strip():
                            accumulated = 0
                        else:
                            accumulated = _to_int_minutes(acc_raw, default=_to_int_minutes(row.get("Duration report", ""), default=0))
                        segment_min = 0
                        try:
                            if cycle_dt is not None:
                                segment_min = int(max(0, (now_dt - cycle_dt).total_seconds() // 60))
                        except Exception:
                            segment_min = 0

                        total_min = int(max(0, accumulated + segment_min))
                        duration_report_new = str(total_min)
                        report_accum_new = str(total_min)
                        report_cycle_start_new = ""  # completed; no active cycle
                        completed_at_new = now_ts
                        completed_by_new = _current_user_display_name()

                    performed_by = _performed_by_label()
                    if old_has_pn:
                        for pn in set(old_map.keys()) | set(new_map.keys()):
                            diff = int(new_map.get(pn, 0) - old_map.get(pn, 0))
                            if diff > 0:
                                stock_out_task(part_number=pn, qty_used=diff, performed_by=performed_by, note=f"JobID={note_job_id} EDIT")
                            elif diff < 0:
                                stock_in_add(part_number=pn, qty_in=abs(diff), performed_by=performed_by, note=f"JobID={note_job_id} EDIT")

                    # Write updates
                    base.loc[idx, "Job Type"] = str(edit_job_type).strip()
                    if edit_job_type == "Maintenance":
                        base.loc[idx, "Maintenance Frequency"] = str(st.session_state.get("br_edit_maint_freq") or "").strip() or "Daily"
                    else:
                        base.loc[idx, "Maintenance Frequency"] = "NA"
                    if is_submit and job_type_new_id and not new_job_id:
                        base.loc[idx, "Job ID"] = job_type_new_id
                    if new_job_id:
                        base.loc[idx, "Job ID"] = new_job_id
                    # Keep selector stable after Job ID change
                    try:
                        final_id = str(new_job_id or job_type_new_id or edit_job_id)
                        st.session_state["br_edit_job_id"] = final_id
                        st.session_state["br_edit_loaded_job_id"] = final_id
                    except Exception:
                        pass
                    base.loc[idx, "Shift"] = str(shift_v).strip()
                    base.loc[idx, "Severity"] = str(edit_sev or "").strip()
                    base.loc[idx, "Priority"] = str(edit_pri or "").strip()
                    base.loc[idx, "Location"] = str(edit_loc).strip()
                    base.loc[idx, "JobStatus"] = next_status
                    base.loc[idx, "Assign by"] = str(edit_assign or "").strip()
                    base.loc[idx, "Verify By"] = str(edit_verify_by or "").strip()
                    base.loc[idx, "Associates"] = str(edit_associates or "").strip()
                    base.loc[idx, "Date_Time Start"] = start_dt.strftime("%Y-%m-%d %H:%M:%S")
                    base.loc[idx, "Date_Time End"] = end_dt.strftime("%Y-%m-%d %H:%M:%S")
                    base.loc[idx, "Duration E"] = str(int(duration_min))
                    if duration_report_new is not None:
                        base.loc[idx, "Duration report"] = duration_report_new
                    # Tracking fields are written on submit completion
                    if report_started_at_new is not None:
                        base.loc[idx, "Report Started At"] = report_started_at_new
                    if report_cycle_start_new is not None:
                        base.loc[idx, "Report Cycle Start At"] = report_cycle_start_new
                    if report_accum_new is not None:
                        base.loc[idx, "Report Accumulated Min"] = report_accum_new
                    if completed_at_new is not None:
                        base.loc[idx, "Completed At"] = completed_at_new
                    if completed_by_new is not None:
                        base.loc[idx, "Completed By"] = str(completed_by_new or "").strip()

                    base.loc[idx, "Machine ID"] = str(edit_mid).strip()
                    base.loc[idx, "Machine/Equipment"] = str(edit_mname).strip()
                    base.loc[idx, "Machine ID not in list"] = "Yes" if bool(use_manual or not machine_id_options) else "No"

                    base.loc[idx, "Problem_Task_Job Description"] = str(edit_prob).strip()
                    base.loc[idx, "Immediate Action_Action"] = str(edit_immediate).strip()
                    base.loc[idx, "Root Cause"] = str(edit_root).strip() if edit_job_type == "Breakdown" else ""
                    base.loc[idx, "Preventive Action"] = str(edit_prev).strip() if edit_job_type == "Breakdown" else ""

                    base.loc[idx, "Remark"] = str(edit_remark or "").strip()

                    if old_has_pn:
                        base.loc[idx, "Spare Parts Used"] = _spares_to_str(edit_spares_list)

                    base.loc[idx, "Approval Status"] = next_approval

                    if next_approval == "In Review":
                        # Clear prior approval/rejection metadata on submit/resubmit
                        base.loc[idx, "Approved By"] = ""
                        base.loc[idx, "Approved At"] = ""
                        base.loc[idx, "Rejected By"] = ""
                        base.loc[idx, "Rejected At"] = ""
                        base.loc[idx, "Rejection Justification"] = ""

                    if not _save_breakdown_data(base):
                        st.error("Failed to save changes.")
                        st.stop()

                    # Post-save verification (helps catch silent DB write issues / mismatched Job ID updates)
                    try:
                        verify_df = load_breakdown_data()
                        verify_id = str(new_job_id or job_type_new_id or edit_job_id).strip()
                        hit = verify_df[verify_df["Job ID"].astype(str) == verify_id].head(1)
                        if hit.empty:
                            show_system_error(
                                "Saved, but could not find the updated Job ID in the database.",
                                RuntimeError(f"Missing Job ID after save: {verify_id}"),
                                context="TaskUpdate.UpdateEdit.post_save_verify",
                            )
                            st.stop()

                        vrow = hit.iloc[0]
                        v_status = _normalize_job_status(vrow.get("JobStatus", ""))
                        v_approval = _normalize_approval_status(vrow.get("Approval Status", ""))
                        if v_status != _normalize_job_status(next_status) or v_approval != _normalize_approval_status(next_approval):
                            show_system_error(
                                "Saved, but the database did not reflect the latest status/approval values.",
                                RuntimeError(
                                    f"Expected status={next_status}, approval={next_approval} | Got status={v_status}, approval={v_approval}"
                                ),
                                context="TaskUpdate.UpdateEdit.post_save_verify",
                            )
                            st.stop()
                    except Exception as e:
                        show_system_error(
                            "Could not verify the database update.",
                            e,
                            context="TaskUpdate.UpdateEdit.post_save_verify",
                        )
                        st.stop()

                    # Save report image (optional). Use final Job ID after any renumber/resubmission.
                    if edit_report_image is not None:
                        try:
                            final_id_for_image = str(new_job_id or job_type_new_id or edit_job_id).strip()
                            # If the Job ID changed and a new image is being uploaded, remove the old image file.
                            try:
                                old_id = str(edit_job_id or "").strip()
                                if old_id and final_id_for_image and old_id != final_id_for_image:
                                    for old in REPORT_IMAGES_DIR.glob(f"{old_id}.*"):
                                        if old.is_file():
                                            old.unlink(missing_ok=True)
                            except Exception:
                                pass
                            _save_report_image(final_id_for_image, edit_report_image)
                        except Exception as e:
                            st.warning(f"Image upload failed (changes still saved): {e}")
                        finally:
                            # Avoid keeping the uploaded file around across reruns.
                            st.session_state.pop("br_edit_report_image", None)

                    if is_submit:
                        st.success("Submitted for SuperUser review.")
                        for k in [
                            "br_edit_job_id",
                            "br_edit_loaded_job_id",
                            "br_edit_autoselected",
                            "br_edit_spares_job",
                            "br_edit_spares_list",
                            "br_edit_spare_used",
                            "br_edit_sp_select_name",
                            "br_edit_sp_qty",
                            "br_edit_report_image",
                            "br_edit_job_type",
                            "br_edit_shift",
                            "br_edit_maint_freq",
                            "br_edit_priority",
                            "br_edit_time_start",
                            "br_edit_time_end",
                            "br_edit_severity",
                            "br_edit_location",
                            "br_edit_job_status",
                            "br_edit_assign_by",
                            "br_edit_machine_manual",
                            "br_edit_machine_id_manual",
                            "br_edit_machine_id_sel",
                            "br_edit_machine_name_manual",
                            "br_edit_start_date",
                            "br_edit_end_date",
                            "br_edit_problem",
                            "br_edit_verify_by",
                            "br_edit_associates",
                            "br_edit_immediate",
                            "br_edit_root",
                            "br_edit_prev",
                            "br_edit_remark",
                        ]:
                            st.session_state.pop(k, None)
                    else:
                        if selected_status_for_btns == "Completed":
                            st.success("Saved changes (kept as In Progress). Use Submit to send to SuperUser.")
                        else:
                            st.success("Saved changes.")
                    st.rerun()

# ================= TAB 2: REVIEW ENTRIES =================
with tab_review:
    st.markdown("### Review task reports")

    def _approval_badge(value: object) -> str:
        s = _normalize_approval_status(value)
        if s == "Approved":
            return "✅ Approved"
        if s == "In Review":
            return "🕒 In Review"
        if s == "Rejected":
            return "❌ Rejected"
        if s == "Not Submitted":
            return "⏳ Not Submitted"
        return str(s or "").strip()

    # Manual refresh/reset (useful when multiple users edit concurrently)
    c_ref_1, c_ref_2 = st.columns([1, 4])
    with c_ref_1:
        if st.button("🔄 Refresh / Reset", key="br_review_refresh_btn", use_container_width=True):
            for k in [
                "br_filter_date",
                "br_filter_jobtype",
                "br_filter_approval",
                "br_filter_kw",
                "show_approver_actions",
                "br_review_approve_selector",
                "br_reject_job_id",
                "br_reject_reason",
                "show_task_report_view",
                "br_view_job_id",
                "br_pdf_job_id",
            ]:
                st.session_state.pop(k, None)
            st.rerun()
    with c_ref_2:
        st.caption("Reload latest data and reset Review filters/view.")

    df = load_breakdown_data()
    if df.empty:
        st.info("No task entries yet. Use **Task Entry** to add one.")
    else:
        # Normalize job type + approval defaults
        if "Job Type" in df.columns:
            df["Job Type"] = df["Job Type"].astype(str).map(_normalize_job_type)
        if "JobStatus" not in df.columns:
            df["JobStatus"] = "In Progress"
        df["JobStatus"] = df["JobStatus"].map(_normalize_job_status)
        # Keep UI to the 3 JobStatus values (legacy Close -> Completed)
        try:
            df.loc[df["JobStatus"].astype(str) == "Close", "JobStatus"] = "Completed"
        except Exception:
            pass

        if "Approval Status" not in df.columns:
            df["Approval Status"] = "In Review"

        df["Approval Status"] = df["Approval Status"].map(_normalize_approval_status)

        # Ensure expected columns exist
        for col in [
            "Approved By",
            "Approved At",
            "Rejected By",
            "Rejected At",
            "Rejection Justification",
        ]:
            if col not in df.columns:
                df[col] = ""

        # Best-effort alignment for legacy rows
        try:
            mask_active = df["JobStatus"].astype(str).isin(["Pending", "In Progress"])
            mask_not_final = ~df["Approval Status"].astype(str).isin(["Approved", "Rejected"])
            df.loc[mask_active & mask_not_final, "Approval Status"] = "Not Submitted"
        except Exception:
            pass
        if "Approved By" not in df.columns:
            df["Approved By"] = ""
        if "Approved At" not in df.columns:
            df["Approved At"] = ""

        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            filter_date = st.date_input("Filter by date", value=None, key="br_filter_date")
        with col_f2:
            filter_job_type = st.selectbox("Filter by Job Type", options=["All", "Breakdown", "Maintenance", "General"], key="br_filter_jobtype")
        with col_f3:
            filter_approval = st.selectbox(
                "Filter by Approval",
                options=["All", "Not Submitted", "In Review", "Rejected", "Approved"],
                key="br_filter_approval",
            )
        with col_f4:
            filter_keyword = st.text_input("Keyword", placeholder="Machine ID / Location / Description...", key="br_filter_kw")

        review_df = df.copy()
        if filter_date:
            review_df["Date"] = pd.to_datetime(review_df["Date"], errors="coerce").dt.date
            review_df = review_df[review_df["Date"] == filter_date]
        if filter_job_type and filter_job_type != "All":
            review_df = review_df[review_df["Job Type"].astype(str) == str(filter_job_type)]
        if filter_approval and filter_approval != "All":
            review_df = review_df[review_df["Approval Status"].astype(str).map(_normalize_approval_status) == str(filter_approval)]
        if filter_keyword and str(filter_keyword).strip():
            kw = str(filter_keyword).strip()
            hay = (
                review_df.get("Machine ID", "").astype(str)
                + " "
                + review_df.get("Machine/Equipment", "").astype(str)
                + " "
                + review_df.get("Location", "").astype(str)
                + " "
                + review_df.get("Problem_Task_Job Description", "").astype(str)
            )
            review_df = review_df[hay.str.contains(kw, case=False, na=False)]

        # Main view
        show_cols = [
            c
            for c in [
                "Approval Status",
                "JobStatus",
                "Date",
                "Job ID",
                "Job Type",
                "Rejected By",
                "Rejected At",
                "Severity",
                "Priority",
                "Shift",
                "Date_Time Start",
                "Date_Time End",
                "Location",
                "Machine ID",
                "Machine/Equipment",
                "Assign by",
                "Verify By",
                "Reported by",
                "Create at",
                "Completed By",
            ]
            if c in review_df.columns
        ]
        # Fallback to core columns present
        core_cols = [c for c in ["Date", "Job ID", "Job Type", "JobStatus", "Approval Status", "Problem_Task_Job Description"] if c in review_df.columns]
        view_cols = show_cols if show_cols else core_cols
        # Sort newest-first (best-effort)
        view_df = review_df.copy()
        try:
            if "Create at" in view_df.columns:
                view_df["__created_sort"] = pd.to_datetime(view_df["Create at"], errors="coerce")
            if "Date" in view_df.columns:
                view_df["__date_sort"] = pd.to_datetime(view_df["Date"], errors="coerce")
            sort_cols = [c for c in ["__date_sort", "__created_sort"] if c in view_df.columns]
            if sort_cols:
                view_df = view_df.sort_values(by=sort_cols, ascending=False)
        except Exception:
            pass

        st.caption(f"Showing {len(view_df)} of {len(df)} row(s)")

        # Table (read-only)
        table_df = view_df[view_cols].copy() if view_cols else view_df.copy()
        # Display-only: add symbol badges for approval status
        try:
            if "Approval Status" in table_df.columns:
                table_df["Approval Status"] = table_df["Approval Status"].map(_approval_badge)
        except Exception:
            pass
        st.dataframe(table_df, use_container_width=True, hide_index=True)

        # Approver action (SuperUser) - tick selection only shows eligible rows
        if _current_level_rank() >= 3 and not df.empty and "Job ID" in df.columns:
            tmp = df.copy()
            tmp["JobStatus"] = tmp.get("JobStatus", "").map(_normalize_job_status)
            tmp["Approval Status"] = tmp.get("Approval Status", "").map(_normalize_approval_status)
            eligible_df = tmp.loc[
                (tmp["JobStatus"].astype(str) == "Completed")
                & (tmp["Approval Status"].astype(str) == "In Review")
            ].copy()

            eligible_ids = _unique_job_ids(eligible_df)
            if eligible_ids:
                st.markdown("#### Bulk Approver Actions")
                col_ap_1, col_ap_2 = st.columns(2)
                with col_ap_1:
                    st.caption("Approve/Reject reports in In Review.")
                with col_ap_2:
                    if st.button("Open/Close", key="br_approver_toggle_btn"):
                        st.session_state.show_approver_actions = not st.session_state.get("show_approver_actions", False)
                        st.rerun()

                if not st.session_state.get("show_approver_actions", False):
                    st.info("Approver section is hidden.")
                    st.markdown("---")
                else:
                    st.caption("Tick rows below to approve. Rejection requires justification.")

                    # Build a compact selection table from the current view (filtered) when possible.
                    selector_source = view_df.copy()
                    if "Job ID" in selector_source.columns:
                        selector_source = selector_source[selector_source["Job ID"].astype(str).isin(eligible_ids)]
                    else:
                        selector_source = eligible_df

                    eligible_ids_in_view = _unique_job_ids(selector_source)
                    if not eligible_ids_in_view:
                        st.info("No In Review reports in the current filter.")
                        st.markdown("---")
                    else:

                        selector_cols = [
                            c
                            for c in [
                                "Job ID",
                                "Job Type",
                                "JobStatus",
                                "Approval Status",
                                "Location",
                                "Machine ID",
                                "Machine/Equipment",
                                "Reported by",
                                "Create at",
                                "Completed By",
                            ]
                            if c in selector_source.columns
                        ]
                        # Rearrange: Approval Status first, JobStatus second
                        selector_cols = [c for c in ["Approval Status", "JobStatus"] if c in selector_cols] + [
                            c for c in selector_cols if c not in {"Approval Status", "JobStatus"}
                        ]
                        selector_df = selector_source[selector_cols].copy() if selector_cols else selector_source.copy()
                        selector_df.insert(0, "Approve", False)

                        # Display-only: add symbol badges for approval status
                        try:
                            if "Approval Status" in selector_df.columns:
                                selector_df["Approval Status"] = selector_df["Approval Status"].map(_approval_badge)
                        except Exception:
                            pass

                        edited = st.data_editor(
                            selector_df,
                            use_container_width=True,
                            hide_index=True,
                            disabled=[c for c in selector_df.columns if c != "Approve"],
                            key="br_review_approve_selector",
                        )

                        selected_ids: list[str] = []
                        try:
                            selected_ids = (
                                edited.loc[edited["Approve"] == True, "Job ID"]
                                .astype(str)
                                .fillna("")
                                .map(lambda x: str(x).strip())
                                .tolist()
                            )
                            selected_ids = [x for x in selected_ids if x]
                        except Exception:
                            selected_ids = []

                        if st.button("✅ Approve selected", type="primary", key="br_approve_selected_btn"):
                            if _current_level_rank() < 3:
                                show_user_error("Only SuperUser can approve reports.")
                                st.stop()
                            if not selected_ids:
                                show_user_error("Tick at least one row.")
                                st.stop()

                            allowed = [jid for jid in selected_ids if jid in set(eligible_ids_in_view)]
                            if not allowed:
                                show_user_error("Selected rows are not eligible.")
                                st.stop()

                            base = load_breakdown_data()
                            mask = base["Job ID"].astype(str).isin([str(x) for x in allowed])
                            if not mask.any():
                                st.error("No matching Job IDs found.")
                                st.stop()

                            base.loc[mask, "Approval Status"] = "Approved"
                            base.loc[mask, "Approved By"] = _current_user_display_name()
                            base.loc[mask, "Approved At"] = format_ts_sg()
                            base.loc[mask, "Rejected By"] = ""
                            base.loc[mask, "Rejected At"] = ""
                            base.loc[mask, "Rejection Justification"] = ""
                            if not _save_breakdown_data(base):
                                st.error("Failed to approve submission(s).")
                                st.stop()

                            st.success(f"Approved {int(mask.sum())} submission(s).")
                            st.rerun()

                        st.markdown("##### 🚫 Reject (SuperUser)")
                        rej_job_id = st.selectbox(
                            "Select Job ID to reject",
                            options=eligible_ids_in_view,
                            key="br_reject_job_id",
                        )
                        rej_reason = st.text_area(
                            "Rejection justification *",
                            placeholder="Explain why the submission is rejected and what to revise...",
                            key="br_reject_reason",
                        )
                        if st.button("🚫 Reject", type="secondary", key="br_reject_btn"):
                            if _current_level_rank() < 3:
                                show_user_error("Only SuperUser can reject reports.")
                                st.stop()
                            if not str(rej_job_id).strip():
                                show_user_error("Select a Job ID.")
                                st.stop()
                            if not str(rej_reason).strip():
                                show_user_error("Rejection justification is required.")
                                st.stop()

                            base = load_breakdown_data()
                            mask = base["Job ID"].astype(str) == str(rej_job_id)
                            if not mask.any():
                                st.error("Job ID not found.")
                                st.stop()

                            cur_status = _normalize_job_status(base.loc[mask, "JobStatus"].iloc[0] if "JobStatus" in base.columns else "")
                            cur_approval = _normalize_approval_status(base.loc[mask, "Approval Status"].iloc[0] if "Approval Status" in base.columns else "")
                            if cur_approval == "Approved":
                                show_user_error("This report is already approved.")
                                st.stop()
                            if cur_status != "Completed" or cur_approval != "In Review":
                                show_user_error("Only In Review reports can be rejected.")
                                st.stop()

                            rej_ts = format_ts_sg()
                            base.loc[mask, "Approval Status"] = "Rejected"
                            base.loc[mask, "Rejected By"] = _current_user_display_name()
                            base.loc[mask, "Rejected At"] = rej_ts
                            base.loc[mask, "Rejection Justification"] = str(rej_reason).strip()
                            base.loc[mask, "Approved By"] = ""
                            base.loc[mask, "Approved At"] = ""

                            # Duration report tracking: start the "remaining time" cycle at rejection.
                            try:
                                idx0 = base.index[mask][0]
                                started_raw = base.loc[idx0, "Report Started At"] if "Report Started At" in base.columns else ""
                                started_str = str(started_raw or "").strip()
                                if not started_str:
                                    started_str = str(base.loc[idx0, "Create at"] or base.loc[idx0, "Reported at"] or rej_ts).strip()
                                    base.loc[idx0, "Report Started At"] = started_str
                                acc_raw = base.loc[idx0, "Report Accumulated Min"] if "Report Accumulated Min" in base.columns else ""
                                acc_val = _to_int_minutes(acc_raw, default=_to_int_minutes(base.loc[idx0, "Duration report"], default=0))
                                base.loc[idx0, "Report Accumulated Min"] = str(acc_val)
                                base.loc[idx0, "Report Cycle Start At"] = rej_ts
                            except Exception:
                                pass
                            if not _save_breakdown_data(base):
                                st.error("Failed to reject submission.")
                                st.stop()
                            st.success("Rejected.")
                            st.rerun()

        st.markdown("---")

        # View task report content (toggle like Inventory History)
        st.markdown("#### View task report")
        col_view_1, col_view_2 = st.columns(2)
        with col_view_1:
            st.caption("Select a Job ID to review full content.")
        with col_view_2:
            if st.button("📋 View Selected Report"):
                st.session_state.show_task_report_view = not st.session_state.get("show_task_report_view", False)
                st.rerun()

        selected_view_job_id = ""
        if st.session_state.get("show_task_report_view", False):
            view_ids = _unique_job_ids(review_df)
            if not view_ids:
                st.info("No Job ID values available in the current filter.")
            else:
                selected_view_job_id = st.selectbox("Select Job ID", options=view_ids, key="br_view_job_id")
                view_row_df = df[df["Job ID"].astype(str) == str(selected_view_job_id)].head(1)
                if view_row_df.empty:
                    st.info("Job not found.")
                else:
                    r = view_row_df.iloc[0].to_dict()

                    st.markdown(f"##### Task Report: {str(r.get('Job ID','') or '').strip()}")

                    # Report image (if uploaded)
                    try:
                        jid_img = str(r.get("Job ID", "") or "").strip()
                        img_path = None
                        if jid_img:
                            for p in REPORT_IMAGES_DIR.glob(f"{jid_img}.*"):
                                if p.is_file() and p.suffix.lower() in _REPORT_IMG_EXTS:
                                    img_path = p
                                    break
                        st.markdown("##### 📷 Report Image")
                        if img_path is None:
                            st.info("No image uploaded for this report.")
                        else:
                            st.image(str(img_path), caption=img_path.name, use_container_width=True)
                    except Exception as e:
                        st.markdown("##### 📷 Report Image")
                        st.warning(f"Could not load image: {e}")

                    # Approve action (SuperUser) for the currently viewed report
                    job_id_view = str(r.get("Job ID", "") or "").strip()
                    job_status_view = _normalize_job_status(r.get("JobStatus", ""))
                    approval_view = _normalize_approval_status(r.get("Approval Status", ""))

                    if _current_level_rank() >= 3 and job_id_view:
                        a1, a2, a3 = st.columns([1, 1, 3])
                        can_review = (job_status_view == "Completed") and (approval_view == "In Review")
                        already_final = approval_view in {"Approved"}

                        with a1:
                            if st.button(
                                "✅ Approve",
                                type="primary",
                                disabled=(not can_review) or already_final,
                                key=f"br_view_approve_{job_id_view}",
                            ):
                                if _current_level_rank() < 3:
                                    show_user_error("Only SuperUser can approve reports.")
                                    st.stop()
                                base = load_breakdown_data()
                                mask = base["Job ID"].astype(str) == str(job_id_view)
                                if not mask.any():
                                    st.error("Job ID not found.")
                                    st.stop()

                                cur_status = _normalize_job_status(base.loc[mask, "JobStatus"].iloc[0] if "JobStatus" in base.columns else "")
                                cur_approval = _normalize_approval_status(base.loc[mask, "Approval Status"].iloc[0] if "Approval Status" in base.columns else "")
                                if cur_approval == "Approved":
                                    st.info("This report is already approved.")
                                    st.stop()
                                if cur_status != "Completed" or cur_approval != "In Review":
                                    show_user_error("Only In Review reports can be approved.")
                                    st.stop()

                                base.loc[mask, "Approval Status"] = "Approved"
                                base.loc[mask, "Approved By"] = _current_user_display_name()
                                base.loc[mask, "Approved At"] = format_ts_sg()
                                base.loc[mask, "Rejected By"] = ""
                                base.loc[mask, "Rejected At"] = ""
                                base.loc[mask, "Rejection Justification"] = ""
                                if not _save_breakdown_data(base):
                                    st.error("Failed to approve submission.")
                                    st.stop()
                                st.success("Approved.")
                                st.rerun()

                        with a2:
                            rej_reason_view = st.text_area(
                                "Rejection justification *",
                                placeholder="Explain what to revise...",
                                disabled=(not can_review) or already_final,
                                key=f"br_view_reject_reason_{job_id_view}",
                            )
                            if st.button(
                                "🚫 Reject",
                                disabled=(not can_review) or already_final,
                                key=f"br_view_reject_{job_id_view}",
                            ):
                                if _current_level_rank() < 3:
                                    show_user_error("Only SuperUser can reject reports.")
                                    st.stop()
                                if not str(rej_reason_view).strip():
                                    show_user_error("Rejection justification is required.")
                                    st.stop()
                                base = load_breakdown_data()
                                mask = base["Job ID"].astype(str) == str(job_id_view)
                                if not mask.any():
                                    st.error("Job ID not found.")
                                    st.stop()
                                cur_status = _normalize_job_status(base.loc[mask, "JobStatus"].iloc[0] if "JobStatus" in base.columns else "")
                                cur_approval = _normalize_approval_status(base.loc[mask, "Approval Status"].iloc[0] if "Approval Status" in base.columns else "")
                                if cur_status != "Completed" or cur_approval != "In Review":
                                    show_user_error("Only In Review reports can be rejected.")
                                    st.stop()
                                rej_ts = format_ts_sg()
                                base.loc[mask, "Approval Status"] = "Rejected"
                                base.loc[mask, "Rejected By"] = _current_user_display_name()
                                base.loc[mask, "Rejected At"] = rej_ts
                                base.loc[mask, "Rejection Justification"] = str(rej_reason_view).strip()
                                base.loc[mask, "Approved By"] = ""
                                base.loc[mask, "Approved At"] = ""

                                # Duration report tracking: start the "remaining time" cycle at rejection.
                                try:
                                    idx0 = base.index[mask][0]
                                    started_raw = base.loc[idx0, "Report Started At"] if "Report Started At" in base.columns else ""
                                    started_str = str(started_raw or "").strip()
                                    if not started_str:
                                        started_str = str(base.loc[idx0, "Create at"] or base.loc[idx0, "Reported at"] or rej_ts).strip()
                                        base.loc[idx0, "Report Started At"] = started_str
                                    acc_raw = base.loc[idx0, "Report Accumulated Min"] if "Report Accumulated Min" in base.columns else ""
                                    acc_val = _to_int_minutes(acc_raw, default=_to_int_minutes(base.loc[idx0, "Duration report"], default=0))
                                    base.loc[idx0, "Report Accumulated Min"] = str(acc_val)
                                    base.loc[idx0, "Report Cycle Start At"] = rej_ts
                                except Exception:
                                    pass
                                if not _save_breakdown_data(base):
                                    st.error("Failed to reject submission.")
                                    st.stop()
                                st.success("Rejected.")
                                st.rerun()

                        with a3:
                            if approval_view == "Approved":
                                st.success("Approved")
                            elif approval_view == "Rejected":
                                st.error("Rejected")
                            elif can_review:
                                st.info("In Review")
                            else:
                                st.info("Review actions are available only when Job Status = Completed and Approval = In Review.")

                    # Table-style view (like before)
                    ordered_fields = [
                        "Job Type",
                        "Date",
                        "Job ID",
                        "Severity",
                        "Priority",
                        "Shift",
                        "Location",
                        "Machine ID",
                        "Machine/Equipment",
                        "Date_Time Start",
                        "Date_Time End",
                        "JobStatus",
                        "Assign by",
                        "Verify By",
                        "Problem_Task_Job Description",
                        "Immediate Action_Action",
                        "Root Cause",
                        "Preventive Action",
                        "Spare Parts Used",
                        "Approval Status",
                        "Approved By",
                        "Approved At",
                        "Rejected By",
                        "Rejected At",
                        "Rejection Justification",
                        "Create by",
                        "Create at",
                        "Reported by",
                        "Reported at",
                    ]
                    rows = []
                    for f in ordered_fields:
                        if f not in r:
                            continue
                        val = r.get(f, "")
                        if f == "JobStatus":
                            val = _normalize_job_status(val)
                        if f == "Approval Status":
                            val = _normalize_approval_status(val)
                        rows.append({"Field": f, "Value": str(val or "").strip()})

                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    # Headlines only (no long text content here)

        st.markdown("---")

        # Download report section
        st.markdown("#### Download report")
        d1, d2 = st.columns(2)
        with d1:
            st.caption("CSV export")
            csv_filtered = review_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ Download (filtered CSV)",
                data=csv_filtered,
                file_name=f"task_report_filtered_{format_ts_sg(fmt='%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
            csv_all = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ Download (full CSV)",
                data=csv_all,
                file_name=f"task_report_full_{format_ts_sg(fmt='%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with d2:
            st.caption("PDF export (single job)")
            if review_df.empty:
                st.info("No rows in the current filter. Clear filters to export.")
            else:
                job_ids = _unique_job_ids(review_df)
                if not job_ids:
                    st.info("No Job ID values found to export.")
                else:
                    default_index = 0
                    if selected_view_job_id and selected_view_job_id in job_ids:
                        default_index = job_ids.index(selected_view_job_id)
                    sel_job_id = st.selectbox("Select Job ID", options=job_ids, index=default_index, key="br_pdf_job_id")
                    row_df = review_df[review_df["Job ID"].astype(str) == str(sel_job_id)].head(1)
                    row_dict = row_df.iloc[0].to_dict() if not row_df.empty else {}

                    try:
                        pdf_bytes = build_task_report_pdf(row_dict)
                        st.download_button(
                            "⬇️ Download PDF",
                            data=pdf_bytes,
                            file_name=f"TaskReport_{sel_job_id.replace('/', '-')}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.error(f"PDF export failed: {e}")

