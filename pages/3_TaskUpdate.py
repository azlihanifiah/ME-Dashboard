import streamlit as st
import pandas as pd
from pathlib import Path
import sqlite3
from io import BytesIO
from datetime import datetime, date, time
from utils import (
    ensure_data_directory,
    initialize_stock_log_database,
    log_stock_operation,
    initialize_inventory_history_database,
    log_inventory_history,
    persist_repo_changes,
    require_login,
    load_existing_data,
    load_breakdown_report,
    save_breakdown_report,
    list_regdata_display_names,
    show_system_error,
    show_user_error,
)

st.set_page_config(page_title="Task Update", page_icon="🔧", layout="wide")
auth = require_login()


def _performed_by_label() -> str:
    name = str(auth.get("name", "") or "").strip()
    user_id = str(auth.get("user_id", "") or "").strip()
    return name or user_id or "System"


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
        return "InProgress"
    key = s.casefold()
    if key in {"open", "pending", "in progress", "in_progress", "inprogress"}:
        return "InProgress"
    if key in {"completed", "complete"}:
        return "Completed"
    if key in {"close", "closed"}:
        return "Close"
    return s


def _normalize_approval_status(value: object) -> str:
    s = str(value or "").strip()
    if not s or s.casefold() in {"none", "nan"}:
        return "Completed"
    key = s.casefold()
    # Legacy mappings
    if key == "pending":
        return "Completed"
    if key == "approved":
        return "Approved"
    # Current states
    if key in {"inprogress", "in progress", "in_progress"}:
        return "InProgress"
    if key in {"close", "closed"}:
        return "Approved"
    if key == "completed":
        return "Completed"
    return s


def _unique_job_ids(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty or "Job ID" not in df.columns:
        return []
    ids = df["Job ID"].astype(str).fillna("").map(lambda x: str(x).strip())
    ids = [x for x in ids.tolist() if x]
    return sorted(set(ids))
APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = APP_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "main_data.db"

# UPDATED: breakdown_report structure (stored in main_data.db)
COLUMNS = [
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
    "Duration",  # minutes (auto)
    "JobStatus",
    "Problem Description",
    "Job Title",
    "Job Description",
    "Remark",
    "Assign By",
    "Immediate Action",
    "Root Cause",
    "Preventive Action",
    "Spare Parts Used",
    "Reported By",
    "Created At",
    "Approval Status",
    "Approved By",
    "Approved At",
]

# Usage reference only (free text is allowed)
USAGE_OPTIONS = ["Equipment", "Machine", "Jig", "Fixture", "Tester"]


@st.cache_data(show_spinner=False)
def _get_machine_catalog() -> tuple[list[str], dict[str, str]]:
    df = load_existing_data()
    if df is None or df.empty:
        return ([], {})

    id_col = "Department ID" if "Department ID" in df.columns else None
    name_col = "Description of Asset" if "Description of Asset" in df.columns else None
    if not id_col:
        return ([], {})

    ids = df[id_col].astype(str).fillna("").map(lambda s: str(s).strip())
    names = df[name_col].astype(str).fillna("").map(lambda s: str(s).strip()) if name_col else pd.Series([""] * len(df))

    machine_map: dict[str, str] = {}
    for mid, mname in zip(ids.tolist(), names.tolist()):
        if not mid:
            continue
        if mid in machine_map:
            continue
        machine_map[mid] = mname

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
        ("Date/Time Start", _fmt_datetime_ddmmyy_hhmm(row.get("Date/Time Start"))),
    ]
    right_lines = [
        ("Job ID", str(row.get("Job ID", "") or "").strip()),
        ("Severity", str(row.get("Severity", "") or "").strip()),
        ("Location", str(row.get("Location", "") or "").strip()),
        ("Machine ID", str(row.get("Machine ID", "") or "").strip()),
        ("Date/Time End", _fmt_datetime_ddmmyy_hhmm(row.get("Date/Time End"))),
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
        ("Problem Description", row.get("Problem Description", "")),
        ("Immediate Action", row.get("Immediate Action", "")),
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
    reported_by = str(row.get("Reported By", "") or "").strip()
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

    # Storage Table
    # Rule:
    #   total_quantity = total_used + total_add
    # where:
    #   total_add  = current stock in store (available)
    #   total_used = total issued/consumed
    c.execute("""
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
    """)

    # --- schema migration for existing DBs ---
    c.execute("PRAGMA table_info(storage)")
    cols = [r[1] for r in c.fetchall()]

    if "usage" not in cols:
        c.execute("ALTER TABLE storage ADD COLUMN usage TEXT")
        c.execute("UPDATE storage SET usage = '' WHERE usage IS NULL")

    if "brand" not in cols:
        c.execute("ALTER TABLE storage ADD COLUMN brand TEXT")
        c.execute("UPDATE storage SET brand = '' WHERE brand IS NULL")

    if "model" not in cols:
        c.execute("ALTER TABLE storage ADD COLUMN model TEXT")
        c.execute("UPDATE storage SET model = '' WHERE model IS NULL")

    if "total_add" not in cols:
        c.execute("ALTER TABLE storage ADD COLUMN total_add INTEGER")
        # Backfill:
        # old logic used: available = total_quantity - total_used
        # now store available into total_add
        c.execute("""
            UPDATE storage
            SET total_add = COALESCE(total_quantity, 0) - COALESCE(total_used, 0)
            WHERE total_add IS NULL
        """)
        # Ensure non-negative
        c.execute("UPDATE storage SET total_add = 0 WHERE total_add < 0 OR total_add IS NULL")

    # Normalize totals to match rule: total_quantity = total_used + total_add
    c.execute("""
        UPDATE storage
        SET total_used = COALESCE(total_used, 0),
            total_add = COALESCE(total_add, 0),
            total_quantity = COALESCE(total_used, 0) + COALESCE(total_add, 0)
        WHERE total_quantity IS NULL
           OR total_used IS NULL
           OR total_add IS NULL
           OR total_quantity != (COALESCE(total_used, 0) + COALESCE(total_add, 0))
    """)

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
    if "usage" not in df.columns:
        df["usage"] = ""
    if "brand" not in df.columns:
        df["brand"] = ""
    if "model" not in df.columns:
        df["model"] = ""
    if "total_add" not in df.columns:
        # Backward compatibility: compute from old columns
        df["total_add"] = (df.get("total_quantity", 0) - df.get("total_used", 0)).clip(lower=0)

    # Enforce rule in memory (display consistency)
    df["total_quantity"] = (df["total_used"].fillna(0).astype(int) + df["total_add"].fillna(0).astype(int)).astype(int)
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
    total_add = int(total_add)
    total_used = 0
    total_quantity = total_used + total_add

    c.execute(
        """
        INSERT INTO storage (part_number, item_name, brand, model, specification, total_quantity, total_used, total_add, part_type, usage)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            part_number,
            item_name,
            str(brand or "").strip(),
            str(model or "").strip(),
            specification,
            total_quantity,
            total_used,
            total_add,
            part_type,
            usage,
        ),
    )
    conn.commit()

    # Inventory history (ADD)
    after_state = _fetch_storage_row(conn, part_number)
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
    total_add = int(total_add)
    total_used = int(total_used)
    total_quantity = total_used + total_add

    c.execute(
        """
        UPDATE storage
        SET item_name = ?,
            brand = ?,
            model = ?,
            specification = ?,
            total_quantity = ?,
            total_used = ?,
            total_add = ?,
            part_type = ?,
            usage = ?
        WHERE part_number = ?
        """,
        (
            item_name,
            str(brand or "").strip(),
            str(model or "").strip(),
            specification,
            total_quantity,
            total_used,
            total_add,
            part_type,
            usage,
            part_number,
        ),
    )
    conn.commit()

    # Inventory history (UPDATE)
    after_state = _fetch_storage_row(conn, part_number)
    log_inventory_history(
        action="UPDATE_PART",
        part_number=part_number,
        performed_by=performed_by,
        note=note,
        before_state=before_state,
        after_state=after_state,
    )

    try:
        persist_repo_changes([str(DB_PATH)], reason=f"Inventory UPDATE_PART {part_number}")
    except Exception:
        pass
    conn.close()


def _get_storage_totals(conn: sqlite3.Connection, part_number: str) -> tuple[int, int]:
    """
    Helper required by stock_in_add / stock_out_adjust / stock_out_task
    Returns (total_used, total_add) for a part_number.
    """
    pn = str(part_number or "").strip()
    if not pn:
        raise ValueError("Part Number is required")

    cur = conn.cursor()
    cur.execute(
        """
        SELECT COALESCE(total_used, 0), COALESCE(total_add, 0)
        FROM storage
        WHERE part_number = ?
        LIMIT 1
        """,
        (pn,),
    )
    row = cur.fetchone()
    if not row:
        raise ValueError(f"Part not found: {pn}")

    return int(row[0]), int(row[1])


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
        before_used, before_add = _get_storage_totals(conn, part_number)

        after_add = before_add + qty_in
        after_used = before_used
        after_qty = after_used + after_add

        cur = conn.cursor()
        cur.execute(
            "UPDATE storage SET total_add = ?, total_quantity = ? WHERE part_number = ?",
            (after_add, after_qty, part_number),
        )
        conn.commit()

        after_state = _fetch_storage_row(conn, part_number)

        # Merge stock_log into inventory history
        log_inventory_history(
            action="IN_ADD",
            part_number=part_number,
            performed_by=performed_by,
            note=note,
            before_state=before_state,
            after_state=after_state,
        )

        log_stock_operation(
            action="IN_ADD",
            part_number=part_number,
            qty=qty_in,
            before_total_add=before_add,
            after_total_add=after_add,
            before_total_used=before_used,
            after_total_used=after_used,
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
        before_used, before_add = _get_storage_totals(conn, part_number)
        if qty_out > before_add:
            raise ValueError("Not enough available stock")

        after_add = before_add - qty_out
        after_used = before_used
        after_qty = after_used + after_add

        cur = conn.cursor()
        cur.execute(
            "UPDATE storage SET total_add = ?, total_quantity = ? WHERE part_number = ?",
            (after_add, after_qty, part_number),
        )
        conn.commit()

        after_state = _fetch_storage_row(conn, part_number)

        # Merge stock_log into inventory history
        log_inventory_history(
            action="OUT_ADJUST",
            part_number=part_number,
            performed_by=performed_by,
            note=note,
            before_state=before_state,
            after_state=after_state,
        )

        log_stock_operation(
            action="OUT_ADJUST",
            part_number=part_number,
            qty=qty_out,
            before_total_add=before_add,
            after_total_add=after_add,
            before_total_used=before_used,
            after_total_used=after_used,
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
        before_used, before_add = _get_storage_totals(conn, part_number)
        if qty_used > before_add:
            raise ValueError("Not enough available stock")

        after_used = before_used + qty_used
        after_add = before_add - qty_used
        after_qty = after_used + after_add

        cur = conn.cursor()
        cur.execute(
            "UPDATE storage SET total_used = ?, total_add = ?, total_quantity = ? WHERE part_number = ?",
            (after_used, after_add, after_qty, part_number),
        )
        conn.commit()

        after_state = _fetch_storage_row(conn, part_number)

        # Merge stock_log into inventory history
        log_inventory_history(
            action="OUT_TASK",
            part_number=part_number,
            performed_by=performed_by,
            note=note,
            before_state=before_state,
            after_state=after_state,
        )

        log_stock_operation(
            action="OUT_TASK",
            part_number=part_number,
            qty=qty_used,
            before_total_add=before_add,
            after_total_add=after_add,
            before_total_used=before_used,
            after_total_used=after_used,
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
            SELECT COALESCE(total_used, 0), COALESCE(total_add, 0)
            FROM storage
            WHERE part_number = ?
            LIMIT 1
            """,
            (pn,),
        )
        row = c.fetchone()
        if not row:
            raise ValueError(f"Part not found: {pn}")

        before_used = int(row[0])
        before_add = int(row[1])

        c.execute("DELETE FROM storage WHERE part_number = ?", (pn,))
        if c.rowcount <= 0:
            raise ValueError(f"Part not found: {pn}")

        conn.commit()

        log_inventory_history(
            action="DELETE_PART",
            part_number=pn,
            performed_by=performed_by,
            note=note,
            before_state=before_state,
            after_state=None,
        )

        log_stock_operation(
            action="DELETE",
            part_number=pn,
            qty=0,
            before_total_add=before_add,
            after_total_add=0,
            before_total_used=before_used,
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
        df = load_breakdown_report()
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[COLUMNS]
    except Exception as e:
        show_system_error("Failed to load breakdown report from database.", e, context="TaskReport.load_breakdown_data")
        return pd.DataFrame(columns=COLUMNS)


def _save_breakdown_data(df: pd.DataFrame) -> bool:
    try:
        out = df.copy() if df is not None else pd.DataFrame(columns=COLUMNS)
        for col in COLUMNS:
            if col not in out.columns:
                out[col] = ""
        out = out[COLUMNS]
        return bool(save_breakdown_report(out))
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

    yy = str(entry_date.year)[-2:]
    mm = f"{entry_date.month:02d}"
    dd = f"{entry_date.day:02d}"
    date_str = f"{yy}/{mm}/{dd}"
    prefix = f"{date_str}_{type_code}_"

    if existing_df is None or existing_df.empty or "Job ID" not in existing_df.columns:
        return f"{prefix}01"

    job_ids = existing_df["Job ID"].astype(str).fillna("").tolist()
    max_n = 0
    for jid in job_ids:
        jid = str(jid or "").strip()
        if not jid.startswith(prefix):
            continue
        tail = jid[len(prefix):].strip()
        try:
            max_n = max(max_n, int(tail))
        except Exception:
            continue

    return f"{prefix}{(max_n + 1):02d}"


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

    total_add = int(total_add)
    total_used = int(total_used)
    total_quantity = total_used + total_add

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
                total_quantity = ?,
                total_used = ?,
                total_add = ?,
                part_type = ?,
                usage = ?
            WHERE part_number = ?
            """,
            (
                new_pn,
                item_name,
                str(brand or "").strip(),
                str(model or "").strip(),
                specification,
                int(total_quantity),
                int(total_used),
                int(total_add),
                str(part_type or "").strip(),
                str(usage or "").strip(),
                old_pn,
            ),
        )
        if cur.rowcount <= 0:
            raise ValueError(f"Part not found: {old_pn}")

        conn.commit()

        after_state = _fetch_storage_row(conn, new_pn)

        action = "RENUMBER_PART" if old_pn != new_pn else "UPDATE_PART"
        extra = f"old_pn={old_pn}" if old_pn != new_pn else ""
        combined_note = (str(note or "").strip() + (" | " + extra if extra else "")).strip(" |")

        log_inventory_history(
            action=action,
            part_number=new_pn,
            performed_by=performed_by,
            note=combined_note,
            before_state=before_state,
            after_state=after_state,
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
st.markdown("---")

tab_generate, tab_review = st.tabs(["📝 Task Entry", "📋 Review Entries"])

# ================= TAB 1: TASK ENTRY =================
with tab_generate:
    st.markdown("### New Task")

    if "spare_parts" not in st.session_state:
        st.session_state.spare_parts = []

    existing_df = load_breakdown_data()

    # Job Type
    # Session-state migration: old value "Other(s)" -> "General"
    if str(st.session_state.get("br_job_type", "") or "").strip().casefold() in {"other", "others"}:
        st.session_state["br_job_type"] = "General"

    job_type = st.selectbox(
        "Job Type *",
        options=["Breakdown", "Maintenance", "General"],
        key="br_job_type",
    )
    job_type_norm = _normalize_job_type(job_type)

    # Date | Job ID (auto) | Shift
    col_d, col_jid, col_shift = st.columns(3)
    with col_d:
        entry_date = st.date_input("Date *", value=date.today(), key="br_date")
    with col_jid:
        job_id = generate_job_id(entry_date, job_type_norm, existing_df)
        st.text_input("Job ID (auto)", value=job_id, disabled=True, key="br_job_id_display")
    with col_shift:
        shift = st.selectbox("Shift *", options=["Day", "Night"], key="br_shift")

    severity_options = ["", "Low", "Medium", "High", "Critical"]

    # Machine catalogue (Asset list)
    machine_id_options, machine_map = _get_machine_catalog()

    assign_by = ""

    # === Breakdown ===
    if job_type_norm == "Breakdown":
        col_ts, col_te, col_sev, col_loc = st.columns(4)
        with col_ts:
            time_start = st.time_input("Time Start *", value=datetime.now().time(), key="br_time_start")
        with col_te:
            time_end = st.time_input("Time End *", value=datetime.now().time(), key="br_time_end")
        with col_sev:
            severity = st.selectbox("Severity", options=severity_options, key="br_severity")
        with col_loc:
            location = st.text_input("Location *", key="br_location")

        start_dt = datetime.combine(entry_date, time_start)
        end_dt = datetime.combine(entry_date, time_end)
        duration_err = None
        duration_min = 0
        if end_dt < start_dt:
            duration_err = "Time End must be after Time Start."
        else:
            duration_min = int((end_dt - start_dt).total_seconds() // 60)

        col_mid, col_mname, col_status = st.columns(3)
        with col_mid:
            machine_not_in_list = st.checkbox("Machine ID not in list", value=False, key="br_machine_not_in_list")
            if machine_not_in_list or not machine_id_options:
                machine_id = st.text_input("Machine ID *", key="br_machine_id_manual")
            else:
                machine_id = st.selectbox("Machine ID *", options=machine_id_options, key="br_machine_id_sel")

        with col_mname:
            if machine_not_in_list or not machine_id_options:
                machine_name = st.text_input("Machine Name *", key="br_machine_name_manual")
            else:
                machine_name = str(machine_map.get(str(machine_id).strip(), "") or "").strip()
                st.text_input("Machine Name", value=machine_name, disabled=True, key="br_machine_name_display")

        with col_status:
            job_status = st.selectbox("Job Status *", options=["InProgress", "Completed"], key="br_job_status")

        problem_description = st.text_area("Problem Description *", height=120, key="br_problem_desc")
        root_cause = st.text_area("Root Cause *", height=120, key="br_root")
        immediate_action = st.text_area("Immediate Action *", height=120, key="br_immediate")
        preventive_action = st.text_area("Preventive Action *", height=120, key="br_preventive")

    # === Maintenance ===
    elif job_type_norm == "Maintenance":
        col_ts, col_te, col_sev, col_loc = st.columns(4)
        with col_ts:
            time_start = st.time_input("Time Start *", value=datetime.now().time(), key="br_time_start")
        with col_te:
            time_end = st.time_input("Time End *", value=datetime.now().time(), key="br_time_end")
        with col_sev:
            severity = st.selectbox("Severity", options=severity_options, key="br_severity")
        with col_loc:
            location = st.text_input("Location *", key="br_location")

        start_dt = datetime.combine(entry_date, time_start)
        end_dt = datetime.combine(entry_date, time_end)
        duration_err = None
        duration_min = 0
        if end_dt < start_dt:
            duration_err = "Time End must be after Time Start."
        else:
            duration_min = int((end_dt - start_dt).total_seconds() // 60)

        col_mid, col_mname, col_status = st.columns(3)
        with col_mid:
            machine_not_in_list = st.checkbox("Machine ID not in list", value=False, key="br_machine_not_in_list")
            if machine_not_in_list or not machine_id_options:
                machine_id = st.text_input("Machine ID *", key="br_machine_id_manual")
            else:
                machine_id = st.selectbox("Machine ID *", options=machine_id_options, key="br_machine_id_sel")

        with col_mname:
            if machine_not_in_list or not machine_id_options:
                machine_name = st.text_input("Machine Name *", key="br_machine_name_manual")
            else:
                machine_name = str(machine_map.get(str(machine_id).strip(), "") or "").strip()
                st.text_input("Machine Name", value=machine_name, disabled=True, key="br_machine_name_display")

        with col_status:
            job_status = st.selectbox("Job Status *", options=["InProgress", "Completed"], key="br_job_status")

        task_description = st.text_area("Task Description *", height=120, key="br_task_desc")
        action = st.text_area("Action *", height=120, key="br_action")

        # Keep non-applicable breakdown fields empty
        problem_description = str(task_description or "").strip()
        root_cause = ""
        immediate_action = str(action or "").strip()
        preventive_action = ""

    # === General ===
    else:
        col_ts, col_te, col_sev = st.columns(3)
        with col_ts:
            time_start = st.time_input("Time Start *", value=datetime.now().time(), key="br_time_start")
        with col_te:
            time_end = st.time_input("Time End *", value=datetime.now().time(), key="br_time_end")
        with col_sev:
            severity = st.selectbox("Severity", options=severity_options, key="br_severity")

        start_dt = datetime.combine(entry_date, time_start)
        end_dt = datetime.combine(entry_date, time_end)
        duration_err = None
        duration_min = 0
        if end_dt < start_dt:
            duration_err = "Time End must be after Time Start."
        else:
            duration_min = int((end_dt - start_dt).total_seconds() // 60)

        col_loc, col_status, col_assign = st.columns(3)
        with col_loc:
            location = st.text_input("Location *", key="br_location")
        with col_status:
            job_status = st.selectbox("Job Status *", options=["InProgress", "Completed"], key="br_job_status")
        with col_assign:
            if _current_level_rank() >= 2:
                options = list_regdata_display_names() or []
                if not options:
                    assign_by = _performed_by_label()
                    st.text_input("Assign By", value=assign_by, disabled=True, key="br_assign_by_display")
                else:
                    default_val = _performed_by_label()
                    if default_val not in options:
                        options = [default_val] + options
                    assign_by = st.selectbox("Assign By *", options=options, key="br_assign_by")
            else:
                assign_by = _performed_by_label()
                st.text_input("Assign By", value=assign_by, disabled=True, key="br_assign_by_display")

        task_description = st.text_area("Task Description *", height=120, key="br_task_desc")
        action = st.text_area("Action *", height=120, key="br_action")

        # General has no machine fields
        machine_id = ""
        machine_name = ""
        machine_not_in_list = False

        problem_description = str(task_description or "").strip()
        root_cause = ""
        immediate_action = str(action or "").strip()
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

    if st.button("✅ Submit Report", type="primary"):
        if duration_err:
            show_user_error(duration_err)
        elif not str(shift).strip():
            show_user_error("Shift is required.")
        elif not str(location).strip():
            show_user_error("Location is required.")
        elif job_type_norm in {"Breakdown", "Maintenance"} and not str(machine_id).strip():
            show_user_error("Machine ID is required.")
        elif job_type_norm in {"Breakdown", "Maintenance"} and (machine_not_in_list or not machine_id_options) and not str(machine_name).strip():
            show_user_error("Machine Name is required.")
        elif job_type_norm == "Breakdown" and not str(problem_description).strip():
            show_user_error("Problem Description is required.")
        elif job_type_norm == "Breakdown" and not str(root_cause).strip():
            show_user_error("Root Cause is required.")
        elif job_type_norm == "Breakdown" and not str(immediate_action).strip():
            show_user_error("Immediate Action is required.")
        elif job_type_norm == "Breakdown" and not str(preventive_action).strip():
            show_user_error("Preventive Action is required.")
        elif job_type_norm in {"Maintenance", "General"} and not str(problem_description).strip():
            show_user_error("Task Description is required.")
        elif job_type_norm in {"Maintenance", "General"} and not str(immediate_action).strip():
            show_user_error("Action is required.")
        elif job_type_norm == "General" and _current_level_rank() >= 2 and not str(assign_by).strip():
            show_user_error("Assign By is required.")
        elif spare_used and not st.session_state.spare_parts:
            show_user_error("You ticked 'Spare parts used?' but did not add any parts.")
        else:
            reported_by_name = _performed_by_label()

            df = load_breakdown_data()
            now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            new_entry = {
                "Date": entry_date.strftime("%Y-%m-%d"),
                "Job ID": job_id,
                "Job Type": job_type_norm,
                "Severity": str(severity or "").strip(),
                "Shift": str(shift).strip(),
                "Location": str(location).strip(),
                "Machine/Equipment": str(machine_name or "").strip(),
                "Machine ID": str(machine_id or "").strip(),
                "Date/Time Start": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "Date/Time End": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "Duration": int(duration_min),
                "JobStatus": _normalize_job_status(job_status),
                "Problem Description": str(problem_description or "").strip(),
                "Job Title": "",
                "Job Description": str(problem_description or "").strip(),
                "Remark": "",
                "Assign By": str(assign_by or "").strip() if job_type_norm == "General" else "",
                "Immediate Action": str(immediate_action or "").strip(),
                "Root Cause": str(root_cause or "").strip(),
                "Preventive Action": str(preventive_action or "").strip(),
                "Spare Parts Used": (
                    " | ".join([f"{p['part_number']}:{p['name']} x{p['qty']}" for p in st.session_state.spare_parts])
                    if spare_used
                    else ""
                ),
                "Reported By": reported_by_name,
                "Created At": now_ts,
                "Approval Status": ("InProgress" if _normalize_job_status(job_status) == "InProgress" else "Completed"),
                "Approved By": "",
                "Approved At": "",
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
                        performed_by=reported_by_name,
                        note=f"JobID={job_id}",
                    )
                except Exception as e:
                    st.error(f"Stock OUT (Task) failed for {part['part_number']}: {e}")
                    st.stop()

            for key in [
                "br_job_type",
                "br_date",
                "br_job_id_display",
                "br_shift",
                "br_time_start",
                "br_time_end",
                "br_severity",
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
                "br_task_desc",
                "br_action",
                "br_assign_by",
                "br_assign_by_display",
                "br_spare_used",
                "br_sp_select_name",
                "br_sp_qty_used",
                "spare_parts",
            ]:
                if key in st.session_state:
                    del st.session_state[key]

            if _normalize_job_status(job_status) == "InProgress":
                st.success("Task submitted (InProgress) & stock updated!")
            else:
                st.success("Task submitted (Completed - needs approver) & stock updated!")
            st.rerun()

# ================= TAB 2: REVIEW ENTRIES =================
with tab_review:
    st.markdown("### Review task reports")
    df = load_breakdown_data()
    if df.empty:
        st.info("No task entries yet. Use **Task Entry** to add one.")
    else:
        # Normalize job type + approval defaults
        if "Job Type" in df.columns:
            df["Job Type"] = df["Job Type"].astype(str).map(_normalize_job_type)
        if "JobStatus" not in df.columns:
            df["JobStatus"] = "InProgress"
        df["JobStatus"] = df["JobStatus"].map(_normalize_job_status)
        if "Approval Status" not in df.columns:
            df["Approval Status"] = "Completed"

        df["Approval Status"] = df["Approval Status"].map(_normalize_approval_status)

        # Keep approval status consistent with job status:
        # - InProgress: approval not required
        # - Completed: requires approver to Close
        # - Close: already approved/closed
        try:
            df.loc[df["JobStatus"] == "InProgress", "Approval Status"] = "InProgress"
            df.loc[df["JobStatus"] == "Close", "Approval Status"] = "Approved"
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
            filter_approval = st.selectbox("Filter by Approval", options=["All", "InProgress", "Completed", "Approved"], key="br_filter_approval")
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
                + review_df.get("Problem Description", "").astype(str)
            )
            review_df = review_df[hay.str.contains(kw, case=False, na=False)]

        # Main view
        show_cols = [
            c
            for c in [
                "Date",
                "Job ID",
                "Job Type",
                "JobStatus",
                "Approval Status",
                "Severity",
                "Shift",
                "Date/Time Start",
                "Date/Time End",
                "Location",
                "Machine ID",
                "Machine/Equipment",
                "Assign By",
                "Reported By",
                "Created At",
            ]
            if c in review_df.columns
        ]
        # Fallback to core columns present
        core_cols = [c for c in ["Date", "Job ID", "Job Type", "JobStatus", "Approval Status", "Problem Description"] if c in review_df.columns]
        view_cols = show_cols if show_cols else core_cols
        # Sort newest-first (best-effort)
        view_df = review_df.copy()
        try:
            if "Created At" in view_df.columns:
                view_df["__created_sort"] = pd.to_datetime(view_df["Created At"], errors="coerce")
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
        st.dataframe(table_df, use_container_width=True, hide_index=True)

        # Approver action (SuperUser) - tick selection only shows eligible rows
        if _current_level_rank() >= 2 and not df.empty and "Job ID" in df.columns:
            tmp = df.copy()
            tmp["JobStatus"] = tmp.get("JobStatus", "").map(_normalize_job_status)
            tmp["Approval Status"] = tmp.get("Approval Status", "").map(_normalize_approval_status)
            eligible_df = tmp.loc[
                (tmp["JobStatus"].astype(str) == "Completed")
                & (tmp["Approval Status"].astype(str) == "Completed")
            ].copy()

            eligible_ids = _unique_job_ids(eligible_df)
            if eligible_ids:
                st.markdown("#### ✅ Approver (SuperUser)")
                st.caption("Tick rows below to approve. Approved rows are not selectable.")

                # Build a compact selection table from the current view (filtered) when possible.
                selector_source = view_df.copy()
                if "Job ID" in selector_source.columns:
                    selector_source = selector_source[selector_source["Job ID"].astype(str).isin(eligible_ids)]
                else:
                    selector_source = eligible_df

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
                        "Reported By",
                        "Created At",
                    ]
                    if c in selector_source.columns
                ]
                selector_df = selector_source[selector_cols].copy() if selector_cols else selector_source.copy()
                selector_df.insert(0, "Approve", False)

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
                    if _current_level_rank() < 2:
                        show_user_error("Only SuperUser can approve reports.")
                        st.stop()
                    if not selected_ids:
                        show_user_error("Tick at least one row.")
                        st.stop()

                    allowed = [jid for jid in selected_ids if jid in set(eligible_ids)]
                    if not allowed:
                        show_user_error("Selected rows are not eligible.")
                        st.stop()

                    base = load_breakdown_data()
                    mask = base["Job ID"].astype(str).isin([str(x) for x in allowed])
                    if not mask.any():
                        st.error("No matching Job IDs found.")
                        st.stop()

                    base.loc[mask, "JobStatus"] = "Close"
                    base.loc[mask, "Approval Status"] = "Approved"
                    base.loc[mask, "Approved By"] = _performed_by_label()
                    base.loc[mask, "Approved At"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if not _save_breakdown_data(base):
                        st.error("Failed to approve submission(s).")
                        st.stop()

                    st.success(f"Approved {int(mask.sum())} submission(s).")
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

                    # Approve action (SuperUser) for the currently viewed report
                    job_id_view = str(r.get("Job ID", "") or "").strip()
                    job_status_view = _normalize_job_status(r.get("JobStatus", ""))
                    approval_view = _normalize_approval_status(r.get("Approval Status", ""))

                    if _current_level_rank() >= 2 and job_id_view:
                        a1, a2 = st.columns([1, 3])
                        with a1:
                            can_approve = (job_status_view == "Completed") and (approval_view == "Completed")
                            already_approved = (approval_view == "Approved") or (job_status_view == "Close")

                            approve_disabled = (not can_approve) or already_approved
                            if st.button(
                                "✅ Approve",
                                type="primary",
                                disabled=approve_disabled,
                                key=f"br_view_approve_{job_id_view}",
                            ):
                                if _current_level_rank() < 2:
                                    show_user_error("Only SuperUser can approve reports.")
                                    st.stop()
                                base = load_breakdown_data()
                                mask = base["Job ID"].astype(str) == str(job_id_view)
                                if not mask.any():
                                    st.error("Job ID not found.")
                                    st.stop()

                                # Re-check eligibility against latest DB
                                cur_status = _normalize_job_status(base.loc[mask, "JobStatus"].iloc[0] if "JobStatus" in base.columns else "")
                                cur_approval = _normalize_approval_status(base.loc[mask, "Approval Status"].iloc[0] if "Approval Status" in base.columns else "")
                                if cur_approval == "Approved" or cur_status == "Close":
                                    st.info("This report is already approved.")
                                    st.stop()
                                if cur_status != "Completed" or cur_approval != "Completed":
                                    show_user_error("Only Completed reports can be approved.")
                                    st.stop()

                                base.loc[mask, "JobStatus"] = "Close"
                                base.loc[mask, "Approval Status"] = "Approved"
                                base.loc[mask, "Approved By"] = _performed_by_label()
                                base.loc[mask, "Approved At"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                if not _save_breakdown_data(base):
                                    st.error("Failed to approve submission.")
                                    st.stop()
                                st.success("Approved.")
                                st.rerun()

                        with a2:
                            if approval_view == "Approved" or job_status_view == "Close":
                                st.success("Approved")
                            elif job_status_view != "Completed" or approval_view != "Completed":
                                st.info("Approve is available only when Job Status = Completed.")

                    # Table-style view (like before)
                    ordered_fields = [
                        "Date",
                        "Job ID",
                        "Job Type",
                        "JobStatus",
                        "Approval Status",
                        "Approved By",
                        "Approved At",
                        "Severity",
                        "Shift",
                        "Location",
                        "Machine ID",
                        "Machine/Equipment",
                        "Date/Time Start",
                        "Date/Time End",
                        "Duration",
                        "Assign By",
                        "Reported By",
                        "Created At",
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

                    # Long text fields below (read-only)
                    st.markdown("##### Content")
                    st.text_area("Problem / Task Description", value=str(r.get("Problem Description", "") or ""), height=120, disabled=True)
                    st.text_area("Immediate Action / Action", value=str(r.get("Immediate Action", "") or ""), height=120, disabled=True)
                    if _normalize_job_type(str(r.get("Job Type", "") or "").strip()) == "Breakdown":
                        st.text_area("Root Cause", value=str(r.get("Root Cause", "") or ""), height=120, disabled=True)
                        st.text_area("Preventive Action", value=str(r.get("Preventive Action", "") or ""), height=120, disabled=True)
                    st.text_area("Spare Parts Used", value=str(r.get("Spare Parts Used", "") or ""), height=80, disabled=True)

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
                file_name=f"task_report_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
            csv_all = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ Download (full CSV)",
                data=csv_all,
                file_name=f"task_report_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
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

        st.markdown("---")

        # Edit InProgress (toggle)
        st.markdown("#### InProgress - Edit report")
        col_edit_1, col_edit_2 = st.columns(2)
        with col_edit_1:
            st.caption("Editable only when Job Status = InProgress.")
        with col_edit_2:
            if st.button("✏️ Edit"):
                st.session_state.show_task_report_edit = not st.session_state.get("show_task_report_edit", False)
                st.rerun()

        if not st.session_state.get("show_task_report_edit", False):
            st.info("Edit section is hidden.")
            st.stop()

        st.markdown("##### Select Report to Edit")

        inprogress_df = df.copy()
        if "JobStatus" in inprogress_df.columns:
            inprogress_df["JobStatus"] = inprogress_df["JobStatus"].map(_normalize_job_status)
            inprogress_df = inprogress_df[inprogress_df["JobStatus"].astype(str) == "InProgress"]
        else:
            inprogress_df = inprogress_df.iloc[0:0]

        inprogress_ids = _unique_job_ids(inprogress_df)
        if not inprogress_ids:
            st.info("No InProgress reports available to edit.")
        else:
            edit_job_id = st.selectbox("Select Job ID", options=inprogress_ids, key="br_edit_job_id")
            base = load_breakdown_data()
            row_df = base[base["Job ID"].astype(str) == str(edit_job_id)].head(1)
            if row_df.empty:
                st.info("Job not found.")
            else:
                row = row_df.iloc[0].to_dict()
                row_job_type = _normalize_job_type(str(row.get("Job Type", "") or "").strip())
                row_status = _normalize_job_status(row.get("JobStatus", ""))
                row_reported_by = str(row.get("Reported By", "") or "").strip()
                can_edit = row_status == "InProgress" and (
                    _current_level_rank() >= 2 or row_reported_by == _performed_by_label()
                )

                st.caption(f"Status: {row_status or '—'} | Approval: {_normalize_approval_status(row.get('Approval Status',''))}")

                # Common editable fields
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.text_input("Job Type", value=row_job_type, disabled=True)
                with col2:
                    st.text_input("Job ID", value=str(edit_job_id), disabled=True)
                with col3:
                    shift_v = st.selectbox("Shift", options=["Day", "Night"], index=0 if str(row.get("Shift", "Day")) == "Day" else 1, disabled=not can_edit, key="br_edit_shift")

                # Time Start/End
                def _safe_time(val: str, fallback: time) -> time:
                    try:
                        dt = pd.to_datetime(val, errors="coerce")
                        if pd.isna(dt):
                            return fallback
                        return dt.to_pydatetime().time()
                    except Exception:
                        return fallback

                entry_date = pd.to_datetime(row.get("Date", ""), errors="coerce").date() if str(row.get("Date", "")).strip() else date.today()
                ts_default = _safe_time(row.get("Date/Time Start", ""), datetime.now().time())
                te_default = _safe_time(row.get("Date/Time End", ""), datetime.now().time())

                col_ts, col_te, col_sev = st.columns(3)
                with col_ts:
                    edit_time_start = st.time_input("Time Start", value=ts_default, disabled=not can_edit, key="br_edit_time_start")
                with col_te:
                    edit_time_end = st.time_input("Time End", value=te_default, disabled=not can_edit, key="br_edit_time_end")
                with col_sev:
                    edit_sev = st.selectbox(
                        "Severity",
                        options=["", "Low", "Medium", "High", "Critical"],
                        index=["", "Low", "Medium", "High", "Critical"].index(str(row.get("Severity", "") or "")) if str(row.get("Severity", "") or "") in {"", "Low", "Medium", "High", "Critical"} else 0,
                        disabled=not can_edit,
                        key="br_edit_severity",
                    )

                # Location / Status / Assign By
                col_loc, col_status, col_assign = st.columns(3)
                with col_loc:
                    edit_loc = st.text_input("Location", value=str(row.get("Location", "") or ""), disabled=not can_edit, key="br_edit_location")
                with col_status:
                    if row_status == "Close":
                        edit_status = "Close"
                        st.text_input("Job Status", value="Close", disabled=True)
                    else:
                        edit_status = st.selectbox(
                            "Job Status",
                            options=["InProgress", "Completed"],
                            index=["InProgress", "Completed"].index(row_status) if row_status in {"InProgress", "Completed"} else 0,
                            disabled=not can_edit,
                            key="br_edit_job_status",
                        )
                with col_assign:
                    if row_job_type == "General":
                        if _current_level_rank() >= 2:
                            opts = list_regdata_display_names() or []
                            cur_val = str(row.get("Assign By", "") or "").strip() or _performed_by_label()
                            if cur_val and cur_val not in opts:
                                opts = [cur_val] + opts
                            if not opts:
                                st.text_input("Assign By", value=cur_val, disabled=True)
                                edit_assign = cur_val
                            else:
                                edit_assign = st.selectbox("Assign By", options=opts, index=opts.index(cur_val) if cur_val in opts else 0, disabled=not can_edit, key="br_edit_assign_by")
                        else:
                            edit_assign = str(row.get("Assign By", "") or _performed_by_label()).strip()
                            st.text_input("Assign By", value=edit_assign, disabled=True)
                    else:
                        edit_assign = ""
                        st.write("")

                # Machine fields for Breakdown/Maintenance
                machine_id_options, machine_map = _get_machine_catalog()
                if row_job_type in {"Breakdown", "Maintenance"}:
                    col_mid, col_mname, _col_sp = st.columns(3)
                    with col_mid:
                        existing_mid = str(row.get("Machine ID", "") or "").strip()
                        existing_mname = str(row.get("Machine/Equipment", "") or "").strip()
                        use_manual = st.checkbox("Machine ID not in list", value=existing_mid not in set(machine_id_options), disabled=not can_edit, key="br_edit_machine_manual")
                        if use_manual or not machine_id_options:
                            edit_mid = st.text_input("Machine ID", value=existing_mid, disabled=not can_edit, key="br_edit_machine_id")
                        else:
                            if existing_mid and existing_mid in machine_id_options:
                                edit_mid = st.selectbox("Machine ID", options=machine_id_options, index=machine_id_options.index(existing_mid), disabled=not can_edit, key="br_edit_machine_id")
                            else:
                                edit_mid = st.selectbox("Machine ID", options=machine_id_options, disabled=not can_edit, key="br_edit_machine_id")

                    with col_mname:
                        if use_manual or not machine_id_options:
                            edit_mname = st.text_input("Machine Name", value=existing_mname, disabled=not can_edit, key="br_edit_machine_name")
                        else:
                            edit_mname = str(machine_map.get(str(edit_mid).strip(), "") or "").strip()
                            st.text_input("Machine Name", value=edit_mname, disabled=True)
                    with _col_sp:
                        st.write("")
                else:
                    edit_mid = ""
                    edit_mname = ""

                # Description fields
                if row_job_type == "Breakdown":
                    edit_prob = st.text_area("Problem Description", value=str(row.get("Problem Description", "") or ""), height=120, disabled=not can_edit, key="br_edit_problem")
                    edit_root = st.text_area("Root Cause", value=str(row.get("Root Cause", "") or ""), height=120, disabled=not can_edit, key="br_edit_root")
                    edit_immediate = st.text_area("Immediate Action", value=str(row.get("Immediate Action", "") or ""), height=120, disabled=not can_edit, key="br_edit_immediate")
                    edit_prev = st.text_area("Preventive Action", value=str(row.get("Preventive Action", "") or ""), height=120, disabled=not can_edit, key="br_edit_prev")
                else:
                    edit_prob = st.text_area("Task Description", value=str(row.get("Problem Description", "") or ""), height=120, disabled=not can_edit, key="br_edit_problem")
                    edit_immediate = st.text_area("Action", value=str(row.get("Immediate Action", "") or ""), height=120, disabled=not can_edit, key="br_edit_immediate")
                    edit_root = ""
                    edit_prev = ""

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

                if st.button("Save changes", type="primary", disabled=not can_edit, key="br_edit_save"):
                    # Validate
                    if not str(edit_loc).strip():
                        show_user_error("Location is required.")
                        st.stop()
                    if row_job_type in {"Breakdown", "Maintenance"}:
                        if not str(edit_mid).strip():
                            show_user_error("Machine ID is required.")
                            st.stop()
                        if st.session_state.get("br_edit_machine_manual") and not str(edit_mname).strip():
                            show_user_error("Machine Name is required.")
                            st.stop()
                    if not str(edit_prob).strip():
                        show_user_error("Description is required.")
                        st.stop()
                    if row_job_type == "Breakdown":
                        if not str(edit_root).strip() or not str(edit_immediate).strip() or not str(edit_prev).strip():
                            show_user_error("Root Cause / Immediate Action / Preventive Action are required.")
                            st.stop()
                    else:
                        if not str(edit_immediate).strip():
                            show_user_error("Action is required.")
                            st.stop()

                    start_dt = datetime.combine(entry_date, edit_time_start)
                    end_dt = datetime.combine(entry_date, edit_time_end)
                    if end_dt < start_dt:
                        show_user_error("Time End must be after Time Start.")
                        st.stop()
                    duration_min = int((end_dt - start_dt).total_seconds() // 60)

                    # Update row in base DF
                    mask = base["Job ID"].astype(str) == str(edit_job_id)
                    if not mask.any():
                        st.error("Job ID not found.")
                        st.stop()

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

                    performed_by = _performed_by_label()
                    if old_has_pn:
                        for pn in set(old_map.keys()) | set(new_map.keys()):
                            diff = int(new_map.get(pn, 0) - old_map.get(pn, 0))
                            if diff > 0:
                                stock_out_task(part_number=pn, qty_used=diff, performed_by=performed_by, note=f"JobID={edit_job_id} EDIT")
                            elif diff < 0:
                                stock_in_add(part_number=pn, qty_in=abs(diff), performed_by=performed_by, note=f"JobID={edit_job_id} EDIT")

                    next_status = _normalize_job_status(edit_status)
                    if next_status == "Close":
                        show_user_error("Only SuperUser can approve/close reports. Please ask a SuperUser to approve the submission.")
                        st.stop()

                    base.loc[mask, "Shift"] = str(shift_v).strip()
                    base.loc[mask, "Severity"] = str(edit_sev or "").strip()
                    base.loc[mask, "Location"] = str(edit_loc).strip()
                    base.loc[mask, "JobStatus"] = next_status
                    base.loc[mask, "Assign By"] = str(edit_assign or "").strip() if row_job_type == "General" else ""
                    base.loc[mask, "Date/Time Start"] = start_dt.strftime("%Y-%m-%d %H:%M:%S")
                    base.loc[mask, "Date/Time End"] = end_dt.strftime("%Y-%m-%d %H:%M:%S")
                    base.loc[mask, "Duration"] = int(duration_min)

                    if row_job_type in {"Breakdown", "Maintenance"}:
                        base.loc[mask, "Machine ID"] = str(edit_mid).strip()
                        base.loc[mask, "Machine/Equipment"] = str(edit_mname).strip()
                    else:
                        base.loc[mask, "Machine ID"] = ""
                        base.loc[mask, "Machine/Equipment"] = ""

                    base.loc[mask, "Problem Description"] = str(edit_prob).strip()
                    base.loc[mask, "Job Description"] = str(edit_prob).strip()
                    base.loc[mask, "Immediate Action"] = str(edit_immediate).strip()
                    base.loc[mask, "Root Cause"] = str(edit_root).strip() if row_job_type == "Breakdown" else ""
                    base.loc[mask, "Preventive Action"] = str(edit_prev).strip() if row_job_type == "Breakdown" else ""

                    if old_has_pn:
                        base.loc[mask, "Spare Parts Used"] = _spares_to_str(edit_spares_list)

                    # Reset approval on edit
                    if next_status == "InProgress":
                        base.loc[mask, "Approval Status"] = "InProgress"
                    elif next_status == "Completed":
                        base.loc[mask, "Approval Status"] = "Completed"
                    base.loc[mask, "Approved By"] = ""
                    base.loc[mask, "Approved At"] = ""

                    if not _save_breakdown_data(base):
                        st.error("Failed to save changes.")
                        st.stop()

                    if next_status == "InProgress":
                        st.success("Saved.")
                    else:
                        st.success("Saved. Now needs approver to Close.")
                    st.rerun()