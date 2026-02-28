import streamlit as st
import pandas as pd
from pathlib import Path
import sqlite3
from datetime import datetime, date, time
from utils import (
    ensure_data_directory,
    initialize_stock_log_database,
    log_stock_operation,
    require_login,
    load_breakdown_report,
    save_breakdown_report,
)

st.set_page_config(page_title="Task Report", page_icon="🔧", layout="wide")
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
DATA_DIR = Path("data")
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
    "Immediate Action",
    "Root Cause",
    "Preventive Action",
    "Spare Parts Used",
    "Reported By",
    "Created At",
]

# Usage reference only (free text is allowed)
USAGE_OPTIONS = ["Equipment", "Machine", "Jig", "Fixture", "Tester"]


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
    if "total_add" not in df.columns:
        # Backward compatibility: compute from old columns
        df["total_add"] = (df.get("total_quantity", 0) - df.get("total_used", 0)).clip(lower=0)

    # Enforce rule in memory (display consistency)
    df["total_quantity"] = (df["total_used"].fillna(0).astype(int) + df["total_add"].fillna(0).astype(int)).astype(int)
    return df


def save_part(part_number: str, item_name: str, specification: str, total_add: int, part_type: str, usage: str):
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
        INSERT INTO storage (part_number, item_name, specification, total_quantity, total_used, total_add, part_type, usage)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (part_number, item_name, specification, total_quantity, total_used, total_add, part_type, usage),
    )
    conn.commit()
    conn.close()


def update_storage_row(part_number: str, item_name: str, specification: str, total_add: int, total_used: int, part_type: str, usage: str):
    """
    Always re-calc total_quantity = total_used + total_add.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    total_add = int(total_add)
    total_used = int(total_used)
    total_quantity = total_used + total_add

    c.execute(
        """
        UPDATE storage
        SET item_name = ?,
            specification = ?,
            total_quantity = ?,
            total_used = ?,
            total_add = ?,
            part_type = ?,
            usage = ?
        WHERE part_number = ?
        """,
        (item_name, specification, total_quantity, total_used, total_add, part_type, usage, part_number),
    )
    conn.commit()
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
    finally:
        conn.close()


def delete_part(part_number: str, performed_by: str = "", note: str = ""):
    pn = str(part_number or "").strip()
    if not pn:
        raise ValueError("Part Number is required")

    conn = sqlite3.connect(DB_PATH)
    try:
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
        st.error(f"Error loading breakdown report from main_data.db: {e}")
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


def generate_job_id(entry_date: date, existing_df: pd.DataFrame) -> str:
    yy = str(entry_date.year)[-2:]
    mm = f"{entry_date.month:02d}"
    dd = f"{entry_date.day:02d}"
    prefix = f"{yy}-{mm}-{dd}"
    if existing_df is None or existing_df.empty or "Date" not in existing_df.columns:
        return f"{prefix}/01"
    existing_df = existing_df.copy()
    existing_df["Date"] = pd.to_datetime(existing_df["Date"], errors="coerce").dt.date
    same_day = existing_df[existing_df["Date"].astype(str) == str(entry_date)]
    n = len(same_day) + 1
    return f"{prefix}/{n:02d}"


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


ensure_data_directory()
initialize_stock_log_database()
init_db()

st.title("🔧 TaskReport")
st.markdown("Technical team: report and update breakdown entries.")
st.markdown("---")

tab_generate, tab_review, tab_storage = st.tabs(["📝 Task Entry", "📋 Review Entries", "🛠️Workshop Storage Log"])

# ================= TAB 1: TASK ENTRY =================
with tab_generate:
    st.markdown("### New Task")

    if "spare_parts" not in st.session_state:
        st.session_state.spare_parts = []

    existing_df = load_breakdown_data()

    # Row 1: Date | JobID(auto)
    col_d, col_jid = st.columns(2)
    with col_d:
        entry_date = st.date_input("Date *", value=date.today(), key="br_date")
    with col_jid:
        job_id = generate_job_id(entry_date, existing_df)
        st.text_input("Job ID (auto)", value=job_id, disabled=True, key="br_job_id_display")

    # Row 2: Job type | Severity
    col_jt, col_sev = st.columns(2)
    with col_jt:
        job_type = st.selectbox(
            "Job Type *",
            options=["Breakdown", "Maintenance", "Other"],
            key="br_job_type",
        )
    with col_sev:
        severity = st.text_input("Severity", key="br_severity", placeholder="e.g. Low / Medium / High / Critical")

    # Row 3: Shift | Location
    col_s, col_l = st.columns(2)
    with col_s:
        shift = st.text_input("Shift *", key="br_shift", placeholder="e.g. A / B / C / Night")
    with col_l:
        location = st.text_input("Location *", key="br_location", placeholder="e.g. Plant 1 / Line 2")

    # Row 4: Machine/Equipment | Machine ID
    col_me, col_mid = st.columns(2)
    with col_me:
        machine_equipment = st.text_input(
            "Machine/Equipment *",
            key="br_machine_equipment",
            placeholder="e.g. Press machine",
        )
    with col_mid:
        machine_id = st.text_input("Machine ID *", key="br_machine_id", placeholder="e.g. MC-001")

    # Row 5: Date/Time Start | Date/Time End (user can pick date and time)
    col_dt_s, col_dt_e = st.columns(2)
    with col_dt_s:
        start_date = st.date_input("Date Start *", value=entry_date, key="br_start_date")
        start_time = st.time_input("Time Start *", value=datetime.now().time(), key="br_time_start")
    with col_dt_e:
        end_date = st.date_input("Date End *", value=entry_date, key="br_end_date")
        end_time = st.time_input("Time End *", value=datetime.now().time(), key="br_time_end")

    # Row 6: Duration(auto calculate) | JobStatus  (REVERT to minutes)
    start_dt = datetime.combine(start_date, start_time)
    end_dt = datetime.combine(end_date, end_time)

    duration_min = None
    duration_err = None

    if end_dt < start_dt:
        duration_err = "End Date/Time must be after Start Date/Time."
    else:
        duration_seconds = int((end_dt - start_dt).total_seconds())
        duration_min = int(duration_seconds // 60)  # minutes

    col_dur, col_status = st.columns(2)
    with col_dur:
        if duration_err:
            st.error(duration_err)
            st.text_input("Duration (auto, minutes)", value="—", disabled=True, key="br_duration_display")
        else:
            st.text_input("Duration (auto, minutes)", value=str(duration_min), disabled=True, key="br_duration_display")

    with col_status:
        job_status = st.selectbox("JobStatus *", options=["Open", "Pending", "Close"], key="br_job_status")

    st.markdown("---")

    # Text blocks
    problem_description = st.text_area(
        "Problem Description *",
        placeholder="Describe the breakdown or issue...",
        height=120,
        key="br_problem_desc",
    )
    immediate_action = st.text_area(
        "Immediate Action *",
        placeholder="What was done immediately to contain or fix?",
        height=120,
        key="br_immediate",
    )
    root_cause = st.text_area(
        "Root Cause *",
        placeholder="Identified root cause of the breakdown...",
        height=120,
        key="br_root",
    )
    preventive_action = st.text_area(
        "Preventive Action *",
        placeholder="Actions to prevent recurrence...",
        height=120,
        key="br_preventive",
    )

    # Spare Parts Used section
    st.markdown("### 🧰 Spare Parts Used")
    spare_used = st.checkbox("Spare parts used?", value=False, key="br_spare_used")

    if not spare_used:
        st.session_state.spare_parts = []
        st.info("No spare parts will be recorded for this job.")
        available_parts = pd.DataFrame()
    else:
        storage_df = get_storage()
        # NEW: available stock is total_add
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
            st.error(duration_err)
        elif not str(shift).strip():
            st.error("Shift is required.")
        elif not str(location).strip():
            st.error("Location is required.")
        elif not str(machine_equipment).strip():
            st.error("Machine/Equipment is required.")
        elif not str(machine_id).strip():
            st.error("Machine ID is required.")
        elif not problem_description.strip():
            st.error("Problem Description is required.")
        elif not immediate_action.strip():
            st.error("Immediate Action is required.")
        elif not root_cause.strip():
            st.error("Root Cause is required.")
        elif not preventive_action.strip():
            st.error("Preventive Action is required.")
        elif spare_used and not st.session_state.spare_parts:
            st.error("You ticked 'Spare parts used?' but did not add any parts.")
        else:
            reported_by_name = _performed_by_label()

            pending = {
                "Date": entry_date.strftime("%Y-%m-%d"),
                "Job ID": job_id,
                "Job Type": str(job_type).strip(),
                "Severity": str(severity or "").strip(),
                "Shift": str(shift).strip(),
                "Location": str(location).strip(),
                "Machine/Equipment": str(machine_equipment).strip(),
                "Machine ID": str(machine_id).strip(),
                "Date/Time Start": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "Date/Time End": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "Duration": int(duration_min if duration_min is not None else 0),
                "JobStatus": str(job_status).strip(),
                "Problem Description": problem_description.strip(),
                "Immediate Action": immediate_action.strip(),
                "Root Cause": root_cause.strip(),
                "Preventive Action": preventive_action.strip(),
                "Spare Parts Used": (
                    " | ".join([f"{p['name']} x{p['qty']}" for p in st.session_state.spare_parts])
                    if spare_used
                    else ""
                ),
                "Created At": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Save report to main_data.db (breakdown_report table)
            df = load_breakdown_data()

            new_entry = {
                "Date": pending["Date"],
                "Job ID": pending["Job ID"],
                "Job Type": pending["Job Type"],
                "Severity": pending.get("Severity", ""),
                "Shift": pending["Shift"],
                "Location": pending["Location"],
                "Machine/Equipment": pending["Machine/Equipment"],
                "Machine ID": pending["Machine ID"],
                "Date/Time Start": pending["Date/Time Start"],
                "Date/Time End": pending["Date/Time End"],
                "Duration": int(pending.get("Duration", 0) or 0),
                "JobStatus": pending.get("JobStatus", ""),
                "Problem Description": pending["Problem Description"],
                "Immediate Action": pending["Immediate Action"],
                "Root Cause": pending["Root Cause"],
                "Preventive Action": pending["Preventive Action"],
                "Spare Parts Used": pending.get("Spare Parts Used", ""),
                "Reported By": reported_by_name,
                "Created At": pending["Created At"],
            }

            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)

            for col in COLUMNS:
                if col not in df.columns:
                    df[col] = ""
            df = df[COLUMNS]
            if not _save_breakdown_data(df):
                st.error("Failed to save breakdown report to main_data.db")
                st.stop()

            # AUTO OUT stock (Task Report)
            for part in st.session_state.get("spare_parts", []):
                try:
                    stock_out_task(
                        part_number=part["part_number"],
                        qty_used=part["qty"],
                        performed_by=reported_by_name,
                        note=f"JobID={pending['Job ID']}",
                    )
                except Exception as e:
                    st.error(f"Stock OUT (Task) failed for {part['part_number']}: {e}")
                    st.stop()

            # Clear state
            for key in [
                "br_date",
                "br_job_type",
                "br_severity",
                "br_shift",
                "br_location",
                "br_machine_equipment",
                "br_machine_id",
                "br_start_date",
                "br_end_date",
                "br_time_start",
                "br_time_end",
                "br_job_status",
                "br_problem_desc",
                "br_immediate",
                "br_root",
                "br_preventive",
                "spare_parts",
            ]:
                if key in st.session_state:
                    del st.session_state[key]

            st.success("Breakdown report saved & stock updated!")
            st.rerun()

# ================= TAB 2: REVIEW ENTRIES =================
with tab_review:
    st.markdown("### Review breakdown reports")
    df = load_breakdown_data()
    if df.empty:
        st.info("No breakdown entries yet. Use **Task Entry** to add one.")
    else:
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            filter_date = st.date_input("Filter by date", value=None, key="br_filter_date")
        with col_f2:
            filter_machine_id = st.text_input("Filter by Machine ID", placeholder="e.g. MC-001", key="br_filter_mid")
        with col_f3:
            filter_problem = st.text_input("Filter by Problem (keyword)", placeholder="Keyword...", key="br_filter_problem")

        review_df = df.copy()
        if filter_date:
            review_df["Date"] = pd.to_datetime(review_df["Date"], errors="coerce").dt.date
            review_df = review_df[review_df["Date"] == filter_date]
        if filter_machine_id and str(filter_machine_id).strip():
            review_df = review_df[
                review_df["Machine ID"].astype(str).str.contains(str(filter_machine_id).strip(), case=False, na=False)
            ]
        if filter_problem and str(filter_problem).strip():
            review_df = review_df[
                review_df["Problem Description"].astype(str).str.contains(str(filter_problem).strip(), case=False, na=False)
            ]

        st.dataframe(review_df, use_container_width=True, hide_index=True)

# ================= TAB 3: Workshop Storage Log =================
with tab_storage:
    st.markdown("### 🏭 Workshop Storage Management")

    storage_df = get_storage()

    # Add New Part (layout kept)
    if "show_add_part" not in st.session_state:
        st.session_state.show_add_part = False

    if st.button("➕ Add New Part"):
        st.session_state.show_add_part = not st.session_state.show_add_part

    if st.session_state.show_add_part:
        st.markdown("#### Add New Part")

        col_pt, col_pn = st.columns([1, 1])
        with col_pt:
            add_part_type_label = st.selectbox("Part Type", options=list(PART_TYPE_CONFIG.keys()), key="add_part_type")
        with col_pn:
            auto_pn = generate_part_number(add_part_type_label, storage_df)
            st.text_input("Part Number (auto)", value=auto_pn, disabled=True, key="add_pn_display")

        col_item, col_qty = st.columns([2, 1])
        with col_item:
            add_item_name = st.text_input("Item Name", key="add_item_name")
        with col_qty:
            # This is total_add (available/current stock)
            add_total_qty = st.number_input("Total Quantity", min_value=0, step=1, key="add_total_qty")

        col_spec, col_usage = st.columns([2, 1])
        with col_spec:
            add_specification = st.text_input("Item Specification", key="add_spec")
        with col_usage:
            add_usage = st.text_input(
                "Usage (Equipment/Machine/Jig/Fixture/Tester)",
                key="add_usage",
                placeholder="Enter usage (free text)",
            )

        if st.button("💾 Save Part"):
            if not add_item_name.strip():
                st.error("Item Name is required.")
            else:
                try:
                    save_part(
                        part_number=auto_pn,
                        item_name=add_item_name.strip(),
                        specification=add_specification.strip(),
                        total_add=int(add_total_qty),
                        part_type=PART_TYPE_CONFIG[add_part_type_label]["type_code"],
                        usage=str(add_usage or "").strip(),
                    )
                    st.success("Part saved.")
                    st.session_state.show_add_part = False
                    st.rerun()
                except sqlite3.IntegrityError:
                    st.error(f"Part Number '{auto_pn}' already exists.")
                except Exception as e:
                    st.error(f"Save failed: {e}")

    st.markdown("---")
    st.markdown("## 📋 Existing Parts")

    storage_df = get_storage()

    show_existing_table = st.toggle("Show existing parts table", value=True, key="existing_parts_show_table")
    if not show_existing_table:
        st.info("Existing parts table hidden.")
    else:
        if storage_df.empty:
            st.info("No parts in storage yet.")
        else:
            storage_df = storage_df.copy()

            # Available stock = total_add
            storage_df["available"] = storage_df["total_add"].fillna(0).astype(int)

            c_f1, c_f2 = st.columns([2, 1])
            with c_f1:
                existing_search = st.text_input(
                    "Search parts (Part Number / Item Name / Specification)",
                    key="existing_parts_search",
                    placeholder="Type to filter...",
                ).strip()
            with c_f2:
                show_all_parts = st.checkbox(
                    "Show all parts",
                    value=True,
                    key="existing_parts_show_all",
                    help="Untick to hide parts with 0 available stock.",
                )

            filtered_df = storage_df
            if not show_all_parts:
                filtered_df = filtered_df[filtered_df["available"] > 0]

            if existing_search:
                pn_match = filtered_df["part_number"].astype(str).str.contains(existing_search, case=False, na=False)
                name_match = filtered_df.get("item_name", "").astype(str).str.contains(existing_search, case=False, na=False)
                spec_match = filtered_df.get("specification", "").astype(str).str.contains(existing_search, case=False, na=False)
                filtered_df = filtered_df[pn_match | name_match | spec_match]

            # ONLY show requested columns
            show_cols = ["part_number", "item_name", "specification", "part_type", "usage", "available"]
            for c in show_cols:
                if c not in filtered_df.columns:
                    filtered_df[c] = ""

            st.caption(f"Showing {len(filtered_df)} of {len(storage_df)} parts")
            st.dataframe(
                filtered_df[show_cols],
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("---")
    

    # NEW/UPDATED: In / Out features (User/SuperUser only) with:
    # - Description entry for every IN/OUT
    # - Search bar to filter parts (replaces "Note")

    st.markdown("#### 🔁 Stock IN / OUT")

    with st.expander("Open Stock IN/OUT", expanded=False):
        storage_df = get_storage()
        if storage_df.empty:
            st.warning("No parts available to update.")
        else:
                # Search bar (replaces note)
                search = st.text_input(
                    "Search Part (Part Number / Item Name)",
                    key="stock_part_search",
                    placeholder="Type to filter...",
                ).strip()

                filtered = storage_df.copy()
                if search:
                    pn_match = filtered["part_number"].astype(str).str.contains(search, case=False, na=False)
                    name_match = filtered.get("item_name", "").astype(str).str.contains(search, case=False, na=False)
                    filtered = filtered[pn_match | name_match]

                if filtered.empty:
                    st.error("No matching parts found. Clear the search and try again.")
                    st.stop()

                # Build searchable options
                filtered = filtered.sort_values(["item_name", "part_number"], na_position="last")
                options = [
                    f"{r['part_number']} | {r.get('item_name','')}"
                    for _, r in filtered.iterrows()
                ]
                option_to_pn = {opt: opt.split("|", 1)[0].strip() for opt in options}

                c_in, c_out = st.columns(2)

                with c_in:
                    st.markdown("**Stock IN (add new quantity)**")
                    in_opt = st.selectbox("Select Part (IN)", options=options, key="stock_in_opt")
                    in_pn = option_to_pn[in_opt]
                    in_qty = st.number_input("Qty IN", min_value=1, step=1, key="stock_in_qty")

                    # Description required for every action
                    in_desc = st.text_input(
                        "Description (required)",
                        key="stock_in_desc",
                        placeholder="e.g. Supplier delivery / stock refill / adjustment reason...",
                    )

                    if st.button("✅ Apply IN", key="btn_apply_in"):
                        if not in_desc.strip():
                            st.error("Description is required for Stock IN.")
                            st.stop()
                        try:
                            stock_in_add(
                                in_pn,
                                int(in_qty),
                                performed_by=_performed_by_label(),
                                note=in_desc.strip(),  # stored in main_data.db stock_log.note
                            )
                            st.success("Stock IN applied.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Stock IN failed: {e}")

                with c_out:
                    st.markdown("**Stock OUT (reduce available stock)**")
                    out_opt = st.selectbox("Select Part (OUT)", options=options, key="stock_out_opt")
                    out_pn = option_to_pn[out_opt]

                    try:
                        avail = int(storage_df.loc[storage_df["part_number"] == out_pn, "total_add"].iloc[0])
                    except Exception:
                        avail = 0
                    st.caption(f"Available (total_add): {avail}")

                    out_qty = st.number_input("Qty OUT", min_value=1, step=1, key="stock_out_qty")

                    # Description required for every action
                    out_desc = st.text_input(
                        "Description (required)",
                        key="stock_out_desc",
                        placeholder="e.g. Damaged / returned / stock correction / issued without report...",
                    )

                    if st.button("✅ Apply OUT", key="btn_apply_out"):
                        if not out_desc.strip():
                            st.error("Description is required for Stock OUT.")
                            st.stop()
                        try:
                            stock_out_adjust(
                                out_pn,
                                int(out_qty),
                                performed_by=_performed_by_label(),
                                note=out_desc.strip(),  # stored in main_data.db stock_log.note
                            )
                            st.success("Stock OUT applied.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Stock OUT failed: {e}")

    # Edit + Bulk Delete in one SuperUser editor
    st.markdown("---")
    st.markdown("#### ✏️🗑️ Storage Editor  ")

    with st.expander("Open editor", expanded=False):
        if _current_level_rank() < 2:
            st.info("SuperUser/Admin clearance required to edit/delete storage rows.")
        else:
            st.success("Clearance OK. You can edit rows and tick parts to remove.")

            df_edit = get_storage().copy()
            if df_edit.empty:
                st.info("No parts in storage.")
            else:
                df_edit["total_quantity"] = (df_edit["total_used"].astype(int) + df_edit["total_add"].astype(int)).astype(int)
                df_edit["Remove"] = False

                edited = st.data_editor(
                    df_edit[["Remove", "part_number", "item_name", "specification", "total_add", "total_used", "total_quantity", "part_type", "usage"]],
                    hide_index=True,
                    use_container_width=True,
                    disabled=["part_number", "total_quantity"],
                    column_config={
                        "Remove": st.column_config.CheckboxColumn("Remove", default=False),
                        "total_add": st.column_config.NumberColumn("Total Quantity (Available)", min_value=0, step=1),
                        "total_used": st.column_config.NumberColumn("Total Used", min_value=0, step=1),
                        "part_type": st.column_config.SelectboxColumn(
                            "Part Type",
                            options=[cfg["type_code"] for cfg in PART_TYPE_CONFIG.values()],
                            required=False,
                        ),
                        "usage": st.column_config.TextColumn("Usage (free text)"),
                    },
                    key="storage_table_editor",
                )

                selected_pns = edited[edited["Remove"] == True]["part_number"].astype(str).str.strip().tolist()
                st.caption(f"Selected for delete: {len(selected_pns)}")

                del_note = st.text_input(
                    "Delete Description",
                    key="storage_delete_note",
                    placeholder="Optional reason for deletion...",
                )
                confirm_delete = st.checkbox(
                    "I confirm deleting selected parts",
                    key="storage_delete_confirm",
                )

                if st.button("💾 Apply Changes", type="primary", key="btn_apply_storage_changes"):
                    errors = []

                    # 1) apply updates for rows not marked for delete
                    for _, r in edited.iterrows():
                        pn = str(r["part_number"]).strip()
                        if bool(r.get("Remove", False)):
                            continue
                        try:
                            item = str(r["item_name"]).strip()
                            spec = str(r["specification"]).strip()
                            ptype = str(r["part_type"]).strip()
                            usage = str(r.get("usage", "")).strip()
                            total_add = int(pd.to_numeric(r["total_add"], errors="coerce"))
                            total_used = int(pd.to_numeric(r["total_used"], errors="coerce"))

                            if not pn:
                                errors.append("Row has empty part_number (cannot update).")
                                continue
                            if not item:
                                errors.append(f"{pn}: Item Name is required.")
                                continue
                            if total_add < 0 or total_used < 0:
                                errors.append(f"{pn}: quantities cannot be negative.")
                                continue

                            update_storage_row(pn, item, spec, total_add, total_used, ptype, usage)
                        except Exception as e:
                            errors.append(f"{pn}: update failed: {e}")

                    # 2) apply deletes for selected rows
                    if selected_pns and not confirm_delete:
                        errors.append("Please confirm deleting selected parts.")
                    else:
                        for pn in selected_pns:
                            try:
                                delete_part(
                                    part_number=pn,
                                    performed_by=_performed_by_label(),
                                    note=str(del_note or "").strip(),
                                )
                            except Exception as e:
                                errors.append(f"{pn}: delete failed: {e}")

                    if errors:
                        st.error("Some changes were not applied:")
                        for msg in errors[:40]:
                            st.write(f"- {msg}")
                        st.stop()

                    st.success("Storage changes applied.")
                    st.rerun()