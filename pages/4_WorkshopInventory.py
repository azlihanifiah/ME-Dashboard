import streamlit as st
import pandas as pd
from pathlib import Path
import sqlite3
from datetime import datetime

from utils import (
    ensure_data_directory,
    initialize_stock_log_database,
    log_stock_operation,
    initialize_inventory_history_database,
    log_inventory_history,
    persist_repo_changes,
    require_login,
    render_role_navigation,
)

st.set_page_config(page_title="Workshop Inventory", page_icon="🏭", layout="wide")

auth = require_login(min_level_rank=2)
render_role_navigation(auth)


def _performed_by_label() -> str:
    name = str(auth.get("name", "") or "").strip()
    user_id = str(auth.get("user_id", "") or "").strip()
    return name or user_id or "System"


def _current_level_rank() -> int:
    try:
        return int(auth.get("level_rank") or 0)
    except Exception:
        return 0


APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = APP_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "main_data.db"


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Storage Table (v2)
    # - total_quantity: current available stock
    # - total_in: cumulative stock-in quantity
    # - total_out: cumulative stock-out quantity
    # Rule (normalized): total_in = total_out + total_quantity
    desired_cols = [
        ("part_number", "TEXT"),
        ("part_type", "TEXT"),
        ("item_name", "TEXT"),
        ("brand", "TEXT"),
        ("model", "TEXT"),
        ("specification", "TEXT"),
        ("preferred_supplier", "TEXT"),
        ("item_cost_rm", "REAL"),
        ("total_quantity", "INTEGER"),
        ("usage_area", "TEXT"),
        ("total_in", "INTEGER"),
        ("total_out", "INTEGER"),
    ]

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
        conn.commit()
        conn.close()
        return

    # Decide whether a rebuild is needed (legacy/mismatched columns).
    c.execute("PRAGMA table_info(storage)")
    existing_cols = [r[1] for r in (c.fetchall() or [])]
    desired_names = [n for n, _t in desired_cols]
    needs_rebuild = set(existing_cols) != set(desired_names)

    if not needs_rebuild:
        # Normalize totals without rebuilding.
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
        conn.commit()
        conn.close()
        return

    # Migrate any legacy schema into the v2 schema (best-effort).
    try:
        df_old = pd.read_sql("SELECT * FROM storage", conn)
    except Exception:
        df_old = pd.DataFrame()

    def _to_int_series(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)

    if df_old is None:
        df_old = pd.DataFrame()

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

    # Totals migration:
    # - If old columns exist (total_add/total_used), map them to new.
    # - Otherwise, keep new columns if already present.
    has_legacy_totals = ("total_add" in df_old.columns) or ("total_used" in df_old.columns)
    if has_legacy_totals:
        legacy_add = _to_int_series(df_old.get("total_add", 0))
        legacy_used = _to_int_series(df_old.get("total_used", 0))
        out_qty = legacy_used.clip(lower=0)
        avail_qty = legacy_add.clip(lower=0)
        in_qty = (out_qty + avail_qty).astype(int)
        out["total_out"] = out_qty
        out["total_quantity"] = avail_qty
        out["total_in"] = in_qty
    else:
        avail_qty = _to_int_series(df_old.get("total_quantity", 0)).clip(lower=0)
        out_qty = _to_int_series(df_old.get("total_out", 0)).clip(lower=0)
        in_qty = _to_int_series(df_old.get("total_in", (out_qty + avail_qty))).clip(lower=0)
        # Normalize rule: total_in = total_out + total_quantity
        in_qty = (out_qty + avail_qty).astype(int)
        out["total_quantity"] = avail_qty
        out["total_out"] = out_qty
        out["total_in"] = in_qty

    # Keep only desired columns and ensure they exist
    for col, _typ in desired_cols:
        if col not in out.columns:
            out[col] = "" if _typ == "TEXT" else 0
    out = out[[cname for cname, _ in desired_cols]].copy()

    # Rebuild table to match the requested column set.
    c.execute("DROP TABLE IF EXISTS storage__new")
    _create_storage_table("storage__new")
    if not out.empty:
        out.to_sql("storage__new", conn, if_exists="append", index=False)

    c.execute("DROP TABLE IF EXISTS storage")
    c.execute("ALTER TABLE storage__new RENAME TO storage")

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


def get_storage() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM storage", conn)
    conn.close()

    # Ensure columns exist (for safety)
    if "usage_area" not in df.columns:
        df["usage_area"] = ""
    if "brand" not in df.columns:
        df["brand"] = ""
    if "model" not in df.columns:
        df["model"] = ""
    if "preferred_supplier" not in df.columns:
        df["preferred_supplier"] = ""
    if "item_cost_rm" not in df.columns:
        df["item_cost_rm"] = 0.0
    if "total_in" not in df.columns:
        df["total_in"] = 0
    if "total_out" not in df.columns:
        df["total_out"] = 0
    if "total_quantity" not in df.columns:
        df["total_quantity"] = 0

    # Normalize totals (display consistency)
    df["total_quantity"] = pd.to_numeric(df["total_quantity"], errors="coerce").fillna(0).astype(int).clip(lower=0)
    df["total_out"] = pd.to_numeric(df["total_out"], errors="coerce").fillna(0).astype(int).clip(lower=0)
    df["total_in"] = (df["total_out"] + df["total_quantity"]).astype(int)
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
    """Return (before, after) dicts containing only changed fields.

    If include_keys is provided, only those keys are considered.
    """
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
            # normalize numeric strings
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
    part_type: str,
    usage_area: str,
    total_quantity: int,
    brand: str = "",
    model: str = "",
    preferred_supplier: str = "",
    item_cost_rm: float = 0.0,
    performed_by: str = "",
    note: str = "",
) -> None:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    total_quantity = int(total_quantity)
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
            str(preferred_supplier or "").strip(),
            float(item_cost_rm or 0.0),
            int(max(total_quantity, 0)),
            str(usage_area or "").strip(),
            int(max(total_in, 0)),
            int(max(total_out, 0)),
        ),
    )
    conn.commit()

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


def _get_storage_totals(conn: sqlite3.Connection, part_number: str) -> tuple[int, int, int]:
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


def stock_in_add(part_number: str, qty_in: int, performed_by: str = "", note: str = "") -> None:
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
            """
            UPDATE storage
            SET total_in = ?, total_out = ?, total_quantity = ?
            WHERE part_number = ?
            """,
            (after_in, after_out, after_qty, part_number),
        )
        conn.commit()

        after_state = _fetch_storage_row(conn, part_number)

        # Log only changed totals
        b_diff, a_diff = _diff_state_for_log(
            before_state,
            after_state,
            include_keys=["total_in", "total_out", "total_quantity"],
        )

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


def stock_out_adjust(part_number: str, qty_out: int, performed_by: str = "", note: str = "") -> None:
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
        after_in = after_out + after_qty

        cur = conn.cursor()
        cur.execute(
            """
            UPDATE storage
            SET total_in = ?, total_out = ?, total_quantity = ?
            WHERE part_number = ?
            """,
            (after_in, after_out, after_qty, part_number),
        )
        conn.commit()

        after_state = _fetch_storage_row(conn, part_number)

        # Log only changed totals
        b_diff, a_diff = _diff_state_for_log(
            before_state,
            after_state,
            include_keys=["total_in", "total_out", "total_quantity"],
        )

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


def delete_part(part_number: str, performed_by: str = "", note: str = "") -> None:
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
            # DELETE: keep key fields so audit shows what was removed.
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

    for pn in reserved:
        if not str(pn).startswith(pn_prefix):
            continue
        tail = str(pn)[len(pn_prefix) :]
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
    part_type: str,
    usage_area: str,
    total_quantity: int,
    total_out: int,
    *,
    brand: str = "",
    model: str = "",
    preferred_supplier: str = "",
    item_cost_rm: float = 0.0,
    performed_by: str = "",
    note: str = "",
) -> None:
    old_pn = str(old_part_number or "").strip()
    new_pn = str(new_part_number or "").strip()
    if not old_pn:
        raise ValueError("Old Part Number is required")
    if not new_pn:
        raise ValueError("New Part Number is required")

    total_quantity = int(total_quantity)
    total_out = int(total_out)
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
                str(preferred_supplier or "").strip(),
                float(item_cost_rm or 0.0),
                int(total_quantity),
                str(usage_area or "").strip(),
                int(total_in),
                int(total_out),
                old_pn,
            ),
        )
        if cur.rowcount <= 0:
            raise ValueError(f"Part not found: {old_pn}")

        conn.commit()

        after_state = _fetch_storage_row(conn, new_pn)

        # UPDATE/RENUMBER: only log the fields that changed
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

st.title("🏭 Workshop Inventory")
st.markdown("---")

storage_df = get_storage()

# Add New Part
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
        add_total_qty = st.number_input(
            "Total Quantity (Available)",
            min_value=0,
            step=1,
            key="add_total_qty",
        )

    col_model, col_brand = st.columns([1, 1])
    with col_model:
        add_model = st.text_input("Model", key="add_model")
    with col_brand:
        add_brand = st.text_input("Brand", key="add_brand")

    col_supplier, col_cost = st.columns([2, 1])
    with col_supplier:
        add_supplier = st.text_input("Preferred Supplier", key="add_supplier")
    with col_cost:
        add_item_cost = st.number_input(
            "Item Cost (RM)",
            min_value=0.0,
            step=0.10,
            format="%.2f",
            key="add_item_cost_rm",
        )

    col_spec, col_usage = st.columns([2, 1])
    with col_spec:
        add_specification = st.text_input("Item Specification", key="add_spec")
    with col_usage:
        add_usage = st.text_input(
            "Usage Area",
            key="add_usage",
            placeholder="Enter usage area (free text)",
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
                    part_type=PART_TYPE_CONFIG[add_part_type_label]["type_code"],
                    usage_area=str(add_usage or "").strip(),
                    total_quantity=int(add_total_qty),
                    brand=str(add_brand or "").strip(),
                    model=str(add_model or "").strip(),
                    preferred_supplier=str(add_supplier or "").strip(),
                    item_cost_rm=float(add_item_cost or 0.0),
                    performed_by=_performed_by_label(),
                    note="Add New Part",
                )
                st.success("Part saved.")
                st.session_state.show_add_part = False
                st.rerun()
            except sqlite3.IntegrityError:
                st.error(f"Part Number '{auto_pn}' already exists.")
            except Exception as e:
                st.error(f"Save failed: {e}")

st.markdown("---")
st.markdown("#### 📋 Existing Parts")

storage_df = get_storage()

show_existing_table = st.toggle("Show existing parts table", value=True, key="existing_parts_show_table")
if not show_existing_table:
    st.info("Existing parts table hidden.")
else:
    if storage_df.empty:
        st.info("No parts in storage yet.")
    else:
        storage_df = storage_df.copy()
        storage_df["available"] = storage_df["total_quantity"].fillna(0).astype(int)

        c_f1, c_f2 = st.columns([2, 1])
        with c_f1:
            existing_search = st.text_input(
                "Search parts (Part Number / Part Type / Item Name / Model / Brand / Specification / Supplier / Usage Area)",
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
            brand_match = filtered_df.get("brand", "").astype(str).str.contains(existing_search, case=False, na=False)
            model_match = filtered_df.get("model", "").astype(str).str.contains(existing_search, case=False, na=False)
            spec_match = filtered_df.get("specification", "").astype(str).str.contains(existing_search, case=False, na=False)
            supp_match = filtered_df.get("preferred_supplier", "").astype(str).str.contains(existing_search, case=False, na=False)
            usage_match = filtered_df.get("usage_area", "").astype(str).str.contains(existing_search, case=False, na=False)
            filtered_df = filtered_df[pn_match | name_match | brand_match | model_match | spec_match | supp_match | usage_match]

        show_cols = [
            "part_number",
            "part_type",
            "item_name",
            "model",
            "brand",
            "specification",
            "preferred_supplier",
            "item_cost_rm",
            "total_quantity",
            "usage_area",
            "total_in",
            "total_out",
        ]
        for c in show_cols:
            if c not in filtered_df.columns:
                filtered_df[c] = ""

        st.caption(f"Showing {len(filtered_df)} of {len(storage_df)} parts")
        display_df = filtered_df[show_cols].copy().rename(
            columns={
                "part_number": "Part Number",
                "part_type": "Part Type",
                "item_name": "Item Name",
                "model": "Model",
                "brand": "Brand",
                "specification": "Specification",
                "preferred_supplier": "Preferred Supplier",
                "item_cost_rm": "Item Cost (RM)",
                "total_quantity": "Total Quantity",
                "usage_area": "Usage Area",
                "total_in": "Total In",
                "total_out": "Total Out",
            }
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("#### 🔁 Stock IN / OUT")

with st.expander("Open Stock IN/OUT", expanded=False):
    storage_df = get_storage()
    if storage_df.empty:
        st.warning("No parts available to update.")
    else:
        search = st.text_input(
            "Search Part (Part Number / Item Name)",
            key="stock_part_search",
            placeholder="Type to filter...",
        ).strip()

        filtered = storage_df.copy()
        if search:
            pn_match = filtered["part_number"].astype(str).str.contains(search, case=False, na=False)
            name_match = filtered.get("item_name", "").astype(str).str.contains(search, case=False, na=False)
            brand_match = filtered.get("brand", "").astype(str).str.contains(search, case=False, na=False)
            model_match = filtered.get("model", "").astype(str).str.contains(search, case=False, na=False)
            filtered = filtered[pn_match | name_match | brand_match | model_match]

        if filtered.empty:
            st.error("No matching parts found. Clear the search and try again.")
            st.stop()

        filtered = filtered.sort_values(["item_name", "part_number"], na_position="last")
        options = [f"{r['part_number']} | {r.get('item_name','')}" for _, r in filtered.iterrows()]
        option_to_pn = {opt: opt.split("|", 1)[0].strip() for opt in options}

        c_in, c_out = st.columns(2)

        with c_in:
            st.markdown("**Stock IN (add new quantity)**")
            in_opt = st.selectbox("Select Part (IN)", options=options, key="stock_in_opt")
            in_pn = option_to_pn[in_opt]
            in_qty = st.number_input("Qty IN", min_value=1, step=1, key="stock_in_qty")

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
                    stock_in_add(in_pn, int(in_qty), performed_by=_performed_by_label(), note=in_desc.strip())
                    st.success("Stock IN applied.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Stock IN failed: {e}")

        with c_out:
            st.markdown("**Stock OUT (reduce available stock)**")
            out_opt = st.selectbox("Select Part (OUT)", options=options, key="stock_out_opt")
            out_pn = option_to_pn[out_opt]

            try:
                avail = int(storage_df.loc[storage_df["part_number"] == out_pn, "total_quantity"].iloc[0])
            except Exception:
                avail = 0
            st.caption(f"Available: {avail}")

            out_qty = st.number_input("Qty OUT", min_value=1, step=1, key="stock_out_qty")

            out_requestor = st.text_input(
                "Requestor (required)",
                key="stock_out_requestor",
                placeholder="Name / Department / Line...",
            )

            out_desc = st.text_input(
                "Description (required)",
                key="stock_out_desc",
                placeholder="e.g. Damaged / returned / stock correction / issued without report...",
            )

            if st.button("✅ Apply OUT", key="btn_apply_out"):
                if not out_requestor.strip():
                    st.error("Requestor is required for Stock OUT.")
                    st.stop()
                if not out_desc.strip():
                    st.error("Description is required for Stock OUT.")
                    st.stop()
                try:
                    note = f"Requestor={out_requestor.strip()} | {out_desc.strip()}"
                    stock_out_adjust(out_pn, int(out_qty), performed_by=_performed_by_label(), note=note)
                    st.success("Stock OUT applied.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Stock OUT failed: {e}")

st.markdown("---")
st.markdown("#### ✏️🗑️ Storage Editor")

with st.expander("Open editor", expanded=False):
    if _current_level_rank() < 1:
        st.info("User/SuperUser/Admin clearance required to edit/delete storage rows.")
    else:
        st.success("Clearance OK. You can edit rows and tick parts to remove.")

        df_edit = get_storage().copy()
        if df_edit.empty:
            st.info("No parts in storage.")
        else:
            df_edit["total_in"] = (df_edit["total_out"].astype(int) + df_edit["total_quantity"].astype(int)).astype(int)
            df_edit["Remove"] = False

            edited = st.data_editor(
                df_edit[
                    [
                        "Remove",
                        "part_number",
                        "model",
                        "brand",
                        "part_type",
                        "item_name",
                        "specification",
                        "preferred_supplier",
                        "item_cost_rm",
                        "usage_area",
                        "total_quantity",
                        "total_out",
                        "total_in",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
                disabled=["part_number", "total_in"],
                column_config={
                    "Remove": st.column_config.CheckboxColumn("Remove", default=False),
                    "item_cost_rm": st.column_config.NumberColumn("Item Cost (RM)", min_value=0.0, step=0.1, format="%.2f"),
                    "total_quantity": st.column_config.NumberColumn("Total Quantity (Available)", min_value=0, step=1),
                    "total_out": st.column_config.NumberColumn("Total Out", min_value=0, step=1),
                    "part_type": st.column_config.SelectboxColumn(
                        "Part Type",
                        options=[cfg["type_code"] for cfg in PART_TYPE_CONFIG.values()],
                        required=False,
                    ),
                    "usage_area": st.column_config.TextColumn("Usage Area"),
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
            confirm_delete = st.checkbox("I confirm deleting selected parts", key="storage_delete_confirm")

            if st.button("💾 Apply Changes", type="primary", key="btn_apply_storage_changes"):
                errors: list[str] = []
                reserved_pns: set[str] = set()

                def _norm_text(v: object) -> str:
                    return str(v or "").strip()

                def _norm_int(v: object) -> int:
                    try:
                        return int(pd.to_numeric(v, errors="coerce") or 0)
                    except Exception:
                        return 0

                def _norm_float2(v: object) -> float:
                    try:
                        return round(float(pd.to_numeric(v, errors="coerce") or 0.0), 2)
                    except Exception:
                        return 0.0

                # apply updates for rows not marked for delete
                for _, r in edited.iterrows():
                    pn = str(r["part_number"]).strip()
                    if bool(r.get("Remove", False)):
                        continue
                    try:
                        item = str(r["item_name"]).strip()
                        brand = str(r.get("brand", "")).strip()
                        model = str(r.get("model", "")).strip()
                        spec = str(r["specification"]).strip()
                        ptype = str(r["part_type"]).strip()
                        usage_area = str(r.get("usage_area", "")).strip()
                        supplier = str(r.get("preferred_supplier", "")).strip()
                        item_cost_rm = float(pd.to_numeric(r.get("item_cost_rm", 0.0), errors="coerce") or 0.0)
                        total_quantity = int(pd.to_numeric(r["total_quantity"], errors="coerce"))
                        total_out = int(pd.to_numeric(r["total_out"], errors="coerce"))

                        if not pn:
                            errors.append("Row has empty part_number (cannot update).")
                            continue
                        if not item:
                            errors.append(f"{pn}: Item Name is required.")
                            continue
                        if total_quantity < 0 or total_out < 0:
                            errors.append(f"{pn}: quantities cannot be negative.")
                            continue

                        desired_pn = pn

                        try:
                            conn_tmp = sqlite3.connect(DB_PATH)
                            try:
                                before_state = _fetch_storage_row(conn_tmp, pn) or {}
                            finally:
                                conn_tmp.close()
                        except Exception:
                            before_state = {}

                        # Only log/update if something actually changed.
                        before_item = _norm_text(before_state.get("item_name"))
                        before_brand = _norm_text(before_state.get("brand"))
                        before_model = _norm_text(before_state.get("model"))
                        before_spec = _norm_text(before_state.get("specification"))
                        before_ptype = _norm_text(before_state.get("part_type"))
                        before_usage = _norm_text(before_state.get("usage_area") or before_state.get("usage"))
                        before_supplier = _norm_text(before_state.get("preferred_supplier"))
                        before_cost = _norm_float2(before_state.get("item_cost_rm"))
                        before_qty = _norm_int(before_state.get("total_quantity"))
                        before_out = _norm_int(before_state.get("total_out"))

                        after_cost = _norm_float2(item_cost_rm)
                        changed_fields = (
                            item != before_item
                            or brand != before_brand
                            or model != before_model
                            or spec != before_spec
                            or ptype != before_ptype
                            or usage_area != before_usage
                            or supplier != before_supplier
                            or after_cost != before_cost
                            or int(total_quantity) != int(before_qty)
                            or int(total_out) != int(before_out)
                        )

                        old_type = str(before_state.get("part_type", "") or "").strip()
                        if old_type and ptype and old_type != ptype:
                            pn_prefix = TYPE_CODE_TO_PN_PREFIX.get(ptype)
                            if not pn_prefix:
                                errors.append(f"{pn}: unknown Part Type code '{ptype}' (cannot renumber).")
                                continue
                            desired_pn = generate_part_number_by_prefix(
                                pn_prefix, get_storage(), reserved=reserved_pns
                            )
                            reserved_pns.add(desired_pn)

                        # If nothing changed and no renumber, skip update/log.
                        if not changed_fields and desired_pn == pn:
                            continue

                        update_storage_row_allow_renumber(
                            pn,
                            desired_pn,
                            item,
                            spec,
                            ptype,
                            usage_area,
                            total_quantity,
                            total_out,
                            brand=brand,
                            model=model,
                            preferred_supplier=supplier,
                            item_cost_rm=item_cost_rm,
                            performed_by=_performed_by_label(),
                            note="Storage Editor",
                        )
                    except Exception as e:
                        errors.append(f"{pn}: update failed: {e}")

                # apply deletes for selected rows
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

st.markdown("---")

col_hist_1, col_hist_2 = st.columns(2)
with col_hist_1:
    st.caption("Inventory history stored in main_data.db (latest 200).")
with col_hist_2:
    if st.button("📋 View Inventory History"):
        st.session_state.show_inventory_history = not st.session_state.get("show_inventory_history", False)
        st.rerun()

if st.session_state.get("show_inventory_history", False):
    st.markdown("---")
    st.markdown("### 📊 Inventory History Log")

    try:
        initialize_inventory_history_database()
        conn = sqlite3.connect(DB_PATH)
        try:
            hist_df = pd.read_sql(
                "SELECT id, timestamp, action, part_number, performed_by, note, before_state, after_state "
                "FROM inventory_history ORDER BY id DESC LIMIT 200",
                conn,
            )
        finally:
            conn.close()

        if hist_df is not None and not hist_df.empty:
            display_hist = hist_df.copy().rename(
                columns={
                    "id": "Id",
                    "timestamp": "📅 Timestamp",
                    "action": "Action",
                    "part_number": "Part number",
                    "performed_by": "Performed by",
                    "note": "Note",
                    "before_state": "Before state",
                    "after_state": "After state",
                }
            )
            st.dataframe(display_hist, use_container_width=True, hide_index=True)
        else:
            st.info("📝 No inventory history entries yet.")
    except Exception as e:
        st.error(f"Failed to load inventory history: {e}")
