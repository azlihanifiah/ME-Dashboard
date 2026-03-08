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
)

st.set_page_config(page_title="Workshop Inventory", page_icon="🏭", layout="wide")

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


APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = APP_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "main_data.db"


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Storage Table
    # Rule:
    #   total_quantity = total_used + total_add
    # where:
    #   total_add  = current stock in store (available)
    #   total_used = total issued/consumed
    c.execute(
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
        c.execute(
            """
            UPDATE storage
            SET total_add = COALESCE(total_quantity, 0) - COALESCE(total_used, 0)
            WHERE total_add IS NULL
            """
        )
        # Ensure non-negative
        c.execute("UPDATE storage SET total_add = 0 WHERE total_add < 0 OR total_add IS NULL")

    # Normalize totals to match rule: total_quantity = total_used + total_add
    c.execute(
        """
        UPDATE storage
        SET total_used = COALESCE(total_used, 0),
            total_add = COALESCE(total_add, 0),
            total_quantity = COALESCE(total_used, 0) + COALESCE(total_add, 0)
        WHERE total_quantity IS NULL
           OR total_used IS NULL
           OR total_add IS NULL
           OR total_quantity != (COALESCE(total_used, 0) + COALESCE(total_add, 0))
        """
    )

    conn.commit()
    conn.close()


PART_TYPE_CONFIG = {
    "Electrical": {"type_code": "ELEC", "pn_prefix": "PN1"},
    "Mechanical": {"type_code": "MECH", "pn_prefix": "PN2"},
    "Pneumatic": {"type_code": "PNE", "pn_prefix": "PN3"},
}

TYPE_CODE_TO_PN_PREFIX = {cfg["type_code"]: cfg["pn_prefix"] for cfg in PART_TYPE_CONFIG.values()}


def get_storage() -> pd.DataFrame:
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
        df["total_add"] = (df.get("total_quantity", 0) - df.get("total_used", 0)).clip(lower=0)

    df["total_quantity"] = (
        df["total_used"].fillna(0).astype(int) + df["total_add"].fillna(0).astype(int)
    ).astype(int)
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
) -> None:
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


def _get_storage_totals(conn: sqlite3.Connection, part_number: str) -> tuple[int, int]:
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


def stock_in_add(part_number: str, qty_in: int, performed_by: str = "", note: str = "") -> None:
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


def stock_out_adjust(part_number: str, qty_out: int, performed_by: str = "", note: str = "") -> None:
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
    total_add: int,
    total_used: int,
    part_type: str,
    usage: str,
    *,
    brand: str = "",
    model: str = "",
    performed_by: str = "",
    note: str = "",
) -> None:
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
        add_total_qty = st.number_input("Total Quantity", min_value=0, step=1, key="add_total_qty")

    col_brand, col_model = st.columns([1, 1])
    with col_brand:
        add_brand = st.text_input("Brand", key="add_brand")
    with col_model:
        add_model = st.text_input("Model", key="add_model")

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
                    brand=str(add_brand or "").strip(),
                    model=str(add_model or "").strip(),
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
        storage_df["available"] = storage_df["total_add"].fillna(0).astype(int)

        c_f1, c_f2 = st.columns([2, 1])
        with c_f1:
            existing_search = st.text_input(
                "Search parts (Part Number / Item Name / Brand / Model / Specification)",
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
            filtered_df = filtered_df[pn_match | name_match | brand_match | model_match | spec_match]

        show_cols = ["part_number", "item_name", "brand", "model", "specification", "part_type", "usage", "available"]
        for c in show_cols:
            if c not in filtered_df.columns:
                filtered_df[c] = ""

        st.caption(f"Showing {len(filtered_df)} of {len(storage_df)} parts")
        st.dataframe(filtered_df[show_cols], use_container_width=True, hide_index=True)

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
                avail = int(storage_df.loc[storage_df["part_number"] == out_pn, "total_add"].iloc[0])
            except Exception:
                avail = 0
            st.caption(f"Available (total_add): {avail}")

            out_qty = st.number_input("Qty OUT", min_value=1, step=1, key="stock_out_qty")

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
                    stock_out_adjust(out_pn, int(out_qty), performed_by=_performed_by_label(), note=out_desc.strip())
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
            df_edit["total_quantity"] = (
                df_edit["total_used"].astype(int) + df_edit["total_add"].astype(int)
            ).astype(int)
            df_edit["Remove"] = False

            edited = st.data_editor(
                df_edit[
                    [
                        "Remove",
                        "part_number",
                        "item_name",
                        "brand",
                        "model",
                        "specification",
                        "total_add",
                        "total_used",
                        "total_quantity",
                        "part_type",
                        "usage",
                    ]
                ],
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
            confirm_delete = st.checkbox("I confirm deleting selected parts", key="storage_delete_confirm")

            if st.button("💾 Apply Changes", type="primary", key="btn_apply_storage_changes"):
                errors: list[str] = []
                reserved_pns: set[str] = set()

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

                        desired_pn = pn

                        try:
                            conn_tmp = sqlite3.connect(DB_PATH)
                            try:
                                before_state = _fetch_storage_row(conn_tmp, pn) or {}
                            finally:
                                conn_tmp.close()
                        except Exception:
                            before_state = {}

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

                        update_storage_row_allow_renumber(
                            pn,
                            desired_pn,
                            item,
                            spec,
                            total_add,
                            total_used,
                            ptype,
                            usage,
                            brand=brand,
                            model=model,
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
                "SELECT timestamp, action, part_number, note, performed_by "
                "FROM inventory_history ORDER BY id DESC LIMIT 200",
                conn,
            )
        finally:
            conn.close()

        if hist_df is not None and not hist_df.empty:
            display_hist = hist_df.copy().rename(
                columns={
                    "timestamp": "📅 Timestamp",
                    "action": "🔄 Action",
                    "part_number": "🔧 Part Number",
                    "note": "📄 Details",
                    "performed_by": "👤 User",
                }
            )
            st.dataframe(display_hist, use_container_width=True)
        else:
            st.info("📝 No inventory history entries yet.")
    except Exception as e:
        st.error(f"Failed to load inventory history: {e}")
