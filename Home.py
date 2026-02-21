# Home.py

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="ME Dashboard", page_icon="📊", layout="wide")
st.title("📊 ME Asset List")

# ======================================
#   LOAD DATA (ASSET)
# ======================================
csv_file = Path("data/DataBase_ME_Asset.csv")

if not csv_file.exists():
    st.error("❌ Data Base not found.")
    st.stop()

df = pd.read_csv(csv_file)

# ======================================
#   FILTER DATA BY STATUS
# ======================================
statuses = ["Good", "Idle", "NG", "Expired", "Expired Soon"]

if "Status" in df.columns:
    df_filter = df[df["Status"].isin(statuses)]
else:
    st.warning("⚠️ Column 'Status' not found. Using entire dataset.")
    df_filter = df.copy()

# ======================================
#   ASSET ANALYSIS (SIDE-BY-SIDE)
# ======================================
st.markdown("## 📦 Asset Analysis")

col_left, col_right = st.columns(2)

with col_left:
    if "Type" in df.columns:
        type_count = df["Type"].value_counts().reset_index()
        type_count.columns = ["Type", "Count"]

        fig_type = px.pie(
            type_count,
            names="Type",
            values="Count",
            title="Type of Asset",
        )
        st.plotly_chart(fig_type, use_container_width=True)
    else:
        st.warning("⚠️ Column 'Type' not found.")

with col_right:
    if "Status" in df.columns:
        status_count = df["Status"].value_counts().reset_index()
        status_count.columns = ["Status", "Count"]

        fig_status = px.bar(
            status_count,
            x="Status",
            y="Count",
            title="Asset Status",
            text="Count",
            color="Status",
        )
        st.plotly_chart(fig_status, use_container_width=True)
    else:
        st.warning("⚠️ Column 'Status' not found.")

# ======================================
#   EQUIPMENT QUANTITY SUMMARY
# ======================================
if "Description of Asset" in df_filter.columns:
    Eq_Quantity = (
        df_filter["Description of Asset"]
        .value_counts()
        .rename_axis("Description of Asset")
        .reset_index(name="Quantity")
        .sort_values(by="Description of Asset")
        .reset_index(drop=True)
    )
else:
    st.error("❌ Column 'Description of Asset' not found.")
    st.stop()

st.markdown("### Asset Quantity Summary")
st.dataframe(Eq_Quantity, use_container_width=True, hide_index=True)

# ======================================
#   TASK REPORT ANALYSIS
# ======================================
st.markdown("---")
st.markdown("## 🛠️ Task Report Analysis")

task_csv = Path("data/BreakdownReport.csv")
if not task_csv.exists():
    st.info("No BreakdownReport.csv found yet (data/BreakdownReport.csv).")
else:
    try:
        tdf = pd.read_csv(task_csv)

        def _find_col(df, candidates):
            if df is None or df.empty:
                return None
            norm_map = {str(c).strip().lower(): c for c in df.columns}
            for cand in candidates:
                key = str(cand).strip().lower()
                if key in norm_map:
                    return norm_map[key]
            # Fallback: startswith match (e.g., "Duration (min)")
            for cand in candidates:
                key = str(cand).strip().lower()
                for norm_name, real_name in norm_map.items():
                    if norm_name.startswith(key):
                        return real_name
            return None

        job_type_col = _find_col(tdf, ["Job Type", "Task Type", "JobType", "TaskType"])
        duration_col = _find_col(tdf, ["Duration", "Duration (min)", "Downtime", "Down Time"])

        def _duration_to_minutes(value):
            # expects minutes (int/float). Keeps compatibility if any old rows were stored as HH:MM:SS / MM:SS.
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return None
            if isinstance(value, (int, float)) and not pd.isna(value):
                return float(value)

            s = str(value).strip()
            if not s:
                return None

            if ":" in s:
                parts = s.split(":")
                try:
                    if len(parts) == 3:
                        h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
                        return (h * 60) + m + (sec / 60.0)
                    if len(parts) == 2:
                        m, sec = int(parts[0]), int(parts[1])
                        return m + (sec / 60.0)
                except Exception:
                    return None

            try:
                return float(s)
            except Exception:
                return None

        c1, c2 = st.columns(2)

        # ---- Chart 1: Task count (PIE) ----
        with c1:
            if not job_type_col:
                st.warning("⚠️ Task report column not found: 'Job Type' (or 'Task Type').")
            else:
                job_type_series = (
                    tdf[job_type_col]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .replace({"": "UNKNOWN"})
                )

                type_counts = job_type_series.value_counts().reset_index()
                type_counts.columns = ["Job Type", "Count"]

                fig_task_count = px.pie(
                    type_counts,
                    names="Job Type",
                    values="Count",
                    title="Task Count",
                )
                st.plotly_chart(fig_task_count, use_container_width=True)

        # ---- Chart 2: Duration (Breakdown vs Maintenance) (PIE) ----
        with c2:
            if not (job_type_col and duration_col):
                st.warning("⚠️ Task report columns not found: need 'Job Type' and 'Duration'.")
            else:
                tmp = tdf[[job_type_col, duration_col]].copy()
                tmp[job_type_col] = (
                    tmp[job_type_col]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .str.replace(r"\s+", " ", regex=True)
                )

                def _normalize_job_type(raw: str) -> str:
                    v = str(raw or "").strip().lower()
                    if not v:
                        return "Unknown"
                    if "break" in v:
                        return "Breakdown"
                    if "maint" in v:
                        return "Maintenance"
                    return str(raw).strip().title()

                tmp["_JobTypeNorm"] = tmp[job_type_col].apply(_normalize_job_type)
                tmp["_DurationMin"] = tmp[duration_col].apply(_duration_to_minutes)
                tmp["_DurationMin"] = pd.to_numeric(tmp["_DurationMin"], errors="coerce")

                # Only Breakdown vs Maintenance (as requested)
                tmp = tmp[tmp["_JobTypeNorm"].isin(["Breakdown", "Maintenance"])]
                tmp = tmp[tmp["_DurationMin"].notna() & (tmp["_DurationMin"] >= 0)]

                if tmp.empty:
                    st.info("No duration data for Breakdown/Maintenance.")
                    # Helps verify what's in the CSV without breaking UX
                    available_types = (
                        tdf[job_type_col]
                        .fillna("")
                        .astype(str)
                        .str.strip()
                        .replace({"": "UNKNOWN"})
                        .value_counts()
                        .head(10)
                    )
                    st.caption("Top Job Types in report (first 10):")
                    st.dataframe(available_types.reset_index().rename(columns={"index": "Job Type", job_type_col: "Count"}), hide_index=True, use_container_width=True)
                else:
                    dur_sum = (
                        tmp.groupby("_JobTypeNorm", as_index=False)["_DurationMin"].sum()
                        .rename(columns={"_JobTypeNorm": "Job Type", "_DurationMin": "Total Duration (min)"})
                    )

                    fig_duration = px.pie(
                        dur_sum,
                        names="Job Type",
                        values="Total Duration (min)",
                        title="Duration (Breakdown vs Maintenance)",
                    )
                    st.plotly_chart(fig_duration, use_container_width=True)

                    stats = (
                        tmp.groupby("_JobTypeNorm")["_DurationMin"]
                        .agg(["count", "sum", "mean", "median", "min", "max"])
                        .reset_index()
                        .rename(
                            columns={
                                "_JobTypeNorm": "Job Type",
                                "count": "No. of Jobs",
                                "sum": "Total Duration (min)",
                                "mean": "Avg Duration (min)",
                                "median": "Median (min)",
                                "min": "Min (min)",
                                "max": "Max (min)",
                            }
                        )
                    )
                    st.markdown("**Duration Statistics (minutes)**")
                    st.dataframe(stats, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Failed to load/plot BreakdownReport.csv: {e}")