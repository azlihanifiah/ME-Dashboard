# Home.py

import streamlit as st
import pandas as pd
import plotly.express as px
import re
from utils import load_existing_data, load_breakdown_report, require_login

st.set_page_config(page_title="ME Dashboard", page_icon="📊", layout="wide")
require_login()
st.title("📊 ME Asset List")

# ======================================
#   LOAD DATA (ASSET) from main_data.db
# ======================================
df = load_existing_data()
if df is None or df.empty:
    st.error("❌ No asset data found in main_data.db.")
    st.stop()
    raise SystemExit

# ======================================
#   FILTER DATA BY STATUS
# ======================================
statuses = ["Good", "Idle", "NG", "Expired", "Expired Soon", "Obsolete"]

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
    raise SystemExit

st.markdown("### Asset Quantity Summary")
st.dataframe(Eq_Quantity, use_container_width=True, hide_index=True)

# ======================================
#   TASK REPORT ANALYSIS
# ======================================
st.markdown("---")
st.markdown("## 🛠️ Task Report Analysis")

try:
    tdf = load_breakdown_report()

    if tdf is None or tdf.empty:
        st.info("No breakdown report data found yet.")
    else:

        job_type_col = "Job Type" if "Job Type" in tdf.columns else ("Task Type" if "Task Type" in tdf.columns else None)
        duration_col = "Duration" if "Duration" in tdf.columns else None

        def _duration_to_minutes(value):
            # Keeps compatibility with minutes, HH:MM:SS / MM:SS, and common legacy text formats.
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return None
            if isinstance(value, (int, float)) and not pd.isna(value):
                return float(value)

            s = str(value).strip()
            if not s:
                return None

            # Handles pandas timedelta-like text, e.g. "1 days 02:10:00" / "0 days 00:30:00"
            m = re.match(r"^\s*(\d+)\s+days?\s+(\d{1,2}):(\d{1,2}):(\d{1,2})\s*$", s, flags=re.IGNORECASE)
            if m:
                days = int(m.group(1))
                h = int(m.group(2))
                minute = int(m.group(3))
                sec = int(m.group(4))
                return (days * 24 * 60) + (h * 60) + minute + (sec / 60.0)

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

            # Handles text like "120 min", "90 minutes", "45.5"
            num_match = re.search(r"-?\d+(?:\.\d+)?", s)
            if num_match:
                try:
                    return float(num_match.group(0))
                except Exception:
                    return None

            try:
                return float(s)
            except Exception:
                return None

        def _normalize_job_type(value: str) -> str:
            v = str(value or "").strip().lower()
            if not v:
                return ""
            if "break" in v:
                return "Breakdown"
            if "maint" in v:
                return "Maintenance"
            return ""

        # =====================
        # Prepare figures/stats
        # =====================
        fig_task_count = None
        fig_duration = None
        stats_display = None
        stats_metrics = None

        if job_type_col:
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
                title="Pie Chart: Task Count",
            )
            fig_task_count.update_traces(textinfo="percent+label")

        if job_type_col and duration_col:
            tmp = tdf[[job_type_col, duration_col]].copy()
            tmp[job_type_col] = tmp[job_type_col].fillna("").astype(str).str.strip()
            tmp["Job Type Norm"] = tmp[job_type_col].apply(_normalize_job_type)
            tmp[duration_col] = tmp[duration_col].apply(_duration_to_minutes)
            tmp[duration_col] = pd.to_numeric(tmp[duration_col], errors="coerce")

            tmp = tmp[tmp["Job Type Norm"].isin(["Breakdown", "Maintenance"])].copy()
            tmp = tmp[tmp[duration_col].notna() & (tmp[duration_col] >= 0)].copy()

            if not tmp.empty:
                tmp = tmp.rename(columns={duration_col: "Duration (min)"})
                tmp["Duration (min)"] = pd.to_numeric(tmp["Duration (min)"], errors="coerce")
                tmp = tmp[tmp["Duration (min)"].notna()].copy()

                dur_sum = tmp.groupby("Job Type Norm", as_index=False)["Duration (min)"].sum()
                dur_sum.columns = ["Job Type", "Total Duration (min)"]

                fig_duration = px.pie(
                    dur_sum,
                    names="Job Type",
                    values="Total Duration (min)",
                    title="Pie Chart: Duration (Breakdown vs Maintenance)",
                )
                fig_duration.update_traces(
                    hole=0.45,
                    textinfo="percent+label",
                    hovertemplate="%{label}<br>Total: %{value:,.0f} min<br>Share: %{percent}<extra></extra>",
                )
                fig_duration.update_layout(legend_title_text="")

                stats = (
                    tmp.groupby("Job Type Norm")["Duration (min)"]
                    .agg(
                        Jobs="count",
                        Total_Min="sum",
                        Avg_Min="mean",
                        Median_Min="median",
                        Min_Min="min",
                        Max_Min="max",
                    )
                    .reset_index()
                    .rename(columns={"Job Type Norm": "Job Type"})
                )

                for jt in ["Breakdown", "Maintenance"]:
                    if jt not in set(stats["Job Type"].tolist()):
                        stats = pd.concat(
                            [
                                stats,
                                pd.DataFrame(
                                    [
                                        {
                                            "Job Type": jt,
                                            "Jobs": 0,
                                            "Total_Min": 0.0,
                                            "Avg_Min": 0.0,
                                            "Median_Min": 0.0,
                                            "Min_Min": 0.0,
                                            "Max_Min": 0.0,
                                        }
                                    ]
                                ),
                            ],
                            ignore_index=True,
                        )
                stats = stats.sort_values("Job Type")

                stats_display = stats.copy()
                for c in ["Total_Min", "Avg_Min", "Median_Min", "Min_Min", "Max_Min"]:
                    stats_display[c] = pd.to_numeric(stats_display[c], errors="coerce").fillna(0).round(1)

                def _v(stats_df: pd.DataFrame, job_type: str, col: str) -> float:
                    try:
                        return float(stats_df.loc[stats_df["Job Type"] == job_type, col].iloc[0])
                    except Exception:
                        return 0.0

                b_total = _v(stats, "Breakdown", "Total_Min")
                m_total = _v(stats, "Maintenance", "Total_Min")
                total_all = float(b_total + m_total)
                diff = float(b_total - m_total)
                b_share = (b_total / total_all * 100.0) if total_all > 0 else 0.0
                b_jobs = int(_v(stats, "Breakdown", "Jobs"))
                m_jobs = int(_v(stats, "Maintenance", "Jobs"))

                stats_metrics = {
                    "b_total": b_total,
                    "m_total": m_total,
                    "diff": diff,
                    "b_share": b_share,
                    "b_jobs": b_jobs,
                    "m_jobs": m_jobs,
                    "total_all": total_all,
                }

        # =====================
        # Layout
        # =====================
        left_col, right_col = st.columns(2)

        with left_col:
            if fig_task_count is None:
                st.warning("⚠️ Task report column not found: 'Job Type' (or 'Task Type').")
            else:
                st.plotly_chart(fig_task_count, use_container_width=True)

        with right_col:
            if fig_duration is None:
                if not (job_type_col and duration_col):
                    st.warning("⚠️ Task report columns not found: need 'Job Type' and 'Duration'.")
                else:
                    st.info("No duration data for Breakdown/Maintenance.")
            else:
                st.plotly_chart(fig_duration, use_container_width=True)

        st.markdown("---")
        st.markdown("### Statistical Analysis (Duration)")

        if stats_display is None or stats_metrics is None:
            st.info("No statistical duration analysis available.")
        else:
            s1, s2, s3 = st.columns(3)
            with s1:
                st.metric(
                    "Breakdown (Total min)",
                    f"{stats_metrics['b_total']:,.0f}",
                    f"{stats_metrics['b_share']:.1f}% share",
                )
                st.caption(f"Jobs: {stats_metrics['b_jobs']}")
            with s2:
                st.metric("Maintenance (Total min)", f"{stats_metrics['m_total']:,.0f}")
                st.caption(f"Jobs: {stats_metrics['m_jobs']}")
            with s3:
                st.metric("Difference (B - M)", f"{stats_metrics['diff']:,.0f}")
                if stats_metrics["total_all"] > 0:
                    st.caption(f"Total (B+M): {stats_metrics['total_all']:,.0f} min")

            st.dataframe(
                stats_display.rename(
                    columns={
                        "Total_Min": "Total (min)",
                        "Avg_Min": "Avg (min)",
                        "Median_Min": "Median (min)",
                        "Min_Min": "Min (min)",
                        "Max_Min": "Max (min)",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

except Exception as e:
    st.error(f"Failed to load/plot breakdown report data: {e}")