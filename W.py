
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO
import requests
import re
import datetime
import base64
from typing import Tuple, List, Dict

# ------------------------------
# Page config
# ------------------------------
st.set_page_config(
    page_title="DC | Task Dashboard (Corporate)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------
# Corporate CSS
# ------------------------------
st.markdown(
    """
<style>
/* Root, fonts & body */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"]  {
    font-family: 'Inter', system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    color: #0f172a;
}

/* Page background */
.stApp {
    background: linear-gradient(180deg,#f8fafc 0%, #f1f5f9 100%) !important;
}

/* Top header / nav */
.corp-topbar {
    background: linear-gradient(90deg, #0f172a, #0b3a66);
    color: #ffffff;
    padding: 18px 24px;
    border-radius: 8px;
    margin-bottom: 18px;
    box-shadow: 0 6px 18px rgba(12,18,31,0.12);
}
.corp-topbar h1 {
    margin: 0;
    font-weight: 700;
    letter-spacing: 0.2px;
    font-size: 20px;
}
.corp-topbar .muted {
    opacity: 0.85;
    font-size: 13px;
    color: rgba(255,255,255,0.9);
}

/* KPI Card */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 18px;
    margin-bottom: 18px;
}
.kpi-card {
    background: white;
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: 0 6px 14px rgba(12,18,31,0.06);
    border-left: 4px solid #e6eef8;
}
.kpi-title {
    font-size: 13px;
    color: #64748b;
    margin-bottom: 6px;
    font-weight: 600;
}
.kpi-value {
    font-size: 20px;
    font-weight: 700;
    color: #0f172a;
}

/* Sidebar custom */
section[data-testid="stSidebar"] .stButton>button {
    width: 100%;
}
.stSidebar {
    background: linear-gradient(180deg,#ffffff 0%, #f8fafc 100%) !important;
    padding: 18px !important;
    border-radius: 8px;
    box-shadow: 0 6px 18px rgba(12,18,31,0.04);
}

/* Table styling for custom HTML table */
.styled-table {
    border-collapse: collapse;
    width: 100%;
    font-size: 0.92rem;
    background: white;
    border-radius: 6px;
    overflow: hidden;
}
.styled-table th, .styled-table td {
    padding: 10px 12px;
    text-align: left;
    border-bottom: 1px solid #f1f5f9;
}
.styled-table thead th {
    background: #0b3a66;
    color: white;
    font-weight: 600;
    font-size: 13px;
}
.styled-table tbody tr:hover {
    background: #fbfdff;
}
/* urgent / overdue */
.urgent-highlight { background-color: #fff7ed; font-weight: 600; }
.overdue-highlight { background-color: #fff1f2; color: #7f1d1d; font-weight: 700; }

/* small helpers */
.section-box {
    background: white;
    padding: 14px;
    border-radius: 8px;
    box-shadow: 0 6px 18px rgba(12,18,31,0.04);
    margin-bottom: 16px;
}
.small-muted { color: #64748b; font-size: 13px; }
.link { color: #0b69d6; text-decoration: none; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------
# Constants and defaults
# ------------------------------
DEFAULT_SHEET_GVIZ_CSV = (
    "https://docs.google.com/spreadsheets/d/14-idXJHzHKCUQxxaqGZi-6S0G20gvPUhK4G16ci2FwI"
    "/gviz/tq?tqx=out:csv&gid=213021534"
)

PRIORITY_CANONICAL = {
    "most urgent": "Most Urgent",
    "mosturgent": "Most Urgent",
    "most_urgent": "Most Urgent",
    "urgent": "Most Urgent",
    "highest": "Most Urgent",
    "high": "High",
    "medium": "Medium",
    "med": "Medium",
    "low": "Low",
    "not urgent": "Low",
}

# ------------------------------
# Utility functions (kept & refined)
# ------------------------------
def safe_request_csv(url: str, timeout: int = 12) -> pd.DataFrame:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        text = resp.text
        return pd.read_csv(StringIO(text))
    except Exception as e:
        st.sidebar.warning(f"Failed to fetch CSV: {e}")
        return pd.DataFrame()

def normalize_string(val: object) -> str:
    if pd.isna(val):
        return ""
    s = str(val)
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

def canonical_priority(val: str) -> str:
    if val is None:
        return "Medium"
    val_norm = normalize_string(val)
    if val_norm == "":
        return "Medium"
    if val_norm in PRIORITY_CANONICAL:
        return PRIORITY_CANONICAL[val_norm]
    if "urgent" in val_norm:
        return "Most Urgent"
    if "high" in val_norm:
        return "High"
    if "medium" in val_norm or "med" in val_norm:
        return "Medium"
    if "low" in val_norm:
        return "Low"
    return "Medium"

def valid_sr_filter(df: pd.DataFrame) -> pd.DataFrame:
    if "Sr" not in df.columns:
        return df
    mask = df["Sr"].notna() & (df["Sr"].astype(str).str.strip() != "") & (df["Sr"].astype(str).str.strip().str.lower() != "sr")
    return df[mask].copy()

def create_clickable_file_link(file_value: str, sr_number: object) -> str:
    if pd.isna(file_value) or str(file_value).strip() == "" or str(file_value).strip().lower() == "file":
        return "No file"
    file_str = str(file_value).strip()
    if "drive.google.com" in file_str:
        match = re.search(r"[-\w]{25,}", file_str)
        if match:
            file_id = match.group(0)
            return f'<a class="link" href="https://drive.google.com/file/d/{file_id}/view" target="_blank">Open File</a>'
        else:
            return f'<a class="link" href="{file_str}" target="_blank">Open File</a>'
    if file_str.startswith("http://") or file_str.startswith("https://"):
        return f'<a class="link" href="{file_str}" target="_blank">Open Link</a>'
    return "No file"

def df_to_csv_download_link(df: pd.DataFrame, filename: str = "export.csv") -> str:
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a class="link" href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    return href

def parse_date_flexible(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    patterns = [
        ("%d/%m/%Y", r"^\d{1,2}/\d{1,2}/\d{4}$"),
        ("%d-%m-%Y", r"^\d{1,2}-\d{1,2}-\d{4}$"),
        ("%Y-%m-%d", r"^\d{4}-\d{1,2}-\d{1,2}$"),
        ("%d %b %Y", r"^\d{1,2} [A-Za-z]{3} \d{4}$"),
    ]
    for fmt, pat in patterns:
        if re.match(pat, s):
            try:
                dt = datetime.datetime.strptime(s, fmt)
                return dt.date().isoformat()
            except Exception:
                pass
    try:
        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if not pd.isna(dt):
            return dt.date().isoformat()
    except Exception:
        pass
    return s

# ------------------------------
# Sample fallback
# ------------------------------
def create_sample_data_large() -> pd.DataFrame:
    n = 120
    officers = ["CMFO", "DRO", "ADC (RD)", "ADC G", "Legal Cell", "AC G", "DyESA", "Election Tehsildar", "ADC (W)", "EO"]
    priorities = ["Most Urgent", "High", "Medium", "Low"]
    statuses = ["In progress", "Completed", "In progress", "In progress"]
    rows = []
    today = datetime.date.today()
    for i in range(1, n + 1):
        entry_date = (today - datetime.timedelta(days=np.random.randint(1, 60)))
        deadline_date = entry_date + datetime.timedelta(days=np.random.choice([7, 14, 30]))
        if i % 5 == 0:
            deadline_date = entry_date - datetime.timedelta(days=np.random.randint(1, 10))
        status = statuses[i % len(statuses)]
        response_date = np.nan
        if status == "Completed":
            response_date = (entry_date + datetime.timedelta(days=np.random.randint(1, 20))).strftime("%d/%m/%Y")
        rows.append({
            "Sr": i,
            "Marked to Officer": officers[i % len(officers)],
            "Priority": priorities[i % len(priorities)],
            "Status": status,
            "Subject": f"Task {i} - Administrative item regarding process {i%7}",
            "File": f"document_{i:03d}.pdf" if i % 3 != 0 else f"https://example.com/doc_{i}.pdf",
            "Entry Date": entry_date.strftime("%d/%m/%Y"),
            "Deadline": deadline_date.strftime("%d/%m/%Y"),
            "Response Recieved on": response_date,
            "Remarks": "Auto-generated sample data" if i % 5 else "Requires signature"
        })
    return pd.DataFrame(rows)

# ------------------------------
# Data loader
# ------------------------------
@st.cache_data(ttl=300)
def load_and_process(sheet_url: str) -> pd.DataFrame:
    raw = safe_request_csv(sheet_url)
    if raw.empty:
        raw = create_sample_data_large()

    raw.columns = [str(c).strip() for c in raw.columns]

    if "Sr" in raw.columns:
        first_row_vals = raw.iloc[0].astype(str).str.strip().str.lower().tolist()
        if "sr" in first_row_vals:
            raw = raw.iloc[1:].reset_index(drop=True)

    raw = valid_sr_filter(raw)

    expected_cols = ["Marked to Officer", "Priority", "Status", "File", "Subject", "Entry Date", "Remarks", "Sr", "Deadline", "Response Recieved on"]
    for col in expected_cols:
        if col not in raw.columns:
            raw[col] = np.nan
            st.sidebar.warning(f"Missing expected column: '{col}'. Added with NAs.")

    raw["Marked to Officer"] = raw["Marked to Officer"].fillna("Unknown").astype(str).str.strip()
    raw["Priority"] = raw["Priority"].apply(lambda v: canonical_priority(v))
    raw["Status"] = raw["Status"].fillna("In progress").astype(str).str.strip()

    raw["Entry Date (Parsed)"] = pd.to_datetime(raw["Entry Date"].apply(parse_date_flexible), errors="coerce")
    raw["Deadline (Parsed)"] = pd.to_datetime(raw["Deadline"].apply(parse_date_flexible), errors="coerce")
    raw["Response Date (Parsed)"] = pd.to_datetime(raw["Response Recieved on"].apply(parse_date_flexible), errors="coerce")

    raw["File Link"] = raw.apply(lambda r: create_clickable_file_link(r["File"], r.get("Sr", "")), axis=1)
    raw["Sr_original"] = raw["Sr"].astype(str)

    today = pd.Timestamp.today().normalize()
    is_pending = raw["Status"].str.lower() == "in progress"
    is_completed = raw["Status"].str.lower() == "completed"
    is_overdue = (raw["Deadline (Parsed)"] < today) & is_pending & raw["Deadline (Parsed)"].notna()
    is_due_soon = (raw["Deadline (Parsed)"] >= today) & (raw["Deadline (Parsed)"] <= today + pd.Timedelta(days=3)) & is_pending

    conditions = [is_completed, is_overdue, is_due_soon, is_pending]
    choices = ["Completed", "Overdue", "Due Soon", "Pending"]
    raw["Task_Status"] = np.select(conditions, choices, default="Pending")

    cols_to_keep = ["Sr_original", "Marked to Officer", "Priority", "Status", "Task_Status", "Subject", "Entry Date", "Deadline", "Response Recieved on", "File Link", "Remarks", "Entry Date (Parsed)", "Deadline (Parsed)", "Response Date (Parsed)"]
    raw = raw[[c for c in cols_to_keep if c in raw.columns]]

    return raw

# ------------------------------
# UI: Sidebar controls
# ------------------------------
def sidebar_controls():
    st.sidebar.markdown("## Controls & Data")
    sheet_url = st.sidebar.text_input("Google Sheet CSV URL (gviz CSV recommended)", value=DEFAULT_SHEET_GVIZ_CSV)
    show_debug = st.sidebar.checkbox("Show debug info (raw head)", value=False)
    highlight_urgent = st.sidebar.checkbox("Highlight Urgent/Overdue tasks", value=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Quick Actions**")
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()
    st.sidebar.markdown("---")
    return {
        "sheet_url": sheet_url.strip(),
        "show_debug": show_debug,
        "highlight_urgent": highlight_urgent,
    }

# ------------------------------
# UI: Header & KPI render
# ------------------------------
def render_header_and_kpis(df: pd.DataFrame):
    # Topbar
    st.markdown(
        f"""
        <div class="corp-topbar">
            <div style="display:flex;align-items:center;justify-content:space-between">
                <div>
                    <h1>District Collector Office — Task Dashboard</h1>
                    <div class="muted">Corporate-style view • Data refreshed: {pd.Timestamp.today().date().isoformat()}</div>
                </div>
                <div style="text-align:right">
                    <div class="muted">Built by Deep Shah</div>
                    <div class="muted">DC Ludhiana</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI numbers
    total_tasks = len(df)
    pending_df = df[df["Task_Status"].isin(["Pending", "Due Soon", "Overdue"])]
    total_pending = len(pending_df)
    total_overdue = len(df[df["Task_Status"] == "Overdue"])
    unique_officers = df["Marked to Officer"].nunique()
    most_urgent_total = len(df[(df["Priority"] == "Most Urgent") & (df["Task_Status"] != "Completed")])

    kpi_html = f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-title">Total Tasks</div>
            <div class="kpi-value">{total_tasks}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Pending (Incl. Due Soon/Overdue)</div>
            <div class="kpi-value">{total_pending}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Total Overdue</div>
            <div class="kpi-value">{total_overdue}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Unique Officers</div>
            <div class="kpi-value">{unique_officers}</div>
        </div>
    </div>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)

# ------------------------------
# Dashboard summary (charts & performance)
# ------------------------------
def dashboard_summary_page(df: pd.DataFrame, settings: dict):
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("### Overview & Officer Performance")
    pending_df = df[df["Task_Status"].isin(["Pending", "Due Soon", "Overdue"])].copy()
    completed_df = df[df["Task_Status"] == "Completed"].copy()

    all_officers = df["Marked to Officer"].unique()
    officer_stats_base = pd.DataFrame({"Marked to Officer": all_officers})

    officer_pending_counts_raw = pending_df.groupby("Marked to Officer")["Task_Status"].value_counts().unstack(fill_value=0)
    for col in ["Overdue", "Due Soon", "Pending"]:
        if col not in officer_pending_counts_raw.columns:
            officer_pending_counts_raw[col] = 0
    officer_pending_counts_raw = officer_pending_counts_raw.reindex(columns=["Overdue", "Due Soon", "Pending"], fill_value=0)
    officer_pending_counts_raw["Total Pending"] = officer_pending_counts_raw.sum(axis=1)
    officer_pending_counts_raw = officer_pending_counts_raw.reset_index()

    today = pd.Timestamp.today().normalize()
    last_week = today - pd.Timedelta(days=7)
    recent_completed = completed_df[
        (completed_df["Response Date (Parsed)"].notna()) &
        (completed_df["Response Date (Parsed)"] >= last_week) &
        (completed_df["Response Date (Parsed)"] <= today)
    ]
    completed_counts_7d = recent_completed.groupby("Marked to Officer").size().reset_index(name="Completed (Last 7 Days)")
    completed_counts_total = completed_df.groupby("Marked to Officer").size().reset_index(name="Completed (Total)")

    officer_summary = officer_stats_base.merge(officer_pending_counts_raw, on="Marked to Officer", how="left")
    officer_summary = officer_summary.merge(completed_counts_total, on="Marked to Officer", how="outer")
    officer_summary = officer_summary.merge(completed_counts_7d, on="Marked to Officer", how="outer")

    for col in ["Overdue", "Due Soon", "Pending", "Total Pending", "Completed (Last 7 Days)", "Completed (Total)"]:
        if col not in officer_summary.columns:
            officer_summary[col] = 0
        officer_summary[col] = officer_summary[col].fillna(0).astype(int)

    officer_summary["Total_Tasks_Handled"] = officer_summary["Completed (Total)"] + officer_summary["Total Pending"]
    officer_summary["Performance_%"] = officer_summary.apply(
        lambda row: (row["Completed (Total)"] / row["Total_Tasks_Handled"] * 100) if row["Total_Tasks_Handled"] > 0 else 0,
        axis=1
    )

    officer_bar_chart_data = officer_summary[officer_summary["Total Pending"] > 0].copy()
    officer_bar_chart_data = officer_bar_chart_data.sort_values("Total Pending", ascending=True)

    # Layout: left wide chart, right summary table
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Pending Tasks by Officer")
        if not officer_bar_chart_data.empty:
            fig = px.bar(
                officer_bar_chart_data,
                x="Total Pending",
                y="Marked to Officer",
                orientation="h",
                labels={"Total Pending": "Number of Tasks", "Marked to Officer": "Officer"},
                text_auto=True,
                height=420,
                color="Total Pending",
                color_continuous_scale="Blues"
            )
            fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No pending tasks currently.")

        st.markdown("### Overall status")
        total_pending = officer_summary["Total Pending"].sum()
        total_overdue = officer_summary["Overdue"].sum()
        total_tasks_in_df = len(df)
        percent_pending = (total_pending / total_tasks_in_df * 100) if total_tasks_in_df > 0 else 0

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Pending", f"{total_pending}")
        with m2:
            st.metric("% Pending", f"{percent_pending:.1f}%")
        with m3:
            st.metric("Total Overdue", f"{total_overdue}")

    with col2:
        st.subheader("Top / Bottom Performers")
        rankable_officers = officer_summary[officer_summary["Total_Tasks_Handled"] > 0].copy()

        best_5 = rankable_officers.sort_values(by=["Performance_%", "Overdue", "Total Pending"], ascending=[False, True, True]).head(5)
        best_5_display = best_5[["Marked to Officer", "Performance_%", "Completed (Total)", "Total Pending"]].copy()
        best_5_display["Performance_%"] = best_5_display["Performance_%"].map('{:,.1f}%'.format)
        st.markdown("**Best (Top 5)**")
        st.dataframe(best_5_display, use_container_width=True, hide_index=True)

        worst_5 = rankable_officers.sort_values(by=["Performance_%", "Overdue", "Total Pending"], ascending=[True, False, False]).head(5)
        worst_5_display = worst_5[["Marked to Officer", "Performance_%", "Completed (Total)", "Total Pending"]].copy()
        worst_5_display["Performance_%"] = worst_5_display["Performance_%"].map('{:,.1f}%'.format)
        st.markdown("**Worst (Top 5)**")
        st.dataframe(worst_5_display, use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# Officer Pie Charts section
# ------------------------------
def render_officer_pie_charts(df: pd.DataFrame):
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.subheader("Officer: Pending Task Distribution")
    pending_df = df[df["Task_Status"].isin(["Pending", "Due Soon", "Overdue"])].copy()
    officers_with_pending = sorted(pending_df["Marked to Officer"].unique().tolist())
    if not officers_with_pending:
        st.info("No officers currently have pending tasks.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    selected_officer = st.selectbox("Select Officer", options=officers_with_pending)
    if selected_officer:
        officer_tasks = pending_df[pending_df["Marked to Officer"] == selected_officer]
        if officer_tasks.empty:
            st.info(f"{selected_officer} has no pending tasks.")
        else:
            status_counts = officer_tasks["Task_Status"].value_counts().reset_index()
            status_counts.columns = ["Task Status", "Count"]
            fig = px.pie(
                status_counts,
                values="Count",
                names="Task Status",
                title=f"{selected_officer} — Pending Distribution",
                color="Task Status",
                color_discrete_map={"Overdue": "#dc2626", "Due Soon": "#f59e0b", "Pending": "#0b69d6"}
            )
            fig.update_traces(textposition='inside', textinfo='percent+label+value')
            st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# All tasks table
# ------------------------------
def render_all_tasks_table(df: pd.DataFrame, settings: dict):
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.subheader("All Tasks (Filtered View)")
    col1, col2, col3, col4 = st.columns(4)
    officers = sorted(df["Marked to Officer"].fillna("Unknown").unique().tolist())
    with col1:
        officer_filter = st.multiselect("Officer", options=["All Officers"] + officers, default="All Officers")
    with col2:
        priority_options = sorted(df["Priority"].unique().tolist())
        priority_filter = st.multiselect("Priority", options=["All"] + priority_options, default="All")
    with col3:
        status_options = sorted(df["Task_Status"].unique().tolist())
        status_filter = st.multiselect("Task Status", options=["All"] + status_options, default="All")
    with col4:
        q = st.text_input("Search subject / remarks:", value="")

    filtered = df.copy()
    if "All Officers" not in officer_filter:
        filtered = filtered[filtered["Marked to Officer"].isin(officer_filter)]
    if "All" not in priority_filter:
        filtered = filtered[filtered["Priority"].isin(priority_filter)]
    if "All" not in status_filter:
        filtered = filtered[filtered["Task_Status"].isin(status_filter)]
    if q.strip():
        qlow = q.strip().lower()
        mask_subject = filtered["Subject"].astype(str).str.lower().str.contains(qlow, na=False)
        mask_remarks = filtered["Remarks"].astype(str).str.lower().str.contains(qlow, na=False)
        filtered = filtered[mask_subject | mask_remarks]

    # Sort: Overdue, Due Soon, Pending, Completed
    status_order = {"Overdue": 3, "Due Soon": 2, "Pending": 1, "Completed": 0}
    filtered["__status_rank"] = filtered["Task_Status"].map(status_order).fillna(0)
    filtered = filtered.sort_values(by=["__status_rank", "Priority"], ascending=[False, False]).drop(columns="__status_rank")

    st.markdown(f"**Showing {len(filtered)} rows**")
    display_cols = ["Sr_original", "Marked to Officer", "Task_Status", "Priority", "Subject", "Entry Date", "Deadline", "File Link", "Remarks"]
    available_cols = [c for c in display_cols if c in filtered.columns]

    # Build HTML table
    if settings["highlight_urgent"]:
        def style_row_html(row):
            cls = ""
            if row["Task_Status"] == "Overdue":
                cls = "overdue-highlight"
            elif row["Priority"] == "Most Urgent":
                cls = "urgent-highlight"
            cells = f"<tr class='{cls}'>"
            for col in available_cols:
                cell_value = row[col] if pd.notna(row[col]) else ""
                cells += f"<td>{cell_value}</td>"
            cells += "</tr>"
            return cells

        header_html = "".join(f"<th>{col}</th>" for col in available_cols)
        rows_html = "".join(filtered.apply(style_row_html, axis=1))
        table_html = f"""
        <table class="styled-table">
            <thead><tr>{header_html}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        """
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.dataframe(filtered[available_cols], use_container_width=True, hide_index=True)

    st.markdown(df_to_csv_download_link(filtered, filename="all_tasks_filtered.csv"), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# Main
# ------------------------------
def main():
    settings = sidebar_controls()
    sheet_url = settings["sheet_url"] or DEFAULT_SHEET_GVIZ_CSV
    df = load_and_process(sheet_url)

    if settings["show_debug"]:
        st.sidebar.markdown("### Debug Info")
        st.sidebar.write("Dataframe shape:", df.shape)
        st.sidebar.write("Columns:", df.columns.tolist())
        st.sidebar.write("Task_Status values:", df["Task_Status"].unique().tolist())
        st.sidebar.write(df.head(8))

    render_header_and_kpis(df)
    dashboard_summary_page(df, settings)
    render_officer_pie_charts(df)
    render_all_tasks_table(df, settings)

    # Sidebar About
    st.sidebar.markdown("---")
    st.sidebar.header("About")
    st.sidebar.markdown(
        """
        **DC Ludhiana — Task Dashboard**

        Built by: Deep Shah  
        Ownership: DC Ludhiana Office

        Contact: +91-8905309441 • 18deep.shah2002@gmail.com
        """
    )

if __name__ == "__main__":
    main()
