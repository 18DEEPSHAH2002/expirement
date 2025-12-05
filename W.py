"""
Improved Light-Version of: Task Management Dashboard - Streamlit App
Applied fixes: date regex, HTML escaping, Task_Status ordering, Performance calculation,
file link handling, column aliasing, safer refresh, and small cleanups.
Author: Updated for Deep Shah
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO
import requests
import re
import datetime
import base64
from html import escape

# ------------------------------
# Page config & CSS (kept from original, minor adjustments allowed)
# ------------------------------
st.set_page_config(
    page_title="Task Management Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
/* (Styles unchanged from the original - omitted here for brevity in comments) */
.stApp {
    background-color: #dbeafe !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: #000000 !important;
}
label[data-baseweb="select"] { color: #0c4a6e !important; font-weight: 600 !important; }
.styled-table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
.styled-table th, .styled-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
.styled-table th { background-color: #3b82f6; color: white; }
.styled-table tr:nth-child(even) { background-color: #f2f2f2; }
.styled-table a { color: #2563eb; text-decoration: none; }
.styled-table a:hover { text-decoration: underline; }
.urgent-highlight { background-color: #fee2e2; font-weight: bold; }
.overdue-highlight { background-color: #fecaca; font-weight: bold; color: #b91c1c; }
.metric-card, .stMetric { background-color: #eff6ff !important; color: #000000 !important; border-radius: 8px; padding: 1rem; border: 1px solid #bfdbfe; }
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

PRIORITY_ORDER = ["Most Urgent", "High", "Medium", "Low"]
TASK_STATUS_ORDER = ["Overdue", "Due Soon", "Pending", "Completed"]

# ------------------------------
# Utility functions
# ------------------------------
def safe_request_csv(url: str, timeout: int = 12) -> pd.DataFrame:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        text = resp.text
        return pd.read_csv(StringIO(text))
    except Exception as e:
        st.warning(f"Failed to fetch CSV from URL: {e}")
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
    # Google Drive link detection (support both /d/<id>/ and open?id= formats)
    if "drive.google.com" in file_str:
        # Try to capture file id from either /d/<id> or id=<id>
        m1 = re.search(r"/d/([-\w]{25,})", file_str)
        m2 = re.search(r"[?&]id=([-\w]{25,})", file_str)
        file_id = m1.group(1) if m1 else (m2.group(1) if m2 else None)
        if file_id:
            return f'<a class="file-link" href="https://drive.google.com/file/d/{file_id}/view" target="_blank">📎 Open File</a>'
        else:
            return f'<a class="file-link" href="{escape(file_str)}" target="_blank">📎 Open File</a>'
    if file_str.startswith("http://") or file_str.startswith("https://"):
        return f'<a class="file-link" href="{escape(file_str)}" target="_blank">📎 Open Link</a>'
    # plain filename - show as text
    return escape(file_str)

def df_to_csv_download_link(df: pd.DataFrame, filename: str = "export.csv") -> str:
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download CSV</a>'
    return href

def summarize_priority_counts(df: pd.DataFrame) -> pd.Series:
    counts = df["Priority"].value_counts()
    ordered = PRIORITY_ORDER
    return pd.Series({k: int(counts.get(k, 0)) for k in ordered})

def parse_date_flexible(x):
    if pd.isna(x) or str(x).strip() == "":
        return pd.NaT
    s = str(x).strip()
    patterns = [
        ("%d/%m/%Y", r"^\d{1,2}/\d{1,2}/\d{4}$"),
        ("%d-%m-%Y", r"^\d{1,2}-\d{1,2}-\d{4}$"),  # FIXED: corrected regex for DD-MM-YYYY
        ("%Y-%m-%d", r"^\d{4}-\d{1,2}-\d{1,2}$"),
        ("%d %b %Y", r"^\d{1,2} [A-Za-z]{3} \d{4}$"),
    ]
    for fmt, pat in patterns:
        if re.match(pat, s):
            try:
                dt = datetime.datetime.strptime(s, fmt)
                return pd.Timestamp(dt.date())
            except Exception:
                pass
    try:
        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if not pd.isna(dt):
            return pd.Timestamp(dt.date())
    except Exception:
        pass
    return pd.NaT

# ------------------------------
# Sample data generator (kept)
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
# Column helpers (alias detection)
# ------------------------------
def find_first_matching_col(df: pd.DataFrame, candidates: list):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# ------------------------------
# Data loading & processing
# ------------------------------
@st.cache_data(ttl=300)
def load_and_process(sheet_url: str) -> pd.DataFrame:
    raw = safe_request_csv(sheet_url)
    if raw.empty:
        raw = create_sample_data_large()

    raw.columns = [str(c).strip() for c in raw.columns]

    # Detect header rows repeated
    if "Sr" in raw.columns and len(raw) > 0:
        first_row_vals = raw.iloc[0].astype(str).str.strip().str.lower().tolist()
        if "sr" in first_row_vals:
            raw = raw.iloc[1:].reset_index(drop=True)

    raw = valid_sr_filter(raw)

    # Column aliasing: provide expected names if variations exist
    # We'll map common misspellings to canonical names in the dataframe itself
    canonical_map = {}
    # possible variants for response date
    resp_candidates = ["Response Recieved on", "Response Received on", "Response Recieved", "Response Received"]
    found_resp = find_first_matching_col(raw, resp_candidates)
    if found_resp and found_resp != "Response Recieved on":
        canonical_map[found_resp] = "Response Recieved on"  # keep original canonical used elsewhere to minimize changes

    # Normalize names (also preserve any columns already matching expected)
    # Apply renaming if necessary
    if canonical_map:
        raw = raw.rename(columns=canonical_map)

    expected_cols = ["Marked to Officer", "Priority", "Status", "File", "Subject", "Entry Date", "Remarks", "Sr", "Deadline", "Response Recieved on"]
    for col in expected_cols:
        if col not in raw.columns:
            raw[col] = np.nan
            # don't spam sidebar with many warnings; only show once if debug is enabled (handled by caller)

    # Standardize officer field
    raw["Marked to Officer"] = raw["Marked to Officer"].fillna("Unknown").astype(str).str.strip()

    # Normalize priority
    raw["Priority"] = raw["Priority"].apply(lambda v: canonical_priority(v))

    # Clean status strings
    raw["Status"] = raw["Status"].fillna("In progress").astype(str).str.strip()

    # Parse key dates robustly
    raw["Entry Date (Parsed)"] = raw["Entry Date"].apply(parse_date_flexible)
    raw["Deadline (Parsed)"] = raw["Deadline"].apply(parse_date_flexible)
    raw["Response Date (Parsed)"] = raw["Response Recieved on"].apply(parse_date_flexible) if "Response Recieved on" in raw.columns else pd.NaT

    # File link column as HTML or escaped text
    raw["File Link"] = raw.apply(lambda r: create_clickable_file_link(r.get("File", ""), r.get("Sr", "")), axis=1)

    # Keep original Sr textual form
    raw["Sr_original"] = raw["Sr"].astype(str)

    # Calculate Task_Status
    today = pd.Timestamp.today().normalize()
    is_pending = raw["Status"].str.lower() == "in progress"
    is_completed = raw["Status"].str.lower() == "completed"
    is_overdue = (raw["Deadline (Parsed)"] < today) & is_pending & raw["Deadline (Parsed)"].notna()
    is_due_soon = (raw["Deadline (Parsed)"] >= today) & (raw["Deadline (Parsed)"] <= today + pd.Timedelta(days=3)) & is_pending

    conditions = [is_completed, is_overdue, is_due_soon, is_pending]
    choices = ["Completed", "Overdue", "Due Soon", "Pending"]
    raw["Task_Status"] = np.select(conditions, choices, default="Pending")

    # Convert Task_Status to ordered categorical for correct sorting
    raw["Task_Status"] = pd.Categorical(raw["Task_Status"], categories=TASK_STATUS_ORDER, ordered=True)

    # Convert Priority to ordered categorical for consistent sorting
    raw["Priority"] = pd.Categorical(raw["Priority"], categories=PRIORITY_ORDER, ordered=True)

    # Reorder and keep useful parsed date columns
    cols_to_keep = ["Sr_original", "Marked to Officer", "Priority", "Status", "Task_Status", "Subject", "Entry Date", "Deadline", "Response Recieved on", "File Link", "Remarks",
                    "Entry Date (Parsed)", "Deadline (Parsed)", "Response Date (Parsed)"]
    cols_to_keep = [c for c in cols_to_keep if c in raw.columns]
    raw = raw[cols_to_keep]

    return raw

# ------------------------------
# UI helper components
# ------------------------------
def sidebar_controls():
    st.sidebar.title("Controls & Settings")
    sheet_url = st.sidebar.text_input("Google Sheet CSV URL (gviz CSV recommended)", value=DEFAULT_SHEET_GVIZ_CSV)
    show_debug = st.sidebar.checkbox("Show debug info (raw head)", value=False)
    highlight_urgent = st.sidebar.checkbox("Highlight Urgent/Overdue tasks", value=True)

    # Safer refresh mechanism using session state to avoid accidental loops
    if "last_refreshed" not in st.session_state:
        st.session_state["last_refreshed"] = False

    if st.sidebar.button(" Refresh Data Now"):
        # Clear cache and mark refreshed; single rerun will show updated data
        st.cache_data.clear()
        st.session_state["last_refreshed"] = True
        st.experimental_rerun()

    return {
        "sheet_url": sheet_url.strip(),
        "show_debug": show_debug,
        "highlight_urgent": highlight_urgent,
    }

def render_global_metrics(df: pd.DataFrame):
    st.markdown('<h1 class="main-header"> Task Management Dashboard</h1>', unsafe_allow_html=True)
    total_tasks = len(df)
    pending_df = df[df["Task_Status"].isin(["Pending", "Due Soon", "Overdue"])]
    total_pending = len(pending_df)
    total_overdue = len(df[df["Task_Status"] == "Overdue"])
    unique_officers = df["Marked to Officer"].nunique()
    most_urgent_total = len(df[(df["Priority"] == "Most Urgent") & (df["Task_Status"] != "Completed")])

    c1, c2, c3, c4, c5 = st.columns(5, gap="medium")
    with c1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Tasks (All)", total_tasks)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Pending Tasks", total_pending)
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Overdue", total_overdue)
        st.markdown("</div>", unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Pending 'Most Urgent'", most_urgent_total)
        st.markdown("</div>", unsafe_allow_html=True)
    with c5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Officers", unique_officers)
        st.markdown("</div>", unsafe_allow_html=True)

def dashboard_summary_page(df: pd.DataFrame, settings: dict):
    st.header("Dashboard Summary")
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

    officer_stats = officer_stats_base.merge(officer_pending_counts_raw, on="Marked to Officer", how="left")

    today = pd.Timestamp.today().normalize()
    last_week = today - pd.Timedelta(days=7)
    recent_completed = completed_df[
        (completed_df["Response Date (Parsed)"].notna()) &
        (completed_df["Response Date (Parsed)"] >= last_week) &
        (completed_df["Response Date (Parsed)"] <= today)
    ]
    completed_counts_7d = recent_completed.groupby("Marked to Officer").size().reset_index(name="Completed (Last 7 Days)")

    completed_counts_total = completed_df.groupby("Marked to Officer").size().reset_index(name="Completed (Total)")

    officer_summary = officer_stats.merge(completed_counts_total, on="Marked to Officer", how="outer")
    officer_summary = officer_summary.merge(completed_counts_7d, on="Marked to Officer", how="outer")

    for col in ["Overdue", "Due Soon", "Pending", "Total Pending", "Completed (Last 7 Days)", "Completed (Total)"]:
        if col not in officer_summary.columns:
            officer_summary[col] = 0
        officer_summary[col] = officer_summary[col].fillna(0).astype(int)

    officer_summary["Total_Tasks_Handled"] = officer_summary["Completed (Total)"] + officer_summary["Total Pending"]

    # Correct performance calculation, avoid division by zero and show 0% when no tasks ever handled
    def perf(row):
        total = row["Total_Tasks_Handled"]
        completed = row["Completed (Total)"]
        if total == 0:
            return 0.0
        return (completed / total) * 100.0
    officer_summary["Performance_%"] = officer_summary.apply(perf, axis=1)

    officer_bar_chart_data = officer_summary[officer_summary["Total Pending"] > 0].copy()
    officer_bar_chart_data = officer_bar_chart_data.sort_values("Total Pending", ascending=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Officer-wise Pending Tasks")
        if not officer_bar_chart_data.empty:
            fig = px.bar(
                officer_bar_chart_data,
                x="Total Pending",
                y="Marked to Officer",
                orientation="h",
                title="Total Pending Tasks (Overdue + Due Soon + Pending)",
                labels={"Total Pending": "Number of Tasks", "Marked to Officer": "Officer"},
                color="Total Pending",
                color_continuous_scale="Blues",
                height=450,
                text_auto=True
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No pending tasks to show.")

        st.subheader("Overall Status")
        total_pending = officer_summary["Total Pending"].sum()
        total_overdue = officer_summary["Overdue"].sum()
        total_tasks_in_df = len(df)
        percent_pending = (total_pending / total_tasks_in_df * 100) if total_tasks_in_df > 0 else 0

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Tasks Pending", f"{total_pending} / {total_tasks_in_df}")
        with m2:
            st.metric("% of Tasks Pending", f"{percent_pending:.1f}%")
        with m3:
            st.metric("Total Overdue", total_overdue)

    with col2:
        rankable_officers = officer_summary[officer_summary["Total_Tasks_Handled"] > 0].copy()
        st.subheader("Top 5 Best Performance")
        st.markdown("<small>(Based on: Highest Completion %)</small>", unsafe_allow_html=True)
        best_5 = rankable_officers.sort_values(by=["Performance_%", "Overdue", "Total Pending"], ascending=[False, True, True]).head(5)
        best_5_display = best_5[["Marked to Officer", "Performance_%", "Completed (Total)", "Total Pending"]].copy()
        best_5_display["Performance_%"] = best_5_display["Performance_%"].map('{:,.1f}%'.format)
        st.dataframe(best_5_display, use_container_width=True, hide_index=True)

        st.subheader("Top 5 Worst Performance")
        st.markdown("<small>(Based on: Lowest Completion %)</small>", unsafe_allow_html=True)
        worst_5 = rankable_officers.sort_values(by=["Performance_%", "Overdue", "Total Pending"], ascending=[True, False, False]).head(5)
        worst_5_display = worst_5[["Marked to Officer", "Performance_%", "Completed (Total)", "Total Pending"]].copy()
        worst_5_display["Performance_%"] = worst_5_display["Performance_%"].map('{:,.1f}%'.format)
        st.dataframe(worst_5_display, use_container_width=True, hide_index=True)

def render_officer_pie_charts(df: pd.DataFrame):
    st.markdown("---")
    st.header("Officer-wise Pending Distribution")
    st.markdown("Select an officer to see the breakdown of their pending tasks.")
    pending_df = df[df["Task_Status"].isin(["Pending", "Due Soon", "Overdue"])].copy()
    officers_with_pending = sorted(pending_df["Marked to Officer"].unique().tolist())
    if not officers_with_pending:
        st.info("No officers currently have pending tasks.")
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
                title=f"Pending Task Distribution for: {selected_officer}"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label+value')
            st.plotly_chart(fig, use_container_width=True)

def render_all_tasks_table(df: pd.DataFrame, settings: dict):
    st.markdown("---")
    st.header(" All Tasks (Filtered View)")
    st.markdown("Use the filters below to inspect all rows. The table is sorted by Task Status (Overdue first).")

    col1, col2, col3, col4 = st.columns(4)
    officers = sorted(df["Marked to Officer"].fillna("Unknown").unique().tolist())
    with col1:
        officer_filter = st.multiselect("Filter by Officer", options=["All Officers"] + officers, default="All Officers")
    with col2:
        priority_options = [p for p in PRIORITY_ORDER if p in df["Priority"].cat.categories] if pd.api.types.is_categorical_dtype(df["Priority"]) else sorted(df["Priority"].unique().tolist())
        priority_filter = st.multiselect("Filter by Priority", options=["All"] + priority_options, default="All")
    with col3:
        status_options = TASK_STATUS_ORDER
        status_filter = st.multiselect("Filter by Task Status", options=["All"] + status_options, default="All")
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

    # Sort by Task_Status (ordered categorical) and Priority (ordered categorical)
    # We want Overdue (first), then Due Soon, then Pending, then Completed.
    # Because Task_Status is ordered with Overdue first, ascending=True places Overdue at top.
    sort_cols = []
    ascending = []
    if "Task_Status" in filtered.columns:
        sort_cols.append("Task_Status")
        ascending.append(True)
    if "Priority" in filtered.columns:
        sort_cols.append("Priority")
        ascending.append(True)
    if sort_cols:
        filtered = filtered.sort_values(by=sort_cols, ascending=ascending)

    st.markdown(f"**Showing {len(filtered)} rows**")

    display_cols = ["Sr_original", "Marked to Officer", "Task_Status", "Priority", "Subject", "Entry Date", "Deadline", "File Link", "Remarks"]
    available_cols = [c for c in display_cols if c in filtered.columns]

    if settings["highlight_urgent"]:
        def style_row_html(row):
            cls = ""
            if str(row.get("Task_Status")) == "Overdue":
                cls = "overdue-highlight"
            elif str(row.get("Priority")) == "Most Urgent":
                cls = "urgent-highlight"

            row_cells = f"<tr class='{cls}'>"
            for col in available_cols:
                cell_value = row[col] if pd.notna(row[col]) else ""
                # Do not escape HTML for File Link column (it contains safe HTML anchors),
                # but escape everything else to avoid accidental HTML injection.
                if col == "File Link":
                    row_cells += f"<td>{cell_value}</td>"
                else:
                    row_cells += f"<td>{escape(str(cell_value))}</td>"
            row_cells += "</tr>"
            return row_cells

        header = "".join(f"<th>{escape(col)}</th>" for col in available_cols)
        rows_html = "".join(filtered.apply(style_row_html, axis=1))
        table_html = f"""
        <table class="styled-table">
            <thead><tr>{header}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        """
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.dataframe(filtered[available_cols], use_container_width=True, hide_index=True)

    st.markdown(df_to_csv_download_link(filtered, filename="all_tasks_filtered.csv"), unsafe_allow_html=True)

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
        st.sidebar.write("Task_Status values:", df["Task_Status"].cat.categories.tolist() if pd.api.types.is_categorical_dtype(df["Task_Status"]) else df["Task_Status"].unique().tolist())
        st.sidebar.write("Sample rows:")
        st.sidebar.dataframe(df.head(10))

    render_global_metrics(df)
    dashboard_summary_page(df, settings)
    render_officer_pie_charts(df)
    render_all_tasks_table(df, settings)

    st.sidebar.markdown("---")
    st.sidebar.header(" About ")
    st.sidebar.markdown(
        """
        THIS IS WORKING DASHBOARD FOR LUDHIANA ADMINISTRATION ONLY 

        THIS DASHBOARD IS BUILD BY DEEP SHAH  
        THE OWNERSHIP IS UNDER DC LUDHIANA OFFICE 
        """
    )
    st.sidebar.markdown("### Contact / Notes")
    st.sidebar.markdown("If any changes happen in the excel and get any bug or loophole, contact: +918905309441; gmail:18deep.shah2002@gmail.com")

if __name__ == "__main__":
    main()
