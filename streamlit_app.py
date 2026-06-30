from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = APP_DIR / "draft_data.csv"
REQUIRED_COLUMNS = {
    "Fantasy Team",
    "Round",
    "Ov Pick",
    "Pos",
    "Player",
    "DraftGP",
    "DraftPts",
    "DraftRank",
    "DrafRankDelta",
    "SeasonPts",
    "SeasonRank",
    "SeasonDelta",
    "Utilisation",
}
NUMERIC_COLUMNS = [
    "Round",
    "Pick",
    "Ov Pick",
    "DraftGP",
    "DraftPts",
    "DraftPPG",
    "DraftRank",
    "DrafRankDelta",
    "SeasonGP",
    "SeasonPts",
    "SeasonPPG",
    "SeasonRank",
    "SeasonDelta",
    "Utilisation",
]
POSITION_NAMES = {"G": "Goalkeeper", "D": "Defender", "M": "Midfielder", "F": "Forward"}


st.set_page_config(
    page_title="Cheese Draft Review 25/26",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.7rem; padding-bottom: 3rem; max-width: 1500px;}
      [data-testid="stMetric"] {background: rgba(128,128,128,.07); border: 1px solid rgba(128,128,128,.16); padding: .8rem 1rem; border-radius: .75rem;}
      .draft-callout {padding: .85rem 1rem; border-left: 4px solid #f0b429; background: rgba(240,180,41,.08); border-radius: .35rem; margin: .5rem 0 1rem 0;}
      .small-note {font-size: .88rem; opacity: .78;}
      .team-badge {display:inline-block; padding:.25rem .55rem; border-radius:999px; font-weight:700; background:rgba(240,180,41,.15); border:1px solid rgba(240,180,41,.3);}
    </style>
    """,
    unsafe_allow_html=True,
)


def repair_text(value: object) -> object:
    if not isinstance(value, str) or not any(marker in value for marker in ("Ã", "Â", "â")):
        return value
    try:
        return value.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame:
    return pd.read_csv(DEFAULT_DATA)


@st.cache_data(show_spinner=False)
def load_uploaded_data(file_bytes: bytes, filename: str) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    source = BytesIO(file_bytes)
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".xlsx", ".xlsm"}:
        return pd.read_excel(source, sheet_name="Draft", engine="openpyxl")
    raise ValueError("Please upload a CSV or XLSX workbook.")


def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError("The Draft data is missing: " + ", ".join(sorted(missing)))

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["Fantasy Team", "Player", "Team", "Pos"]:
        if col in df.columns:
            df[col] = df[col].map(repair_text).astype("string").str.strip()

    df["Utilisation"] = df["Utilisation"].clip(lower=0, upper=1)
    df["Position"] = df["Pos"].map(POSITION_NAMES).fillna(df["Pos"])
    df["Draft Value"] = np.where(df["DrafRankDelta"] > 0, "Positive", "Negative")
    return df.dropna(subset=["Fantasy Team", "Player", "Ov Pick"]).copy()


def minmax(series: pd.Series, invert: bool = False) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    low, high = values.min(), values.max()
    scaled = pd.Series(0.5, index=values.index) if high == low else (values - low) / (high - low)
    return 1 - scaled if invert else scaled


def calculate_team_scores(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("Fantasy Team", as_index=False)
        .agg(
            DraftPts=("DraftPts", "sum"),
            DraftDelta=("DrafRankDelta", "sum"),
            SeasonPts=("SeasonPts", "sum"),
            PositivePicks=("DrafRankDelta", lambda s: int((s > 0).sum())),
            Picks=("Player", "count"),
            ZeroPointPicks=("DraftPts", lambda s: int((s <= 0).sum())),
        )
    )
    utilised = df.groupby("Fantasy Team").apply(
        lambda x: x["DraftPts"].sum() / x["SeasonPts"].sum() if x["SeasonPts"].sum() else 0,
        include_groups=False,
    )
    grouped["Utilisation"] = grouped["Fantasy Team"].map(utilised).clip(0, 1)

    grouped["Composite"] = (
        0.40 * minmax(grouped["DraftPts"])
        + 0.25 * minmax(grouped["DraftDelta"])
        + 0.15 * minmax(grouped["SeasonPts"])
        + 0.10 * minmax(grouped["Utilisation"])
        + 0.10 * minmax(grouped["PositivePicks"])
    )
    # A league-report-card scale: even the weakest full draft receives a score above zero,
    # while an exceptional draft can approach but not automatically receive 10.
    grouped["Score"] = (2.2 + 7.2 * grouped["Composite"]).round(1)
    grouped = grouped.sort_values(["Score", "DraftPts"], ascending=False).reset_index(drop=True)
    grouped.insert(0, "Rank", grouped.index + 1)
    return grouped


def pick_rankings(df: pd.DataFrame, early_max: int = 5, late_min: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    early = df[df["Round"] <= early_max].copy()
    late = df[df["Round"] >= late_min].copy()

    early["Rating"] = 100 * (
        0.45 * minmax(early["DrafRankDelta"], invert=True)
        + 0.25 * minmax(early["SeasonDelta"], invert=True)
        + 0.15 * minmax(early["Utilisation"], invert=True)
        + 0.15 * minmax(early["DraftPts"], invert=True)
    )
    late["Rating"] = 100 * (
        0.40 * minmax(late["DrafRankDelta"])
        + 0.25 * minmax(late["SeasonDelta"])
        + 0.20 * minmax(late["DraftPts"])
        + 0.15 * minmax(late["Utilisation"])
    )

    early = early.sort_values(["Rating", "DrafRankDelta"], ascending=[False, True])
    late = late.sort_values(["Rating", "DraftPts"], ascending=[False, False])
    return early, late


def worst_rationale(row: pd.Series) -> str:
    parts = [
        f"cost pick {int(row['Ov Pick'])} but ranked {int(row['DraftRank'])} for points contributed",
        f"a draft-value delta of {int(row['DrafRankDelta']):+d}",
    ]
    if row["DraftPts"] <= 0:
        parts.append("returned no points to the drafting team")
    elif row["DraftGP"] <= 3:
        parts.append(f"contributed in only {int(row['DraftGP'])} appearances")
    else:
        parts.append(f"contributed {row['DraftPts']:.1f} points")
    if row["SeasonRank"] > 0:
        parts.append(f"and finished only {int(row['SeasonRank'])}th overall for the season")
    return "; ".join(parts).capitalize() + "."


def best_rationale(row: pd.Series) -> str:
    parts = [
        f"selected {int(row['Ov Pick'])}th but ranked {int(row['DraftRank'])} for points contributed",
        f"creating {int(row['DrafRankDelta']):+d} places of draft value",
        f"with {row['DraftPts']:.1f} points",
    ]
    if row["Utilisation"] >= 0.9:
        parts.append(f"and the manager captured {row['Utilisation']:.0%} of the player's season output")
    elif row["SeasonRank"] > 0:
        parts.append(f"while the player finished {int(row['SeasonRank'])}th overall for the season")
    return "; ".join(parts).capitalize() + "."


def team_summary(row: pd.Series, table: pd.DataFrame) -> str:
    def percentile(column: str) -> float:
        return table[column].rank(pct=True).loc[row.name]

    strengths: list[str] = []
    weaknesses: list[str] = []

    if percentile("DraftPts") >= 0.75:
        strengths.append("strong total production")
    elif percentile("DraftPts") <= 0.25:
        weaknesses.append("low total production")

    if percentile("DraftDelta") >= 0.75:
        strengths.append("excellent value against draft position")
    elif percentile("DraftDelta") <= 0.25:
        weaknesses.append("heavy losses against draft position")

    if percentile("Utilisation") >= 0.75:
        strengths.append("captured a high share of the players' season output")
    elif percentile("Utilisation") <= 0.25:
        weaknesses.append("failed to retain or benefit from much of the value drafted")

    if row["PositivePicks"] >= 10:
        strengths.append("unusually consistent depth")
    if row["ZeroPointPicks"] >= 3:
        weaknesses.append(f"carried {int(row['ZeroPointPicks'])} zero-point selections")

    if not strengths:
        strengths.append("some useful individual selections")
    if not weaknesses:
        weaknesses.append("no single metric was disastrous, but the draft lacked an elite edge")

    return f"Strengths: {', '.join(strengths)}. Main concern: {', '.join(weaknesses)}."


def display_pick_table(frame: pd.DataFrame, rationale_type: str, rows: int) -> None:
    display = frame.head(rows).copy()
    display.insert(0, "Rank", range(1, len(display) + 1))
    display["Rationale"] = display.apply(
        worst_rationale if rationale_type == "worst" else best_rationale,
        axis=1,
    )
    display["Utilisation"] = display["Utilisation"].map(lambda x: f"{x:.0%}")
    display["Rating"] = display["Rating"].round(1)
    columns = [
        "Rank",
        "Player",
        "Fantasy Team",
        "Round",
        "Ov Pick",
        "DraftPts",
        "DraftRank",
        "DrafRankDelta",
        "SeasonRank",
        "Utilisation",
        "Rationale",
    ]
    st.dataframe(
        display[columns],
        hide_index=True,
        width="stretch",
        column_config={
            "DraftPts": st.column_config.NumberColumn("Draft points", format="%.2f"),
            "DrafRankDelta": st.column_config.NumberColumn("Draft delta", format="%+d"),
            "Utilisation": st.column_config.TextColumn("Points captured"),
            "Rationale": st.column_config.TextColumn("Why it stands out", width="large"),
        },
    )


def csv_download(frame: pd.DataFrame, filename: str, label: str) -> None:
    st.download_button(
        label,
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


# ---------- Data source ----------
st.sidebar.title("⚽ Draft controls")
uploaded = st.sidebar.file_uploader(
    "Use an updated Draft sheet",
    type=["csv", "xlsx", "xlsm"],
    help="An Excel upload must contain a tab named 'Draft'. Leave empty to use the included 25/26 data.",
)

try:
    raw_df = load_uploaded_data(uploaded.getvalue(), uploaded.name) if uploaded else load_default_data()
    df = prepare_data(raw_df)
except Exception as exc:
    st.error(f"The data could not be loaded: {exc}")
    st.stop()

all_teams = sorted(df["Fantasy Team"].dropna().unique().tolist())
all_positions = sorted(df["Position"].dropna().unique().tolist())
selected_teams = st.sidebar.multiselect("Teams", all_teams, default=all_teams)
selected_positions = st.sidebar.multiselect("Positions", all_positions, default=all_positions)
filtered_df = df[df["Fantasy Team"].isin(selected_teams) & df["Position"].isin(selected_positions)].copy()

if filtered_df.empty:
    st.warning("The selected filters contain no players.")
    st.stop()

team_scores = calculate_team_scores(df)
early_ranked, late_ranked = pick_rankings(df)

st.sidebar.caption(
    f"Using {'uploaded data' if uploaded else 'the included Draft dataset'} · "
    f"{len(df)} selections · {df['Fantasy Team'].nunique()} teams"
)

# ---------- Header ----------
st.title("Cheese Fantasy Draft Review · 2025/26")
st.markdown(
    "<div class='draft-callout'><b>Results-based analysis:</b> the app judges what each pick eventually produced, not whether the choice was defensible on draft night.</div>",
    unsafe_allow_html=True,
)

pages = st.tabs([
    "League overview",
    "Best & worst picks",
    "Team report cards",
    "Team breakdown",
    "Player explorer",
    "Methodology",
])

# ---------- Overview ----------
with pages[0]:
    best_team = team_scores.iloc[0]
    biggest_steal = df.loc[df["DrafRankDelta"].idxmax()]
    biggest_bust = df[df["Round"] <= 5].loc[df[df["Round"] <= 5]["DrafRankDelta"].idxmin()]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top-rated draft", best_team["Fantasy Team"], f"{best_team['Score']:.1f}/10")
    c2.metric("Total selections", f"{len(df):,}", f"{df['Fantasy Team'].nunique()} teams")
    c3.metric("Biggest late steal", biggest_steal["Player"], f"{biggest_steal['DrafRankDelta']:+.0f} places")
    c4.metric("Worst early return", biggest_bust["Player"], f"{biggest_bust['DrafRankDelta']:+.0f} places")

    left, right = st.columns([1.05, 1])
    with left:
        st.subheader("Team draft rating")
        score_chart = team_scores.sort_values("Score")
        fig = px.bar(
            score_chart,
            x="Score",
            y="Fantasy Team",
            orientation="h",
            text="Score",
            range_x=[0, 10],
            labels={"Fantasy Team": "Team", "Score": "Score out of 10"},
        )
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_layout(height=470, margin=dict(l=10, r=35, t=10, b=10), showlegend=False)
        st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("Draft cost vs contributed rank")
        plot_df = filtered_df.copy()
        fig = px.scatter(
            plot_df,
            x="Ov Pick",
            y="DraftRank",
            color="Fantasy Team",
            symbol="Position",
            hover_name="Player",
            hover_data={"Round": True, "DraftPts": ":.2f", "DrafRankDelta": ":+d", "Ov Pick": True, "DraftRank": True},
            labels={"Ov Pick": "Overall pick", "DraftRank": "Rank for contributed points"},
        )
        max_axis = int(max(plot_df["Ov Pick"].max(), plot_df["DraftRank"].max()))
        fig.add_shape(type="line", x0=1, y0=1, x1=max_axis, y1=max_axis, line=dict(dash="dash", width=1))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(height=470, margin=dict(l=10, r=10, t=10, b=10), legend_title_text="Team")
        st.plotly_chart(fig, width="stretch")

    st.subheader("League table")
    overview_table = team_scores.copy()
    overview_table["Utilisation"] = overview_table["Utilisation"].map(lambda x: f"{x:.1%}")
    st.dataframe(
        overview_table[["Rank", "Fantasy Team", "Score", "DraftPts", "DraftDelta", "SeasonPts", "Utilisation", "PositivePicks", "ZeroPointPicks"]],
        hide_index=True,
        width="stretch",
        column_config={
            "Fantasy Team": "Team",
            "Score": st.column_config.ProgressColumn("Score / 10", min_value=0, max_value=10, format="%.1f"),
            "DraftPts": st.column_config.NumberColumn("Draft points", format="%.2f"),
            "DraftDelta": st.column_config.NumberColumn("Total draft delta", format="%+d"),
            "SeasonPts": st.column_config.NumberColumn("Season points", format="%.2f"),
            "Utilisation": "Points captured",
            "PositivePicks": "Positive picks",
            "ZeroPointPicks": "Zero-point picks",
        },
    )

# ---------- Best / worst ----------
with pages[1]:
    st.subheader("Selection rankings")
    c1, c2, c3 = st.columns(3)
    early_max = c1.slider("Early-round cutoff", 1, 8, 5)
    late_min = c2.slider("Late-round starting point", 2, int(df["Round"].max()), 6)
    pick_count = c3.slider("Number of selections", 5, 20, 10)
    early_custom, late_custom = pick_rankings(df, early_max=early_max, late_min=late_min)

    worst_tab, best_tab = st.tabs(["Worst early picks", "Best later picks"])
    with worst_tab:
        st.caption(
            "Balanced ranking: 45% contributed-rank loss, 25% season-rank loss, 15% low points captured, and 15% low points contributed."
        )
        display_pick_table(early_custom, "worst", pick_count)
        csv_download(early_custom.head(pick_count), "worst_early_picks.csv", "Download this ranking")

    with best_tab:
        st.caption(
            "Balanced ranking: 40% contributed-rank gain, 25% season-rank gain, 20% points contributed, and 15% points captured."
        )
        display_pick_table(late_custom, "best", pick_count)
        csv_download(late_custom.head(pick_count), "best_late_picks.csv", "Download this ranking")

# ---------- Report cards ----------
with pages[2]:
    st.subheader("League draft report cards")
    st.caption("Scores are comparative within this league and recalculate automatically when a new Draft sheet is uploaded.")

    report = team_scores.copy()
    report["Explanation"] = [team_summary(row, report) for _, row in report.iterrows()]
    report["Utilisation"] = report["Utilisation"].map(lambda x: f"{x:.1%}")
    st.dataframe(
        report[["Rank", "Fantasy Team", "Score", "DraftPts", "DraftDelta", "Utilisation", "PositivePicks", "Explanation"]],
        hide_index=True,
        width="stretch",
        column_config={
            "Fantasy Team": "Team",
            "Score": st.column_config.ProgressColumn("Score / 10", min_value=0, max_value=10, format="%.1f"),
            "DraftPts": st.column_config.NumberColumn("Draft points", format="%.2f"),
            "DraftDelta": st.column_config.NumberColumn("Draft delta", format="%+d"),
            "Utilisation": "Points captured",
            "PositivePicks": "Positive picks",
            "Explanation": st.column_config.TextColumn("Assessment", width="large"),
        },
    )
    csv_download(report, "team_draft_report_cards.csv", "Download report cards")

# ---------- Team breakdown ----------
with pages[3]:
    selected_team = st.selectbox("Choose a fantasy team", all_teams, index=all_teams.index(team_scores.iloc[0]["Fantasy Team"]))
    team_df = df[df["Fantasy Team"] == selected_team].copy()
    team_row = team_scores[team_scores["Fantasy Team"] == selected_team].iloc[0]

    st.markdown(f"<span class='team-badge'>{selected_team}</span>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Draft score", f"{team_row['Score']:.1f}/10", f"League rank {int(team_row['Rank'])}")
    c2.metric("Points contributed", f"{team_row['DraftPts']:,.2f}")
    c3.metric("Total draft delta", f"{int(team_row['DraftDelta']):+d}")
    c4.metric("Points captured", f"{team_row['Utilisation']:.1%}")
    c5.metric("Positive picks", f"{int(team_row['PositivePicks'])}/{int(team_row['Picks'])}")
    st.write(team_summary(team_row, team_scores))

    left, right = st.columns([1.15, 1])
    with left:
        round_summary = team_df.groupby("Round", as_index=False).agg(DraftPts=("DraftPts", "sum"), DraftDelta=("DrafRankDelta", "sum"))
        fig = px.bar(round_summary, x="Round", y="DraftPts", hover_data={"DraftDelta": ":+d"}, labels={"DraftPts": "Points contributed"})
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), title="Contribution by draft round")
        st.plotly_chart(fig, width="stretch")

    with right:
        best_pick = team_df.sort_values(["DrafRankDelta", "DraftPts"], ascending=False).iloc[0]
        worst_pick = team_df.sort_values(["DrafRankDelta", "DraftPts"], ascending=True).iloc[0]
        st.markdown("#### Best value selection")
        st.success(f"**{best_pick['Player']}** · Round {int(best_pick['Round'])}, pick {int(best_pick['Ov Pick'])}\n\n{best_rationale(best_pick)}")
        st.markdown("#### Weakest value selection")
        st.error(f"**{worst_pick['Player']}** · Round {int(worst_pick['Round'])}, pick {int(worst_pick['Ov Pick'])}\n\n{worst_rationale(worst_pick)}")

    st.markdown("#### All selections")
    team_display = team_df.sort_values("Ov Pick").copy()
    team_display["Utilisation"] = team_display["Utilisation"].map(lambda x: f"{x:.0%}")
    st.dataframe(
        team_display[["Round", "Ov Pick", "Player", "Position", "DraftGP", "DraftPts", "DraftRank", "DrafRankDelta", "SeasonRank", "SeasonDelta", "Utilisation"]],
        hide_index=True,
        width="stretch",
        column_config={
            "Ov Pick": "Overall pick",
            "DraftGP": "Games held",
            "DraftPts": st.column_config.NumberColumn("Draft points", format="%.2f"),
            "DrafRankDelta": st.column_config.NumberColumn("Draft delta", format="%+d"),
            "SeasonDelta": st.column_config.NumberColumn("Season delta", format="%+d"),
            "Utilisation": "Points captured",
        },
    )

# ---------- Explorer ----------
with pages[4]:
    st.subheader("Player explorer")
    search = st.text_input("Search for a player", placeholder="e.g. Saka, Senesi, Haaland")
    explorer = filtered_df.copy()
    if search:
        explorer = explorer[explorer["Player"].str.contains(search, case=False, na=False)]

    sort_options = {
        "Overall pick": ("Ov Pick", True),
        "Draft points": ("DraftPts", False),
        "Draft value": ("DrafRankDelta", False),
        "Season value": ("SeasonDelta", False),
        "Points captured": ("Utilisation", False),
    }
    sort_label = st.selectbox("Sort by", list(sort_options))
    sort_col, ascending = sort_options[sort_label]
    explorer = explorer.sort_values(sort_col, ascending=ascending)
    explorer_display = explorer.copy()
    explorer_display["Utilisation"] = explorer_display["Utilisation"].map(lambda x: f"{x:.0%}")
    st.dataframe(
        explorer_display[["Fantasy Team", "Round", "Ov Pick", "Player", "Position", "Team", "DraftGP", "DraftPts", "DraftRank", "DrafRankDelta", "SeasonPts", "SeasonRank", "SeasonDelta", "Utilisation"]],
        hide_index=True,
        width="stretch",
    )
    csv_download(explorer, "draft_player_explorer.csv", "Download filtered player data")

# ---------- Method ----------
with pages[5]:
    st.subheader("How the analysis works")
    st.markdown(
        """
        **Draft points** are the points a player actually contributed while associated with their original drafting team.  
        **Draft delta** is `overall pick − contributed-points rank`; positive numbers represent value.  
        **Season delta** is `overall pick − final season rank`; it helps separate a poor player outcome from a good player the manager released too early.  
        **Points captured** is `draft points ÷ season points`, capped between 0% and 100% for presentation.

        #### Team score
        Each team is ranked relative to the other teams in the uploaded dataset. The composite score uses:

        - **40%** total draft points
        - **25%** cumulative draft delta
        - **15%** total underlying season points
        - **10%** points captured
        - **10%** number of positive-value selections

        Each component is min-max normalised within the league, combined, and displayed on a **2.2–9.4 report-card scale**. This is deliberately comparative: uploading a different season or league recalculates every score.

        #### Important limitation
        This is a retrospective performance review. It does not account for injuries known only after the draft, transfer timing, waiver-wire replacements, trades, starting-line-up decisions, or how reasonable market expectations were on draft day.
        """
    )
    st.markdown("<p class='small-note'>The ranking sliders and downloadable tables are intended to make league arguments easier, not to end them.</p>", unsafe_allow_html=True)
