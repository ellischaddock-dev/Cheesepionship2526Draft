# Cheese Fantasy Draft Review 2025/26

A shareable Streamlit dashboard for reviewing the league's fantasy-football draft.

## Included views

- League overview and draft-score table
- Ten best later-round selections and ten worst early selections
- Team report cards with a transparent score out of ten
- Team-by-team breakdowns
- Searchable player explorer
- CSV downloads for league chat and follow-up analysis
- Optional upload of a replacement CSV or Excel workbook with a `Draft` tab

## Run locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload every file and folder from this project, preserving the `.streamlit` and `data` folders.
3. Sign in to Streamlit Community Cloud and connect the GitHub account.
4. Create a new app and select:
   - the repository
   - the `main` branch
   - `streamlit_app.py` as the entrypoint
5. Deploy and share the resulting `streamlit.app` address with the league.

No secrets or external database are required. Updating the included `data/draft_data.csv` file in GitHub will update the default dashboard after redeployment. League members can also temporarily analyse a replacement workbook through the sidebar uploader; uploads are session-specific and do not overwrite the repository data.

## Expected data

The default dataset is `data/draft_data.csv`. An uploaded Excel file must contain a worksheet named `Draft` with these key columns:

- Fantasy Team
- Round
- Ov Pick
- Pos
- Player
- DraftGP / DraftPts / DraftRank / DrafRankDelta
- SeasonPts / SeasonRank / SeasonDelta
- Utilisation

## Scoring

The team score is comparative within the loaded league:

- 40% total DraftPts
- 25% cumulative DrafRankDelta
- 15% total SeasonPts
- 10% weighted utilisation
- 10% number of positive-value selections

Metrics are min-max normalised, combined, and placed on a report-card scale.
