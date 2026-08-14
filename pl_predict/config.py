"""Central configuration: seasons, team-name mappings, stadium coordinates."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = PROJECT_ROOT / "output"

# football-data.co.uk season codes, oldest first. "0001" -> 2000-01.
# Includes the in-progress season; its file 404s until the first matches are played.
SEASONS = [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(2000, 2027)]
UPCOMING_SEASON = SEASONS[-1]  # the season being predicted (2026-27)
DIVISIONS = ["E0", "E1"]  # Premier League, Championship

FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

USER_AGENT = "Mozilla/5.0 (pl-predict; personal research project)"

# FPL team name -> football-data.co.uk team name (canonical).
FPL_TO_FD = {
    "Man Utd": "Man United",
    "Spurs": "Tottenham",
    "Coventry City": "Coventry",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Nott'm Forest": "Nott'm Forest",
    "Sheffield Utd": "Sheffield United",
    "Luton": "Luton",
    "Leeds": "Leeds",
}


def fpl_to_fd(name: str) -> str:
    return FPL_TO_FD.get(name, name)


# Approximate stadium coordinates for 2026-27 clubs (for weather lookups).
STADIUMS = {
    "Arsenal": (51.5549, -0.1084),
    "Aston Villa": (52.5092, -1.8847),
    "Bournemouth": (50.7352, -1.8384),
    "Brentford": (51.4907, -0.2889),
    "Brighton": (50.8616, -0.0837),
    "Chelsea": (51.4817, -0.1910),
    "Coventry": (52.4481, -1.4956),
    "Crystal Palace": (51.3983, -0.0857),
    "Everton": (53.4419, -2.9989),  # Hill Dickinson Stadium
    "Fulham": (51.4749, -0.2217),
    "Hull": (53.7466, -0.3675),
    "Ipswich": (52.0550, 1.1447),
    "Leeds": (53.7778, -1.5721),
    "Liverpool": (53.4308, -2.9608),
    "Man City": (53.4831, -2.2004),
    "Man United": (53.4631, -2.2913),
    "Newcastle": (54.9756, -1.6216),
    "Nott'm Forest": (52.9399, -1.1329),
    "Tottenham": (51.6043, -0.0664),
    "Sunderland": (54.9146, -1.3882),
}
