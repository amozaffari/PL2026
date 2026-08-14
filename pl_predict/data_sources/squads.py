"""Squad and transfer-window signals from free FPL data.

Current rosters come from the live FPL API; last season's rosters come from the
community-maintained vaastav/Fantasy-Premier-League GitHub dataset. Diffing the
two (players are stable across seasons via their FPL `code`) reveals real
transfers: intra-league moves, arrivals from abroad, and departures.
"""

import io

import pandas as pd
import requests

from ..config import FPL_BOOTSTRAP_URL, RAW_DIR, USER_AGENT, fpl_to_fd

VAASTAV_BASE = ("https://raw.githubusercontent.com/vaastav/"
                "Fantasy-Premier-League/master/data/2025-26")

STATUS_LABELS = {"i": "injured", "s": "suspended", "d": "doubtful", "u": "unavailable"}


def _get_csv(url: str, cache_name: str) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = RAW_DIR / cache_name
    if not cache.exists():
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        resp.raise_for_status()
        cache.write_bytes(resp.content)
    return pd.read_csv(io.BytesIO(cache.read_bytes()))


def current_players() -> pd.DataFrame:
    data = requests.get(FPL_BOOTSTRAP_URL,
                        headers={"User-Agent": USER_AGENT}, timeout=30).json()
    teams = {t["id"]: fpl_to_fd(t["name"]) for t in data["teams"]}
    df = pd.DataFrame(data["elements"])
    df["team_name"] = df["team"].map(teams)
    return df[["code", "web_name", "team_name", "now_cost", "status", "news",
               "total_points"]]


def last_season_players() -> pd.DataFrame:
    players = _get_csv(f"{VAASTAV_BASE}/players_raw.csv", "vaastav_2526_players.csv")
    teams = _get_csv(f"{VAASTAV_BASE}/teams.csv", "vaastav_2526_teams.csv")
    team_names = {r.id: fpl_to_fd(r.name) for r in teams.itertuples()}
    players["team_name"] = players["team"].map(team_names)
    return players[["code", "web_name", "team_name", "total_points"]]


# Heuristic scales for turning FPL-point flows into Elo. Deliberately modest:
# the signal understates clubs buying from abroad (those arrivals carry 0 pts).
TRANSFER_PTS_PER_ELO = 12.0
TRANSFER_ELO_CAP = 40.0
INJURY_PTS_PER_ELO = 10.0
INJURY_ELO_CAP = 30.0
STATUS_WEIGHT = {"i": 1.0, "u": 1.0, "s": 1.0, "d": 0.5}


def squad_elo_offsets() -> dict[str, float]:
    """Season-long Elo offset per club from net transfer-window quality flow.

    A transparent, market-free alternative to the Polymarket calibration
    (which already prices transfers in — do not stack the two).
    """
    summary, _ = transfer_activity()
    return {
        r.team: max(-TRANSFER_ELO_CAP,
                    min(TRANSFER_ELO_CAP, r.net_pts / TRANSFER_PTS_PER_ELO))
        for r in summary.itertuples()
    }


def injury_elo_penalties() -> dict[str, float]:
    """Short-horizon Elo penalty per club for currently unavailable players,
    weighted by their last-season FPL points. Meant for upcoming-match
    predictions, not season-long simulation."""
    now = current_players()
    prev = last_season_players()[["code", "total_points"]].rename(
        columns={"total_points": "pts_prev"})
    merged = now.merge(prev, on="code", how="left")
    merged["pts_prev"] = merged["pts_prev"].fillna(0)
    out: dict[str, float] = {}
    for team, group in merged.groupby("team_name"):
        weighted = sum(r.pts_prev * STATUS_WEIGHT.get(r.status, 0.0)
                       for r in group.itertuples() if r.status != "a")
        out[team] = -min(INJURY_ELO_CAP, weighted / INJURY_PTS_PER_ELO)
    return out


def transfer_activity() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-club window summary, individual moves), from last-season roster diff.

    Player quality is proxied by last season's FPL points, so arrivals from
    abroad carry 0 by construction — the per-club numbers understate clubs that
    buy from foreign leagues.
    """
    now = current_players()
    prev = last_season_players()
    merged = prev.merge(now, on="code", how="outer", suffixes=("_prev", "_now"))

    moves = []
    for r in merged.itertuples():
        prev_team = getattr(r, "team_name_prev", None)
        now_team = getattr(r, "team_name_now", None)
        prev_pts = 0 if pd.isna(r.total_points_prev) else int(r.total_points_prev)
        if pd.isna(prev_team) and not pd.isna(now_team):
            moves.append({"player": r.web_name_now, "from": "(outside league)",
                          "to": now_team, "last_season_pts": 0})
        elif pd.isna(now_team) and not pd.isna(prev_team):
            moves.append({"player": r.web_name_prev, "from": prev_team,
                          "to": "(left league)", "last_season_pts": prev_pts})
        elif prev_team != now_team:
            moves.append({"player": r.web_name_now, "from": prev_team,
                          "to": now_team, "last_season_pts": prev_pts})
    moves_df = pd.DataFrame(moves).sort_values("last_season_pts", ascending=False)

    clubs = sorted(set(now["team_name"]))
    rows = []
    for club in clubs:
        ins = moves_df[moves_df["to"] == club]
        outs = moves_df[moves_df["from"] == club]
        unavailable = now[(now["team_name"] == club) & (now["status"] != "a")]
        rows.append({
            "team": club,
            "players_in": len(ins),
            "players_out": len(outs),
            "pts_in": int(ins["last_season_pts"].sum()),
            "pts_out": int(outs["last_season_pts"].sum()),
            "net_pts": int(ins["last_season_pts"].sum() - outs["last_season_pts"].sum()),
            "unavailable": len(unavailable),
        })
    summary = (pd.DataFrame(rows)
               .sort_values("net_pts", ascending=False)
               .reset_index(drop=True))
    return summary, moves_df.reset_index(drop=True)
