#!/usr/bin/env python3
"""
fetch_spurs_manager_data.py

Pulls Tottenham Hotspur Premier League stats for two manager eras and writes
spurs_manager_data.json.  Data sources are tried in priority order:

  Layer 1  soccerdata + FBref    schedule, results, goals, xG, possession, shots
  Layer 2  soccerdata + Understat xG / xGA cross-check / gap-fill
  Layer 3  understat package     async direct pull from understat.com
  Layer 4  API-Football          broad fallback (RapidAPI free tier)
  Layer 5  Hardcoded estimates   last resort — flagged as ESTIMATE in output

Manager tenures covered:
  Ange Postecoglou  2024-25 full Premier League season  (GW1-38)
  Thomas Frank      2025-26 Premier League GW1-26       (sacked Feb 2026)

Install dependencies:
  pip install soccerdata understat requests pandas lxml html5lib

Usage:
  python fetch_spurs_manager_data.py
  python fetch_spurs_manager_data.py --debug          # print raw column names
  python fetch_spurs_manager_data.py --rapidapi-key YOUR_KEY
  RAPIDAPI_KEY=xxx python fetch_spurs_manager_data.py

Output:
  spurs_manager_data.json
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# pandas — required for soccerdata DataFrames
# ---------------------------------------------------------------------------
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None  # type: ignore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_FILE = "spurs_manager_data.json"
DEBUG = False  # set via --debug flag

# Manager tenure definitions
MANAGERS: Dict[str, Dict[str, Any]] = {
    "postecoglou": {
        "name": "Ange Postecoglou",
        "label": "2024-25 full season",
        "season": 2024,   # soccerdata/FBref/API-Football season key (start year)
        "max_gw": 38,     # full season
    },
    "frank": {
        "name": "Thomas Frank",
        "label": "2025-26 GW1-GW26",
        "season": 2025,
        "max_gw": 26,     # sacked after GW26, Feb 2026
    },
}

# Team name variants used by each source
SPURS_FBREF            = "Tottenham Hotspur"
SPURS_UNDERSTAT        = "Tottenham"
SPURS_API_FOOTBALL_ID  = 47  # API-Football team id

LEAGUE_FBREF           = "ENG-Premier League"
LEAGUE_UNDERSTAT       = "EPL"
LEAGUE_API_FOOTBALL    = 39  # Premier League

# ---------------------------------------------------------------------------
# Last-resort hardcoded fallback estimates
# Used ONLY when every live source fails; flagged as ESTIMATE in JSON output.
# ---------------------------------------------------------------------------
FALLBACK: Dict[str, Dict[str, Any]] = {
    "postecoglou": {
        "games": 38, "wins": 11, "draws": 8, "losses": 19,
        "points": 41,
        "goals_for": 64, "goals_against": 65,
        "xg_total": 48.5, "xga_total": 52.3,
        "xg_per_game": 1.28, "xga_per_game": 1.38,
        "possession_pct": 53.8,
        "shots": 476,
        # Opta "big chances created" not public; FBref SCA used as proxy
        "big_chances_created": 52,
        "clean_sheets": 9,
    },
    "frank": {
        "games": 26, "wins": 8, "draws": 6, "losses": 12,
        "points": 30,
        "goals_for": 38, "goals_against": 48,
        "xg_total": 32.1, "xga_total": 38.4,
        "xg_per_game": 1.24, "xga_per_game": 1.48,
        "possession_pct": 51.2,
        "shots": 310,
        "big_chances_created": 34,
        "clean_sheets": 6,
    },
}

ALL_STATS: List[str] = [
    "games", "wins", "draws", "losses", "points",
    "goals_for", "goals_against",
    "xg_total", "xga_total", "xg_per_game", "xga_per_game",
    "possession_pct", "shots", "big_chances_created", "clean_sheets",
]

# ---------------------------------------------------------------------------
# Era — stat accumulator with source attribution
# ---------------------------------------------------------------------------

class Era:
    """Accumulates stats for one manager tenure, tracking source per stat."""

    def __init__(self, key: str) -> None:
        self.key  = key
        self._d:   Dict[str, Any]  = {}
        self._src: Dict[str, str]  = {}
        self._est: List[str]        = []

    def add(self, stat: str, value: Any, source: str) -> None:
        self._d[stat]   = value
        self._src[stat] = source
        print(f"  [OK] {self.key}.{stat:25s} = {value!s:10}  [source: {source}]")

    def estimate(self, stat: str, value: Any, reason: str) -> None:
        self._d[stat]   = value
        self._src[stat] = f"ESTIMATE — {reason}"
        self._est.append(stat)
        print(f"  [!!] ESTIMATE {self.key}.{stat:20s} = {value!s:10}  [reason: {reason}]")

    def has(self, stat: str) -> bool:
        return stat in self._d

    def needs(self, *stats: str) -> bool:
        return any(s not in self._d for s in stats)

    def to_json(self) -> Dict[str, Any]:
        return {**self._d, "_sources": self._src, "_estimated_stats": self._est}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sleep(s: float, reason: str = "") -> None:
    print(f"  [sleep {s:.0f}s" + (f": {reason}" if reason else "") + "]")
    time.sleep(s)


def _extract_gw(val: Any) -> int:
    """Parse gameweek number from values like 'Matchweek 12' or 12."""
    try:
        return int(val)
    except (ValueError, TypeError):
        m = re.search(r"\d+", str(val))
        return int(m.group()) if m else 9999


def _safe_float(row: Any, *keys: str) -> Optional[float]:
    for k in keys:
        try:
            v = row[k]
            if HAS_PANDAS and pd.notna(v):
                return float(v)
            elif not HAS_PANDAS and v is not None:
                return float(v)
        except (KeyError, TypeError, ValueError):
            pass
    return None


def _safe_int(row: Any, *keys: str) -> Optional[int]:
    v = _safe_float(row, *keys)
    return int(round(v)) if v is not None else None


def _find_col(df: Any, *candidates: str) -> Optional[str]:
    """Return first matching column name (exact then case-insensitive)."""
    for c in candidates:
        if c in df.columns:
            return c
    lower = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _find_col_containing(df: Any, *terms: str) -> Optional[str]:
    """Return first column whose name contains ALL of *terms (case-insensitive)."""
    for col in df.columns:
        col_l = col.lower()
        if all(t.lower() in col_l for t in terms):
            return col
    return None


def _debug_cols(label: str, df: Any) -> None:
    if DEBUG and df is not None:
        print(f"    [debug] {label} columns: {list(df.columns)[:30]}")


# ---------------------------------------------------------------------------
# FBref MultiIndex column flattening
#
# soccerdata's FBref read_team_season_stats() returns DataFrames with
# MultiIndex columns, e.g.:
#   ('Performance', 'W')  ('Performance', 'D')  ('Expected', 'xG') ...
# We flatten these to plain strings like 'Performance_W', 'xG', etc.,
# and also register bare leaf names so _safe_float('xG') works directly.
# ---------------------------------------------------------------------------

def _flatten_df(df: Any) -> Any:
    """
    If df has MultiIndex columns, flatten to single-level strings.
    Leaf names that are unique across the whole table are also kept as-is
    (e.g. 'xG', 'Poss') so existing lookup code works without changes.
    """
    if not HAS_PANDAS or df is None:
        return df
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    # Build flat names: prefer the bare leaf if it's unique, else 'Group_Leaf'
    leaf_counts: Dict[str, int] = {}
    for col in df.columns:
        leaf = str(col[-1]).strip()
        leaf_counts[leaf] = leaf_counts.get(leaf, 0) + 1

    new_cols = []
    for col in df.columns:
        parts = [str(p).strip() for p in col]
        leaf  = parts[-1]
        if leaf_counts[leaf] == 1:
            new_cols.append(leaf)
        else:
            new_cols.append("_".join(p for p in parts if p and p != leaf) + "_" + leaf)

    flat = df.copy()
    flat.columns = new_cols
    return flat


# ---------------------------------------------------------------------------
# DataFrame filtering helpers
# ---------------------------------------------------------------------------

def _filter_team_row(df: Any, spurs_name: str) -> Optional[Any]:
    """Find the Spurs row in a (possibly MultiIndex) team DataFrame."""
    if df is None or df.empty:
        return None
    df = _flatten_df(df)
    _debug_cols("team_season_stats", df)
    # Try index levels
    if hasattr(df.index, "names"):
        for lvl in df.index.names:
            if lvl and str(lvl).lower() in ("team", "squad"):
                mask = df.index.get_level_values(lvl).astype(str).str.contains(
                    "Tottenham", case=False, na=False)
                sub = df[mask]
                return sub.iloc[0] if len(sub) else None
    # Try columns
    for col in ("team", "Team", "squad", "Squad"):
        if col in df.columns:
            mask = df[col].astype(str).str.contains("Tottenham", case=False, na=False)
            sub  = df[mask]
            return sub.iloc[0] if len(sub) else None
    print(f"    [warn] Cannot find team column. Available: {list(df.columns)[:15]}")
    return None


def _filter_team_df(df: Any, spurs_name: str, max_gw: int = 999) -> Optional[Any]:
    """Return DataFrame of Spurs per-match rows up to max_gw."""
    if df is None or df.empty:
        return None
    df = _flatten_df(df)
    filtered = None
    if hasattr(df.index, "names"):
        for lvl in df.index.names:
            if lvl and str(lvl).lower() in ("team", "squad"):
                mask = df.index.get_level_values(lvl).astype(str).str.contains(
                    "Tottenham", case=False, na=False)
                filtered = df[mask].copy()
                break
    if filtered is None:
        for col in ("team", "Team", "squad", "Squad"):
            if col in df.columns:
                mask = df[col].astype(str).str.contains("Tottenham", case=False, na=False)
                filtered = df[mask].copy()
                break
    if filtered is None:
        return None
    if max_gw < 38:
        for gw_col in ("round", "Round", "gameweek", "Gameweek", "Wk", "week"):
            if gw_col in filtered.columns:
                filtered = filtered[
                    filtered[gw_col].apply(_extract_gw) <= max_gw].copy()
                break
    return filtered if len(filtered) else None


def _filter_schedule(sched: Any, max_gw: int) -> Optional[Any]:
    """Return Spurs rows from a schedule DataFrame up to max_gw."""
    if sched is None or sched.empty:
        return None
    _debug_cols("schedule", sched)
    h_col = _find_col_containing(sched, "home", "team") or \
            _find_col(sched, "home_team", "HomeTeam")
    a_col = _find_col_containing(sched, "away", "team") or \
            _find_col(sched, "away_team", "AwayTeam")
    if not h_col or not a_col:
        print(f"    [warn] Schedule cols (first 20): {list(sched.columns)[:20]}")
        return None
    mask = (
        sched[h_col].astype(str).str.contains("Tottenham", case=False, na=False)
        | sched[a_col].astype(str).str.contains("Tottenham", case=False, na=False)
    )
    sub = sched[mask].copy()
    if max_gw < 38:
        gw_col = _find_col(sub, "round", "Round", "Wk", "gameweek",
                           "Gameweek", "week", "matchweek", "Matchweek")
        if gw_col:
            sub = sub[sub[gw_col].apply(_extract_gw) <= max_gw]
    return sub if len(sub) else None


# ---------------------------------------------------------------------------
# Layer 1: soccerdata / FBref
# ---------------------------------------------------------------------------

def pull_fbref(eras: Dict[str, Era]) -> None:
    """
    Primary data source.
      games/wins/draws/losses/points/goals_for/goals_against/clean_sheets
        <- FBref schedule (match results)
      xg_total/xga_total/xg_per_game/xga_per_game
        <- FBref schedule xg columns (fastest path) OR standard season stats
      possession_pct  <- FBref standard season stats
      shots           <- FBref shooting season stats
      big_chances_created <- FBref GCA/SCA (proxy for Opta "big chances")
    """
    try:
        import soccerdata as sd  # noqa: F401
    except ImportError:
        print("[FBref] soccerdata not installed — skipping Layer 1")
        print("  Install: pip install soccerdata pandas lxml html5lib")
        return
    if not HAS_PANDAS:
        print("[FBref] pandas not installed — skipping Layer 1")
        return

    for key, cfg in MANAGERS.items():
        era    = eras[key]
        season = cfg["season"]
        max_gw = cfg["max_gw"]
        print(f"\n[FBref] {cfg['name']} ({cfg['label']}, season={season})")

        try:
            import soccerdata as sd
            fbref = sd.FBref(leagues=LEAGUE_FBREF, seasons=season)

            # ── 1a. Schedule: match results + xG (when cols present) ──────────
            # FBref schedule rows include home_xg / away_xg for most seasons.
            print("  Fetching schedule...")
            try:
                sched = fbref.read_schedule()
                _sleep(4, "FBref rate limit")
                spurs_sched = _filter_schedule(sched, max_gw)
                if spurs_sched is not None and len(spurs_sched):
                    _parse_schedule_results(spurs_sched, era, "FBref/schedule")
                else:
                    print("    [warn] No schedule rows found for Spurs")
            except Exception as e:
                print(f"    [FBref schedule] {type(e).__name__}: {e}")

            # ── 1b. Standard season stats: possession, xG (season totals) ─────
            if era.needs("possession_pct", "xg_total", "xga_total"):
                print("  Fetching standard season stats...")
                try:
                    std = fbref.read_team_season_stats(stat_type="standard")
                    _sleep(4, "FBref rate limit")
                    row = _filter_team_row(std, SPURS_FBREF)
                    if row is not None:
                        _parse_standard_season_row(row, era, "FBref/standard-season")
                    else:
                        print("    [warn] Spurs row not found in standard season stats")
                except Exception as e:
                    print(f"    [FBref standard] {type(e).__name__}: {e}")

            # ── 1c. Keeper season stats: clean sheets ───────────────────────
            if era.needs("clean_sheets"):
                print("  Fetching keeper season stats...")
                try:
                    kp  = fbref.read_team_season_stats(stat_type="keeper")
                    _sleep(4, "FBref rate limit")
                    row = _filter_team_row(kp, SPURS_FBREF)
                    if row is not None:
                        _parse_keeper_row(row, era, "FBref/keeper-season")
                except Exception as e:
                    print(f"    [FBref keeper] {type(e).__name__}: {e}")

            # ── 1d. Shooting season stats: total shots ─────────────────────
            if era.needs("shots"):
                print("  Fetching shooting season stats...")
                try:
                    sh  = fbref.read_team_season_stats(stat_type="shooting")
                    _sleep(4, "FBref rate limit")
                    row = _filter_team_row(sh, SPURS_FBREF)
                    if row is not None:
                        _parse_shooting_row(row, era, "FBref/shooting-season")
                except Exception as e:
                    print(f"    [FBref shooting] {type(e).__name__}: {e}")

            # ── 1e. GCA/SCA: proxy for big chances created ─────────────────
            if era.needs("big_chances_created"):
                print("  Fetching GCA stats (SCA as big-chances proxy)...")
                try:
                    cc  = fbref.read_team_season_stats(stat_type="gca")
                    _sleep(4, "FBref rate limit")
                    row = _filter_team_row(cc, SPURS_FBREF)
                    if row is not None:
                        _parse_gca_row(row, era,
                                       "FBref/gca-season (SCA proxy — not identical to Opta big chances)")
                except Exception as e:
                    print(f"    [FBref gca] {type(e).__name__}: {e}")

            # ── 1f. Per-match stats for partial season (Frank era) ──────────
            if max_gw < 38 and era.needs("xg_total", "possession_pct", "shots"):
                for method_name in ("read_team_match_stats", "read_match_stats"):
                    method = getattr(fbref, method_name, None)
                    if method is None:
                        continue
                    print(f"  Fetching per-match stats ({method_name})...")
                    try:
                        for stat_type in ("standard", "shooting"):
                            ms = method(stat_type=stat_type)
                            _sleep(4, "FBref rate limit")
                            spurs_ms = _filter_team_df(ms, SPURS_FBREF, max_gw)
                            if spurs_ms is not None and len(spurs_ms):
                                _parse_match_stats_agg(
                                    spurs_ms, era,
                                    f"FBref/{method_name}/{stat_type}")
                    except Exception as e:
                        print(f"    [FBref {method_name}] {type(e).__name__}: {e}")

        except Exception as e:
            print(f"[FBref] Unexpected error for '{key}': {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()


# --- parse helpers ----------------------------------------------------------

def _parse_schedule_results(matches: Any, era: Era, source: str) -> None:
    """
    From a per-match schedule DataFrame:
      - Derive W/D/L/GF/GA/pts/clean_sheets from score columns
      - Also extract xG/xGA if home_xg/away_xg columns are present
        (FBref schedule includes these for recent seasons)
    """
    if len(matches) == 0:
        return

    h_col  = _find_col_containing(matches, "home", "team") or \
              _find_col(matches, "home_team", "HomeTeam")
    a_col  = _find_col_containing(matches, "away", "team") or \
              _find_col(matches, "away_team", "AwayTeam")
    hg_col = _find_col(matches, "home_goals", "HomeGoals", "FTHG") or \
              _find_col_containing(matches, "home", "goal")
    ag_col = _find_col(matches, "away_goals", "AwayGoals", "FTAG") or \
              _find_col_containing(matches, "away", "goal")

    # Single "score" column fallback ("2-1" format)
    if not hg_col or not ag_col:
        score_col = _find_col(matches, "score", "Score", "result", "Result")
        if score_col:
            parsed = matches[score_col].astype(str).str.extract(r"(\d+)\D+(\d+)")
            matches = matches.copy()
            matches["_hg"] = pd.to_numeric(parsed[0], errors="coerce")
            matches["_ag"] = pd.to_numeric(parsed[1], errors="coerce")
            hg_col, ag_col = "_hg", "_ag"

    if not hg_col or not ag_col:
        print(f"    [warn] Cannot find goal columns. "
              f"Available cols: {list(matches.columns)[:20]}")
        return

    # xG columns — present in FBref schedule for most recent seasons
    hxg_col = (_find_col(matches, "home_xg", "xg_home", "xG_home") or
                _find_col_containing(matches, "home", "xg"))
    axg_col = (_find_col(matches, "away_xg", "xg_away", "xG_away") or
                _find_col_containing(matches, "away", "xg"))

    wins = draws = losses = gf = ga = cs = pts = 0
    xg_total = xga_total = 0.0
    valid = xg_valid = 0

    for _, row in matches.iterrows():
        is_home = ("Tottenham" in str(row.get(h_col, ""))) if h_col else True
        try:
            hg = int(row[hg_col])
            ag = int(row[ag_col])
        except (ValueError, TypeError):
            continue  # skip unplayed / postponed
        sg = hg if is_home else ag
        og = ag if is_home else hg
        gf += sg; ga += og; valid += 1
        if og == 0: cs += 1
        if sg > og:    wins += 1; pts += 3
        elif sg == og: draws += 1; pts += 1
        else:          losses += 1

        # xG from schedule
        if hxg_col and axg_col:
            try:
                hxg = float(row[hxg_col])
                axg = float(row[axg_col])
                xg_total  += hxg if is_home else axg
                xga_total += axg if is_home else hxg
                xg_valid  += 1
            except (TypeError, ValueError):
                pass

    if valid == 0:
        return

    if not era.has("games"):         era.add("games",         valid,  source)
    if not era.has("wins"):          era.add("wins",          wins,   source)
    if not era.has("draws"):         era.add("draws",         draws,  source)
    if not era.has("losses"):        era.add("losses",        losses, source)
    if not era.has("points"):        era.add("points",        pts,    source)
    if not era.has("goals_for"):     era.add("goals_for",     gf,     source)
    if not era.has("goals_against"): era.add("goals_against", ga,     source)
    if not era.has("clean_sheets"):  era.add("clean_sheets",  cs,     source)

    # xG from schedule cols (source labeled clearly)
    xg_src = source + "/xg-cols"
    if xg_valid and not era.has("xg_total"):
        era.add("xg_total",    round(xg_total, 2),           xg_src)
        era.add("xg_per_game", round(xg_total / xg_valid, 3), xg_src)
    if xg_valid and not era.has("xga_total"):
        era.add("xga_total",    round(xga_total, 2),            xg_src)
        era.add("xga_per_game", round(xga_total / xg_valid, 3), xg_src)

    if not xg_valid and hxg_col:
        print("    [warn] xG columns found but all values NaN/unparseable")
    elif not hxg_col:
        print("    [info] No xG columns in schedule — will try season stats next")


def _parse_standard_season_row(row: Any, era: Era, source: str) -> None:
    """Parse FBref standard season stats (flattened MultiIndex row)."""
    mp = _safe_int(row, "MP", "Games", "Matches")
    if mp and era.needs("games"): era.add("games", mp, source)

    w = _safe_int(row, "W", "Wins")
    d = _safe_int(row, "D", "Draws")
    l = _safe_int(row, "L", "Losses", "Loses")
    if w is not None and era.needs("wins"):   era.add("wins",   w, source)
    if d is not None and era.needs("draws"):  era.add("draws",  d, source)
    if l is not None and era.needs("losses"): era.add("losses", l, source)
    if w is not None and d is not None and era.needs("points"):
        era.add("points", w * 3 + d, source)

    gf = _safe_int(row, "GF", "GoalsFor")
    ga = _safe_int(row, "GA", "GoalsAgainst")
    if gf is not None and era.needs("goals_for"):     era.add("goals_for",     gf, source)
    if ga is not None and era.needs("goals_against"): era.add("goals_against", ga, source)

    xg  = _safe_float(row, "xG", "xg", "Expected_xG")
    xga = _safe_float(row, "xGA", "xga", "Expected_xGA")
    games = era._d.get("games") or mp or 0
    if xg is not None and era.needs("xg_total"):
        era.add("xg_total",    round(xg, 2),  source)
        if games and era.needs("xg_per_game"):
            era.add("xg_per_game", round(xg / games, 3), source)
    if xga is not None and era.needs("xga_total"):
        era.add("xga_total",    round(xga, 2), source)
        if games and era.needs("xga_per_game"):
            era.add("xga_per_game", round(xga / games, 3), source)

    poss = _safe_float(row, "Poss", "Possession", "poss")
    if poss is not None and era.needs("possession_pct"):
        era.add("possession_pct", round(poss, 1), source)


def _parse_keeper_row(row: Any, era: Era, source: str) -> None:
    cs = _safe_int(row, "CS", "CleanSheets", "Clean Sheets", "Performance_CS")
    if cs is not None and era.needs("clean_sheets"):
        era.add("clean_sheets", cs, source)


def _parse_shooting_row(row: Any, era: Era, source: str) -> None:
    shots = _safe_int(row, "Sh", "Shots", "shots", "Standard_Sh")
    if shots is not None and era.needs("shots"):
        era.add("shots", shots, source)


def _parse_gca_row(row: Any, era: Era, source: str) -> None:
    # SCA = shot-creating actions; closest public proxy for Opta "big chances"
    sca = _safe_int(row, "SCA", "sca", "SCA90", "SCA_SCA")
    if sca is not None and era.needs("big_chances_created"):
        era.add("big_chances_created", sca, source)


def _parse_match_stats_agg(ms: Any, era: Era, source: str) -> None:
    """Sum / average per-match stat rows for a partial season."""
    games = len(ms)
    if games == 0:
        return
    xg_col   = _find_col(ms, "xG", "xg", "xGoals", "expected_goals")
    xga_col  = _find_col(ms, "xGA", "xga", "xGoalsAgainst")
    poss_col = _find_col(ms, "Poss", "poss", "Possession", "possession")
    sh_col   = _find_col(ms, "Sh", "shots", "Shots", "Standard_Sh")
    if xg_col and era.needs("xg_total"):
        t = float(ms[xg_col].sum())
        era.add("xg_total",    round(t, 2),         source)
        era.add("xg_per_game", round(t / games, 3), source)
    if xga_col and era.needs("xga_total"):
        t = float(ms[xga_col].sum())
        era.add("xga_total",    round(t, 2),         source)
        era.add("xga_per_game", round(t / games, 3), source)
    if poss_col and era.needs("possession_pct"):
        era.add("possession_pct", round(float(ms[poss_col].mean()), 1), source)
    if sh_col and era.needs("shots"):
        era.add("shots", int(ms[sh_col].sum()), source)


# ---------------------------------------------------------------------------
# Layer 2: soccerdata / Understat — xG cross-check
# ---------------------------------------------------------------------------

def pull_understat_soccerdata(eras: Dict[str, Era]) -> None:
    """xG source #2: soccerdata Understat scraper."""
    if not any(era.needs("xg_total", "xga_total") for era in eras.values()):
        return
    try:
        import soccerdata as sd
    except ImportError:
        print("[Understat/soccerdata] soccerdata not installed — skipping Layer 2")
        return
    if not HAS_PANDAS:
        return

    for key, cfg in MANAGERS.items():
        era = eras[key]
        if not era.needs("xg_total", "xga_total", "xg_per_game", "xga_per_game"):
            continue
        print(f"\n[Understat/soccerdata] xG for {cfg['name']} (season {cfg['season']})")
        try:
            us    = sd.Understat(leagues=LEAGUE_UNDERSTAT, seasons=cfg["season"])
            sched = us.read_schedule()
            _sleep(3, "Understat rate limit")
            _debug_cols("understat schedule", sched)

            matches = _filter_schedule(sched, cfg["max_gw"])
            if matches is None or matches.empty:
                print("    [warn] No Understat matches found")
                continue

            h_col   = _find_col_containing(matches, "home", "team")
            hxg_col = (_find_col_containing(matches, "home", "xg") or
                       _find_col(matches, "xg_home", "home_xg"))
            axg_col = (_find_col_containing(matches, "away", "xg") or
                       _find_col(matches, "xg_away", "away_xg"))

            if not hxg_col or not axg_col:
                print(f"    [warn] xG cols not found. All cols: {list(matches.columns)}")
                continue

            xg_total = xga_total = 0.0
            valid = 0
            for _, row in matches.iterrows():
                is_home = h_col and "Tottenham" in str(row.get(h_col, ""))
                try:
                    hxg = float(row[hxg_col])
                    axg = float(row[axg_col])
                    xg_total  += hxg if is_home else axg
                    xga_total += axg if is_home else hxg
                    valid += 1
                except (TypeError, ValueError):
                    pass

            src = "Understat/soccerdata"
            if valid and era.needs("xg_total"):
                era.add("xg_total",     round(xg_total, 2),          src)
                era.add("xg_per_game",  round(xg_total / valid, 3),  src)
            if valid and era.needs("xga_total"):
                era.add("xga_total",    round(xga_total, 2),          src)
                era.add("xga_per_game", round(xga_total / valid, 3),  src)

        except Exception as e:
            print(f"[Understat/soccerdata] {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Layer 3: understat package — async direct pull from understat.com
# ---------------------------------------------------------------------------

def pull_understat_direct(eras: Dict[str, Era]) -> None:
    """xG source #3: async direct pull via the 'understat' pip package."""
    if not any(era.needs("xg_total", "xga_total") for era in eras.values()):
        return
    try:
        import understat as _up  # noqa: F401
    except ImportError:
        print("[Understat/direct] 'understat' package not installed — skipping Layer 3")
        print("  Install: pip install understat")
        return

    async def _fetch() -> Dict[str, Tuple[float, float, int]]:
        import understat as _up
        out: Dict[str, Tuple[float, float, int]] = {}
        async with _up.Understat() as us:
            for key, cfg in MANAGERS.items():
                era = eras[key]
                if not era.needs("xg_total", "xga_total"):
                    continue
                print(f"\n[Understat/direct] {cfg['name']} season {cfg['season']}...")
                try:
                    results = await us.get_team_results(SPURS_UNDERSTAT, cfg["season"])
                    await asyncio.sleep(2)
                    pl = [
                        r for r in results
                        if r.get("isResult") and str(r.get("id", "x")).isdigit()
                    ][:cfg["max_gw"]]
                    xg  = sum(float(r.get("xG",  0) or 0) for r in pl)
                    xga = sum(float(r.get("xGA", 0) or 0) for r in pl)
                    out[key] = (xg, xga, len(pl))
                except Exception as e:
                    print(f"    {type(e).__name__}: {e}")
        return out

    try:
        fetched = asyncio.run(_fetch())
        for key, (xg, xga, n) in fetched.items():
            if n == 0:
                continue
            era = eras[key]
            src = "understat.com/direct"
            if era.needs("xg_total"):
                era.add("xg_total",     round(xg, 2),         src)
                era.add("xg_per_game",  round(xg / n, 3),     src)
            if era.needs("xga_total"):
                era.add("xga_total",    round(xga, 2),         src)
                era.add("xga_per_game", round(xga / n, 3),     src)
    except Exception as e:
        print(f"[Understat/direct] async error: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Layer 4: API-Football (RapidAPI free tier)
# ---------------------------------------------------------------------------

APIF_HOST = "api-football-v1.p.rapidapi.com"


def pull_api_football(eras: Dict[str, Era], rapidapi_key: str) -> None:
    """Broad fallback via API-Football. Free tier: 100 req/day, no xG."""
    if not rapidapi_key:
        print("[API-Football] No RAPIDAPI_KEY set — skipping Layer 4")
        return
    try:
        import requests
    except ImportError:
        print("[API-Football] 'requests' not installed — skipping Layer 4")
        return

    headers = {"X-RapidAPI-Key": rapidapi_key, "X-RapidAPI-Host": APIF_HOST}
    base = f"https://{APIF_HOST}"

    for key, cfg in MANAGERS.items():
        era    = eras[key]
        season = cfg["season"]
        max_gw = cfg["max_gw"]
        print(f"\n[API-Football] {cfg['name']} (season {season})")
        try:
            if max_gw == 38:
                resp = requests.get(
                    f"{base}/teams/statistics",
                    headers=headers,
                    params={"league": LEAGUE_API_FOOTBALL, "season": season,
                            "team": SPURS_API_FOOTBALL_ID},
                    timeout=15,
                )
                _sleep(2, "API-Football rate limit")
                if resp.status_code == 200:
                    _apply_apif_season_stats(
                        resp.json().get("response", {}), era,
                        "API-Football/team-statistics")
                else:
                    print(f"    HTTP {resp.status_code}: {resp.text[:120]}")
            else:
                games, wins, draws, losses, gf, ga, cs = \
                    _apif_fixtures(base, headers, season, max_gw)
                if games:
                    _apply_apif_fixture_agg(
                        era, games, wins, draws, losses, gf, ga, cs,
                        "API-Football/fixtures")
        except Exception as e:
            print(f"[API-Football] {type(e).__name__}: {e}")


def _apply_apif_season_stats(data: Dict, era: Era, source: str) -> None:
    fx    = data.get("fixtures", {})
    gd    = data.get("goals", {})
    mp    = fx.get("played", {}).get("total") or 0
    wins  = fx.get("wins",   {}).get("total") or 0
    draws = fx.get("draws",  {}).get("total") or 0
    loses = fx.get("loses",  {}).get("total") or 0
    gf    = gd.get("for",     {}).get("total", {}).get("total") or 0
    ga    = gd.get("against", {}).get("total", {}).get("total") or 0
    cs    = data.get("clean_sheet", {}).get("total") or 0
    if mp    and era.needs("games"):         era.add("games",         int(mp),    source)
    if wins  and era.needs("wins"):          era.add("wins",          int(wins),  source)
    if draws is not None and era.needs("draws"):  era.add("draws",   int(draws), source)
    if loses is not None and era.needs("losses"): era.add("losses",  int(loses), source)
    if gf    and era.needs("goals_for"):     era.add("goals_for",     int(gf),    source)
    if ga    and era.needs("goals_against"): era.add("goals_against", int(ga),    source)
    if wins is not None and draws is not None and era.needs("points"):
        era.add("points", int(wins) * 3 + int(draws), source)
    if cs    and era.needs("clean_sheets"): era.add("clean_sheets",  int(cs),    source)


def _apply_apif_fixture_agg(
    era: Era, games: int, wins: int, draws: int,
    losses: int, gf: int, ga: int, cs: int, source: str
) -> None:
    if era.needs("games"):         era.add("games",         games,           source)
    if era.needs("wins"):          era.add("wins",          wins,            source)
    if era.needs("draws"):         era.add("draws",         draws,           source)
    if era.needs("losses"):        era.add("losses",        losses,          source)
    if era.needs("points"):        era.add("points",        wins * 3 + draws, source)
    if era.needs("goals_for"):     era.add("goals_for",     gf,              source)
    if era.needs("goals_against"): era.add("goals_against", ga,              source)
    if era.needs("clean_sheets"): era.add("clean_sheets",  cs,              source)


def _apif_fixtures(
    base: str, headers: Dict, season: int, max_gw: int
) -> Tuple[int, int, int, int, int, int, int]:
    import requests
    try:
        resp = requests.get(
            f"{base}/fixtures",
            headers=headers,
            params={"league": LEAGUE_API_FOOTBALL, "season": season,
                    "team": SPURS_API_FOOTBALL_ID},
            timeout=20,
        )
        _sleep(2, "API-Football rate limit")
        if resp.status_code != 200:
            print(f"    HTTP {resp.status_code}")
            return 0, 0, 0, 0, 0, 0, 0
        fixtures = resp.json().get("response", [])
        pl = [f for f in fixtures
              if f.get("league", {}).get("id") == LEAGUE_API_FOOTBALL]
        pl.sort(key=lambda x: _extract_gw(x.get("league", {}).get("round", "0")))
        pl = pl[:max_gw]
        wins = draws = losses = gf = ga = cs = 0
        for f in pl:
            goals   = f.get("goals", {})
            teams   = f.get("teams", {})
            hg      = goals.get("home") or 0
            ag      = goals.get("away") or 0
            is_home = teams.get("home", {}).get("id") == SPURS_API_FOOTBALL_ID
            sg = hg if is_home else ag
            og = ag if is_home else hg
            gf += sg; ga += og
            if og == 0: cs += 1
            if sg > og:    wins   += 1
            elif sg == og: draws  += 1
            else:          losses += 1
        return len(pl), wins, draws, losses, gf, ga, cs
    except Exception as e:
        print(f"    [API-Football fixtures] {type(e).__name__}: {e}")
        return 0, 0, 0, 0, 0, 0, 0


# ---------------------------------------------------------------------------
# Layer 5: Hardcoded fallback estimates
# ---------------------------------------------------------------------------

def apply_fallbacks(eras: Dict[str, Era]) -> None:
    """Fill any stat still missing after all live sources."""
    for key, era in eras.items():
        for stat in ALL_STATS:
            if era.needs(stat) and stat in FALLBACK[key]:
                era.estimate(stat, FALLBACK[key][stat],
                             "all live sources failed — verify manually")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    global DEBUG
    parser = argparse.ArgumentParser(
        description="Fetch Tottenham Hotspur Premier League manager-era stats")
    parser.add_argument("--rapidapi-key",
                        default=os.environ.get("RAPIDAPI_KEY", ""),
                        help="RapidAPI key for API-Football fallback")
    parser.add_argument("--debug", action="store_true",
                        help="Print raw DataFrame column names to diagnose mapping issues")
    args  = parser.parse_args()
    DEBUG = args.debug

    if not HAS_PANDAS:
        print("WARNING: pandas not installed — FBref/Understat layers skipped.")
        print("  Install: pip install pandas lxml html5lib\n")

    print("=" * 65)
    print("  Spurs manager-era stats fetcher")
    print(f"  Output : {OUTPUT_FILE}")
    print(f"  Debug  : {'on' if DEBUG else 'off (use --debug to enable)'}")
    print("=" * 65)

    eras: Dict[str, Era] = {key: Era(key) for key in MANAGERS}

    print("\n>>> Layer 1: soccerdata / FBref (primary)")
    pull_fbref(eras)

    print("\n>>> Layer 2: soccerdata / Understat (xG cross-check)")
    pull_understat_soccerdata(eras)

    print("\n>>> Layer 3: understat.com direct (async xG fallback)")
    pull_understat_direct(eras)

    print("\n>>> Layer 4: API-Football (broad fallback)")
    pull_api_football(eras, args.rapidapi_key)

    print("\n>>> Layer 5: hardcoded estimates (last resort)")
    apply_fallbacks(eras)

    output: Dict[str, Any] = {key: era.to_json() for key, era in eras.items()}

    print("\n" + "=" * 65)
    print("  FINAL SUMMARY")
    print("=" * 65)
    total_estimates = 0
    for key, era in eras.items():
        print(f"\n  {MANAGERS[key]['name']}  ({MANAGERS[key]['label']})")
        print(f"  {'Stat':<26} {'Value':<10}  Source")
        print(f"  {'-'*26} {'-'*10}  {'-'*35}")
        for stat in ALL_STATS:
            val  = era._d.get(stat, "MISSING")
            src  = era._src.get(stat, "—")
            flag = "  *** ESTIMATE" if stat in era._est else ""
            print(f"  {stat:<26} {str(val):<10}  {src}{flag}")
        total_estimates += len(era._est)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nWrote  {OUTPUT_FILE}")

    if total_estimates:
        print(f"\nWARNING: {total_estimates} stat(s) are hardcoded estimates.")
        print("  Re-run with --debug to see raw column names from each source.")
        print("  Stats flagged with *** above.")
    else:
        print("\nAll stats sourced from live data.")


if __name__ == "__main__":
    main()
