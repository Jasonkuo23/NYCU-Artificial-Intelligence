#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import requests
from bs4 import BeautifulSoup


SBR_BASE = "https://www.sportsbookreview.com"
MLB_POINTSPREAD_URL = (
    SBR_BASE + "/betting-odds/mlb-baseball/pointspread/full-game/?date={date_iso}"
)
MLB_MONEYLINE_URL = (
    SBR_BASE + "/betting-odds/mlb-baseball/money-line/full-game/?date={date_iso}"
)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
)


SPORTSBOOK_SLUG = "bet365"

DEFAULT_DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
DEFAULT_OUTPUT_CSV = DEFAULT_DATASET_DIR / "sbr_mlb_odds.csv"


class ParseError(RuntimeError):
    def __init__(self, step: str, message: str, *, url: str = "") -> None:
        super().__init__(f"{step}: {message}")
        self.step = step
        self.message = message
        self.url = url


def _dbg(enabled: bool, msg: str) -> None:
    if enabled:
        print(f"[DEBUG] {msg}", file=sys.stderr)


def _daterange_inclusive(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _parse_date_iso(s: str) -> date:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError as e:
        raise ParseError("args", f"invalid date {s!r}; expected YYYY-MM-DD") from e


def american_to_decimal(american: Optional[float]) -> Optional[float]:
    if american is None:
        return None
    try:
        a = float(american)
    except Exception:
        return None
    if abs(a) < 1e-9:
        return None
    if a > 0:
        return 1.0 + (a / 100.0)
    return 1.0 + (100.0 / abs(a))


def _fetch(url: str, *, timeout_s: int, max_retries: int, debug: bool) -> str:
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}
    last_exc: Optional[BaseException] = None
    for attempt in range(1, max_retries + 2):
        try:
            _dbg(debug, f"GET {url} (attempt {attempt})")
            resp = requests.get(url, headers=headers, timeout=timeout_s)
            if resp.status_code >= 400:
                raise ParseError("fetch", f"HTTP {resp.status_code}", url=url)
            return resp.text
        except ParseError:
            raise
        except BaseException as e:
            last_exc = e
            time.sleep(1.0)

    raise ParseError("fetch", f"failed after retries: {last_exc}", url=url)


def _parse_next_data(html: str, *, url: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    script = soup.select_one("script#__NEXT_DATA__")
    if not script or not script.string:
        raise ParseError("next_data", "missing __NEXT_DATA__ script", url=url)
    try:
        return json.loads(script.string)
    except json.JSONDecodeError as e:
        raise ParseError("next_data", f"invalid JSON: {e}", url=url) from e


@dataclass(frozen=True)
class GameMeta:
    game_id: int
    date: str  # YYYY-MM-DD (local page date)
    away_team: str
    home_team: str
    away_score: Optional[int]
    home_score: Optional[int]
    result: str


@dataclass(frozen=True)
class Moneyline:
    away_odds_dec: Optional[float]
    home_odds_dec: Optional[float]


@dataclass(frozen=True)
class PointSpread:
    away_spread: Optional[float]
    home_spread: Optional[float]
    away_odds_dec: Optional[float]
    home_odds_dec: Optional[float]


def _extract_odds_table_model(next_data: dict[str, Any], *, url: str) -> Optional[dict[str, Any]]:
    """Returns oddsTableModel dict, or None when the page has no odds table (often no games)."""
    props = next_data.get("props")
    if not isinstance(props, dict):
        return None
    page_props = props.get("pageProps")
    if not isinstance(page_props, dict):
        return None

    odds_tables = page_props.get("oddsTables")
    if not isinstance(odds_tables, list) or not odds_tables:
        return None

    first = odds_tables[0]
    if not isinstance(first, dict):
        return None
    model = first.get("oddsTableModel")
    if not isinstance(model, dict):
        return None
    return model


def _parse_market_page(
    *,
    url: str,
    market: str,
    date_iso: str,
    timeout_s: int,
    max_retries: int,
    debug: bool,
) -> tuple[dict[tuple[int, str], GameMeta], dict[tuple[int, str], Any]]:
    """Returns (meta_by_key, market_by_key) keyed by (game_id, sportsbook_slug)."""

    # Parse SBR's embedded Next.js JSON instead of scraping rendered table DOM.
    html = _fetch(url, timeout_s=timeout_s, max_retries=max_retries, debug=debug)
    data = _parse_next_data(html, url=url)
    model = _extract_odds_table_model(data, url=url)
    if model is None:
        _dbg(debug, f"No odds table model for {market} (likely no games)")
        return {}, {}

    game_rows = model.get("gameRows")
    if game_rows is None:
        _dbg(debug, f"No gameRows for {market} (likely no games)")
        return {}, {}
    if not isinstance(game_rows, list):
        raise ParseError("parse", f"oddsTableModel.gameRows has unexpected type {type(game_rows).__name__}", url=url)
    if len(game_rows) == 0:
        _dbg(debug, f"Empty gameRows for {market} (no games)")
        return {}, {}

    meta_by_key: dict[tuple[int, str], GameMeta] = {}
    market_by_key: dict[tuple[int, str], Any] = {}

    for row in game_rows:
        gv = (row or {}).get("gameView")
        if not isinstance(gv, dict):
            continue
        game_id = gv.get("gameId")
        if not isinstance(game_id, int):
            continue

        away_team = ((gv.get("awayTeam") or {}) if isinstance(gv.get("awayTeam"), dict) else {}).get("shortName")
        home_team = ((gv.get("homeTeam") or {}) if isinstance(gv.get("homeTeam"), dict) else {}).get("shortName")
        if not away_team or not home_team:
            continue

        away_score_raw = gv.get("awayTeamScore")
        home_score_raw = gv.get("homeTeamScore")

        def to_int(x: Any) -> Optional[int]:
            if isinstance(x, int):
                return x
            if isinstance(x, float) and x.is_integer():
                return int(x)
            return None

        away_score = to_int(away_score_raw)
        home_score = to_int(home_score_raw)

        result = ""
        if away_score is not None and home_score is not None:
            if home_score > away_score:
                result = "home_win"
            elif away_score > home_score:
                result = "away_win"

        odds_views = row.get("oddsViews")
        if not isinstance(odds_views, list):
            continue

        for ov in odds_views:
            if not isinstance(ov, dict):
                continue
            sportsbook = ov.get("sportsbook")
            if not isinstance(sportsbook, str) or not sportsbook:
                continue
            if sportsbook != SPORTSBOOK_SLUG:
                continue

            cur = ov.get("currentLine")
            if not isinstance(cur, dict):
                # Some sportsbooks may not have a line.
                cur = {}

            key = (game_id, sportsbook)
            meta_by_key[key] = GameMeta(
                game_id=game_id,
                date=date_iso,
                away_team=str(away_team),
                home_team=str(home_team),
                away_score=away_score,
                home_score=home_score,
                result=result,
            )

            # Convert American odds to decimal odds for downstream EV calculations.
            if market == "moneyline":
                market_by_key[key] = Moneyline(
                    away_odds_dec=american_to_decimal(cur.get("awayOdds")),
                    home_odds_dec=american_to_decimal(cur.get("homeOdds")),
                )
            elif market == "pointspread":
                market_by_key[key] = PointSpread(
                    away_spread=cur.get("awaySpread"),
                    home_spread=cur.get("homeSpread"),
                    away_odds_dec=american_to_decimal(cur.get("awayOdds")),
                    home_odds_dec=american_to_decimal(cur.get("homeOdds")),
                )
            else:
                raise ParseError("market", f"unknown market {market!r}", url=url)

    _dbg(debug, f"Parsed {market}: {len(market_by_key)} sportsbook lines")
    return meta_by_key, market_by_key


def crawl(
    *,
    start_date: date,
    end_date: date,
    output_csv: str,
    delay_s: float,
    max_retries: int,
    timeout_s: int,
    debug: bool,
) -> int:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_out: list[dict[str, Any]] = []

    for d in _daterange_inclusive(start_date, end_date):
        date_iso = d.strftime("%Y-%m-%d")
        ps_url = MLB_POINTSPREAD_URL.format(date_iso=date_iso)
        ml_url = MLB_MONEYLINE_URL.format(date_iso=date_iso)

        print(f"[INFO] Date {date_iso}", file=sys.stderr)

        # Fetch spread market for the date.
        meta_ps, ps = _parse_market_page(
            url=ps_url,
            market="pointspread",
            date_iso=date_iso,
            timeout_s=timeout_s,
            max_retries=max_retries,
            debug=debug,
        )
        if delay_s > 0:
            time.sleep(delay_s)

        # Fetch moneyline market for the same date.
        meta_ml, ml = _parse_market_page(
            url=ml_url,
            market="moneyline",
            date_iso=date_iso,
            timeout_s=timeout_s,
            max_retries=max_retries,
            debug=debug,
        )

        if not ps and not ml:
            print(f"[INFO] No games for {date_iso}", file=sys.stderr)
            if delay_s > 0:
                time.sleep(delay_s)
            continue

        # Join spread + moneyline rows using (game_id, sportsbook) as the stable key.
        keys = set(ps.keys()) | set(ml.keys())
        _dbg(debug, f"Join keys (game_id,sportsbook): {len(keys)}")

        for key in sorted(keys):
            meta = meta_ps.get(key) or meta_ml.get(key)
            if meta is None:
                continue

            ps_row: Optional[PointSpread] = ps.get(key)
            ml_row: Optional[Moneyline] = ml.get(key)

            # Derive spread label from final score and home spread when both are available.
            spread_result = ""
            if (
                meta.home_score is not None
                and meta.away_score is not None
                and ps_row is not None
                and ps_row.home_spread is not None
            ):
                margin = float(meta.home_score) + float(ps_row.home_spread) - float(meta.away_score)
                if margin > 0:
                    spread_result = "home_cover"
                elif margin < 0:
                    spread_result = "away_cover"
                else:
                    spread_result = "push"

            rows_out.append(
                {
                    "date": meta.date,
                    "game_id": meta.game_id,
                    "away_team": meta.away_team,
                    "home_team": meta.home_team,
                    "away_score": meta.away_score,
                    "home_score": meta.home_score,
                    "result": meta.result,
                    "spread_result": spread_result,
                    "moneyline_away": (None if ml_row is None else ml_row.away_odds_dec),
                    "moneyline_home": (None if ml_row is None else ml_row.home_odds_dec),
                    "away_spread": (None if ps_row is None else ps_row.away_spread),
                    "home_spread": (None if ps_row is None else ps_row.home_spread),
                    "spread_away": (None if ps_row is None else ps_row.away_odds_dec),
                    "spread_home": (None if ps_row is None else ps_row.home_odds_dec),
                }
            )

        if delay_s > 0:
            time.sleep(delay_s)

    # Write one normalized row per game with both markets and optional labels.
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "date",
                "game_id",
                "away_team",
                "home_team",
                "away_score",
                "home_score",
                "result",
                "spread_result",
                "moneyline_away_decimal",
                "moneyline_home_decimal",
                "away_spread",
                "home_spread",
                "spread_away_decimal",
                "spread_home_decimal",
            ]
        )

        def fmt(x: Any) -> str:
            if x is None:
                return ""
            if isinstance(x, float):
                return f"{x:.4f}".rstrip("0").rstrip(".")
            return str(x)

        for r in rows_out:
            w.writerow(
                [
                    r["date"],
                    r["game_id"],
                    r["away_team"],
                    r["home_team"],
                    fmt(r["away_score"]),
                    fmt(r["home_score"]),
                    r["result"],
                    r["spread_result"],
                    fmt(r["moneyline_away"]),
                    fmt(r["moneyline_home"]),
                    fmt(r["away_spread"]),
                    fmt(r["home_spread"]),
                    fmt(r["spread_away"]),
                    fmt(r["spread_home"]),
                ]
            )

    print(f"Wrote {len(rows_out)} rows -> {output_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Scrape SportsbookReview MLB odds for a date range, collecting Moneyline and "
            "Point Spread (full game). Output odds are Decimal (converted from American)."
        )
    )
    ap.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    ap.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_CSV),
        help="Output CSV path (default: workspace dataset/sbr_mlb_odds.csv).",
    )
    ap.add_argument("--delay", type=float, default=0.2, help="Delay between requests (seconds).")
    ap.add_argument("--retries", type=int, default=2, help="Retries per page fetch.")
    ap.add_argument("--timeout", type=int, default=30, help="HTTP request timeout (seconds).")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    start = _parse_date_iso(args.start_date)
    end = _parse_date_iso(args.end_date)
    if end < start:
        raise SystemExit("end-date must be >= start-date")

    try:
        return crawl(
            start_date=start,
            end_date=end,
            output_csv=args.output,
            delay_s=float(args.delay),
            max_retries=int(args.retries),
            timeout_s=int(args.timeout),
            debug=bool(args.debug),
        )
    except ParseError as e:
        print(f"[ERROR] {e} (url={e.url})", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
