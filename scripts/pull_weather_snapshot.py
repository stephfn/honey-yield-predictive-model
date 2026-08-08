"""Build `data/honey_weather.parquet` -- daily reanalysis weather for every hive site.

Section 7.3 item 3 of `Milestone_4_alt.ipynb`: the period feature sets carry no weather at
all, and a week's nectar flow is far more plausibly weather-driven than tomorrow's gram
count is. `honeymodel.data.WEATHER_TABLE` has pointed at a file that did not exist since
Milestone 4.

**This is the only script here that touches the network**, in the same way
`pull_source_snapshot.py` is the only one that touches Tailscale. It runs once, writes a
Parquet file, and every notebook reads that file. Re-running it is not part of reproducing
a result.

Source: the Open-Meteo Historical Weather API, which serves ERA5 / ERA5-Land reanalysis on
a ~9 km grid. Reanalysis rather than GHCN-Daily station records, which the Milestone 4 plan
named as the first choice and listed sparse German station coverage as the matching risk:
34 sites spread over 6 degrees of longitude would each need a nearest-station search, a
distance threshold, and a gap-filling rule. Reanalysis is gridded, complete, and has no
missing days, which removes three judgement calls from the feature layer. The cost is that
these are model values, not measurements -- stated wherever the features are used.

    Open-Meteo (2023). Historical Weather API. https://open-meteo.com/  (CC-BY-4.0)
    ERA5 hourly data, Hersbach et al. (2023), Copernicus Climate Change Service.

Usage
-----
    python scripts/pull_weather_snapshot.py            # all sites, full record
    python scripts/pull_weather_snapshot.py --limit 3  # smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from honeymodel.data import WEATHER_TABLE, load_model_table  # noqa: E402

ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"

#: Daily variables requested. Chosen for a mechanism, not for coverage:
#: temperature and sunshine gate whether bees fly at all, precipitation stops foraging
#: outright, and evapotranspiration is the closest free proxy for nectar secretion.
DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "precipitation_hours",
    "sunshine_duration",
    "shortwave_radiation_sum",
    "wind_speed_10m_max",
    "et0_fao_evapotranspiration",
]

#: Pulled wider than the modelling window so rolling weather features have a warm-up.
START_DATE = "2019-05-01"
END_DATE = "2022-12-31"


def hive_sites(frame: pd.DataFrame) -> pd.DataFrame:
    """Distinct (latitude, longitude) pairs with the hives that sit on each.

    The published coordinates are rounded to 0.1 degrees, so 78 hives collapse to 34
    sites -- roughly 11 km of latitude, which is finer than the reanalysis grid anyway.
    Fetching per site rather than per hive is 34 requests instead of 78 for identical data.
    """
    sites = (
        frame.groupby(["latitude", "longitude"], as_index=False)
        .agg(hives=("hive_id", "nunique"), hive_days=("hive_id", "size"))
        .sort_values(["latitude", "longitude"])
        .reset_index(drop=True)
    )
    sites["site_id"] = sites.index
    return sites


#: One Parquet per site, so an interrupted or rate-limited run resumes instead of
#: re-fetching what it already has. Gitignored -- only the merged table is committed.
CACHE_DIR = WEATHER_TABLE.parent / "_weather_cache"


def fetch_site(latitude: float, longitude: float, retries: int = 6) -> pd.DataFrame:
    """One site, full record. Backs off hard on 429.

    The free tier meters by *cost*, not by request count: one call here asks for 10
    variables over 1,341 days, which is worth far more than one unit. A flat courtesy
    sleep is not enough, so 429 is retried with escalating waits rather than treated as
    a failure.
    """
    query = urllib.parse.urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "daily": ",".join(DAILY_VARIABLES),
            "timezone": "UTC",
        }
    )
    url = f"{ENDPOINT}?{query}"
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                payload = json.load(response)
            break
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == retries:
                raise RuntimeError(f"{latitude},{longitude}: HTTP {error.code}") from error
            wait = min(300, 30 * attempt)
            print(f"    rate limited, waiting {wait}s (attempt {attempt})", flush=True)
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            if attempt == retries:
                raise RuntimeError(f"{latitude},{longitude} failed after {retries} tries") from error
            time.sleep(5 * attempt)

    daily = pd.DataFrame(payload["daily"])
    daily = daily.rename(columns={"time": "measurement_date"})
    daily["measurement_date"] = pd.to_datetime(daily["measurement_date"])
    daily["latitude"] = latitude
    daily["longitude"] = longitude
    return daily


def build(limit: int | None = None, out_path: Path = WEATHER_TABLE) -> pd.DataFrame:
    model = load_model_table()
    sites = hive_sites(model)
    if limit:
        sites = sites.head(limit)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for row in sites.itertuples():
        cached = CACHE_DIR / f"site_{row.site_id:02d}.parquet"
        if cached.exists():
            frame = pd.read_parquet(cached)
            status = "cached"
        else:
            frame = fetch_site(row.latitude, row.longitude)
            frame["site_id"] = row.site_id
            frame.to_parquet(cached, index=False)
            status = "fetched"
            time.sleep(8.0)  # pacing; the free tier meters by request cost, not count
        frames.append(frame)
        print(
            f"  site {row.site_id:2d}  {row.latitude:5.1f},{row.longitude:5.1f}  "
            f"{len(frame):,} days  ({row.hives} hives)  [{status}]",
            flush=True,
        )

    weather = pd.concat(frames, ignore_index=True)

    # Derived-at-source rather than in the feature layer, because both the daily and the
    # period feature builders want them and neither should own the definition.
    weather["growing_degree_days_10c"] = (weather["temperature_2m_mean"] - 10.0).clip(lower=0)
    weather["foraging_hours_proxy"] = weather["sunshine_duration"] / 3600.0
    weather["is_foraging_day"] = (
        (weather["temperature_2m_max"] > 12.0) & (weather["precipitation_sum"] < 1.0)
    ).astype(float)

    expected_days = (pd.Timestamp(END_DATE) - pd.Timestamp(START_DATE)).days + 1
    per_site = weather.groupby("site_id").measurement_date.nunique()
    if not (per_site == expected_days).all():
        raise AssertionError(
            f"expected {expected_days} days per site, got {per_site.min()}..{per_site.max()}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    weather.to_parquet(out_path, index=False)
    print(
        f"\nwrote {out_path} -- {len(weather):,} rows, {weather.site_id.nunique()} sites, "
        f"{weather.measurement_date.min().date()}..{weather.measurement_date.max().date()}"
    )
    return weather


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="fetch only the first N sites")
    parser.add_argument("--out", type=Path, default=WEATHER_TABLE)
    args = parser.parse_args()
    build(limit=args.limit, out_path=args.out)


if __name__ == "__main__":
    main()
