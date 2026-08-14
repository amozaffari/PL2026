"""Kickoff weather at the home stadium via the keyless Open-Meteo API.

Weather has only a marginal, well-documented effect on match outcomes; it is
attached to upcoming-fixture predictions as context, not used inside the model.
"""

import pandas as pd
import requests

from ..config import OPEN_METEO_FORECAST_URL, STADIUMS, USER_AGENT


def kickoff_weather(home_team: str, kickoff_utc) -> dict:
    """Hourly forecast (temp °C, precip mm, wind km/h) nearest to kickoff.

    Returns an empty dict when the stadium is unknown or kickoff is outside
    the 16-day forecast horizon.
    """
    coords = STADIUMS.get(home_team)
    kickoff = pd.Timestamp(kickoff_utc)
    horizon = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=15)
    if coords is None or kickoff > horizon:
        return {}
    lat, lon = coords
    try:
        resp = requests.get(
            OPEN_METEO_FORECAST_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m,precipitation,wind_speed_10m",
                "start_date": kickoff.strftime("%Y-%m-%d"),
                "end_date": kickoff.strftime("%Y-%m-%d"),
                "timezone": "UTC",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=20,
        )
        resp.raise_for_status()
        hourly = resp.json()["hourly"]
    except (requests.RequestException, KeyError):
        return {}
    times = pd.to_datetime(hourly["time"]).tz_localize("UTC")
    idx = int(abs(times - kickoff).argmin())
    return {
        "temp_c": hourly["temperature_2m"][idx],
        "precip_mm": hourly["precipitation"][idx],
        "wind_kmh": hourly["wind_speed_10m"][idx],
    }
