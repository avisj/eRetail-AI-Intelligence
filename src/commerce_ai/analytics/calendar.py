"""Calendar and Business Event Feature Engineering Engine.

Extracts calendar features, cyclical trigonometric encodings, and business event markers
(holidays, promotions, paydays, mega-sales) with pluggable international calendar support.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


@dataclass
class BusinessEvent:
    """Configurable commercial event definition."""

    event_date: Union[date, str]
    event_name: str
    event_type: str  # "Holiday", "Promotion", "Payday", "Mega Sale", "Festival"
    impact_factor: float = 1.0  # Estimated demand impact multiplier


def get_standard_ecommerce_events(years: Optional[List[int]] = None) -> List[BusinessEvent]:
    """Provides a sample library of major retail and seasonal events.

    Agnostic architecture: events are configurable data models rather than hardcoded logic.
    """
    target_years = years or [2024, 2025, 2026]
    events: List[BusinessEvent] = []

    for y in target_years:
        # Fixed-date events
        events.extend([
            BusinessEvent(date(y, 1, 1), "New Year's Day", "Holiday", 1.2),
            BusinessEvent(date(y, 2, 14), "Valentine's Day", "Holiday", 1.4),
            BusinessEvent(date(y, 7, 4), "Summer Sale / Independence Day", "Promotion", 1.5),
            BusinessEvent(date(y, 12, 24), "Christmas Eve", "Holiday", 1.3),
            BusinessEvent(date(y, 12, 25), "Christmas Day", "Holiday", 0.7),
            BusinessEvent(date(y, 12, 31), "New Year's Eve", "Holiday", 1.2),
        ])

        # Monthly payday cycles (28th of every month)
        for m in range(1, 13):
            events.append(
                BusinessEvent(date(y, m, 28), "Month-End Payday", "Payday", 1.25)
            )

    # Specific historical Cyber Week dates for 2024 & 2025
    events.extend([
        # 2024
        BusinessEvent(date(2024, 11, 29), "Black Friday 2024", "Mega Sale", 3.5),
        BusinessEvent(date(2024, 11, 30), "Cyber Weekend 2024", "Mega Sale", 2.8),
        BusinessEvent(date(2024, 12, 2), "Cyber Monday 2024", "Mega Sale", 3.8),
        BusinessEvent(date(2024, 10, 31), "Diwali Festive 2024", "Festival", 2.5),
        # 2025
        BusinessEvent(date(2025, 11, 28), "Black Friday 2025", "Mega Sale", 3.5),
        BusinessEvent(date(2025, 11, 29), "Cyber Weekend 2025", "Mega Sale", 2.8),
        BusinessEvent(date(2025, 12, 1), "Cyber Monday 2025", "Mega Sale", 3.8),
        BusinessEvent(date(2025, 10, 20), "Diwali Festive 2025", "Festival", 2.5),
    ])

    return events


def build_calendar_features(
    df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """Extract standard calendar and cyclical trigonometric features from a date column.

    Features generated:
        - year, month, quarter, week, day_of_week (0=Mon, 6=Sun), day_of_month, day_of_year
        - is_weekend, is_month_start, is_month_end, is_quarter_start, is_quarter_end
        - Cyclical encodings: sin_day_of_week, cos_day_of_week, sin_month, cos_month

    Args:
        df: DataFrame containing date column.
        date_col: Name of the date column.

    Returns:
        pd.DataFrame: Augmented DataFrame with calendar features.
    """
    result = df.copy()
    dt_series = pd.to_datetime(result[date_col])

    result["year"] = dt_series.dt.year
    result["month"] = dt_series.dt.month
    result["quarter"] = dt_series.dt.quarter
    result["week"] = dt_series.dt.isocalendar().week.astype(int)
    result["day_of_week"] = dt_series.dt.dayofweek
    result["day_of_month"] = dt_series.dt.day
    result["day_of_year"] = dt_series.dt.dayofyear

    # Binary flags
    result["is_weekend"] = result["day_of_week"].isin([5, 6]).astype(int)
    result["is_month_start"] = dt_series.dt.is_month_start.astype(int)
    result["is_month_end"] = dt_series.dt.is_month_end.astype(int)
    result["is_quarter_start"] = dt_series.dt.is_quarter_start.astype(int)
    result["is_quarter_end"] = dt_series.dt.is_quarter_end.astype(int)

    # Cyclical trigonometric encodings
    # Day of week (period = 7)
    result["sin_day_of_week"] = np.round(np.sin(2 * np.pi * result["day_of_week"] / 7.0), 6)
    result["cos_day_of_week"] = np.round(np.cos(2 * np.pi * result["day_of_week"] / 7.0), 6)

    # Month of year (period = 12, month 1 mapped to 0)
    result["sin_month"] = np.round(np.sin(2 * np.pi * (result["month"] - 1) / 12.0), 6)
    result["cos_month"] = np.round(np.cos(2 * np.pi * (result["month"] - 1) / 12.0), 6)

    return result


def build_event_features(
    df: pd.DataFrame,
    events: Optional[List[BusinessEvent]] = None,
    date_col: str = "date",
) -> pd.DataFrame:
    """Enrich dataset with business event flags and impact multipliers.

    Args:
        df: DataFrame containing date column.
        events: List of BusinessEvent instances (defaults to standard ecommerce events).
        date_col: Name of the date column.

    Returns:
        pd.DataFrame: DataFrame with [is_event, event_name, event_type, event_impact_factor].
    """
    result = df.copy()
    event_list = events if events is not None else get_standard_ecommerce_events()

    # Create event lookup dictionary by normalized date
    event_rows = []
    for ev in event_list:
        ev_dt = pd.to_datetime(ev.event_date).normalize()
        event_rows.append({
            "_event_date": ev_dt,
            "is_event": 1,
            "event_name": ev.event_name,
            "event_type": ev.event_type,
            "event_impact_factor": ev.impact_factor,
        })

    if not event_rows:
        result["is_event"] = 0
        result["event_name"] = "None"
        result["event_type"] = "Normal"
        result["event_impact_factor"] = 1.0
        return result

    event_df = pd.DataFrame(event_rows).drop_duplicates(subset=["_event_date"], keep="first")

    temp_date = pd.to_datetime(result[date_col]).dt.normalize()
    result["_temp_merge_date"] = temp_date

    merged = pd.merge(
        result,
        event_df,
        left_on="_temp_merge_date",
        right_on="_event_date",
        how="left",
    )

    merged["is_event"] = merged["is_event"].fillna(0).astype(int)
    merged["event_name"] = merged["event_name"].fillna("None")
    merged["event_type"] = merged["event_type"].fillna("Normal")
    merged["event_impact_factor"] = merged["event_impact_factor"].fillna(1.0).astype(float)

    merged.drop(columns=["_temp_merge_date", "_event_date"], inplace=True, errors="ignore")
    return merged
