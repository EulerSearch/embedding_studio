from datetime import datetime, timezone
from typing import Optional


def current_time() -> datetime:
    """
    Returns the current UTC time with timezone.

    Used for passing it to Pydantic models
    for proper handling with the `freeze_time` function from the `freezegun`
    module.
    """
    return datetime.now(timezone.utc)


def unaware_utc_to_aware_utc(stamp: datetime):
    """
    :param stamp: datetime.datetime tz unaware object in utc
    :return: datetime.datetime tz aware object in utc
    """
    return stamp.replace(tzinfo=timezone.utc)


def utc_with_tz() -> datetime:
    """Return datetime with tz in utc"""
    return unaware_utc_to_aware_utc(datetime.utcnow())


def utc_timestamp() -> int:
    """Return timestamp in utc"""
    return int(utc_with_tz().timestamp())


def check_utc_timestamp(
    timestamp: int,
    delta_sec: Optional[int] = None,
    delta_minus_sec: Optional[int] = None,
    delta_plus_sec: Optional[int] = None,
) -> bool:
    """Check utc timestamp

    :param timestamp: utc timestamp (seconds)
    :param delta_sec: max delta between passed timestamp and current utc timestamp (seconds)
    :param delta_minus_sec:  max delta for timestamp in past
    :param delta_plus_sec: max delta for timestamp in future
    :return: True if all checks passed
    """
    cur_ts = utc_timestamp()
    ts_delta = timestamp - cur_ts
    delta_minus_sec = delta_minus_sec or delta_sec
    delta_plus_sec = delta_plus_sec or delta_sec
    if delta_minus_sec is not None and ts_delta < -delta_minus_sec:
        return False
    if delta_plus_sec is not None and ts_delta > delta_plus_sec:
        return False
    return True
