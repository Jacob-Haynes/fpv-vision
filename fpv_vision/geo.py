"""Operator-entered positions: "lat, lon" decimal degrees or an MGRS grid (WGS84).

Self-contained UTM/MGRS maths (no native deps) so it installs cleanly on the Pi.
Polar UPS zones (beyond 84N / 80S) are not supported.
"""

from __future__ import annotations

import math
import re

_A = 6378137.0
_F = 1 / 298.257223563
_E2 = _F * (2 - _F)
_EP2 = _E2 / (1 - _E2)
_K0 = 0.9996

_BANDS = "CDEFGHJKLMNPQRSTUVWX"  # 8-degree latitude bands from 80S
_COL_SETS = ("ABCDEFGH", "JKLMNPQR", "STUVWXYZ")
_ROW_LETTERS = "ABCDEFGHJKLMNPQRSTUV"  # 20 letters, cycle every 2,000 km


class PositionError(ValueError):
    pass


def _zone_for(lat: float, lon: float) -> int:
    zone = int((lon + 180) // 6) + 1
    if 56 <= lat < 64 and 3 <= lon < 12:
        zone = 32
    if 72 <= lat < 84:
        if 0 <= lon < 9:
            zone = 31
        elif 9 <= lon < 21:
            zone = 33
        elif 21 <= lon < 33:
            zone = 35
        elif 33 <= lon < 42:
            zone = 37
    return min(zone, 60)


def _band_for(lat: float) -> str:
    if lat >= 72:
        return "X"
    return _BANDS[int((lat + 80) // 8)]


def latlon_to_utm(lat: float, lon: float) -> tuple[int, float, float]:
    zone = _zone_for(lat, lon)
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    phi, lam = math.radians(lat), math.radians(lon)
    n = _A / math.sqrt(1 - _E2 * math.sin(phi) ** 2)
    t = math.tan(phi) ** 2
    c = _EP2 * math.cos(phi) ** 2
    a = math.cos(phi) * (lam - lon0)
    m = _A * ((1 - _E2 / 4 - 3 * _E2**2 / 64 - 5 * _E2**3 / 256) * phi
              - (3 * _E2 / 8 + 3 * _E2**2 / 32 + 45 * _E2**3 / 1024) * math.sin(2 * phi)
              + (15 * _E2**2 / 256 + 45 * _E2**3 / 1024) * math.sin(4 * phi)
              - (35 * _E2**3 / 3072) * math.sin(6 * phi))
    easting = _K0 * n * (a + (1 - t + c) * a**3 / 6 + (5 - 18 * t + t**2 + 72 * c - 58 * _EP2) * a**5 / 120) + 500000
    northing = _K0 * (m + n * math.tan(phi) * (a**2 / 2 + (5 - t + 9 * c + 4 * c**2) * a**4 / 24
                                               + (61 - 58 * t + t**2 + 600 * c - 330 * _EP2) * a**6 / 720))
    if lat < 0:
        northing += 10000000
    return zone, easting, northing


def utm_to_latlon(zone: int, easting: float, northing: float, northern: bool) -> tuple[float, float]:
    x = easting - 500000
    y = northing if northern else northing - 10000000
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    m = y / _K0
    mu = m / (_A * (1 - _E2 / 4 - 3 * _E2**2 / 64 - 5 * _E2**3 / 256))
    e1 = (1 - math.sqrt(1 - _E2)) / (1 + math.sqrt(1 - _E2))
    phi1 = (mu + (3 * e1 / 2 - 27 * e1**3 / 32) * math.sin(2 * mu)
            + (21 * e1**2 / 16 - 55 * e1**4 / 32) * math.sin(4 * mu)
            + (151 * e1**3 / 96) * math.sin(6 * mu) + (1097 * e1**4 / 512) * math.sin(8 * mu))
    n1 = _A / math.sqrt(1 - _E2 * math.sin(phi1) ** 2)
    t1 = math.tan(phi1) ** 2
    c1 = _EP2 * math.cos(phi1) ** 2
    r1 = _A * (1 - _E2) / (1 - _E2 * math.sin(phi1) ** 2) ** 1.5
    d = x / (n1 * _K0)
    lat = phi1 - (n1 * math.tan(phi1) / r1) * (d**2 / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * _EP2) * d**4 / 24
                                               + (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 252 * _EP2 - 3 * c1**2) * d**6 / 720)
    lon = lon0 + (d - (1 + 2 * t1 + c1) * d**3 / 6
                  + (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * _EP2 + 24 * t1**2) * d**5 / 120) / math.cos(phi1)
    return math.degrees(lat), math.degrees(lon)


def to_mgrs(lat: float, lon: float, precision: int = 5) -> str:
    if not -80 <= lat < 84:
        raise PositionError("MGRS outside UTM latitudes (80S-84N) is not supported")
    zone, e, n = latlon_to_utm(lat, lon)
    col = _COL_SETS[(zone - 1) % 3][int(e // 100000) - 1]
    row_offset = 5 if zone % 2 == 0 else 0
    row = _ROW_LETTERS[(int(n // 100000) + row_offset) % 20]
    div = 10 ** (5 - precision)
    ee = int(e % 100000) // div
    nn = int(n % 100000) // div
    return f"{zone}{_band_for(lat)}{col}{row}{ee:0{precision}d}{nn:0{precision}d}"


def format_mgrs(grid: str) -> str:
    """'30UWB1234567890' -> '30U WB 12345 67890'."""
    m = _MGRS_RE.match(grid.replace(" ", "").upper())
    if not m:
        return grid
    zone, band, sq, digits = m.groups()
    h = len(digits) // 2
    return f"{int(zone)}{band} {sq} {digits[:h]} {digits[h:]}".strip()


_MGRS_RE = re.compile(r"^(\d{1,2})([C-HJ-NP-X])([A-HJ-NP-Z]{2})(\d{0,10})$")


def from_mgrs(grid: str) -> tuple[float, float]:
    g = grid.replace(" ", "").upper()
    m = _MGRS_RE.match(g)
    if not m:
        raise PositionError(f"not an MGRS grid: {grid!r}")
    zone_s, band, sq, digits = m.groups()
    zone = int(zone_s)
    if not 1 <= zone <= 60 or len(digits) % 2:
        raise PositionError(f"bad MGRS grid: {grid!r}")
    prec = len(digits) // 2
    scale = 10 ** (5 - prec)
    e_in = int(digits[:prec] or 0) * scale
    n_in = int(digits[prec:] or 0) * scale
    # centre of the stated precision square
    if prec < 5:
        e_in += scale // 2
        n_in += scale // 2

    col_set = _COL_SETS[(zone - 1) % 3]
    if sq[0] not in col_set:
        raise PositionError(f"column letter {sq[0]} not valid in zone {zone}")
    easting = (col_set.index(sq[0]) + 1) * 100000 + e_in
    row_offset = 5 if zone % 2 == 0 else 0
    row_idx = (_ROW_LETTERS.index(sq[1]) - row_offset) % 20
    northing_100k = row_idx * 100000

    northern = band >= "N"
    band_lat_min = -80 + 8 * _BANDS.index(band)
    # Lowest northing of the band; add 2,000 km cycles until we reach it.
    _, _, n_min = latlon_to_utm(band_lat_min, (zone - 1) * 6 - 180 + 3)
    northing = northing_100k + n_in
    while northing < n_min - 1000:
        northing += 2000000
    return utm_to_latlon(zone, easting, northing, northern)


_LATLON_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)\s*$")


def parse_position(text: str) -> tuple[float, float]:
    """Accept '51.15, -1.75' or an MGRS grid like '30U WB 12345 67890'."""
    if not text or not text.strip():
        raise PositionError("no position given")
    m = _LATLON_RE.match(text)
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise PositionError("lat/lon out of range")
        return lat, lon
    return from_mgrs(text)
