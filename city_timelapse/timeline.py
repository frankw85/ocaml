"""Shared timeline for the fictional city of Aldermouth.

Used by both the Blender scene builder and the caption overlay so that the
year shown on screen always matches what the 3D city looks like.
"""
import numpy as np

CITY = "ALDERMOUTH"
SUBTITLE = "FROM A RIVER FORD TO A METROPOLIS"

FPS = 24
TITLE_FRAMES = 3 * FPS          # title card at the start
ERA_FRAMES = 2 * FPS            # time spent travelling between two keyframe years
HOLD_FRAMES = 3 * FPS           # hold on the final year

# year, built-up radius (scene units), era label, population
KEYS = [
    (60,   0.35, "IRON AGE",         600),
    (150,  0.85, "IMPERIAL OUTPOST", 9_000),
    (400,  0.90, "LATE ANTIQUITY",   5_000),
    (700,  0.95, "EARLY MEDIEVAL",   2_500),
    (950,  1.15, "EARLY MEDIEVAL",   6_000),
    (1150, 1.55, "HIGH MEDIEVAL",    18_000),
    (1350, 1.95, "LATE MEDIEVAL",    32_000),
    (1500, 2.40, "RENAISSANCE",      60_000),
    (1650, 3.40, "EARLY MODERN",     190_000),
    (1760, 4.70, "GEORGIAN ERA",     420_000),
    (1830, 7.00, "INDUSTRIAL ERA",   1_100_000),
    (1880, 10.5, "INDUSTRIAL ERA",   2_900_000),
    (1920, 14.0, "INTERWAR YEARS",   5_300_000),
    (1960, 17.0, "POST-WAR ERA",     7_100_000),
    (1995, 18.6, "GLOBAL CITY",      7_600_000),
    (2025, 19.6, "GLOBAL CITY",      8_700_000),
]

# year range, title, description
EVENTS = [
    (130, 260, "AD 150 · THE STONE WALL", "A 3 km wall with five gates encloses the imperial town"),
    (1340, 1380, "1348 · THE PESTILENCE", "Plague arrives by ship; within two years a third of the town has died"),
    (1680, 1720, "1694 · THE GREAT FIRE", "Four days of fire level most of the walled town"),
    (1825, 1860, "1836 · THE RAILWAY", "The first line reaches the city and the fields start to fill"),
    (1938, 1960, "1940-44 · THE WAR YEARS", "Air raids flatten the docks; a green belt follows in 1952"),
]

YEARS = np.array([k[0] for k in KEYS], float)
RADII = np.array([k[1] for k in KEYS], float)
POPS = np.array([k[3] for k in KEYS], float)

FRAME_START = 1
ANIM_START = FRAME_START + TITLE_FRAMES
ANIM_END = ANIM_START + ERA_FRAMES * (len(KEYS) - 1)
FRAME_END = ANIM_END + HOLD_FRAMES


def key_frames():
    """Frame number of each keyframe year."""
    return [ANIM_START + i * ERA_FRAMES for i in range(len(KEYS))]


def year_at(frame):
    return float(np.interp(frame, key_frames(), YEARS))


def era_at(year):
    i = int(np.searchsorted(YEARS, year, side="right")) - 1
    return KEYS[max(0, min(i, len(KEYS) - 1))][2]


def population_at(year):
    return float(np.interp(year, YEARS, POPS))


def event_at(year):
    for y0, y1, title, desc in EVENTS:
        if y0 <= year <= y1:
            return title, desc
    return None


def birth_from_distance(d):
    """Year a spot at effective distance d from the core gets built up."""
    b = np.interp(d, RADII, YEARS)
    return np.where(d > RADII[-1], 1e5, b)
