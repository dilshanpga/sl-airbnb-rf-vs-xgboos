"""
Feature engineering for Sri Lankan Airbnb listings.

The raw dataset only has a free-text title (`name`), `numberOfGuests` and
`stars`. Everything else the models use is derived here from the title,
using local knowledge of how Sri Lankan hosts write listings and where
the tourist areas are.

Both the notebook and the front end import this file, so the pickled model
and the app always build features the same way.
"""
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# --------------------------------------------------------------------------
# Sri Lankan tourist regions, matched against the listing title.
# Order matters: the first region that matches wins.
# --------------------------------------------------------------------------
REGIONS = {
    "Colombo & suburbs": [
        "colombo", "havelock", "mount lavinia", "dehiwala", "nugegoda",
        "kotte", "battaramulla", "rajagiriya", "boralesgamuwa", "wellawatte",
        "bambalapitiya", "kollupitiya", "maharagama", "moratuwa", "malabe",
    ],
    "South coast": [
        "galle", "unawatuna", "mirissa", "weligama", "ahangama", "hikkaduwa",
        "bentota", "tangalle", "hiriketiya", "koggala", "talpe", "midigama",
        "madiha", "matara", "dikwella", "kalutara", "beruwala", "induruwa",
        "ambalangoda", "habaraduwa", "polhena", "talalla", "dickwella",
        "kabalana", "thalpe", "aluthgama", "balapitiya", "kosgoda",
    ],
    "Hill country": [
        "kandy", "ella", "nuwara eliya", "haputale", "bandarawela", "hatton",
        "knuckles", "kitulgala", "badulla", "nuwara", "eliya", "adams peak",
        "nallathanniya", "kandapola", "peradeniya", "gampola", "tea estate",
    ],
    "Cultural triangle": [
        "sigiriya", "dambulla", "habarana", "anuradhapura", "polonnaruwa",
        "matale", "kandalama", "kurunegala",
    ],
    "North & East": [
        "jaffna", "trincomalee", "trinco", "nilaveli", "arugam", "pasikudah",
        "batticaloa", "kalkudah", "uppuveli", "nallur", "mannar", "pottuvil",
    ],
    "West coast & airport": [
        "negombo", "airport", "katunayake", "kalpitiya", "wattala", "ja-ela",
        "chilaw", "marawila", "waikkal",
    ],
    "Wildlife & safari": [
        "yala", "udawalawe", "tissa", "kirinda", "tissamaharama", "wilpattu",
        "minneriya", "kataragama", "safari",
    ],
}

# --------------------------------------------------------------------------
# Keyword groups. Each becomes a 0/1 feature.
# --------------------------------------------------------------------------
KEYWORDS = {
    # Words that describe a single room inside a bigger property
    "kw_room": r"\b(room|rooms|chamber|chambre|zimmer|ensuite|en-suite)\b",
    "kw_bed_type": r"\b(double|twin|triple|single|king|queen|dbl|quad|family room)\b",
    "kw_room_grade": r"\b(deluxe|superior|standard|executive|economy|premier)\b",
    # Words that describe a whole property
    "kw_whole_property": (
        r"\b(villa|villas|house|home|bungalow|cottage|apartment|apt|condo|"
        r"entire|penthouse|chalet|loft|residence|whole|cabin|townhouse|"
        r"maison|haus|flat)\b"
    ),
    "kw_bedrooms": r"\b(\d+\s*(br|bed|beds|bedroom|bedrooms|bhk))\b|\bbedrooms?\b",
    # Words for hotel-style businesses that rent rooms
    "kw_hotel_business": (
        r"\b(hotel|resort|boutique|inn|lodge|hostel|guesthouse|guest house|"
        r"homestay|home stay|b&b|bnb|bed and breakfast|motel)\b"
    ),
    "kw_suite": r"\b(suite|suites)\b",
    "kw_cabana": r"\b(cabana|cabanas|hut|tent|glamping|treehouse|tree house)\b",
    # Amenities / selling points
    "kw_pool": r"\b(pool|infinity|jacuzzi|swimming)\b",
    "kw_luxury": r"\b(luxury|luxurious|luxe|premium|exclusive|grand|royal)\b",
    "kw_beach": r"\b(beach|beachfront|sea|ocean|seaview|surf|bay|lagoon)\b",
    "kw_nature": r"\b(nature|jungle|forest|garden|river|lake|eco|paddy|mountain|hill|hills|tea)\b",
    "kw_kitchen": r"\b(kitchen|kitchenette|self catering|self-catering|fully furnished|furnished)\b",
    "kw_breakfast": r"\b(breakfast|half board|full board|meals|chef)\b",
    "kw_ac": r"\b(ac|a/c|air conditioned|air-conditioned|aircon)\b",
    "kw_budget": r"\b(budget|cheap|affordable|economy|backpacker)\b",
    "kw_family": r"\b(family|group|groups|sleeps)\b",
}

FOREIGN_WORDS = r"\b(chambre|maison|vacances|avec|proche|zimmer|haus|mit|casa|camera|habitaci[oó]n)\b"

NUMERIC_FEATURES = (
    ["numberOfGuests", "stars", "bedrooms_in_title", "title_length",
     "title_words", "title_has_number", "title_foreign_language"]
    + list(KEYWORDS)
)
CATEGORICAL_FEATURES = ["region"]


def find_region(title: str) -> str:
    t = title.lower()
    for region, places in REGIONS.items():
        for place in places:
            if re.search(r"\b" + re.escape(place) + r"\b", t):
                return region
    return "Unknown"


def bedrooms_from_title(title: str) -> float:
    """'Cozy 2 BR apartment' -> 2.0. Returns NaN when the title has no count."""
    m = re.search(r"(\d+)\s*-?\s*(br|bed|beds|bedroom|bedrooms|bhk|bdr)\b", title.lower())
    if m:
        n = float(m.group(1))
        return n if 0 < n <= 15 else np.nan
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
    m = re.search(r"\b(one|two|three|four|five|six)\s*-?\s*(br|bed|beds|bedroom|bedrooms)\b", title.lower())
    if m:
        return float(words[m.group(1)])
    return np.nan


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw columns (name, numberOfGuests, stars) into model features."""
    df = df.copy()
    title = df["name"].fillna("").astype(str)
    low = title.str.lower()

    out = pd.DataFrame(index=df.index)
    out["numberOfGuests"] = pd.to_numeric(df["numberOfGuests"], errors="coerce")
    # stars stays NaN for unrated listings; the pipeline imputes it
    out["stars"] = pd.to_numeric(df.get("stars", np.nan), errors="coerce")
    out["bedrooms_in_title"] = title.apply(bedrooms_from_title)
    out["title_length"] = title.str.len()
    out["title_words"] = title.str.split().str.len().fillna(0)
    out["title_has_number"] = title.str.contains(r"\d", regex=True).astype(int)
    out["title_foreign_language"] = low.apply(lambda t: int(bool(re.search(FOREIGN_WORDS, t))))
    for name, pattern in KEYWORDS.items():
        out[name] = low.apply(lambda t, p=pattern: int(bool(re.search(p, t))))
    out["region"] = title.apply(find_region)
    return out[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """sklearn wrapper so feature engineering is saved inside the pipeline/pickle."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return engineer_features(X)


def room_type_group(room_type: str) -> str:
    """Collapse Airbnb's 83 room types into 5 groups."""
    r = str(room_type).lower()
    if r.startswith("entire") or r in ("tiny home", "earthen home", "casa particular", "cycladic home"):
        return "Entire place"
    if r.startswith("private room"):
        return "Private room"
    if r.startswith("room in"):
        return "Hotel-style room"
    if r.startswith("shared room"):
        return "Shared room"
    return "Unique stay"
