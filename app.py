"""
Streamlit front end for the Sri Lankan Airbnb listing classifier.

Run:  pip install -r requirements.txt
      streamlit run app.py

Needs model.pkl and features.py in the same folder.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from features import KEYWORDS, engineer_features  # also needed to unpickle the pipeline

HERE = Path(__file__).parent

st.set_page_config(page_title="Entire Place or Room?", page_icon="🏝️", layout="centered")


@st.cache_resource
def load_bundle():
    with open(HERE / "model.pkl", "rb") as f:
        return pickle.load(f)


bundle = load_bundle()
model = bundle["model"]
classes = bundle["classes"]

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("About the model")
    st.write(f"**Model:** {bundle['model_name']}")
    st.write("**Trained on:** 3,895 Sri Lankan Airbnb listings (Kaggle: *Destination Sri Lanka*)")
    st.write("**Test-set performance**")
    m = bundle["test_metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{m['Test accuracy']:.1%}")
    c2.metric("F1", f"{m['F1']:.3f}")
    c3.metric("ROC-AUC", f"{m['Test ROC-AUC']:.3f}")
    st.caption("ML Lecture 7 class activity: Random Forest + XGBoost soft-voting ensemble.")

# ---------------------------------------------------------------- examples
EXAMPLES = {
    "Galle villa": ("Luxury 4 Bedroom Villa with Private Pool, Galle", 8, 4.95, False),
    "Mirissa hotel room": ("Deluxe Double Room with Sea View - Mirissa", 2, None, True),
    "Kandy homestay": ("Kandy Hill View Homestay - Family Room with breakfast", 4, 4.8, False),
}
if "title" not in st.session_state:
    st.session_state.update(title="", guests=2, stars=4.8, no_rating=False)


def fill(example):
    title, guests, stars, no_rating = EXAMPLES[example]
    st.session_state.update(title=title, guests=guests,
                            stars=stars if stars is not None else 4.8, no_rating=no_rating)


# ---------------------------------------------------------------- main form
st.title("🏝️ Entire place or room?")
st.write("Enter a Sri Lankan accommodation listing the way a host would post it on Airbnb, "
         "ikman.lk or Facebook. The model predicts whether the **whole property** is being "
         "rented or **just a room**.")

st.write("Try an example:")
cols = st.columns(len(EXAMPLES))
for col, name in zip(cols, EXAMPLES):
    col.button(name, on_click=fill, args=(name,), use_container_width=True)

st.text_input("Listing title", key="title",
              placeholder="e.g. Cozy 2 BR apartment near Galle Fort")
c1, c2 = st.columns(2)
c1.number_input("Number of guests", min_value=1, max_value=16, step=1, key="guests")
no_rating = c2.checkbox("No rating yet (new listing)", key="no_rating")
c2.number_input("Star rating", min_value=1.0, max_value=5.0, step=0.01, format="%.2f",
                key="stars", disabled=no_rating)

if st.button("Predict", type="primary", use_container_width=True):
    title = st.session_state.title.strip()
    if not title:
        st.warning("Please enter a listing title.")
        st.stop()

    row = pd.DataFrame([{
        "name": title,
        "numberOfGuests": int(st.session_state.guests),
        "stars": np.nan if no_rating else float(st.session_state.stars),
    }])
    p_entire = float(model.predict_proba(row)[0, 1])
    label = classes[int(p_entire >= 0.5)]
    confidence = p_entire if p_entire >= 0.5 else 1 - p_entire

    st.divider()
    if label == "Entire place":
        st.success(f"### 🏡 {label}")
    else:
        st.info(f"### 🛏️ {label}")
    st.write(f"Confidence: **{confidence:.1%}**")
    st.progress(p_entire, text=f"P(entire place) = {p_entire:.1%}")
    if 0.35 < p_entire < 0.65:
        st.warning("The title is ambiguous. Adding words such as *room*, *villa*, *apartment* "
                   "or a bedroom count would make the listing clearer to guests.")

    # What the feature builder picked up from the title
    feats = engineer_features(row).iloc[0]
    found = [k.replace("kw_", "").replace("_", " ") for k in KEYWORDS if feats[k] == 1]
    with st.expander("What the model detected in your title"):
        st.write(f"**Region:** {feats['region']}")
        if not np.isnan(feats["bedrooms_in_title"]):
            st.write(f"**Bedrooms mentioned:** {int(feats['bedrooms_in_title'])}")
        st.write("**Keyword groups found:** " + (", ".join(found) if found else "none"))
        st.write("**Rating:** " + ("missing, so the model imputes the training median"
                                   if no_rating else f"{st.session_state.stars:.2f}"))
