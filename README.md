# Entire Place or Room? Classifying Sri Lankan Airbnb Listings

**ML Lecture 7 Class Activity:** Random Forest vs XGBoost, with soft voting and stacking, on a local Sri Lankan dataset, plus a GenAI-generated front end.

## Problem
Many Sri Lankan accommodation listings (on Airbnb, ikman.lk and Facebook groups) only give a title and a guest count. "Sea View Villa – Mirissa" could be the whole villa or one room in it. Given only the **listing title**, **number of guests** and **star rating** (if any), the model predicts whether the listing is an **entire place** or a **room**.

## Dataset
*Destination Sri Lanka* (Kaggle, user **kanchana1990**): 5,000 Sri Lankan accommodations from Airbnb plus personal contacts. Columns: `name`, `numberOfGuests`, `roomType`, `stars` (59% missing).
After removing 46 duplicates and 84 "unique stays" (tents, treehouses, domes), **4,870 listings** remain: 52.6% entire places, 47.4% rooms.

## Results (test set, 974 listings)

| Model | CV ROC-AUC (5-fold) | Test ROC-AUC | Test accuracy | F1 | Train–test gap |
|---|---|---|---|---|---|
| Random Forest | 0.8912 ± 0.0154 | 0.898 | 81.4% | 0.825 | 4.2 pts |
| XGBoost | 0.8899 ± 0.0158 | 0.898 | 81.3% | 0.825 | 2.4 pts |
| **Soft Voting (RF + XGB)**, exported | 0.8917 ± 0.0156 | 0.899 | 81.3% | 0.825 | 3.2 pts |
| Stacking (RF + XGB → LogReg) | 0.8917 ± 0.0155 | 0.899 | 81.4% | 0.825 | 3.6 pts |

All four models are a statistical tie: the differences are about one tenth of the CV standard deviation. XGBoost overfit least and was fastest; Random Forest was easiest to tune. The listing **title** is the most informative input, followed by guest count; the rating adds almost nothing. The full interpretation, strengths and limitations are in the notebook (§11).

## Files

| File | What it is |
|---|---|
| `SL_Airbnb_RF_vs_XGBoost.ipynb` | Main notebook: EDA, preprocessing, parameter design + tuning, comparison, interpretation, single-sample inference, pickle export |
| `features.py` | Feature engineering from the title (keywords, bedroom count, Sri Lankan region). Shared by the notebook, the pickle and the app |
| `model.pkl` | Trained soft-voting pipeline (raw fields in → prediction out) |
| `app.py` | Streamlit front end |
| `airbnb_plus_personal_contacts.csv` | Dataset |
| `requirements.txt` | Package versions |

## Run it

**Notebook (Google Colab):** upload `SL_Airbnb_RF_vs_XGBoost.ipynb`, then upload `airbnb_plus_personal_contacts.csv` to the Colab file panel and choose *Runtime → Run all* (about 5 minutes). The notebook writes `features.py` itself and saves a fresh `model.pkl`.

**Front end (on your computer):**
```bash
pip install -r requirements.txt
streamlit run app.py
```
The browser opens at http://localhost:8501. Type a listing title, the number of guests and an optional rating, then press **Predict**.

> **Version note:** a pickle only loads with the same scikit-learn / XGBoost versions it was built with. The included `model.pkl` was built with scikit-learn 1.9.1 and XGBoost 3.4.1. If you retrain in Colab and use *that* `model.pkl`, run `import sklearn, xgboost; print(sklearn.__version__, xgboost.__version__)` in Colab and put those versions in `requirements.txt`.

## GenAI prompt used for the front end
> I have trained a scikit-learn ML model and saved it with pickle as `model.pkl`. The pickle is a dictionary: `model` is a full sklearn Pipeline (it imports a custom transformer from `features.py`, which sits in the same folder), `classes` maps 0 → "Room" and 1 → "Entire place", and `test_metrics` holds the accuracy, F1 and ROC-AUC. The pipeline takes a pandas DataFrame with exactly these raw columns: `name` (text: the listing title), `numberOfGuests` (integer 1–16) and `stars` (float 1–5, or empty/NaN if the listing has no rating yet). Please write a Streamlit app (`app.py`) that loads the pickle once with caching, shows text boxes / number inputs for the three fields (a checkbox for "no rating yet" that sends NaN), a Predict button, and displays the predicted class with its probability as a progress bar. Also show the model name and test metrics in the sidebar, provide 3 example Sri Lankan listings the user can click to auto-fill, and show which keywords and region `features.engineer_features` detected in the title. Keep it in one file and add a `requirements.txt`.

## Activity checklist
- [x] Local Sri Lankan dataset (Kaggle: Destination Sri Lanka)
- [x] Classification problem defined (entire place vs room), with a data-driven reason for choosing this target
- [x] Preprocessing: null detection, duplicate removal, median imputation + missing indicators, scaling, one-hot encoding, feature engineering
- [x] Random Forest and XGBoost with designed (non-default) parameters, tuned by 5-fold CV
- [x] Performance compared; strengths, limitations and interpretation written up
- [x] Single-sample inference on a held-out listing
- [x] Model exported as `.pkl`
- [x] GenAI-generated pickle loader + front-end UI
- [x] Optional: soft voting and stacking added to the comparison
