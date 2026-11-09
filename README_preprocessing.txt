CBECS preprocessing

Open notebooks/01_data_preprocessing_ysmagulov.ipynb and run cells in order.
Install requirements.txt in your environment first. All 21 cells are
numbered; 9 code cells have executed outputs.

23 original inputs are retained. Missing flags: 23. Additional threshold flags:
SQFT > 1m, NWKER >= 5000, ceiling height > 50. Eight combinations are added.
Source values are retained separately from clipped/imputed model values.

data/processed/X_<split>_tree.csv: engineered inputs, unscaled.
data/processed/X_<split>_scaled.csv: engineered inputs, continuous values scaled.
data/processed/X_<split>_base.csv and _base_scaled.csv: no engineered combinations.
Both variants contain all 23 original inputs and all missing/threshold flags.
Base model columns: 175. Engineered model columns: 183.
y_<split>.csv and metadata_<split>.csv match feature rows exactly.
In cbecs2018_processed.csv, exclude PUBID, split, and MFBTU from X.
MFBTU remains in thousands of Btu (kBtu). No test scores are computed.

documentation/feature_availability.csv: intended analyst, sources, and limitations.
documentation/feature_engineering_validation.csv: fixed 15% validation comparison.
documentation/data_dictionary.csv: all retained source definitions.
figures/eda/eda_overview.png: concise training-only EDA.

Splits: 4,449 training, 954 validation, 954 test; seed 42. No cross-validation
or screening-based feature deletion. Rerunning replaces generated files.
New inputs must use the EIA codebook categories and the same cap/flag rules.
Use reference-year weather, not unobserved future weather.
Missing indicators represent raw blanks, not EIA's prior imputations.
