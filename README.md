# Crop Advisory API

Yield, cost and risk estimates for Indian district-crop combinations.

Model: XGBoost, R2 0.740, MAE 0.424 t/ha on a held-out 2010+ split.
Coverage: 234 districts, 1602 district-crop-season combinations, 1997-2015.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | status and model metrics |
| GET | `/options/states` | list of states |
| GET | `/options/districts?state=` | districts in a state |
| GET | `/options/crops?state=&district=` | crops available in a district |
| GET | `/history?state=&district=&crop=&season=` | recorded yields plus next-season estimate |
| POST | `/predict` | yield, economics, risk, advisory |
| POST | `/compare` | every crop in a district, ranked |

Interactive docs at `/docs`.

## Data

Six public sources: district crop production (MoAFW), farm harvest prices, rainfall
and temperature (ICRISAT District Level Database), cost of production C2 (Rajya Sabha
Session 253 Q2740), and minimum support prices (DES). No synthetic data.

## Caveats

Records end in 2015, so this is a demonstration on historical data. Costs are C2, the
full economic cost including imputed land rent, so most district-crop combinations
show a negative margin. That matches the cost-of-cultivation literature.
