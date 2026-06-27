# Healthcare Cost Prediction

## Version

2.0

## Algorithm

Stacking Ensemble (XGBoost + LightGBM)

## Author

MIRZA SHARIF BAIG

## Generated

2026-06-26T20:05:47.548036

## Dataset

- Name: Synthetic Patient Financial Risk Dataset
- Records: 10000
- Description: Synthetic healthcare and financial dataset.

## Performance

| Metric | Value |
|--------|------:|
| MAE | 1316.53 |
| RMSE | 1791.19 |
| R² | 0.9617 |

## Features

- Age
- BMI
- Hospital Visits
- Medication Count
- Annual Claim Amount
- Health Risk Score
- Claims-to-Premium Ratio

## Intended Use

Estimate future healthcare costs to support operational planning, financial forecasting, and risk stratification. This model is intended to assist decision-making and should not be used as the sole basis for clinical diagnosis or treatment.

## Limitations

- Trained on synthetic data.
- Predictions may not generalize to real-world populations without retraining.
- Not intended for clinical diagnosis.


## Ethical Considerations

- Do not use predictions as the sole basis for healthcare decisions.
- Review outputs with qualified healthcare professionals.
- Monitor model performance and fairness after deployment.
