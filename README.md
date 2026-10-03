# Can Bitcoin Be Predicted in the Short Term?

**Prophet vs ARIMA vs LightGBM, each evaluated against a naive baseline**

Many "predict Bitcoin with ML" projects report impressive charts that turn out to be a model copying yesterday's
price. This project asks a stricter question: **does any model beat the simplest possible forecast,
"tomorrow's price = today's price"?** Every model is tested out-of-sample with walk-forward evaluation.

## Results

| Model | Task | Result | Baseline |
|---|---|---|---|
| Prophet (tuned, log price) | Short-term level | Fitted price **~9% below** actual in the last days | Naive: 0% bias |
| ARIMA(2,1,2) | 3-day-ahead price, walk-forward (180 days) | **2.59%** MAPE | Naive: **2.60%** |
| LightGBM (18 features) | Next-day direction, walk-forward (2 years) | **48.9%** accuracy, AUC **0.493** | Majority class: **50.1%**, coin flip: 0.500 |
| LightGBM long/flat strategy | 2-year return (0.1% fee) | **+20.1%** | Buy & hold: **+39.5%** |

**Conclusion:** no model beats the naive baseline in a meaningful way. ARIMA converges to "tomorrow = today",
and LightGBM's direction predictions are no better than a coin flip. With public price data, short-term BTC
movements behave like a **random walk**, which is consistent with the efficient market hypothesis.

## What I learned along the way

- **Prophet:** the default model predicted *negative* prices and missed every major cycle. A log transform,
  `changepoint_range=0.95` and halving dates as holidays fixed the long-run fit. Short-term behaviour was still
  poor:
  - a rigid trend lagged the price by about 9%;
  - a flexible trend (`changepoint_prior_scale=0.5`) fit the recent data almost perfectly but extrapolated a
    **+114% move in 60 days**, which is a textbook case of overfitting.
- **Hyperparameter search:** a grid search over 8 combinations × 31 CV folds showed that parameter choice barely
  mattered: the top 3 configurations were within 0.05 percentage points of each other.
- **ARIMA:** learned on its own that the best forecast is roughly the last price. Its prediction interval widens
  honestly with the horizon.
- **LightGBM:** features covered momentum, volatility, RSI, moving-average distance, volume, ETH and S&P 500.
  None of them carried a usable signal. The feature-importance chart only shows which variables the model used
  to fit noise.

## Method

- **Data:** daily BTC-USD, ETH-USD and S&P 500 from Yahoo Finance (`yfinance`), 2015-01-01 to 2026-10-01.
- **No look-ahead:** every feature uses only information available at the end of that day.
- **Walk-forward testing:**
  - ARIMA is updated one day at a time.
  - LightGBM is retrained every 30 days on past data only.
  - Prophet is evaluated with `cross_validation` (4-year initial window, 60-day horizon).

## Run it

```bash
pip install -r requirements.txt
jupyter notebook btc_forecasting.ipynb
```

The Prophet cross-validation step takes a few minutes.

## Next steps

- Forecast **volatility** instead of direction (GARCH, or LightGBM regression on |returns|). Volatility clusters
  and is known to be far more predictable than direction.
- Add non-price data (on-chain metrics, funding rates, sentiment) and test whether it adds any signal.

---
*Educational project, not financial advice.*
