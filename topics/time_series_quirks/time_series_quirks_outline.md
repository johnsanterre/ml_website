# Time Series Quirks - 15 Minute Lecture Outline

## 1. Introduction (2 minutes)
- What makes time series different from cross-sectional data?
- Common pitfalls that break time series models
- The importance of temporal ordering
- Learning objectives

## 2. Data Leakage and Temporal Ordering (3 minutes)
- Future information contamination
- Proper train-test splitting with time
- Normalization on historical data only
- Rolling window statistics and lag features
- The look-ahead bias trap

## 3. Cross-Validation for Time Series (3 minutes)
- Why standard K-fold fails for time series
- Time series split and expanding window CV
- Purged and embargoed cross-validation
- Walk-forward analysis
- Selecting validation strategy by use case

## 4. Cold Start and Bootstrapping Problems (2 minutes)
- Insufficient historical data for new series
- Transfer learning across time series
- Hierarchical and grouped time series approaches
- Synthetic data generation techniques
- Warm-up periods and initialization strategies

## 5. Handling Non-Stationarity (2 minutes)
- Detecting and testing for stationarity
- Differencing and transformation techniques
- Seasonal decomposition and detrending
- Cointegration for multiple series
- When to use stationary vs non-stationary models

## 6. Dealing with Irregularities (2 minutes)
- Missing data patterns and imputation
- Irregular sampling intervals
- Outlier detection and treatment
- Structural breaks and regime changes
- Holiday and calendar effects

## 7. Performance Evaluation Considerations (1 minute)
- Appropriate metrics for time series
- Out-of-sample vs out-of-time testing
- Backtesting and walk-forward validation
- Statistical significance in temporal context
- Business-relevant evaluation periods

## Key Takeaways
- Time series requires special handling to respect temporal structure
- Data leakage is the most common and dangerous pitfall
- Cross-validation must account for temporal dependencies
- Non-stationarity and irregularities need specific treatments
- Evaluation methods must reflect real-world deployment scenarios
