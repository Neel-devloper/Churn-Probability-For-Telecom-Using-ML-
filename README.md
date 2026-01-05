# Telecom Customer Churn Prediction

## Overview

This is a realistic churn modeling project that demonstrates practical machine learning work on a common business problem. The project hits fundamental data limitations and explains why further improvements require better data sources, not just model tweaks. It prioritizes recall over accuracy, acknowledging the business cost of missing churners.

## Problem Statement

The telecom industry faces significant challenges with customer churn, where customers switch to competitors or discontinue services. This project aims to predict which customers are likely to churn based on their usage patterns, demographics, and service-related data. Early identification of at-risk customers allows telecom companies to implement retention strategies, reducing revenue loss.

## Dataset

The analysis uses the Telco Customer Churn dataset, which contains information about:
- Customer demographics (gender, age, dependents, etc.)
- Service subscriptions (phone, internet, streaming services)
- Account information (tenure, monthly charges, total charges)
- Churn status (Yes/No)

## Metrics Used

### Primary Metrics
- **Recall (Churners Caught Accuracy)**: The percentage of actual churners correctly identified by the model. This is crucial for churn prediction as missing a potential churner is more costly than false alarms.
- **False Positive Rate (Non-churners Flagged)**: The percentage of non-churners incorrectly flagged as churners.

### Why Accuracy is Misleading

Traditional accuracy can be misleading in imbalanced datasets like churn prediction, where non-churners vastly outnumber churners (typically 3:1 ratio). A model that simply predicts "no churn" for everyone would achieve ~73% accuracy but catch 0% of actual churners. This is why we focus on recall to ensure we don't miss potential churners, even if it means accepting some false positives.

## Model Architecture

The model uses a deep neural network with:
- Input layer: 46 features (after preprocessing)
- Hidden layers: 1024 → 512 → 256 neurons
- Dropout (0.3) for regularization
- Batch normalization
- Output: Single logit for binary classification

## Training Details

- **Loss Function**: BCEWithLogitsLoss with positive class weighting to handle imbalance
- **Optimizer**: AdamW with learning rate 0.001
- **Batch Size**: 64
- **Max Epochs**: 100 (with early stopping)
- **Probability Calibration**: Logistic regression applied to raw probabilities for better threshold-based decisions

## Model Iterations and Threshold Tuning

This project demonstrates iterative model development rather than a single "optimal" solution. We explored different thresholds to balance the trade-off between catching churners (recall) and avoiding false alarms (false positive rate).

### Threshold Experiments

| Threshold | Recall (Churners Caught) | False Positive Rate | Overall Accuracy | Notes |
|-----------|--------------------------|---------------------|------------------|-------|
| 0.7 | ~50% | ~15% | ~75% | Conservative approach, misses many churners |
| 0.5 | ~60% | ~20% | ~73% | Moderate balance |
| 0.3 | ~67% | ~22% | ~72% | Better recall |
| 0.2 | ~72% | ~28% | ~70% | Higher recall |
| 0.1 | ~76% | ~33% | ~68% | Good recall |
| 0.05 | **80.43%** | **35.91%** | **68.42%** | Final choice prioritizing recall |

### Metric Trade-offs

- **Higher thresholds** (0.7, 0.5): Lower false positives but poor recall - misses too many churners
- **Lower thresholds** (0.3, 0.2, 0.1, 0.05): Higher recall but more false positives - flags more customers as at-risk
- **Business Decision**: We chose 0.05 threshold because catching churners is more valuable than avoiding some false alarms

### Calibration Attempt

Raw neural network probabilities can be poorly calibrated for threshold-based decisions. We attempted calibration using logistic regression:

1. Train the neural network on raw probabilities
2. Fit a logistic regression model to map raw probabilities to calibrated probabilities
3. Use calibrated probabilities for final predictions

This helped improve the reliability of threshold choices but didn't dramatically change the fundamental trade-offs.

## Performance Analysis

### Model Plateauing

The model reached a performance plateau around epoch 40-50, where training loss stabilized and further epochs showed minimal improvement. This indicates the model has learned the available patterns in the data, and additional training time doesn't yield significant gains. The plateau suggests either:

1. The model architecture has reached its capacity for this dataset
2. The data contains inherent noise or limitations
3. Hyperparameters are near optimal for the given constraints

### Data Limitations

Several factors limit the model's performance:

1. **Feature Quality**: The dataset provides behavioral indicators but lacks real-time usage patterns, recent interactions, or external factors (competition, economic conditions)

2. **Class Imbalance**: With only ~27% churners in the dataset, the model has fewer examples to learn from for the positive class

3. **Temporal Aspects**: The data is static and doesn't capture how customer behavior changes over time leading to churn

4. **Feature Engineering**: While preprocessing includes one-hot encoding and scaling, more sophisticated feature engineering (interaction terms, temporal features) could potentially improve performance

5. **Sample Size**: With ~7,000 customers, the dataset is moderately sized but may not capture all possible churn scenarios

## Why This Model Cannot Improve Further Without Better Data

This project demonstrates the fundamental limits of machine learning when working with realistic, imperfect data. Despite trying different architectures, hyperparameters, and calibration techniques, the model hits a ceiling that cannot be broken without fundamentally better data sources.

### Core Limitations

1. **Static vs. Dynamic Data**: The dataset captures a single snapshot of customer state, but churn is a process that unfolds over time. Without sequential data showing how behavior changes before churn, the model can only make educated guesses based on current state.

2. **Missing Causal Factors**: The dataset lacks critical predictors like:
   - Recent customer service interactions
   - Competitor promotions or pricing changes
   - Economic factors affecting the customer
   - Real-time usage patterns and anomalies

3. **Label Quality**: Churn labels are binary and retrospective - we don't know why customers actually churned or what interventions might have prevented it.

4. **Feature Resolution**: Monthly charges and tenure provide coarse signals, but churn decisions often hinge on recent, specific events that aren't captured.

### What Additional Data Would Help

- **Time-series data**: Customer behavior over the last 6-12 months
- **Interaction logs**: Customer service calls, app usage, billing disputes
- **External factors**: Local competition, economic indicators, regional events
- **Intervention data**: What retention efforts were attempted and their success rates

### Realistic Assessment

This model achieves reasonable performance (80% recall) given the data constraints, but represents a practical baseline rather than an optimal solution. Further "improvements" through model complexity or hyperparameter tuning would likely yield diminishing returns. True advancement requires better data collection and feature engineering at the business level.

The project serves as a case study in the importance of data quality over model sophistication in real-world machine learning applications.

## Results

Current model achieves:
- **Recall**: 80.43% (catches 4 out of 5 churners)
- **False Positive Rate**: 35.91%
- **Overall Accuracy**: 68.42%

These results represent a practical baseline for churn prediction with the available data. The model successfully identifies most at-risk customers while accepting a moderate false positive rate. This is not "state-of-the-art" performance but demonstrates what can be achieved with standard ML techniques on typical business data.

## Usage

1. Run the notebook cells in order
2. The model trains on 80% of the data
3. Evaluation on the held-out 20% test set
4. Adjust the threshold in the test cell to balance recall vs. false positives based on business needs

## Future Improvements

- Implement time-series features for better churn prediction
- Use ensemble methods or more advanced architectures
- Incorporate external data sources (market conditions, competitor analysis)
- Develop cost-sensitive learning approaches
- Implement A/B testing for retention strategy effectiveness
