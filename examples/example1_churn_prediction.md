# Responsible AI Agent - Example 1: Bank Customer Churn Prediction

**Business Problem:** Predict which customers will leave the bank  
**Dataset Size:** 10,000 customers  
**Task Type:** Binary Classification  
**Target:** Exited (1=Churned, 0=Retained)

---

### Q1: Which customer characteristics are the strongest predictors of churn, and which features have little influence?

The strongest predictors of customer churn in this model are Age and Number of Products, as they have the highest influence on the model's predictions. This means that changes in these features significantly impact whether the model predicts a customer will churn or not.

Features with little to no influence include Gender, Tenure, Has Credit Card, and Credit Score, as their importance scores are near zero or negative. This indicates that these characteristics do not meaningfully contribute to the model's predictions and could potentially be excluded without affecting performance.

Other features like Is Active Member, Geography, and Balance have moderate to low influence, meaning they play a smaller but still noticeable role in the predictions.

---

### Q2: Which customer segments does the model struggle to predict accurately, and how large are those error gaps?

The model struggles the most with the following customer segments, based on their higher error rates:

**Age:**
- Customers aged 44 to 85 have the highest error rate at 21%, followed by those aged 37 to 44 at 17%.
- Younger customers aged 18 to 32 have the lowest error rate at 6%.

**Geography:**
- Customers from Germany have an error rate of 17%, while those from France and Spain have lower error rates at 12%.

**Gender:**
- The model is less accurate for Female customers, with an error rate of 16%, compared to 11% for Male customers.

**Balance:**
- Customers with a balance in the range of 127,884 to 250,898 have an error rate of 17%, compared to 11% for those with lower balances.

**Number of Products:**
- Customers with 1 product have the highest error rate at 19%, while those with 2 products have a much lower error rate at 7%. Customers with 4 products have no errors, but this may be due to a small sample size.

**Is Active Member:**
- Customers who are not active members have an error rate of 16%, compared to 11% for active members.

**Error Gaps:**
- The largest error gap is in the Age feature, where the error rate ranges from 6% (youngest group) to 21% (oldest group), a gap of 15 percentage points.
- For Geography, the gap is 5 percentage points between Germany (17%) and France/Spain (12%).
- For Gender, the gap is 5 percentage points between Female (16%) and Male (11%).

These findings suggest that the model struggles more with older customers, customers from Germany, and female customers. These gaps may reflect genuine differences in the data or areas where the model could be improved.

---

### Q3: Is the model fair across Geography and Gender? Are there performance gaps that could indicate bias?

**Model Performance Across Geography Groups:**

**Accuracy:**
- France: 88%
- Spain: 88%
- Germany: 83% (lowest accuracy)
- Gap: 5 percentage points (Germany performs worst).

**True Positive Rate (TPR):**
- Germany: 56% (highest TPR, meaning the model identifies more actual churn cases here).
- France: 34%
- Spain: 37%
- Gap: 22 percentage points (Germany performs best in identifying true churn cases).

**Selection Rate:**
- Germany: 21% (the model predicts churn more often for German customers).
- France: 7%
- Spain: 7%

**Model Performance Across Gender Groups:**

**Accuracy:**
- Male: 89% (higher accuracy).
- Female: 84% (lower accuracy).
- Gap: 5 percentage points (Female customers perform worse).

**True Positive Rate (TPR):**
- Female: 45% (higher TPR, meaning the model identifies more actual churn cases for females).
- Male: 40%
- Gap: 5 percentage points (Female customers perform better in identifying true churn cases).

**Selection Rate:**
- Female: 13% (the model predicts churn more often for females).
- Male: 8%

**Summary of Differences:**
- The model shows a performance gap of 5 percentage points in accuracy for both Geography and Gender groups, with Germany and Female customers performing worse.
- However, Germany and Female customers have higher true positive rates, meaning the model is better at identifying actual churn cases in these groups, even though overall accuracy is lower.
- The selection rate is higher for Germany and Female customers, indicating the model predicts churn more frequently for these groups.

These gaps may reflect genuine differences in the data or areas where the model could be improved for specific groups.

---

### Q4: What would need to change for customer 25 to churn, considering only changes to Balance, NumOfProducts, and IsActiveMember?

For customer 25 to churn, considering only changes to Balance, NumOfProducts, and IsActiveMember, the following changes would need to occur:

**Balance:** Increase the balance to 222,467.60 (from the current 152,303.80) while also increasing the number of products to 3.

**NumOfProducts:** Increase the number of products to 3 (from the current 1) while keeping the balance high.

**Excluded Changes:**
Changes to other features like Age were suggested in some counterfactuals but are not actionable in the short term, so they were excluded from this analysis.

These changes indicate that a higher balance combined with more products could lead the model to predict churn for this customer.

---

### Q5: Compare all trained models for this churn task. Which algorithm was selected and why?

For this churn prediction task, several models were trained, and the LightGBM algorithm with the MaxAbsScaler preprocessor was selected as the best-performing model. Here's a comparison of all the trained models:

**Model Comparison:**

**LightGBM (MaxAbsScaler):**
- Accuracy: 86%
- Fit Time: 0.04 seconds
- Selected as Best: Yes

**LightGBM (MaxAbsScaler) (another run):**
- Accuracy: 86%
- Fit Time: 0.09 seconds
- Selected as Best: No

**XGBoostClassifier (MaxAbsScaler):**
- Accuracy: 85%
- Fit Time: 0.57 seconds
- Selected as Best: No

**XGBoostClassifier (SparseNormalizer):**
- Accuracy: 85%
- Fit Time: 0.09 seconds
- Selected as Best: No

**ExtremeRandomTrees (MaxAbsScaler):**
- Accuracy: 68%
- Fit Time: 0.33 seconds
- Selected as Best: No

**Why LightGBM Was Selected:**
- Highest Accuracy: LightGBM achieved the highest accuracy (86%), which is the primary metric for this classification task.
- Fast Training Time: It had the shortest fit time (0.04 seconds), making it efficient to train.
- Consistency: Both runs of LightGBM achieved the same accuracy, showing stable performance.

In summary, LightGBM was selected because it provided the best balance of high accuracy and fast training time, outperforming other algorithms like XGBoost and ExtremeRandomTrees.
