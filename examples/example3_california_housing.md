# Responsible AI Agent - Example 3: California Housing Price Prediction

**Business Problem:** Predict median house values in California  
**Dataset Size:** 20,640 census districts  
**Task Type:** Regression  
**Target:** median_house_value (in dollars)

---

### Q1: Which property and location features have the highest importance scores in predicting house values?

The most important features for predicting house values in this dataset are:

**Longitude (0.60 importance score):** This is the most influential feature, indicating that the geographic location (east-west position) strongly impacts house values. This could reflect regional differences in property demand or desirability.

**Median Income (0.59 importance score):** This is nearly as important as longitude, showing that the income level of residents in an area is a key driver of house prices. Higher-income areas likely correlate with higher property values.

**Latitude (0.49 importance score):** The north-south position also plays a significant role, suggesting that house values vary significantly across different latitudinal regions.

**Ocean Proximity (0.14 importance score):** While less influential than the above, proximity to the ocean still has a noticeable impact, likely reflecting the premium associated with coastal properties.

These scores highlight that both location (longitude, latitude, and ocean proximity) and socioeconomic factors (median income) are critical in determining house values. Other features like population, total bedrooms, and housing median age have much smaller importance scores, meaning they contribute less to the model's predictions.

---

### Q2: Where does the model make the largest prediction errors?

The model makes the largest prediction errors in the following areas:

**1. Location Features**

**Longitude:** The largest errors occur in the range (-124.191, -121.69], with a mean error of 34,731.34. This suggests that properties in this westernmost region are harder for the model to predict accurately.

**Latitude:** The range (32.539, 33.93] has the highest error at 33,873.00, indicating challenges in predicting house values in the southernmost areas.

**2. Property Features**

**Total Rooms:** Properties with very few rooms (6 to 1,420) have the largest errors, with a mean error of 36,364.18. This could reflect variability in smaller properties that the model struggles to capture.

**Median Income:** The highest errors occur in areas with the highest income levels (4.646 to 15.0), with a mean error of 39,336.29. This suggests that high-income areas may have unique factors influencing house prices that the model does not fully capture.

**Ocean Proximity:** Properties on ISLANDS have the largest errors, with a mean error of 84,689.38. This is likely due to the unique and highly variable nature of island properties.

**3. Population and Households**

**Population:** Areas with the smallest populations (8 to 781) have the largest errors, with a mean error of 40,718.52. This could reflect variability in rural or sparsely populated areas.

**Households:** Areas with the fewest households (2 to 278) have the largest errors, with a mean error of 37,525.28, again suggesting challenges in rural or less dense areas.

**Key Insights:**
- The model struggles most in extreme or unique cases, such as very small or very large properties, high-income areas, and unique locations like islands.
- Errors are also higher in less populated or less dense areas, likely due to greater variability in house prices in these regions.

These findings suggest that the model could benefit from additional features or adjustments to better capture the unique characteristics of these challenging segments.

---

### Q3: Does the model perform differently across income levels and coastal proximity?

**Model Performance Across Income Levels:**

The model's performance varies significantly across different income levels:

**High-Income Areas (Median Income: 4.646 to 15.0):** The model has the largest errors here, with a mean absolute error (MAE) of 39,336.29. This suggests that predicting house values in affluent areas is particularly challenging, likely due to unique factors influencing high-value properties.

**Low-Income Areas (Median Income: 0.499 to 2.548):** The model performs best in these areas, with a lower MAE of 24,924.12. This indicates that house prices in lower-income regions are more predictable.

**Model Performance Across Coastal Proximity:**

The model also shows notable differences in performance based on proximity to the coast:

**Island Properties:** The model struggles the most here, with an extremely high MAE of 84,689.38. This is likely due to the unique and highly variable nature of island properties.

**Near Bay (MAE: 40,072.30) and Near Ocean (MAE: 38,825.77):** The model also has higher errors for properties close to the bay or ocean, reflecting the complexity of pricing coastal properties.

**Inland Properties:** The model performs best for inland properties, with a much lower MAE of 24,420.60. This suggests that house prices in inland areas are more consistent and easier to predict.

**Key Insights:**
- The model struggles most with high-income areas and unique coastal properties, such as those on islands or near the ocean. These segments likely have more variability or unique factors influencing house prices.
- Inland and lower-income areas are more predictable, with smaller errors.

These findings suggest that the model could benefit from additional features or adjustments to better capture the unique characteristics of high-income and coastal properties.

---

### Q4: What changes to a property would increase its predicted value?

The counterfactual analysis suggests changes that could increase or decrease the predicted value of a property. However, in this case, the counterfactuals provided do not show an increase in the predicted value compared to the original prediction of 143,903. Instead, they show scenarios where the predicted value decreases. Here are the key changes and their effects:

**Latitude Increase:**
Moving the property further north (latitude 39.42, 40.32, 41.12, or 41.27) results in lower predicted values. This suggests that properties in the northern regions of the dataset may have lower house values compared to the original location.

**Ocean Proximity:**
Changing the property to be closer to the ocean ("<1H OCEAN") decreases the predicted value to 76,153. This is counterintuitive, as coastal properties are often more valuable, but it may reflect specific patterns in this dataset.

**Total Rooms and Bedrooms:**
Increasing the total number of rooms (e.g., to 24,193 or 10,070) or bedrooms (e.g., to 2,669.6 or 3,142.5) does not lead to higher predicted values. This suggests that simply adding more rooms or bedrooms may not always increase property value, possibly due to diminishing returns or other factors like location and income levels.

**Key Insight:**
The counterfactuals provided do not suggest actionable changes that would increase the predicted value of this specific property. Instead, they highlight scenarios where the value decreases. This could indicate that the property is already optimized for its location and features, or that the model's predictions are influenced by complex interactions between features.
