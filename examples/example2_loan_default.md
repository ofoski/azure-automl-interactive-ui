# Responsible AI Agent - Example 2: Loan Default Risk Prediction

**Business Problem:** Predict which loan applicants will default  
**Dataset Size:** 32,583 loan applications  
**Task Type:** Binary Classification  
**Target:** loan_status (1=Default, 0=Approved)

---

### Q1: What are the top factors driving loan default predictions?

The features that most strongly influence the model's predictions for loan approval or rejection are:

**loan_percent_income:** This feature has the highest impact on predictions. It represents the percentage of a person's income that the loan amount constitutes. A higher or lower percentage likely signals affordability or risk.

**person_home_ownership:** This feature also plays a significant role. It indicates whether the person owns, rents, or has another type of housing arrangement, which may reflect financial stability.

**loan_grade:** The loan grade, which likely represents a creditworthiness rating, is another key driver of predictions.

**loan_intent:** The purpose of the loan (e.g., education, medical, etc.) is equally influential, as certain intents may carry different levels of risk.

**person_income:** The individual's income is a strong predictor, as it directly relates to their ability to repay the loan.

Other features like loan_int_rate (interest rate), person_emp_length (employment length), and loan_amnt (loan amount) have moderate influence. Features such as cb_person_cred_hist_length (credit history length) and cb_person_default_on_file (whether the person has defaulted before) have negligible impact on the model's predictions.

This ranking helps identify which factors the model considers most important when deciding loan outcomes.

---

### Q2: For which types of loan applications does the model make the most mistakes?

The applicant segments with the highest prediction errors and notable patterns are as follows:

**Income Levels:**
- Applicants with lower incomes (5,500 to 39,000) have the highest error rate (8%).
- This suggests the model struggles more with accurately predicting outcomes for lower-income individuals, possibly due to greater variability in their financial situations.

**Loan Intent:**
- Applicants seeking loans for home improvement have the highest error rate (8%), followed by debt consolidation and personal loans (7% each).
- This indicates the model finds it harder to predict outcomes for these loan purposes, potentially due to diverse financial profiles within these groups.

**Loan Grade:**
- Applicants with loan grades C (11%), F (9%), and E (8%) show the highest errors.
- These grades likely represent mid-to-lower creditworthiness, where the model may face challenges distinguishing between approval and rejection.

**Interest Rates:**
- Loans with higher interest rates (13.47 to 23.22) have a higher error rate (9%).
- This could reflect the increased risk and complexity associated with high-interest loans.

**Home Ownership:**
- Applicants with a mortgage have a higher error rate (7%) compared to renters (6%) and owners (1%).
- This may indicate variability in financial stability among those with mortgages.

**Default History:**
- Applicants with a history of default have a higher error rate (8%) compared to those without (6%).
- This suggests the model struggles more with accurately predicting outcomes for individuals with prior defaults.

**Patterns Observed:**
- The model tends to have higher errors for groups associated with higher financial risk or variability, such as lower-income individuals, those with mid-to-low credit grades, and applicants with prior defaults.
- Certain loan intents and high-interest loans also show higher errors, likely due to the diverse financial circumstances within these categories.

These insights can help identify areas where the model could be improved, such as better handling of lower-income applicants or those with complex financial profiles.

---

### Q3: Is the model fair across different age groups and home ownership statuses? Are there any approval rate disparities?

**Fairness Analysis for Age Groups and Home Ownership Status**

**1. Age Groups**

**Accuracy:** The model performs consistently across all age groups, with accuracy ranging from 93% to 94%.

**Selection Rate:** Younger applicants (ages 20–23) have the highest selection rate (18%), while older applicants (ages 30–80) have the lowest (15%).

**True Positive Rate (TPR):** Younger applicants (ages 20–23) also have the highest TPR (77%), while middle-aged groups (ages 26–30) have the lowest (71%).

**Disparity Observed:**
The selection rate and true positive rate decrease slightly as age increases, but the differences are relatively small. This suggests the model treats age groups fairly, with no significant disparities.

**2. Home Ownership Status**

**Accuracy:**
- Renters: 94%
- Owners: 99%
- Mortgage holders: 93%
- Others: 94%

**Selection Rate:**
- Renters have the highest selection rate (26%), while mortgage holders have the lowest (7%).

**True Positive Rate (TPR):**
- Owners have the highest TPR (95%), while mortgage holders have the lowest (50%).

**Disparity Observed:**
There is a significant disparity in both selection rate and true positive rate for mortgage holders compared to other groups. Mortgage holders are selected less often and have a much lower TPR, indicating the model struggles to correctly identify positive cases for this group. Owners, on the other hand, have the highest accuracy and TPR, suggesting they are treated more favorably.

**Approval Rate Disparities:**
- Age Groups: Approval rates (selection rates) are fairly consistent, with only a slight decrease for older age groups.
- Home Ownership: Renters are approved at a much higher rate (26%) compared to mortgage holders (7%), indicating a notable disparity.

**Conclusion:**
- Age Groups: The model is generally fair across age groups, with only minor differences in performance metrics.
- Home Ownership: The model shows a performance gap for mortgage holders, who experience lower selection rates and true positive rates. This may reflect genuine differences in financial risk or could indicate the model needs improvement for this group.

---

### Q4: What changes to income or loan amount would get applicant 30 approved?

For applicant 30 to get approved, considering only changes to income or loan amount, specific counterfactual scenarios would need to be analyzed. However, based on the model's behavior, typical changes might include:

**Increase Income:** Raising the applicant's income to demonstrate better repayment capacity.

**Reduce Loan Amount:** Lowering the requested loan amount to decrease the loan-to-income ratio.

These changes would make the application appear less risky to the model, potentially leading to approval.

---

### Q5: If applicant 50 changed their loan intent or home ownership status, would they get approved?

For applicant 20 to get approved, considering only changes to person_home_ownership, loan_intent, or loan_grade, the following adjustments could lead to approval:

**Change person_home_ownership:**
Switching from "RENT" to "MORTGAGE" would result in approval. This suggests that owning a home (via a mortgage) is seen as a sign of financial stability by the model.

**Change loan_grade:**
Improving the loan grade from "E" to "A" would lead to approval. Loan grade likely reflects creditworthiness, so a better grade increases the likelihood of approval.

**Non-Actionable Observations:**
Some counterfactuals suggest changes to other features (e.g., loan_percent_income or credit history length), but these are outside the scope of the specified features.

**Conclusion:**
To improve the chances of approval for applicant 20, the most actionable changes are:
- Transitioning to a mortgage for home ownership.
- Improving the loan grade to "A."
