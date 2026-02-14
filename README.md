Project Overview
The primary objective of this project is a multivariate classification task to predict income levels based on 14 demographic and professional features. The dataset contains 48,842 instances and features both categorical and integer data types.
Dataset Features
The model utilizes the following attributes extracted from the 1994 Census:
• Continuous Features: age, fnlwgt, education-num, capital-gain, capital-loss, and hours-per-week.
• Categorical Features: workclass, education, marital-status, occupation, relationship, race, sex, and native-country.
• Target Variable: income (labeled as >50K or <=50K).

--------------------------------------------------------------------------------
Included Components
The main.py script provides a full pipeline from data ingestion to final evaluation:
1. Automated Data Loading: Uses the ucimlrepo library to fetch the dataset directly using its unique ID (2).
2. Exploratory Data Analysis (EDA): Generates a 2x2 grid of visualizations covering target class distribution, age histograms, education-level boxplots, and numeric feature correlations.
3. Data Cleaning: Implements logic to strip leading whitespace from categorical strings and replaces '?' placeholders with the mode of the respective columns [2, History].
4. Feature Engineering: Removes redundant columns like education (captured by education-num) and noise features like fnlwgt to improve model generalization [2, History].
5. Optimized Modeling: Utilizes a Gradient Boosting Classifier, which historically provides the best F1-score for this specific census task.
6. Class Imbalance Handling: Includes a custom classification threshold set at 0.2 to significantly improve the Recall of the high-income minority class.

--------------------------------------------------------------------------------
Installation
To run the code, you must install the following dependencies:

pip install ucimlrepo pandas numpy scikit-learn seaborn matplotlib

Note: The ucimlrepo package is required to pull the official data directly from the UCI repository.
How to Run
1. Clone the Repository: Ensure main.py is in your working directory.
2. Execute the Script: Run the following command in your terminal:
3. Review Output: The script will display four EDA plots and then print a detailed Classification Report for the validation set.

--------------------------------------------------------------------------------
Key Logic Decisions
• Handling Missing Values: Because workclass and occupation contain missing data, the script uses mode imputation to maintain the dataset's integrity [2, History].
• Label Normalization: The code automatically handles trailing periods (e.g., >50K.) found in the raw adult.test file to ensure consistent binary mapping.
• Evaluation Metrics: The project prioritizes the F1-score and Recall over simple accuracy.

--- Evaluation Results (Threshold 0.2) ---
              precision    recall  f1-score   support

       <=50K       0.96      0.78      0.86      7414
        >50K       0.56      0.89      0.69      2355

    accuracy                           0.80      9769
   macro avg       0.76      0.83      0.77      9769
weighted avg       0.86      0.80      0.82      9769


