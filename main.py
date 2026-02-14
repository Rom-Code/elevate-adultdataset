import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

# 1. Load Data [1]
adult = fetch_ucirepo(id=2) 
X, y = adult.data.features.copy(), adult.data.targets.copy()
df = pd.concat([X, y], axis=1)

# 2. Cleaning [History]
# Strip whitespace and handle the trailing period in 'adult.test' labels
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
df['income'] = df['income'].str.replace('.', '', regex=False)
df_eda = df.replace('?', np.nan) # Mark missing values [2]

# 3. EDA Visualizations [2-4]
# Create the 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Target Distribution (Top-Left)
sns.countplot(ax=axes[0, 0], x='income', data=df_eda, palette='viridis')
axes[0, 0].set_title('Target Distribution (Income Class)')

# 2. Feature Distribution: Age (Top-Right)
sns.histplot(ax=axes[0, 1], x='age', data=df_eda, bins=20, kde=True, color='teal')
axes[0, 1].set_title('Distribution of Age')

# 3. Feature vs Target: Education-Num (Bottom-Left)
sns.boxplot(ax=axes[1, 0], x='income', y='education-num', data=df_eda, palette='magma')
axes[1, 0].set_title('Education Level vs Income Class')

# 4. Numeric Correlations (Bottom-Right)
sns.heatmap(
    ax=axes[1, 1],
    data=df_eda.select_dtypes(include=[np.number]).corr(),
    annot=True,
    cmap='coolwarm',
    fmt=".2f"
)
axes[1, 1].set_title('Numeric Feature Correlations')

plt.tight_layout()
plt.show()


# 4. Preprocessing & Evaluation [History]
# Remove noise (fnlwgt) and redundant categorical education [2]
X_final = df_eda.drop(columns=['fnlwgt', 'education', 'income']).fillna(X.mode().iloc)
y_final = df_eda['income'].map({'>50K': 1, '<=50K': 0})

X_encoded = pd.get_dummies(X_final, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_final, test_size=0.2, random_state=42)

# Gradient Boosting captured the best F1-score in comparison tests [History]
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Use threshold 0.2 to improve recall for the imbalanced '>50K' class [History]
y_probs = model.predict_proba(X_test)[:, 1]
y_pred_adj = (y_probs >= 0.2).astype(int)

print("\n--- Evaluation Results (Threshold 0.2) ---")
print(classification_report(y_test, y_pred_adj, target_names=['<=50K', '>50K']))