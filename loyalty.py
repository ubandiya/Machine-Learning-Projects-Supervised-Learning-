import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

pd.set_option('display.max_columns', None)

df = pd.read_csv('loyalty.csv')

df['first_month'] = pd.to_numeric(df['first_month'], errors='coerce').ffill()
df['joining_month'] = df['joining_month'].astype('category').cat.reorder_categories(
    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], ordered=True
).ffill()

month_map = {month: idx for idx, month in enumerate(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
                                                      'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 1)}
df['joining_month'] = df['joining_month'].map(month_map).astype('int')

year_map = {'0-1': 1, '1-3': 2, '3-5': 3, '5-10': 4, '10+': 5}
df['loyalty_years'] = df['loyalty_years'].map(year_map)

df['promotion'] = df['promotion'].str.title()
promotion_map = {'Yes': 0, 'No': 1}
df['promotion'] = df['promotion'].map(promotion_map)

region_map = {'Asia/Pacific': 1, 'Middle East/Africa': 3, 'Europe': 2, 'Americas': 0}
df['region'] = df['region'].map(region_map)

df['lifetime_spend'] = df['spend'] / (df['first_month'] + 1e-10)

df.to_csv('loyalty_cleaned.csv', index=False)

"""fig, axes = plt.subplots(2, 2, figsize=(15, 10))

sns.boxplot(data=df, x='spend', ax=axes[0, 0], showfliers=False)
sns.stripplot(data=df, x='spend', ax=axes[0, 0], color="red", alpha=0.5)
axes[0, 0].set_title('Spend Distribution with Outliers')

sns.histplot(data=df, x='lifetime_spend', bins=20, kde=True, alpha=0.5, ax=axes[0, 1])
mean_val = df['lifetime_spend'].mean()
median_val = df['lifetime_spend'].median()
axes[0, 1].axvline(mean_val, color='blue', linestyle='--', label=f'Mean: {mean_val:.1f}')
axes[0, 1].axvline(median_val, color='orange', linestyle='--', label=f'Median: {median_val:.1f}')
axes[0, 1].legend()
axes[0, 1].set_title('Lifetime Spend Distribution')

sns.barplot(data=df, x='loyalty_years', y='spend', hue='promotion', ax=axes[1, 0])
axes[1, 0].set_title('Spend vs Loyalty Years by Promotion')

sns.barplot(data=df, x='joining_month', y='first_month', ax=axes[1, 1])
axes[1, 1].set_title('First Month Spend vs Joining Month')

fig.suptitle('Descriptive Analysis of Customer Spending')
plt.tight_layout()
plt.show()

sns.boxplot(data=df, x='region', y='spend')
plt.title('Spending Distribution by Region')
plt.show()

corr = df[['spend', 'first_month', 'lifetime_spend']].corr()

plt.figure(figsize=(12, 6))
sns.heatmap(corr, cmap='coolwarm', fmt='.2f', annot=True)
plt.title('Correlation Heatmap')
plt.show()

def find_outliers(data):
    q1 = np.quantile(data, 0.25)
    q3 = np.quantile(data, 0.75)

    IQR = q3 - q1

    lower = q1 - 1.5 * IQR
    upper = q3 + 1.5 * IQR

    outliers = data[(data < lower) | (data > upper)]

    return outliers
"""

X = df.drop(columns=['spend', 'customer_id'], axis=1)
y = df['spend']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler().set_output(transform='pandas')

X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

rf = RandomForestRegressor(n_estimators=500, n_jobs=-1)
dt = DecisionTreeRegressor()

rf.fit(X_train, y_train)
dt.fit(X_train, y_train)

r_pred = rf.predict(X_test)
d_pred = dt.predict(X_test)

def analyse(model, true, prediction):
    print(model.__class__.__name__)
    print(f'MSE: {mean_squared_error(true, prediction)}')
    print(f'R2: {r2_score(true, prediction)}\n')

param_grid = {
    'n_estimators': [x * 100 for x in range(1, 6)],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

"""rf_grid_search = GridSearchCV(RandomForestRegressor(n_jobs=-1), param_grid=param_grid, scoring='neg_mean_squared_error')
rf_grid_search.fit(X_train, y_train)

print(f'Best parameters found: {rf_grid_search.best_params_}')
print(f'Best MSE: {rf_grid_search.best_score_}')"""

feature_importances = rf.feature_importances_
feature_names = X.columns

imporance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

imporance_df = imporance_df.sort_values(by='Importance', ascending=False)
sns.barplot(data=imporance_df, x='Importance', y='Feature')
plt.title('Feature Importances')
plt.show()

rf_residuals = y_test - r_pred
dt_residuals = y_test - d_pred

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(rf_residuals, kde=True)
plt.title('Random Forest Residuals')

plt.subplot(1, 2, 2)
sns.histplot(dt_residuals, kde=True)
plt.title('Decision Tree Residuals')

plt.tight_layout()
plt.show()