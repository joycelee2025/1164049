# %% [markdown]
# COMP647 Assignment 2
# Student Name: Joyce Lee
# Student ID: 1164049

# %%
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

pd.set_option('display.max_columns', None)  # Show all columns in DataFrame
pd.set_option('display.max_rows', None)

# %% [markdown]
# Load Data from a CSV

# %%
df = pd.read_csv('data/telco_customer_churn.csv')

# %% [markdown]
# Display the first 5 rows of the DataFrame to understand its structure.

# %%
df.head()

# %% [markdown]
# Display the number of rows and columns in the DataFrame.

# %%
df.shape

# %% [markdown]
# Display information about the DataFrame, including data types and non-null counts. 
# 
# Noted "Total Charges" is classified as object datatype, need to investigate further.

# %%
df.info()

# %% [markdown]
# Summary statistics for numerical columns in the DataFrame.
# 

# %%
df.describe().transpose()

# %% [markdown]
# # 1. Handle duplicates

# %%
df.columns

# %%
#check for duplicates in each column and print the count of duplicates
for column in df.columns:
    duplicate_count = df[column].duplicated().sum()
    print(f"Column '{column}' has {duplicate_count} duplicates.")

# %%
#check for duplicates by multiple columns
duplicate_rows = df[df.duplicated(subset=['Monthly Charges', 'Tenure Months', 'City', 'Gender'], keep=False)]

duplicate_rows.shape

# %%
duplicate_rows.sort_values('Monthly Charges').head()

# %% [markdown]
# The above two records seem to be different/independent, therefore there are no duplicates found in this dataset.

# %% [markdown]
# # 2. Handle irrelevant data

# %%
#find columns where all values are the same-constant features
constant_columns = [col for col in df.columns if df[col].nunique() == 1]
print("Constant columns:", constant_columns)

# %%
#remove constant columns
df_no_constant_cols = df.drop(columns=constant_columns)

# %%
#columns with mostly missing values
threshold = 0  
print(f"total records: {df.shape[0]}")
for column in df.columns:
    missing_count = df[column].isnull().sum()

    #if data is object type, also check for empty strings
    if df[column].dtype == 'object':
        pseudo_missing = df[column].apply(lambda x: isinstance(x, str) and x.strip() == '').sum()
        missing_count += pseudo_missing

    missing_ratio = (missing_count / df.shape[0]) * 100
    if missing_ratio > threshold:
        print(f"Column '{column}' has {missing_count} missing values ({missing_ratio:.2f}%)")

# %% [markdown]
# The column "churn reason" has not been removed because those 5174 missing values represent customers who did not churn.
# 
# The column "Total charges" will be handled in the next section.

# %% [markdown]
# # 3. Handle missing values

# %%
#display dataframe with missing values
df_missing = df[df.isnull().any(axis=1)]
df_missing.shape

# %%
df_missing.tail()

# %% [markdown]
# "Total charges" is classified as object data type which seems not quite right. After checking, there were a few rows containing empty string, need to transform them into float like "monthly charges".

# %%
#transform "Total charges" to numerical
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')

# %%
#identify numerical columns and categorical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
print("Numerical columns:", numerical_cols)
print("Categorical columns:", categorical_cols)

# %%
#get the list of columns with missing values for numerical columns
missing_numerical_cols = df[numerical_cols].isnull().any()
missing_numerical_cols = missing_numerical_cols[missing_numerical_cols].index
print("Numerical columns with missing values:", missing_numerical_cols.tolist())

# %% [markdown]
# After checking the dataset, the missing value rows of "total charges" are all "Tenure Months"=0, so it should be logical to fill in 0.

# %%
#for Total Charges, fill missing values with 0 as tenure months is 0
missing_before = df['Total Charges'].isnull().sum()
df.loc[df['Tenure Months'] == 0 & df['Total Charges'].isnull(), 'Total Charges'] = 0
missing_after = df['Total Charges'].isnull().sum()

print(f"Missing values in 'Total Charges' before: {missing_before}, after: {missing_after}")

# %%
#get the list of columns with missing values for categorical columns
missing_categorical_cols = df[categorical_cols].isnull().any()
print(missing_categorical_cols)
missing_categorical_cols = missing_categorical_cols[missing_categorical_cols].index
print("Categorical columns with missing values:", missing_categorical_cols.tolist())

# %% [markdown]
# not dropping the rows with missing values in the 'Churn Reason' because those rows actually represent the customers who did not churn

# %%
#fill missing values in the 'Churn Reason' column with 'nochurn'
df_filled = df.copy()
df_filled['Churn Reason'] = df_filled['Churn Reason'].fillna('nochurn')

# %%
#check if there are still any missing values in the 'Churn Reason' column
missing_churn_reason = df_filled['Churn Reason'].isnull().any()
print("Are there any missing values in 'Churn Reason' after filling?:", missing_churn_reason)

# %%
#the number of rows of column 'Churn Reason' with value 'nochurn' should equal the number of rows with missing values in 'Churn Reason'
nochurn_count = df_filled['Churn Reason'].value_counts().get('nochurn', 0)
missing_churn_count = df['Churn Reason'].isnull().sum()
print(f"Number of 'nochurn' in 'Churn Reason': {nochurn_count}, Number of missing values in 'Churn Reason': {missing_churn_count}")

# %% [markdown]
# # 4. Handle outliers

# %%
#find outliers using IQR method, remove points outside Q1 - 1.5 * IQR and Q3 + 1.5 * IQR
def find_outliers_IQR_method(input_df, variable):
    Q1 = input_df[variable].quantile(0.25)
    Q3 = input_df[variable].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    return lower_limit, upper_limit

# %%
#find lower and upper limits for targets including "tenure months", "monthly charges", "total charges""
features = ['Tenure Months', 'Monthly Charges', 'Total Charges']
for feature in features:
    lower, upper = find_outliers_IQR_method(df, feature)
    print(f"Lower limit for {feature}: {lower}, Upper limit for {feature}: {upper}")

# %%
#remove outliers using the IQR method
df_cleaned = df[(df[feature] > lower) & (df[feature] < upper)]

print(f"Cleaned dataset : {df_cleaned.shape}")
print(f"outliers count : {len(df) - len(df_cleaned)}")

# %%
#find outliers using Z-score method
def find_outliers_z_score(input_df, variable):
    df_z_scores = input_df.copy()

    z_scores = np.abs(stats.zscore(df_z_scores[variable]))

    df_z_scores[variable + '_Zscore'] = z_scores
    return df_z_scores


# %%
#find outliers using Z-score method
df_z_scores = find_outliers_z_score(df.copy(), feature)
df_z_scores.head()

# %%
#remove outliers using z-score method, data points with Z-score greater than 3 are considered outliers
df_z_scores_cleaned = df_z_scores[df_z_scores[feature + '_Zscore'] < 3]

print(f"Cleaned dataset using Z-score method: {df_z_scores_cleaned.shape}")
print(f"outliers count using Z-score method: {len(df_z_scores) - len(df_z_scores_cleaned)}")

# %% [markdown]
# # 5. EDA - Exploratory Data Analysis

# %% [markdown]
# ## 5.1 Correlation Bar Plot

# %%
#bar plot of correlation between 'Churn Value' and numerical columns
plt.figure(figsize=(10, 6))
df[numerical_cols].corr()['Churn Value'].drop('Churn Value').sort_values(ascending=False).plot(kind='bar')

# %% [markdown]
# ## 5.2 Pair Plot

# %%
#pair plot of numerical columns
pair_plot = sns.pairplot(df[['Churn Value', 'Total Charges', 'Monthly Charges', 'Tenure Months']])
pair_plot.fig.suptitle('Pair Plot of Selected Numerical Features', y=1.02)
plt.show()

# %% [markdown]
# ## 5.3 Line Plot

# %%
#line plot of 'Churn Value' vs 'Tenure Months'
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Tenure Months', y='Churn Value', marker='o')
plt.title('Churn Value vs Tenure Months')
plt.xlabel('Tenure Months')
plt.ylabel('Churn Value')
plt.show()

# %% [markdown]
# Insights:
# - There is a clear negative relationship: as Tenure Months increases, the Churn Value decreases.
# - This indicates that newer customers are more likely to churn, while long-term customers tend to remain loyal.
# - The decrease in Churn Value looks steeper for smaller Tenure Months (up to around 20 months), but gets flatter for larger Tenure Months.
# - The plot suggests that Tenure Months is a strong predictor of Churn Value, but other factors may also have contribution.

# %% [markdown]
# ## 5.4 Boxplot

# %%
# use boxplot to visualise the relationship between 'Churn Label' and 'Monthly Charges'
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Churn Label', y='Monthly Charges')
plt.title('Monthly Charges by Churn Label')
plt.xlabel('Churn Label')
plt.ylabel('Monthly Charges')
plt.show()

# %% [markdown]
# Insights:
# - Customers who churned have a higher median monthly charge aournd 80, compared to customers who stayed with a monthly charge around 65.
# - Both groups share a similar maximum and minimum, but the lower quartile for churned customers is much higher at around 60 and 20 for non-churned customers.
# - Churned customers are more concentrated toward the upper range around 60–100, while non-churned customers show a wider spread, with many paying at the low end around 20–40 as well as higher amounts.
# - Thess suggest that churn is more common among higher-paying customers.

# %% [markdown]
# ## 5.5 Barplots comparing categorical variables

# %%
#to visualise the relationship between 'Churn Label' and other categorical columns including 'Gender', 'Senior Citizen', 'Partner', 'Phone Service', 'Internet Service', 'Contract'
plt.figure(figsize=(15, 10))
cat_cols = ['Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Phone Service', 'Internet Service', 'Online Security', 'Online Backup', 'Device Protection', 'Tech Support', 'Streaming Movies', 'Contract']
for i, col in enumerate(cat_cols):
    plt.subplot(4, 3, i + 1)
    sns.countplot(data=df, x=col, hue='Churn Label')
    plt.title(f'Churn Label vs {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.legend(title='Churn Label', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# Insights:
# 1. Gender
# - Both male and female customers show a similar pattern: No-Churn is higher than Churn.
# - There is no obvious difference between genders, suggesting that gender may not be a strong predictor of churn.
# 2. Senior Citizen
# - Among non-senior customers, No-Churn is almost three times higher than Churn.
# - Among senior customers, the gap between No-Churn and Churn is much smaller.
# - This indicates that senior citizens are more likely to churn compared to younger customers.
# 3. Partner
# - Customers with no partner have a No-Churn to Churn ratio of about 2:1.
# - Customers with a partner show much stronger retention, with a ratio closer to 4:1.
# - This suggests that having a partner may contribute to customer stability/loyalty.
# 4. Dependents
# - Customers without dependents have a No-Churn to Churn ratio of about 2:1.
# - Customers with dependents are far more likely to stay, with ratios greater than 10:1.
# - This implies that customers with families or dependents are far less likely to churn.
# 5. Phone Service
# - Both groups show higher No-Churn, having a No-Churn to Churn ratio of about 3:1.
# - Phone service availability alone may not strongly differentiate customer churn.
# 6. Internet Service
# - For DSL customers, No-Churn is almost four times higher than Churn.
# - For fibre-optic customers, No-Churn is about 30% higher than Churn.
# - For customer without internet service, No-Churn is about much higher than Churn.
# - This suggests fiber-optic customers are much more likely to churn compared to DSL users.
# 7. Online Security
# - The customres without online security have a much more higher churn than no-churn.
# - This suggests providing online security feature may lower churn rates.
# 8. Online Backup
# - The customres without online backup have a much more higher churn than no-churn.
# - This suggests providing online backup feature may lower churn rates.
# 9. Device Protection
# - The customres without device protection have a much more higher churn than no-churn.
# - This suggests providing device protection feature may lower churn rates.
# 10. Tech Support
# - For customers without tech support, No-Churn is about 30% higher than Churn.
# - For customers with tech support, No-Churn is almost four times higher than Churn.
# - This suggests providing tech support is strongly linked to lower churn rates.
# 11. Streaming Movies
# - Customers with or without streaming movies both show a ratio of about 2:1.
# - This suggests that streaming movies availability does not significantly affect churn.
# 12. Contract
# - For customers with month-to-month contract, No-Churn is about 30% higher than Churn.
# - For customer with one-year or two-year contract, No-Churn is significantly higher than Churn.
# - This suggests that longer contracts strongly reduce churn probability.

# %% [markdown]
# ## 5.6 Pie charts comparing categorical variables

# %%
#pie charts
cat_cols = ['Senior Citizen', 'Partner', 'Dependents', 'Internet Service', 'Online Security', 'Online Backup', 'Device Protection', 'Tech Support', 'Contract']

for col in cat_cols:
    # create canvas: 1 row, 4 columns (overall, first category breakdown, second category breakdown)
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    fig.suptitle(f'{col} Distribution and Churn Breakdown', fontsize=14, fontweight='bold')

    # overall pie chart
    overall_counts = df[col].value_counts()
    axes[0].pie(overall_counts, labels=overall_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0].set_title(f'{col} Overall')

    # first category churn breakdown
    first_cat = overall_counts.index[0]
    first_counts = df[df[col] == first_cat]['Churn Label'].value_counts()
    axes[1].pie(first_counts, labels=first_counts.index, autopct='%1.1f%%', startangle=90)
    axes[1].set_title(f'{col} - {first_cat}')

    # second category churn breakdown
    second_cat = overall_counts.index[1]
    second_counts = df[df[col] == second_cat]['Churn Label'].value_counts()
    axes[2].pie(second_counts, labels=second_counts.index, autopct='%1.1f%%', startangle=90)
    axes[2].set_title(f'{col} - {second_cat}')

    # third category churn breakdown if exists
    if len(overall_counts) > 2:
        third_cat = overall_counts.index[2]
        third_counts = df[df[col] == third_cat]['Churn Label'].value_counts()
        axes[3].pie(third_counts, labels=third_counts.index, autopct='%1.1f%%', startangle=90)
        axes[3].set_title(f'{col} - {third_cat}')
    else:
        axes[3].axis('off')

    plt.tight_layout()
    plt.show()

# %% [markdown]
# Further to the barplots comparing Customer Churn with some categorical variables, the above 9 pie charts show more details in the categorical variables which are likely to be strongly linked to our target variable - Customer Churn.
# 
# 1. Senior Citizen: senior customers have a churn rate of 42%, while younger customers have a churn rate of 24%.
# 2. Partner: customers without partner have a churn rate of 33%, while customers with partner have a churn rate of 20%.
# 3. Dependents: customers without partner have a churn rate of 33%, while customers with partner have a churn rate of 7%.
# 4. Internet service: fibre-optic users have a churn rate of 42%, while DSL users have a churn rate of 19%, no-internet users have a a churn rate of 7%.
# 5. Online security: customers without Online security have a churn rate of 42%, while customers with Online security have a churn rate of 15%, no-internet users have a a churn rate of 7%.
# 6. Online backup: customers without Online backup have a churn rate of 40%, while customers with Online backup have a churn rate of 22%, no-internet users have a a churn rate of 7%.
# 7. Device protection: customers without Device protection have a churn rate of 39%, while customers with Device protection have a churn rate of 23%, no-internet users have a a churn rate of 7%.
# 8. Tech support: customers without tech support have a churn rate of 42%, while customers with tech support have a churn rate of 15%, no-internet users have a a churn rate of 7%.
# 9. Contract: month-to-month custmers have a churn rate of 43%, while customers with one-year contract have a churn rate of 11% and customers with two-year contract have a rate of 3%.

# %% [markdown]
# # 6. Potential quesitons to explore with the dataset?
# 1. Tenure and Churn
# - Question: Does shorter customer tenure increase the likelihood of churn?
# - Hypothesis: Customers with lower tenure months are significantly more likely to churn compared to long-term customers.
# 
# 2. Monthly Charges and Churn
# - Question: Do higher monthly charges increase the likelihood of churn?
# - Hypothesis: Customers with higher monthly charges are more likely to churn.
# 
# 3. Senior Citizens
# - Question: Are senior citizens more prone to churn compared to non-senior customers?
# - Hypothesis: Senior citizens have a higher churn rate than younger customers.
# 
# 4. Family Status (Partner and Dependents)
# - Question: Do customers with families (partners and/or dependents) churn less?
# - Hypothesis: Customers with partners or dependents are significantly less likely to churn.
# 
# 5. Internet Service Type
# - Question: Does the type of internet service (DSL vs. Fiber optic) affect churn behavior?
# - Hypothesis: Fiber-optic customers have higher churn rates than DSL customers.
# 
# 6. Extra internet services like Online Security, Online Backup, Device Protection, Tech Support
# - Question: Does bundling extra services with Internet make customers more loyal?
# - Hypothesis: Customers with extra internet services are much less likely to churn than those without.
# 
# 7. Contract Type
# - Question: Does contract length influence churn probability?
# - Hypothesis: Customers on month-to-month contracts are significantly more likely to churn than those on one- or two-year contracts.
# 
# 8. Non-Predictive Factors
# - Question: Do gender, phone service, or streaming TV significantly affect churn?
# - Hypothesis: Gender, phone service, and streaming TV availability do not have a significant effect on churn rates.


