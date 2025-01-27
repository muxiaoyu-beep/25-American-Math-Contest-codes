# Import necessary libraries
import pandas as pd
import numpy as np
from scipy.stats import zscore
import seaborn as sns
sns.set_theme(style="whitegrid")  # Set seaborn theme
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from xgboost import XGBRegressor, XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
import shap
import os

# Define output directory
output_dir = r"C:/Users/admin/Desktop/result"
os.makedirs(output_dir, exist_ok=True)

# Load datasets
hosts_data = pd.read_csv('C:/Users/admin/Desktop/summerOly_hosts.csv')
athletes_data = pd.read_excel('C:/Users/admin/Desktop/summerOly_athletes.xlsx')
medal_counts = pd.read_csv('C:/Users/admin/Desktop/summerOly_medal_counts.csv')

# Preprocess NOC (National Olympic Committee) codes
def preprocess_noc(df, is_host_data=False):
    # Special handling for hosts_data
    if is_host_data:
        # Map country names to NOC codes
        country_to_noc = {
            'France': 'FRA',
            'United States': 'USA',
            'Japan': 'JPN',
            # Add other country mappings...
        }
        df['NOC'] = df['Host'].map(country_to_noc)
    else:
        # Merge German data
        df['NOC'] = df['NOC'].replace({'GDR': 'GER', 'FRG': 'GER'})

    # Remove dissolved countries
    dissolved_nocs = ['URS', 'EUN', 'TCH', 'YUG', 'SCG']
    df = df[~df['NOC'].isin(dissolved_nocs)]

    return df

# Apply preprocessing to each dataset
medal_counts = preprocess_noc(medal_counts)
athletes_data = preprocess_noc(athletes_data)
hosts_data = preprocess_noc(hosts_data, is_host_data=True)

# Get the host country for the 2028 Olympics
host_2028 = hosts_data[hosts_data['Year'] == 2028]['Host'].values[0]

# Aggregate medals by year and country
yearly_medals = medal_counts.groupby(['Year', 'NOC'])['Total'].sum().reset_index()

# Check yearly_medals data
print("Yearly Medals by Country:\n", yearly_medals.head())

# Feature engineering: Quantify athlete performance
def calculate_performance_score(row):
    medal_score = 0
    if pd.notna(row['Medal']):
        if row['Medal'] == 'Gold':
            medal_score = 10
        elif row['Medal'] == 'Silver':
            medal_score = 7
        elif row['Medal'] == 'Bronze':
            medal_score = 5
    return medal_score + (row['Year'] - 1896) / (2028 - 1896) * 3  # Higher weight for recent years

athletes_data['Performance_Score'] = athletes_data.apply(calculate_performance_score, axis=1)

# Calculate the number of years athletes participated
athletes_data['Years_Participated'] = athletes_data.groupby('Name')['Year'].transform('nunique')

# Aggregate athlete performance by country and year
country_performance = athletes_data.groupby(['NOC', 'Year']).agg(
    Performance_Score=('Performance_Score', 'sum'),
    Athlete_Count=('Name', 'nunique'),  # Number of athletes
    Sport_Diversity=('Sport', 'nunique'),  # Diversity of sports
    Avg_Years_Participated=('Years_Participated', 'mean')  # Average participation years
).reset_index()

# Merge with yearly medal data
country_performance = country_performance.merge(yearly_medals, on=['NOC', 'Year'], how='left').fillna(0)

# Validate merged data for Germany
print("Example of merged data for Germany:")
print(country_performance[country_performance['NOC'] == 'GER'].head())

# Check for dissolved countries
print("\nNumber of dissolved countries present:",
      country_performance[country_performance['NOC'].isin(['URS', 'EUN'])].shape[0])

# Add host country feature
country_performance['Is_Host_2028'] = (country_performance['NOC'] == host_2028).astype(int)

# Feature engineering: Interaction terms
country_performance['Athlete_Sport_Interaction'] = country_performance['Athlete_Count'] * country_performance['Sport_Diversity']
country_performance['Athlete_Medal_Interaction'] = country_performance['Athlete_Count'] * country_performance['Total']

# Check for outliers in total medals
plt.figure(figsize=(10, 6))
sns.boxplot(x=country_performance['Total'])
plt.title('Boxplot of Total Medals by Year and Country')
plt.show()

# Detect outliers using Z-score
z_scores = zscore(country_performance['Total'])
outliers = country_performance[np.abs(z_scores) > 3]
print("Outliers detected by Z-score:\n", outliers)

# Handle outliers (replace values > 140 with median)
median_value = country_performance['Total'].median()
country_performance['Total'] = np.where(country_performance['Total'] > 140, median_value, country_performance['Total'])

# Check data after outlier handling
plt.figure(figsize=(10, 6))
sns.boxplot(x=country_performance['Total'])
plt.title('Boxplot of Total Medals (After Outlier Handling)')
plt.show()

# Define features and target variable
X = country_performance[['Performance_Score', 'Athlete_Count', 'Sport_Diversity', 'Is_Host_2028', 'Athlete_Sport_Interaction', 'Athlete_Medal_Interaction', 'Avg_Years_Participated']]
y = country_performance['Total']  # Target variable: Total medals by year and country

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Evaluate regression model using 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
xgb_reg = XGBRegressor(
    random_state=42,
    reg_alpha=0.5,
    reg_lambda=0.5,
    max_depth=5,
    learning_rate=0.1,
)

# Calculate MSE
cv_scores_mse = cross_val_score(xgb_reg, X_scaled, y, cv=kf, scoring='neg_mean_squared_error')
print("XGBoost Regression - Cross-Validation MSE:", -cv_scores_mse.mean())

# Calculate MAE
cv_scores_mae = cross_val_score(xgb_reg, X_scaled, y, cv=kf, scoring='neg_mean_absolute_error')
print("XGBoost Regression - Cross-Validation MAE:", -cv_scores_mae.mean())

# Calculate R²
cv_scores_r2 = cross_val_score(xgb_reg, X_scaled, y, cv=kf, scoring='r2')
print("XGBoost Regression - Cross-Validation R²:", cv_scores_r2.mean())

# Regularization: Add regularization parameters to XGBoost classifier
xgb_clf = XGBClassifier(random_state=42, reg_alpha=0.1, reg_lambda=0.1)
xgb_clf_params = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
xgb_clf_grid = GridSearchCV(xgb_clf, xgb_clf_params, cv=5, scoring='accuracy')
xgb_clf_grid.fit(X_train, (y_train > 0).astype(int))

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, (y_train > 0).astype(int))

# Probability calibration using Platt Scaling
calibrated_clf = CalibratedClassifierCV(xgb_clf_grid.best_estimator_, method='isotonic', cv=5)
calibrated_clf.fit(X_train_smote, y_train_smote)

# Quantile regression: Add more quantiles
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
quantile_predictions = {}
for q in quantiles:
    gbr = GradientBoostingRegressor(loss='quantile', alpha=q, random_state=42)
    y_pred_quantile = cross_val_predict(gbr, X_scaled, y, cv=5)
    quantile_predictions[f'Quantile_{q}'] = y_pred_quantile

# Visualize quantile regression results
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y)), y, color='blue', label='Actual Medals')
for q in quantiles:
    plt.plot(range(len(y)), quantile_predictions[f'Quantile_{q}'], label=f'Quantile {q}')
plt.title('Quantile Regression Predictions')
plt.xlabel('Country Index')
plt.ylabel('Medal Count')
plt.legend()
plt.show()

# Train the XGBoost regression model
xgb_reg.fit(X_train, y_train)

# SHAP analysis
explainer = shap.TreeExplainer(xgb_reg)
shap_values = explainer.shap_values(X_scaled)
shap.summary_plot(shap_values, X_scaled, feature_names=X.columns)

# ROC curve
y_pred_proba = calibrated_clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve((y_test > 0).astype(int), y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Set global plot style
sns.set_palette("deep")  # Use deep color palette
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300, 'savefig.dpi': 300})

# Sensitivity analysis
def plot_sensitivity_analysis():
    # Create composite figure
    fig = plt.figure(figsize=(18, 6))

    # Subplot 1: Learning rate sensitivity
    ax1 = plt.subplot(1, 3, 1)
    learning_rates = [0.01, 0.05, 0.1, 0.2]
    cv_results = []
    for lr in learning_rates:
        model = XGBRegressor(learning_rate=lr, random_state=42)
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        cv_results.append(scores.mean())

    # Plot with error bars
    ax1.plot(learning_rates, cv_results, 'o-', color='#1f77b4', linewidth=2, markersize=8)
    ax1.set_title('(a) Learning Rate Sensitivity', fontweight='bold', pad=15)
    ax1.set_xlabel('Learning Rate', labelpad=10)
    ax1.set_ylabel('Cross-Validated R²', labelpad=10)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Feature perturbation analysis
    ax2 = plt.subplot(1, 3, 2)
    perturbations = np.linspace(-0.2, 0.2, 5)
    medal_changes = []
    for p in perturbations:
        modified_X = X.copy()
        modified_X['Athlete_Count'] *= (1 + p)
        modified_X_scaled = scaler.transform(modified_X)
        pred = xgb_reg.predict(modified_X_scaled)
        medal_changes.append(pred.mean() - xgb_reg.predict(X_scaled).mean())

    # Add regression line
    coeffs = np.polyfit(perturbations * 100, medal_changes, 1)
    trend_line = np.poly1d(coeffs)
    ax2.plot(perturbations * 100, trend_line(perturbations * 100), '--', color='gray', alpha=0.7)

    ax2.plot(perturbations * 100, medal_changes, 'o-', color='#2ca02c', linewidth=2, markersize=8)
    ax2.set_title('(b) Athlete Count Sensitivity', fontweight='bold', pad=15)
    ax2.set_xlabel('Perturbation Percentage (%)', labelpad=10)
    ax2.set_ylabel('Δ Predicted Medals', labelpad=10)
    ax2.axhline(0, color='grey', linestyle='--')
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Regularization parameter analysis
    ax3 = plt.subplot(1, 3, 3)
    alphas = np.logspace(-3, 1, 5)
    results = []
    for alpha in alphas:
        model = XGBRegressor(reg_alpha=alpha, reg_lambda=alpha, random_state=42)
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        results.append(scores.mean())

    ax3.semilogx(alphas, results, 'o-', color='#d62728', linewidth=2, markersize=8)
    ax3.set_title('(c) Regularization Sensitivity', fontweight='bold', pad=15)
    ax3.set_xlabel('Regularization Strength (log scale)', labelpad=10)
    ax3.set_ylabel('Cross-Validated R²', labelpad=10)
    ax3.grid(True, alpha=0.3)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Integrated_Sensitivity_Analysis.png"),
                bbox_inches='tight', pad_inches=0.1)
    plt.show()

# Execute sensitivity analysis
plot_sensitivity_analysis()

# Predict medal counts for 2028
country_performance_2028 = country_performance.copy()
country_performance_2028['Year'] = 2028
country_performance_2028['Is_Host_2028'] = (country_performance_2028['NOC'] == host_2028).astype(int)

# Extract features and standardize
X_2028 = country_performance_2028[['Performance_Score', 'Athlete_Count', 'Sport_Diversity', 'Is_Host_2028',
                                    'Athlete_Sport_Interaction', 'Athlete_Medal_Interaction', 'Avg_Years_Participated']]
X_2028_scaled = scaler.transform(X_2028)

# Predict 2028 medal counts
country_performance_2028['Predicted_Total'] = xgb_reg.predict(X_2028_scaled)

# Sort and extract top 10 countries
top_2028_countries = country_performance_2028[['NOC', 'Predicted_Total']].sort_values(by='Predicted_Total', ascending=False).head(10)
top_2028_countries.rename(columns={'NOC': 'Country', 'Predicted_Total': 'Predicted Medals'}, inplace=True)

# Identify countries that have never won a medal
countries_without_medals = country_performance.groupby('NOC')['Total'].sum()
countries_without_medals = countries_without_medals[countries_without_medals == 0].index

# Predict medal probabilities for these countries in 2028
X_first_medals = country_performance_2028[country_performance_2028['NOC'].isin(countries_without_medals)].copy()
X_first_medals_scaled = scaler.transform(X_first_medals[['Performance_Score', 'Athlete_Count', 'Sport_Diversity',
                                                         'Is_Host_2028', 'Athlete_Sport_Interaction',
                                                         'Athlete_Medal_Interaction', 'Avg_Years_Participated']])
X_first_medals['Medal_Probability'] = calibrated_clf.predict_proba(X_first_medals_scaled)[:, 1]

# Sort countries with higher probabilities
potential_first_medal_countries = X_first_medals[['NOC', 'Medal_Probability']].sort_values(by='Medal_Probability', ascending=False)
potential_first_medal_countries.rename(columns={'NOC': 'Country', 'Medal_Probability': 'Probability of Winning First Medal'}, inplace=True)

# Save 2028 medal predictions
top_2028_countries_path = os.path.join(output_dir, "2028_Medal_Predictions.csv")
top_2028_countries.to_csv(top_2028_countries_path, index=False)

# Save potential first medal winners
potential_first_medal_countries_path = os.path.join(output_dir, "Potential_First_Medal_Winners.csv")
potential_first_medal_countries.to_csv(potential_first_medal_countries_path, index=False)

# Sport analysis: Extract sport information
olympic_sports = athletes_data.groupby(['Year', 'Sport']).size().reset_index(name='Count')
sports_by_year = olympic_sports.groupby('Year')['Sport'].nunique().reset_index(name='Sport_Count')

# Calculate total medals per year
yearly_total_medals = medal_counts.groupby('Year')['Total'].sum().reset_index(name='Total_Medals')

# Merge sport count and total medals data
sports_medal_analysis = sports_by_year.merge(yearly_total_medals, on='Year', how='left')

# Create scaled column
sports_medal_analysis['Total_scaled'] = sports_medal_analysis['Total_Medals'] / 100

# Plot Olympic sports evolution vs medal distribution
plt.figure(figsize=(12, 6))
sns.lineplot(data=sports_medal_analysis, x='Year', y='Sport_Count', label='Sports Count', color='blue')
sns.lineplot(data=sports_medal_analysis, x='Year', y='Total_scaled', label='Total Medals (scaled)', color='red')
plt.title('Olympic Sports Evolution vs Medal Distribution (1896-2028)')
plt.ylabel('Value')
plt.legend()
plt.savefig(os.path.join(output_dir, "Sports_vs_Medals.png"))
plt.show()

# Analyze sport type changes by decade
sports_medal_analysis['Decade'] = (sports_medal_analysis['Year'] // 10) * 10
decadal_sports = sports_medal_analysis.groupby('Decade').agg(
    Sport_Count=('Sport_Count', 'mean'),
    Medal_Total=('Total_Medals', 'sum')
).reset_index()

# Create dual-axis visualization
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.bar(decadal_sports['Decade'], decadal_sports['Sport_Count'], width=8, alpha=0.6, label='Sports Count')
ax1.set_xlabel('Decade')
ax1.set_ylabel('Average Sports Count', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(decadal_sports['Decade'], decadal_sports['Medal_Total'], 'r-', marker='o', label='Total Medals')
ax2.set_ylabel('Total Medals Awarded', color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title('Decadal Evolution of Olympic Sports and Medals')
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
plt.savefig(os.path.join(output_dir, "Decadal_Sports_Medals.png"))
plt.show()

# Analyze country-sport-medal relationships
merged_medal_sports = athletes_data.merge(
    medal_counts[['Year', 'NOC', 'Total']],
    on=['Year', 'NOC'],
    how='left'
).fillna(0)

# Calculate medal contribution by country and sport
country_sport_impact = merged_medal_sports.groupby(['NOC', 'Sport']).agg(
    Medal_Contribution=('Total', 'sum'),
    Participation_Years=('Year', 'nunique')
).reset_index()

# Calculate sport importance index (medal contribution × participation years)
country_sport_impact['Sport_Importance'] = (
        country_sport_impact['Medal_Contribution'] *
        country_sport_impact['Participation_Years']
)

# Example: Analyze top sports for the USA
top_us_sports = country_sport_impact[country_sport_impact['NOC'] == 'USA'].sort_values(
    by='Sport_Importance', ascending=False
).head(5)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_us_sports, x='Sport', y='Sport_Importance',
           hue='Sport', palette='viridis', legend=False)
plt.title('Top 5 Strategic Sports for United States')
plt.ylabel('Sport Importance Index')
plt.xticks(rotation=45)
plt.savefig(os.path.join(output_dir, "USA_Top_Sports.png"))
plt.show()

# Analyze host country impact
def analyze_host_impact(host_noc, host_year):
    # Get host country data
    host_data = merged_medal_sports[
        (merged_medal_sports['NOC'] == host_noc) &
        (merged_medal_sports['Year'].between(host_year - 8, host_year + 8))  # Include surrounding Olympics
        ]

    # Calculate sport participation changes
    sport_participation = host_data.pivot_table(
        index='Sport',
        columns=pd.cut(host_data['Year'], bins=[host_year - 12, host_year - 4, host_year + 4, host_year + 12]),
        values='Total',
        aggfunc='count',
        fill_value=0,
        observed=False  # Explicitly set observed parameter
    )
    # Visualize heatmap (without annotations)
    plt.figure(figsize=(12, 8))
    sns.heatmap(sport_participation.T, annot=False, cmap='YlGnBu')  # Turn off annotations
    plt.title(f'{host_noc} Sport Participation Around Host Year {host_year}')
    plt.ylabel('Time Period')
    plt.xlabel('Sport Category')
    plt.savefig(os.path.join(output_dir, f"Host_Impact_{host_noc}.png"))
    plt.show()

# Analyze 2012 host country (example using UK data)
analyze_host_impact(host_noc='GBR', host_year=2012)

# Create combined plots
def create_combined_plots():
    # Create 1x3 subplot canvas
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))  # Adjust canvas size for horizontal layout
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # Subplot 1: Olympic Sports Evolution vs Medal Distribution
    ax1 = axs[0]  # Use single index
    sns.lineplot(data=sports_medal_analysis, x='Year', y='Sport_Count',
                 label='Sports Count', color='blue', ax=ax1)
    ax1.set_title('(a) Olympic Sports vs Medals Trend', fontweight='bold', pad=12)
    ax1.set_ylabel('Sports Count', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax1b = ax1.twinx()
    sns.lineplot(data=sports_medal_analysis, x='Year', y='Total_Medals',
                 label='Total Medals', color='red', ax=ax1b)
    ax1b.set_ylabel('Total Medals', color='red')
    ax1b.tick_params(axis='y', labelcolor='red')
    ax1b.grid(False)

    # Subplot 2: Top 5 Strategic Sports for United States
    ax3 = axs[1]  # Second subplot
    sns.barplot(data=top_us_sports, x='Sport', y='Sport_Importance',
                palette='viridis', ax=ax3)
    ax3.set_title('(b) USA Top Strategic Sports', fontweight='bold', pad=12)
    ax3.set_ylabel('Sport Importance Index')
    ax3.set_xlabel('Sport Category')
    ax3.tick_params(axis='x', rotation=45)

    # Subplot 3: GBR Sport Participation Around 2012
    ax4 = axs[2]  # Third subplot
    host_data = merged_medal_sports[
        (merged_medal_sports['NOC'] == 'GBR') &
        (merged_medal_sports['Year'].between(2004, 2020))
        ]
    sport_participation = host_data.pivot_table(
        index='Sport',
        columns=pd.cut(host_data['Year'], bins=[2000, 2008, 2016, 2024]),
        values='Total',
        aggfunc='count',
        fill_value=0
    )
    sns.heatmap(sport_participation.T, annot=False, cmap='YlGnBu', ax=ax4)
    ax4.set_title('(c) GBR Sport Participation Around 2012', fontweight='bold', pad=12)
    ax4.set_xlabel('Sport Category')
    ax4.set_ylabel('Time Period')

    # Save combined plot
    plt.savefig(os.path.join(output_dir, "Combined_Analysis.png"),
                bbox_inches='tight', dpi=300)
    plt.show()

# Execute plotting
create_combined_plots()