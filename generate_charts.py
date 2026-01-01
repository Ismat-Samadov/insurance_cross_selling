"""
Insurance Cross-Selling Business Analytics Chart Generator

This script generates all business-focused visualizations for the insurance
cross-selling analysis. Charts are saved to the charts/ directory.

Usage:
    python generate_charts.py
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve
import xgboost as xgb
import lightgbm as lgb

# Configuration
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

CHARTS_DIR = 'charts'
os.makedirs(CHARTS_DIR, exist_ok=True)


def load_data():
    """Load data using Kaggle API (keeps data in memory)"""
    import zipfile
    import kaggle

    competition_name = 'playground-series-s4e7'

    print("Downloading data from Kaggle API...")
    kaggle.api.competition_download_files(competition_name, path='/tmp')

    with zipfile.ZipFile(f'/tmp/{competition_name}.zip', 'r') as zip_ref:
        with zip_ref.open('train.csv') as f:
            train_df = pd.read_csv(f)
        with zip_ref.open('test.csv') as f:
            test_df = pd.read_csv(f)

    os.remove(f'/tmp/{competition_name}.zip')

    print(f"✓ Training samples: {len(train_df):,}")
    print(f"✓ Test samples: {len(test_df):,}")

    return train_df, test_df


def generate_overall_response_rate(train_df):
    """Chart 1: Overall Response Rate"""
    print("\n[1/16] Generating overall response rate chart...")

    fig, ax = plt.subplots(figsize=(10, 6))
    response_pct = train_df['Response'].value_counts(normalize=True) * 100
    bars = ax.bar(['Not Interested', 'Interested'], response_pct.values,
                  color=['#d62728', '#2ca02c'], alpha=0.7, edgecolor='black')

    ax.set_ylabel('Percentage of Customers (%)', fontsize=12, fontweight='bold')
    ax.set_title('Customer Interest in Vehicle Insurance Cross-Sell',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 100)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{CHARTS_DIR}/01_overall_response_rate.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_age_group_analysis(train_df):
    """Chart 2: Response Rate by Age Group"""
    print("[2/16] Generating age group analysis...")

    train_df['Age_Group'] = pd.cut(train_df['Age'],
                                   bins=[0, 25, 35, 45, 55, 100],
                                   labels=['18-25', '26-35', '36-45', '46-55', '55+'])

    age_response = train_df.groupby('Age_Group')['Response'].agg(['mean', 'count'])
    age_response['mean'] = age_response['mean'] * 100

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.bar(age_response.index, age_response['mean'],
            color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Age Group', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Interest Rate (%)', fontsize=12, fontweight='bold', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')

    ax2 = ax1.twinx()
    ax2.plot(age_response.index, age_response['count'],
             color='red', marker='o', linewidth=2, markersize=8)
    ax2.set_ylabel('Number of Customers', fontsize=12, fontweight='bold', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title('Customer Interest by Age Segment', fontsize=14, fontweight='bold', pad=20)

    for i, (idx, row) in enumerate(age_response.iterrows()):
        ax1.text(i, row['mean'], f"{row['mean']:.1f}%",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    fig.tight_layout()
    plt.savefig(f'{CHARTS_DIR}/02_age_group_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_gender_analysis(train_df):
    """Chart 3: Response Rate by Gender"""
    print("[3/16] Generating gender analysis...")

    gender_response = train_df.groupby('Gender')['Response'].agg(['mean', 'count'])
    gender_response['mean'] = gender_response['mean'] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    bars = ax1.bar(gender_response.index, gender_response['mean'],
                   color=['#ff9999', '#66b3ff'], alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Gender', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Interest Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Interest Rate by Gender', fontsize=13, fontweight='bold')

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    bars2 = ax2.bar(gender_response.index, gender_response['count'],
                    color=['#ff9999', '#66b3ff'], alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Gender', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
    ax2.set_title('Customer Distribution by Gender', fontsize=13, fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{CHARTS_DIR}/03_gender_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_vehicle_damage_impact(train_df):
    """Chart 4: Impact of Previous Vehicle Damage"""
    print("[4/16] Generating vehicle damage impact...")

    damage_response = train_df.groupby('Vehicle_Damage')['Response'].agg(['mean', 'count'])
    damage_response['mean'] = damage_response['mean'] * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(damage_response.index, damage_response['mean'],
                  color=['#90ee90', '#ff6b6b'], alpha=0.7, edgecolor='black')
    ax.set_xlabel('Previous Vehicle Damage', fontsize=12, fontweight='bold')
    ax.set_ylabel('Interest Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Impact of Vehicle Damage History on Cross-Sell Interest',
                 fontsize=14, fontweight='bold', pad=20)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        count = damage_response['count'].iloc[i]
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%\n({int(count):,} customers)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{CHARTS_DIR}/04_vehicle_damage_impact.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_previous_insurance_impact(train_df):
    """Chart 5: Previously Insured Status Impact"""
    print("[5/16] Generating previous insurance impact...")

    insured_response = train_df.groupby('Previously_Insured')['Response'].agg(['mean', 'count'])
    insured_response['mean'] = insured_response['mean'] * 100
    insured_response.index = ['Not Previously Insured', 'Previously Insured']

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(insured_response.index, insured_response['mean'],
                  color=['#ffa500', '#4169e1'], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Interest Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Cross-Sell Interest: New vs. Previously Insured Customers',
                 fontsize=14, fontweight='bold', pad=20)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        count = insured_response['count'].iloc[i]
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%\n({int(count):,} customers)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(f'{CHARTS_DIR}/05_previous_insurance_impact.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_vehicle_age_analysis(train_df):
    """Chart 6: Vehicle Age Analysis"""
    print("[6/16] Generating vehicle age analysis...")

    vehicle_age_response = train_df.groupby('Vehicle_Age')['Response'].agg(['mean', 'count'])
    vehicle_age_response['mean'] = vehicle_age_response['mean'] * 100

    age_order = ['< 1 Year', '1-2 Year', '> 2 Years']
    vehicle_age_response = vehicle_age_response.reindex(age_order)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    bars = ax1.bar(range(len(vehicle_age_response)), vehicle_age_response['mean'],
                   color=['#98d8c8', '#6ab7a8', '#3d9688'], alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Vehicle Age', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Interest Rate (%)', fontsize=12, fontweight='bold', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_xticks(range(len(vehicle_age_response)))
    ax1.set_xticklabels(vehicle_age_response.index)

    ax2 = ax1.twinx()
    ax2.plot(range(len(vehicle_age_response)), vehicle_age_response['count'],
             color='red', marker='o', linewidth=2.5, markersize=10)
    ax2.set_ylabel('Number of Customers', fontsize=12, fontweight='bold', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title('Customer Interest by Vehicle Age', fontsize=14, fontweight='bold', pad=20)

    for i, (idx, row) in enumerate(vehicle_age_response.iterrows()):
        ax1.text(i, row['mean'], f"{row['mean']:.1f}%",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    fig.tight_layout()
    plt.savefig(f'{CHARTS_DIR}/06_vehicle_age_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_premium_segment_analysis(train_df):
    """Chart 7: Premium Amount vs Response"""
    print("[7/16] Generating premium segment analysis...")

    train_df['Premium_Segment'] = pd.qcut(train_df['Annual_Premium'],
                                          q=5,
                                          labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

    premium_response = train_df.groupby('Premium_Segment')['Response'].agg(['mean', 'count'])
    premium_response['mean'] = premium_response['mean'] * 100

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(range(len(premium_response)), premium_response['mean'],
                  color=['#d4f1d4', '#a8e6a8', '#7cdb7c', '#50d050', '#24c524'],
                  alpha=0.7, edgecolor='black')
    ax.set_xlabel('Annual Premium Segment', fontsize=12, fontweight='bold')
    ax.set_ylabel('Interest Rate (%)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(premium_response)))
    ax.set_xticklabels(premium_response.index)
    ax.set_title('Cross-Sell Interest by Customer Premium Segment',
                 fontsize=14, fontweight='bold', pad=20)

    for i, (idx, row) in enumerate(premium_response.iterrows()):
        ax.text(i, row['mean'], f"{row['mean']:.1f}%",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{CHARTS_DIR}/07_premium_segment_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_top_sales_channels(train_df):
    """Chart 8: Sales Channel Performance"""
    print("[8/16] Generating top sales channels...")

    channel_response = train_df.groupby('Policy_Sales_Channel')['Response'].agg(['mean', 'count'])
    channel_response['mean'] = channel_response['mean'] * 100
    channel_response = channel_response.sort_values('mean', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(12, 7))

    bars = ax.barh(range(len(channel_response)), channel_response['mean'],
                   color='coral', alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(channel_response)))
    ax.set_yticklabels([f'Channel {idx}' for idx in channel_response.index])
    ax.set_xlabel('Interest Rate (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sales Channel', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 Sales Channels by Cross-Sell Interest Rate',
                 fontsize=14, fontweight='bold', pad=20)

    for i, (idx, row) in enumerate(channel_response.iterrows()):
        ax.text(row['mean'], i, f"  {row['mean']:.1f}% ({int(row['count']):,} customers)",
                va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{CHARTS_DIR}/08_top_sales_channels.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_customer_tenure_analysis(train_df):
    """Chart 9: Customer Relationship Duration Analysis"""
    print("[9/16] Generating customer tenure analysis...")

    train_df['Vintage_Segment'] = pd.cut(train_df['Vintage'],
                                         bins=[0, 50, 100, 150, 200, 300],
                                         labels=['0-50 days', '51-100 days', '101-150 days',
                                                '151-200 days', '200+ days'])

    vintage_response = train_df.groupby('Vintage_Segment')['Response'].agg(['mean', 'count'])
    vintage_response['mean'] = vintage_response['mean'] * 100

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(range(len(vintage_response)), vintage_response['mean'],
                  color='mediumpurple', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Customer Relationship Duration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Interest Rate (%)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(vintage_response)))
    ax.set_xticklabels(vintage_response.index, rotation=30, ha='right')
    ax.set_title('Cross-Sell Interest by Customer Tenure',
                 fontsize=14, fontweight='bold', pad=20)

    for i, (idx, row) in enumerate(vintage_response.iterrows()):
        ax.text(i, row['mean'], f"{row['mean']:.1f}%\n({int(row['count']):,})",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{CHARTS_DIR}/09_customer_tenure_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_driving_license_impact(train_df):
    """Chart 10: Driving License Impact"""
    print("[10/16] Generating driving license impact...")

    license_response = train_df.groupby('Driving_License')['Response'].agg(['mean', 'count'])
    license_response['mean'] = license_response['mean'] * 100
    license_response.index = ['No License', 'Has License']

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(license_response.index, license_response['mean'],
                  color=['#ff7f0e', '#2ca02c'], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Interest Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Impact of Driving License on Cross-Sell Interest',
                 fontsize=14, fontweight='bold', pad=20)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        count = license_response['count'].iloc[i]
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%\n({int(count):,} customers)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{CHARTS_DIR}/10_driving_license_impact.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_customer_risk_segments(train_df):
    """Chart 11: Combined High-Impact Factors"""
    print("[11/16] Generating customer risk segments...")

    segment_analysis = train_df.groupby(['Vehicle_Damage', 'Previously_Insured'])['Response'].agg(['mean', 'count'])
    segment_analysis['mean'] = segment_analysis['mean'] * 100

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(segment_analysis))
    labels = [f"{damage}\n{'Prev. Insured' if ins == 1 else 'Not Insured'}"
              for damage, ins in segment_analysis.index]

    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    bars = ax.bar(x, segment_analysis['mean'], color=colors, alpha=0.7, edgecolor='black')

    ax.set_ylabel('Interest Rate (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Customer Segment', fontsize=12, fontweight='bold')
    ax.set_title('Cross-Sell Interest by Customer Risk Profile',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        count = segment_analysis['count'].iloc[i]
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%\n({int(count):,})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{CHARTS_DIR}/11_customer_risk_segments.png', dpi=300, bbox_inches='tight')
    plt.close()


def train_models(train_df):
    """Train predictive models and return results"""
    print("\n[12/16] Training predictive models...")

    # Prepare data
    train_processed = train_df.copy()
    train_processed = train_processed.drop('id', axis=1)

    # Encode categorical variables
    categorical_cols = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']
    for col in categorical_cols:
        le = LabelEncoder()
        train_processed[col] = le.fit_transform(train_processed[col])

    # Split
    X = train_processed.drop('Response', axis=1)
    y = train_processed['Response']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,
                                                        random_state=42, stratify=y)

    models = {}
    predictions_val = {}
    model_scores = {}

    # Train XGBoost
    print("  Training XGBoost...")
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42,
                                  eval_metric='logloss', use_label_encoder=False)
    xgb_model.fit(X_train, y_train, verbose=False)
    models['XGBoost'] = xgb_model
    predictions_val['XGBoost'] = xgb_model.predict_proba(X_val)[:, 1]
    model_scores['XGBoost'] = roc_auc_score(y_val, predictions_val['XGBoost'])

    # Train LightGBM
    print("  Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    lgb_model.fit(X_train, y_train)
    models['LightGBM'] = lgb_model
    predictions_val['LightGBM'] = lgb_model.predict_proba(X_val)[:, 1]
    model_scores['LightGBM'] = roc_auc_score(y_val, predictions_val['LightGBM'])

    print(f"  XGBoost ROC-AUC: {model_scores['XGBoost']:.4f}")
    print(f"  LightGBM ROC-AUC: {model_scores['LightGBM']:.4f}")

    return models, predictions_val, model_scores, X, y_val


def generate_model_performance_comparison(model_scores):
    """Chart 12: Model Performance Comparison"""
    print("[13/16] Generating model performance comparison...")

    fig, ax = plt.subplots(figsize=(12, 6))

    model_names = list(model_scores.keys())
    scores = list(model_scores.values())

    colors_gradient = ['#f39c12', '#27ae60']
    bars = ax.bar(model_names, scores, color=colors_gradient, alpha=0.7, edgecolor='black')

    ax.set_ylabel('ROC-AUC Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Prediction Model', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison: Accuracy of Customer Interest Prediction',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0.5, 1.0)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Random Guess (0.5)')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.xticks(rotation=15, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{CHARTS_DIR}/12_model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_roc_curves(predictions_val, model_scores, y_val):
    """Chart 13: ROC Curves for All Models"""
    print("[14/16] Generating ROC curves...")

    fig, ax = plt.subplots(figsize=(10, 8))

    colors_roc = ['#2ecc71', '#f39c12']

    for i, (model_name, y_pred) in enumerate(predictions_val.items()):
        fpr, tpr, _ = roc_curve(y_val, y_pred)
        auc_score = model_scores[model_name]
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.4f})',
                linewidth=2.5, color=colors_roc[i])

    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Guess', alpha=0.3)
    ax.set_xlabel('False Positive Rate (Incorrect Predictions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Correct Predictions)', fontsize=12, fontweight='bold')
    ax.set_title('Model Prediction Accuracy: ROC Curves',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{CHARTS_DIR}/13_roc_curves_all_models.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_feature_importance(best_model, X):
    """Chart 14: Feature Importance"""
    print("[15/16] Generating feature importance...")

    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)

        fig, ax = plt.subplots(figsize=(12, 8))

        ax.barh(range(len(feature_importance)), feature_importance['importance'],
                color='teal', alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(feature_importance)))
        ax.set_yticklabels(feature_importance['feature'])
        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Customer Characteristic', fontsize=12, fontweight='bold')
        ax.set_title('Key Factors Driving Cross-Sell Interest',
                     fontsize=14, fontweight='bold', pad=20)

        for i, (idx, row) in enumerate(feature_importance.iterrows()):
            ax.text(row['importance'], i, f"  {row['importance']:.4f}",
                    va='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{CHARTS_DIR}/14_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()


def generate_targeting_strategy_analysis(best_predictions, y_val):
    """Chart 15: Business Impact Analysis - Targeting Efficiency"""
    print("[16/16] Generating targeting strategy analysis...")

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    results = []

    for threshold in thresholds:
        targeted = (best_predictions >= threshold).sum()
        conversion_rate = y_val[best_predictions >= threshold].mean() * 100 if targeted > 0 else 0
        coverage = (y_val[best_predictions >= threshold].sum() / y_val.sum()) * 100
        results.append({
            'threshold': f'{int(threshold*100)}%',
            'customers_targeted': targeted,
            'conversion_rate': conversion_rate,
            'interested_captured': coverage
        })

    results_df = pd.DataFrame(results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot
    ax1_twin = ax1.twinx()
    ax1.bar(range(len(results_df)), results_df['customers_targeted'],
            color='skyblue', alpha=0.7, edgecolor='black')
    ax1_twin.plot(range(len(results_df)), results_df['conversion_rate'],
                  color='red', marker='o', linewidth=2.5, markersize=10)

    ax1.set_xlabel('Minimum Confidence Threshold', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Customers Targeted', fontsize=12, fontweight='bold', color='skyblue')
    ax1_twin.set_ylabel('Conversion Rate (%)', fontsize=12, fontweight='bold', color='red')
    ax1.set_xticks(range(len(results_df)))
    ax1.set_xticklabels(results_df['threshold'])
    ax1.set_title('Targeting Strategy: Volume vs. Precision', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    ax1_twin.tick_params(axis='y', labelcolor='red')

    # Right plot
    bars2 = ax2.bar(range(len(results_df)), results_df['interested_captured'],
                    color='green', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Minimum Confidence Threshold', fontsize=12, fontweight='bold')
    ax2.set_ylabel('% of Interested Customers Reached', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(results_df)))
    ax2.set_xticklabels(results_df['threshold'])
    ax2.set_title('Market Coverage by Threshold', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 100)

    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{CHARTS_DIR}/16_targeting_strategy_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main execution function"""
    print("=" * 80)
    print("INSURANCE CROSS-SELLING BUSINESS ANALYTICS")
    print("Generating all charts...")
    print("=" * 80)

    # Load data
    train_df, test_df = load_data()

    # Generate business insight charts (1-11)
    generate_overall_response_rate(train_df)
    generate_age_group_analysis(train_df.copy())
    generate_gender_analysis(train_df.copy())
    generate_vehicle_damage_impact(train_df.copy())
    generate_previous_insurance_impact(train_df.copy())
    generate_vehicle_age_analysis(train_df.copy())
    generate_premium_segment_analysis(train_df.copy())
    generate_top_sales_channels(train_df.copy())
    generate_customer_tenure_analysis(train_df.copy())
    generate_driving_license_impact(train_df.copy())
    generate_customer_risk_segments(train_df.copy())

    # Train models and generate performance charts (12-16)
    models, predictions_val, model_scores, X, y_val = train_models(train_df.copy())

    generate_model_performance_comparison(model_scores)
    generate_roc_curves(predictions_val, model_scores, y_val)

    best_model_name = max(model_scores, key=model_scores.get)
    best_model = models[best_model_name]
    best_predictions = predictions_val[best_model_name]

    generate_feature_importance(best_model, X)
    generate_targeting_strategy_analysis(best_predictions, y_val)

    print("\n" + "=" * 80)
    print("ALL CHARTS GENERATED SUCCESSFULLY!")
    print(f"Charts saved to: {CHARTS_DIR}/")
    print("=" * 80)
    print(f"\nBest Model: {best_model_name} (ROC-AUC: {model_scores[best_model_name]:.4f})")


if __name__ == "__main__":
    main()
