"""
visualization.py
Professional plotting functions for DHS Early Marriage Study
Author: Bereket Andualem
Date: 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set professional style for publication
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Professional color palette
COLORS = {
    'early_marriage': '#E74C3C',  # Red
    'not_early': '#2ECC71',        # Green
    'primary': '#3498DB',           # Blue
    'secondary': '#9B59B6',         # Purple
    'trend': '#E67E22'              # Orange
}

def set_publication_style():
    """Set matplotlib parameters for publication-ready figures"""
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 11
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16

def plot_early_marriage_trend(df, save_path=None):
    """
    Figure 1: Trend in early marriage prevalence (2000-2016)
    """
    set_publication_style()
    
    # Calculate rates by year (weighted)
    yearly_rates = []
    for year in [2000, 2005, 2011, 2016]:
        year_data = df[df['survey_year'] == year]
        ever_married = year_data[year_data['ever_married'] == 1]
        if len(ever_married) > 0:
            rate = ever_married['early_marriage'].mean() * 100
            yearly_rates.append({'year': year, 'rate': rate})
    
    trend_df = pd.DataFrame(yearly_rates)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Line plot with markers
    ax.plot(trend_df['year'], trend_df['rate'], 
            marker='o', linewidth=2.5, markersize=10,
            color=COLORS['trend'], label='Early marriage rate')
    
    # Add value labels
    for _, row in trend_df.iterrows():
        ax.annotate(f"{row['rate']:.1f}%", 
                   xy=(row['year'], row['rate']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=11, fontweight='bold')
    
    # Customize
    ax.set_xlabel('Survey Year', fontsize=12)
    ax.set_ylabel('Early Marriage Rate (%)', fontsize=12)
    ax.set_title('Decline in Early Marriage Among Ever-Married Women\nEthiopia 2000-2016', 
                fontsize=14, fontweight='bold')
    ax.set_ylim(50, 75)
    ax.set_xticks([2000, 2005, 2011, 2016])
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.text(0.02, 0.98, f"Total decline: {(trend_df['rate'].iloc[0] - trend_df['rate'].iloc[-1]):.1f} percentage points",
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax

def plot_regional_early_marriage(df, save_path=None):
    """
    Figure 2: Regional variation in early marriage
    """
    set_publication_style()
    
    # Calculate rates by region (among ever-married)
    ever_married = df[df['ever_married'] == 1]
    regional_rates = ever_married.groupby('region')['early_marriage'].mean() * 100
    regional_rates = regional_rates.sort_values(ascending=False)
    
    # Exclude 'Unknown' for main figure
    regional_rates = regional_rates[regional_rates.index != 'Unknown']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Horizontal bar chart
    colors = [COLORS['early_marriage'] if rate > 60 else COLORS['primary'] for rate in regional_rates.values]
    bars = ax.barh(range(len(regional_rates)), regional_rates.values, color=colors, alpha=0.8)
    
    # Add value labels
    for i, (region, rate) in enumerate(regional_rates.items()):
        ax.text(rate + 1, i, f"{rate:.1f}%", va='center', fontsize=10)
    
    # Customize
    ax.set_yticks(range(len(regional_rates)))
    ax.set_yticklabels(regional_rates.index, fontsize=10)
    ax.set_xlabel('Early Marriage Rate (%)', fontsize=12)
    ax.set_title('Regional Variation in Early Marriage\nEthiopia 2000-2016 Combined', 
                fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.axvline(x=60, color='red', linestyle='--', alpha=0.5, label='60% threshold')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax

def plot_education_gradient(df, save_path=None):
    """
    Figure 3: Education gradient in early marriage
    """
    set_publication_style()
    
    ever_married = df[df['ever_married'] == 1]
    
    # Calculate rates by education level
    edu_rates = ever_married.groupby('education_level')['early_marriage'].mean() * 100
    edu_rates = edu_rates.reindex(['no_education', 'primary', 'secondary', 'higher'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar plot
    colors = [COLORS['early_marriage'] if i == 0 else COLORS['primary'] for i in range(len(edu_rates))]
    bars = ax.bar(range(len(edu_rates)), edu_rates.values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels
    for i, (edu, rate) in enumerate(edu_rates.items()):
        ax.text(i, rate + 1, f"{rate:.1f}%", ha='center', fontsize=11, fontweight='bold')
    
    # Customize
    ax.set_xticks(range(len(edu_rates)))
    ax.set_xticklabels(['No Education', 'Primary', 'Secondary', 'Higher'], fontsize=11)
    ax.set_ylabel('Early Marriage Rate (%)', fontsize=12)
    ax.set_title('Education is a Powerful Protective Factor\nAgainst Early Marriage', 
                fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add annotation
    reduction = edu_rates['no_education'] - edu_rates['higher']
    ax.text(0.98, 0.98, f"Higher education reduces\nearly marriage by {reduction:.1f} percentage points",
            transform=ax.transAxes, fontsize=10,
            horizontalalignment='right', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax

def plot_wealth_gradient(df, save_path=None):
    """
    Figure 4: Wealth quintile gradient in early marriage
    """
    set_publication_style()
    
    ever_married = df[df['ever_married'] == 1]
    
    # Calculate rates by wealth quintile
    wealth_order = ['Poorest', 'Poorer', 'Middle', 'Richer', 'Richest']
    wealth_rates = ever_married.groupby('wealth_quintile')['early_marriage'].mean() * 100
    wealth_rates = wealth_rates.reindex(wealth_order)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar plot with gradient colors
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, 5))
    bars = ax.bar(range(len(wealth_rates)), wealth_rates.values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels
    for i, (wealth, rate) in enumerate(wealth_rates.items()):
        ax.text(i, rate + 1, f"{rate:.1f}%", ha='center', fontsize=11, fontweight='bold')
    
    # Customize
    ax.set_xticks(range(len(wealth_rates)))
    ax.set_xticklabels(wealth_order, fontsize=11)
    ax.set_ylabel('Early Marriage Rate (%)', fontsize=12)
    ax.set_title('Wealth is Associated with Lower Early Marriage Rates', 
                fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax

def plot_age_at_marriage_distribution(df, save_path=None):
    """
    Figure 5: Distribution of age at first marriage
    """
    set_publication_style()
    
    ever_married = df[df['ever_married'] == 1].copy()
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Histogram
    ax1 = axes[0]
    ages = ever_married['age_first_marriage'].dropna()
    ax1.hist(ages, bins=range(10, 50, 2), edgecolor='black', alpha=0.7, color=COLORS['primary'])
    ax1.axvline(x=18, color='red', linestyle='--', linewidth=2, label='Early marriage threshold (18 years)')
    ax1.set_xlabel('Age at First Marriage (years)', fontsize=12)
    ax1.set_ylabel('Number of Women', fontsize=12)
    ax1.set_title('Distribution of Age at First Marriage', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Cumulative percentage
    ax2 = axes[1]
    ages_sorted = np.sort(ages)
    cumulative = np.arange(1, len(ages_sorted) + 1) / len(ages_sorted) * 100
    ax2.plot(ages_sorted, cumulative, linewidth=2, color=COLORS['trend'])
    ax2.axvline(x=18, color='red', linestyle='--', linewidth=2, label='Early marriage threshold')
    ax2.axhline(y=70.3, color='green', linestyle='--', alpha=0.5, label='70.3% married by 18 (2000)')
    ax2.set_xlabel('Age at First Marriage (years)', fontsize=12)
    ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax2.set_title('Cumulative Distribution: Percentage Married by Age', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Age at First Marriage Among Ever-Married Women\nEthiopia 2000-2016', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, axes

def plot_fertility_consequence(df, save_path=None):
    """
    Figure 6: Children ever born by early marriage status
    """
    set_publication_style()
    
    ever_married = df[df['ever_married'] == 1].copy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Box plot
    bp = ax.boxplot([ever_married[ever_married['early_marriage'] == 0]['children_ever_born'].dropna(),
                     ever_married[ever_married['early_marriage'] == 1]['children_ever_born'].dropna()],
                    labels=['Not Early Marriage\n(n=~{:,})'.format(len(ever_married[ever_married['early_marriage']==0])),
                            'Early Marriage\n(n=~{:,})'.format(len(ever_married[ever_married['early_marriage']==1]))],
                    patch_artist=True,
                    boxprops=dict(alpha=0.7))
    
    # Colors
    bp['boxes'][0].set_facecolor(COLORS['not_early'])
    bp['boxes'][1].set_facecolor(COLORS['early_marriage'])
    
    # Calculate means
    mean_not_early = ever_married[ever_married['early_marriage'] == 0]['children_ever_born'].mean()
    mean_early = ever_married[ever_married['early_marriage'] == 1]['children_ever_born'].mean()
    
    # Add mean markers
    ax.plot(1, mean_not_early, 'D', color='darkgreen', markersize=10, label=f'Mean: {mean_not_early:.1f}')
    ax.plot(2, mean_early, 'D', color='darkred', markersize=10, label=f'Mean: {mean_early:.1f}')
    
    ax.set_ylabel('Children Ever Born', fontsize=12)
    ax.set_title('Early Marriage is Associated with Higher Fertility', 
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add annotation
    difference = mean_early - mean_not_early
    ax.text(0.98, 0.98, f"Early marriage women have\n{difference:.1f} more children on average",
            transform=ax.transAxes, fontsize=10,
            horizontalalignment='right', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax

def plot_urban_rural_comparison(df, save_path=None):
    """
    Figure 8: Urban vs rural comparison
    """
    set_publication_style()
    
    ever_married = df[df['ever_married'] == 1]
    
    # Calculate rates by residence and year
    residence_year = ever_married.groupby(['survey_year', 'residence'])['early_marriage'].mean() * 100
    residence_year = residence_year.unstack()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Line plots
    ax.plot(residence_year.index, residence_year['rural'], 
            marker='o', linewidth=2.5, markersize=8,
            label='Rural', color=COLORS['early_marriage'])
    ax.plot(residence_year.index, residence_year['urban'], 
            marker='s', linewidth=2.5, markersize=8,
            label='Urban', color=COLORS['primary'])
    
    # Add gap annotation
    gap_2000 = residence_year.loc[2000, 'rural'] - residence_year.loc[2000, 'urban']
    gap_2016 = residence_year.loc[2016, 'rural'] - residence_year.loc[2016, 'urban']
    
    ax.set_xlabel('Survey Year', fontsize=12)
    ax.set_ylabel('Early Marriage Rate (%)', fontsize=12)
    ax.set_title('Rural-Urban Gap in Early Marriage\nPersists Over Time', 
                fontsize=14, fontweight='bold')
    ax.set_ylim(30, 85)
    ax.set_xticks([2000, 2005, 2011, 2016])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.text(0.02, 0.98, f"Rural-Urban gap:\n2000: {gap_2000:.1f} percentage points\n2016: {gap_2016:.1f} points",
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax

def generate_all_figures(df, output_dir):
    """
    Generate all figures for the research paper
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating publication-ready figures...")
    print("="*50)
    
    # Figure 1: Trend
    plot_early_marriage_trend(df, output_dir / 'figure1_trend.png')
    
    # Figure 2: Regional variation
    plot_regional_early_marriage(df, output_dir / 'figure2_regional.png')
    
    # Figure 3: Education gradient
    plot_education_gradient(df, output_dir / 'figure3_education.png')
    
    # Figure 4: Wealth gradient
    plot_wealth_gradient(df, output_dir / 'figure4_wealth.png')
    
    # Figure 5: Age distribution
    plot_age_at_marriage_distribution(df, output_dir / 'figure5_age_distribution.png')
    
    # Figure 6: Fertility consequence
    plot_fertility_consequence(df, output_dir / 'figure6_fertility.png')
    
    # Figure 8: Urban-rural comparison
    plot_urban_rural_comparison(df, output_dir / 'figure8_urban_rural.png')
    
    print("="*50)
    print(f"All figures saved to {output_dir}")
    

    return True
def plot_media_exposure(df, save_path=None):
    """
    Figure 11: Early marriage by media exposure
    """
    set_publication_style()
    
    ever_married = df[df['ever_married'] == 1].copy()
    
    # Check if media exposure variable exists
    if 'media_exposure' not in df.columns:
        print("Warning: 'media_exposure' not found in dataset. Creating from available data...")
        # Try to create from v157 if available
        if 'v157' in df.columns:
            df['media_exposure'] = pd.to_numeric(df['v157'], errors='coerce')
            df['media_exposure'] = df['media_exposure'].map({0: 'none', 1: 'weekly', 2: 'daily'})
            ever_married = df[df['ever_married'] == 1].copy()
        else:
            print("Media exposure data not available. Skipping figure.")
            return None, None
    
    # Calculate rates by media exposure
    media_rates = ever_married.groupby('media_exposure')['early_marriage'].mean() * 100
    media_rates = media_rates.reindex(['none', 'weekly', 'daily'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = [COLORS['early_marriage'], COLORS['primary'], COLORS['secondary']]
    bars = ax.bar(range(len(media_rates)), media_rates.values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels
    for i, (media, rate) in enumerate(media_rates.items()):
        ax.text(i, rate + 1, f"{rate:.1f}%", ha='center', fontsize=11, fontweight='bold')
    
    ax.set_xticks(range(len(media_rates)))
    ax.set_xticklabels(['No Exposure', 'Weekly', 'Daily'], fontsize=11)
    ax.set_ylabel('Early Marriage Rate (%)', fontsize=12)
    ax.set_title('Media Exposure is Associated with Lower Early Marriage Rates', 
                fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add annotation
    reduction = media_rates['none'] - media_rates['daily']
    ax.text(0.98, 0.98, f"Daily media exposure reduces\nearly marriage by {reduction:.1f} percentage points",
            transform=ax.transAxes, fontsize=10,
            horizontalalignment='right', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_empowerment_index(df, save_path=None):
    """
    Figure 12: Early marriage by women's empowerment
    Creates an empowerment index from available decision-making variables
    """
    set_publication_style()
    
    ever_married = df[df['ever_married'] == 1].copy()
    
    # Look for empowerment-related variables
    empowerment_vars = [col for col in df.columns if 'v743' in col or 'decision' in col.lower()]
    
    if len(empowerment_vars) == 0:
        print("Warning: No empowerment variables found. Using work status as proxy.")
        # Use work status as empowerment proxy
        empowerment_col = 'currently_working'
        ever_married['empowerment_level'] = ever_married[empowerment_col].map({0: 'Not Working', 1: 'Working'})
    else:
        print(f"Found empowerment variables: {empowerment_vars}")
        # Create simple empowerment index (sum of decision-making indicators)
        for var in empowerment_vars:
            ever_married[var] = pd.to_numeric(ever_married[var], errors='coerce')
        
        # Create index (higher = more empowered)
        ever_married['empowerment_score'] = ever_married[empowerment_vars].sum(axis=1)
        ever_married['empowerment_level'] = pd.qcut(ever_married['empowerment_score'], 
                                                     q=3, 
                                                     labels=['Low', 'Medium', 'High'],
                                                     duplicates='drop')
    
    # Calculate rates by empowerment level
    empowerment_rates = ever_married.groupby('empowerment_level')['early_marriage'].mean() * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = [COLORS['early_marriage'], COLORS['primary'], COLORS['secondary']]
    bars = ax.bar(range(len(empowerment_rates)), empowerment_rates.values, color=colors[:len(empowerment_rates)], alpha=0.8, edgecolor='black')
    
    # Add value labels
    for i, (level, rate) in enumerate(empowerment_rates.items()):
        ax.text(i, rate + 1, f"{rate:.1f}%", ha='center', fontsize=11, fontweight='bold')
    
    ax.set_xticks(range(len(empowerment_rates)))
    ax.set_xticklabels(empowerment_rates.index, fontsize=11)
    ax.set_ylabel('Early Marriage Rate (%)', fontsize=12)
    ax.set_title('Women\'s Empowerment is Associated with Lower Early Marriage Rates', 
                fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_logistic_education_probability(df, save_path=None):
    """
    Figure 13: Predicted probability of early marriage by education years
    Using logistic regression
    """
    from sklearn.linear_model import LogisticRegression
    
    set_publication_style()
    
    # Prepare data (ever-married only)
    ever_married = df[df['ever_married'] == 1].copy()
    ever_married = ever_married.dropna(subset=['education_years', 'early_marriage'])
    
    # Simple logistic regression
    X = ever_married[['education_years']].values
    y = ever_married['early_marriage'].values
    
    model = LogisticRegression()
    model.fit(X, y)
    
    # Create prediction curve
    edu_range = np.arange(0, 18, 0.5).reshape(-1, 1)
    pred_prob = model.predict_proba(edu_range)[:, 1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Scatter plot of actual data (with jitter)
    jitter = np.random.normal(0, 0.1, len(ever_married))
    ax.scatter(ever_married['education_years'] + jitter, 
               ever_married['early_marriage'] + jitter * 0.2,
               alpha=0.05, s=10, color=COLORS['primary'])
    
    # Logistic curve
    ax.plot(edu_range, pred_prob, linewidth=3, color=COLORS['trend'], label='Predicted probability')
    
    # Add confidence interval (simplified)
    from scipy.stats import norm
    std_errors = np.sqrt(pred_prob * (1 - pred_prob) / len(ever_married))
    ax.fill_between(edu_range.flatten(), 
                    pred_prob - 1.96 * std_errors,
                    pred_prob + 1.96 * std_errors,
                    alpha=0.2, color=COLORS['trend'])
    
    ax.set_xlabel('Years of Education', fontsize=12)
    ax.set_ylabel('Probability of Early Marriage', fontsize=12)
    ax.set_title('Each Additional Year of Education Reduces\nProbability of Early Marriage', 
                fontsize=14, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.5, 17)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add annotation with odds ratio
    odds_ratio = np.exp(model.coef_[0][0])
    ax.text(0.98, 0.98, f"Odds Ratio per year: {odds_ratio:.3f}\n(Each year reduces odds by {(1-odds_ratio)*100:.1f}%)",
            transform=ax.transAxes, fontsize=10,
            horizontalalignment='right', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_regional_early_marriage_with_unknown(df, save_path=None):
    """
    Updated Figure 2: Regional variation INCLUDING Unknown category
    """
    set_publication_style()
    
    ever_married = df[df['ever_married'] == 1]
    
    # Calculate rates by region (including Unknown)
    regional_rates = ever_married.groupby('region')['early_marriage'].mean() * 100
    regional_rates = regional_rates.sort_values(ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color coding: Unknown in different color
    colors = []
    for region in regional_rates.index:
        if region == 'Unknown':
            colors.append('#95A5A6')  # Gray for Unknown
        elif regional_rates[region] > 70:
            colors.append(COLORS['early_marriage'])  # Red for high rates
        else:
            colors.append(COLORS['primary'])  # Blue for lower rates
    
    bars = ax.barh(range(len(regional_rates)), regional_rates.values, color=colors, alpha=0.8)
    
    # Add value labels
    for i, (region, rate) in enumerate(regional_rates.items()):
        ax.text(rate + 1, i, f"{rate:.1f}%", va='center', fontsize=10)
    
    ax.set_yticks(range(len(regional_rates)))
    ax.set_yticklabels(regional_rates.index, fontsize=10)
    ax.set_xlabel('Early Marriage Rate (%)', fontsize=12)
    ax.set_title('Regional Variation in Early Marriage (Including Unknown Region)', 
                fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.axvline(x=60, color='red', linestyle='--', alpha=0.5, label='60% threshold')
    ax.legend()
    
    # Add annotation about Unknown
    unknown_rate = regional_rates.get('Unknown', 0)
    ax.text(0.98, 0.02, f"Note: 'Unknown' region ({unknown_rate:.1f}% early marriage)\nrepresents 15.5% of sample with missing region data",
            transform=ax.transAxes, fontsize=9,
            horizontalalignment='right', verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


# Update the generate_all_figures function
def generate_all_figures(df, output_dir):
    """
    Generate all figures for the research paper (UPDATED with new figures)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating publication-ready figures...")
    print("="*50)
    
    # Figure 1: Trend
    plot_early_marriage_trend(df, output_dir / 'figure1_trend.png')
    
    # Figure 2: Regional variation (UPDATED with Unknown)
    plot_regional_early_marriage_with_unknown(df, output_dir / 'figure2_regional_with_unknown.png')
    
    # Figure 3: Education gradient
    plot_education_gradient(df, output_dir / 'figure3_education.png')
    
    # Figure 4: Wealth gradient
    plot_wealth_gradient(df, output_dir / 'figure4_wealth.png')
    
    # Figure 5: Age distribution
    plot_age_at_marriage_distribution(df, output_dir / 'figure5_age_distribution.png')
    
    # Figure 6: Fertility consequence
    plot_fertility_consequence(df, output_dir / 'figure6_fertility.png')
    
    # Figure 8: Urban-rural comparison
    plot_urban_rural_comparison(df, output_dir / 'figure8_urban_rural.png')
    
    # NEW Figure 11: Media exposure
    plot_media_exposure(df, output_dir / 'figure11_media_exposure.png')
    
    # NEW Figure 12: Women empowerment
    plot_empowerment_index(df, output_dir / 'figure12_empowerment.png')
    
    # NEW Figure 13: Logistic probability
    plot_logistic_education_probability(df, output_dir / 'figure13_logistic_education.png')
    
    print("="*50)
    print(f"All figures saved to {output_dir}")
    
    return True
if __name__ == "__main__":
    print("Visualization module loaded successfully")
    print("Available functions:")
    print("  - plot_early_marriage_trend()")
    print("  - plot_regional_early_marriage()")
    print("  - plot_education_gradient()")
    print("  - plot_wealth_gradient()")
    print("  - plot_age_at_marriage_distribution()")
    print("  - plot_fertility_consequence()")
    print("  - plot_urban_rural_comparison()")
    print("  - generate_all_figures()")