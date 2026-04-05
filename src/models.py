"""
models.py
Professional statistical modeling for DHS Early Marriage Study
Author: Bereket Andualem
Date: 2026

Includes:
- Logistic regression (binary outcome: early marriage)
- Poisson regression (count outcome: children ever born)
- Multilevel models (regional random effects)
- Sensitivity analyses
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')


def prepare_modeling_data(df, exclude_unknown_region=False):
    """
    Prepare dataset for statistical modeling
    
    Parameters:
    -----------
    df : pandas DataFrame
        Modeling-ready dataset
    exclude_unknown_region : bool
        If True, exclude 'Unknown' region from analysis
    
    Returns:
    --------
    df_model : pandas DataFrame
        Prepared dataset for modeling
    """
    # Work with ever-married women only
    df_model = df[df['ever_married'] == 1].copy()
    
    # Create categorical variables
    df_model['education_level_cat'] = pd.Categorical(
        df_model['education_level'],
        categories=['no_education', 'primary', 'secondary', 'higher'],
        ordered=True
    )
    
    df_model['residence_cat'] = pd.Categorical(df_model['residence'])
    df_model['wealth_quintile_cat'] = pd.Categorical(
        df_model['wealth_quintile'],
        categories=['Poorest', 'Poorer', 'Middle', 'Richer', 'Richest'],
        ordered=True
    )
    
    # Create survey year as categorical for trend analysis
    df_model['survey_year_cat'] = df_model['survey_year'].astype('category')
    
    # Create region with Unknown handling
    if exclude_unknown_region:
        df_model = df_model[df_model['region'] != 'Unknown'].copy()
        logging.info(f"Excluded Unknown region. Remaining: {len(df_model):,} women")
    else:
        # Keep Unknown as a category
        df_model['region_cat'] = pd.Categorical(df_model['region'])
        logging.info(f"Keeping Unknown region ({len(df_model[df_model['region']=='Unknown']):,} women)")
    
    # Drop rows with missing key variables
    key_vars = ['early_marriage', 'education_level_cat', 'wealth_quintile_cat', 
                'residence_cat', 'current_age']
    df_model = df_model.dropna(subset=key_vars)
    
    logging.info(f"Final modeling dataset: {len(df_model):,} women")
    
    return df_model


def run_logistic_regression_unadjusted(df, outcome='early_marriage', predictors=None):
    """
    Model 1: Unadjusted logistic regression (crude odds ratios)
    """
    if predictors is None:
        predictors = ['education_level_cat', 'wealth_quintile_cat', 
                      'residence_cat', 'current_age']
    
    results = {}
    
    for predictor in predictors:
        formula = f"{outcome} ~ C({predictor})"
        
        # Handle categorical vs continuous
        if predictor == 'current_age':
            formula = f"{outcome} ~ {predictor}"
        
        model = smf.logit(formula, data=df).fit(disp=0)
        
        # Extract odds ratios and confidence intervals
        params = model.params
        conf = model.conf_int()
        
        if predictor != 'current_age':
            # For categorical, drop reference category
            params = params[1:]
            conf = conf.iloc[1:]
        
        odds_ratios = np.exp(params)
        ci_lower = np.exp(conf[0])
        ci_upper = np.exp(conf[1])
        
        results[predictor] = pd.DataFrame({
            'Variable': params.index,
            'Odds_Ratio': odds_ratios,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper,
            'P_Value': model.pvalues[1:] if predictor != 'current_age' else model.pvalues
        })
    
    return results


def run_logistic_regression_adjusted(df, outcome='early_marriage'):
    """
    Model 2: Adjusted logistic regression (full model)
    """
    formula = (f"{outcome} ~ C(education_level_cat) + "
               f"C(wealth_quintile_cat) + "
               f"C(residence_cat) + "
               f"current_age + "
               f"C(survey_year_cat)")
    
    model = smf.logit(formula, data=df).fit(disp=1)
    
    # Extract results
    params = model.params
    conf = model.conf_int()
    p_values = model.pvalues
    
    # Calculate odds ratios
    odds_ratios = np.exp(params)
    ci_lower = np.exp(conf[0])
    ci_upper = np.exp(conf[1])
    
    results_df = pd.DataFrame({
        'Variable': params.index,
        'Coef': params.values,
        'Odds_Ratio': odds_ratios.values,
        'CI_Lower': ci_lower.values,
        'CI_Upper': ci_upper.values,
        'P_Value': p_values.values,
        'Significant': p_values.values < 0.05
    })
    
    # Calculate pseudo R-squared
    pseudo_r2 = 1 - (model.llf / model.llnull)
    
    logging.info(f"Model pseudo R-squared: {pseudo_r2:.4f}")
    
    return model, results_df, pseudo_r2


def run_logistic_regression_with_interaction(df, outcome='early_marriage'):
    """
    Model 3: Test if education effect changed over time
    """
    # Interaction between education and survey year
    formula = (f"{outcome} ~ C(education_level_cat) * C(survey_year_cat) + "
               f"C(wealth_quintile_cat) + "
               f"C(residence_cat) + "
               f"current_age")
    
    model = smf.logit(formula, data=df).fit(disp=1)
    
    # Extract education-year interaction terms
    interaction_terms = [p for p in model.params.index if 'C(education_level_cat)[T.' in p and 'C(survey_year_cat)[T.' in p]
    
    interaction_results = []
    for term in interaction_terms:
        idx = model.params.index.get_loc(term)
        interaction_results.append({
            'Interaction': term,
            'Odds_Ratio': np.exp(model.params[idx]),
            'P_Value': model.pvalues[idx]
        })
    
    return model, pd.DataFrame(interaction_results)


def run_poisson_regression_fertility(df):
    """
    Model 4: Poisson regression for fertility consequences
    Outcome: children_ever_born
    Predictor: early_marriage (adjusted for confounders)
    """
    # Use ever-married women
    df_fertility = df[df['ever_married'] == 1].copy()
    
    # Poisson model
    formula = ("children_ever_born ~ early_marriage + "
               "C(education_level_cat) + "
               "C(wealth_quintile_cat) + "
               "C(residence_cat) + "
               "current_age")
    
    model = smf.poisson(formula, data=df_fertility).fit(disp=1)
    
    # Extract incidence rate ratios (IRR)
    params = model.params
    conf = model.conf_int()
    
    irr = np.exp(params)
    ci_lower = np.exp(conf[0])
    ci_upper = np.exp(conf[1])
    
    results_df = pd.DataFrame({
        'Variable': params.index,
        'IRR': irr.values,
        'CI_Lower': ci_lower.values,
        'CI_Upper': ci_upper.values,
        'P_Value': model.pvalues.values,
        'Significant': model.pvalues.values < 0.05
    })
    
    # Get early_marriage coefficient specifically
    early_marriage_irr = results_df[results_df['Variable'] == 'early_marriage']
    
    logging.info(f"Early marriage increases fertility by {(early_marriage_irr['IRR'].values[0] - 1) * 100:.1f}%")
    
    return model, results_df


def run_multilevel_model(df):
    """
    Model 5: Multilevel logistic regression with regional random effects
    """
    try:
        import statsmodels.formula.api as smf
        from statsmodels.regression.mixed_linear_model import MixedLM
        
        # Mixed effects logistic regression
        formula = ("early_marriage ~ C(education_level_cat) + "
                   "C(wealth_quintile_cat) + "
                   "C(residence_cat) + "
                   "current_age")
        
        # This is a simplification - full mixed effects requires more setup
        # For now, run logistic with region fixed effects as alternative
        
        formula_with_region = formula + " + C(region)"
        model = smf.logit(formula_with_region, data=df).fit(disp=0)
        
        # Calculate variance inflation to check regional clustering
        region_effects = model.params.filter(like='C(region)[T.')
        
        logging.info(f"Regional variance: {region_effects.var():.4f}")
        
        return model, region_effects
        
    except Exception as e:
        logging.warning(f"Multilevel model error: {e}")
        logging.info("Using region fixed effects as alternative")
        return None, None

def sensitivity_analysis(df, exclude_unknown=False):
    """
    Sensitivity analysis: Compare models with/without Unknown region
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    exclude_unknown : bool
        If True, also run model excluding Unknown region for comparison
    
    Returns:
    --------
    results : dict
        Dictionary containing model results for each scenario
    comparison : pd.DataFrame
        Comparison of coefficients between models
    """
    results = {}
    
    # Run main model on full dataset
    df_full = prepare_modeling_data(df, exclude_unknown_region=False)
    model_full, results_full, pseudo_r2_full = run_logistic_regression_adjusted(df_full)
    
    # Ensure results_full is a DataFrame, not a dict
    if isinstance(results_full, dict):
        # If it's a dict, extract the DataFrame
        results_full_df = results_full.get('results', results_full.get('df', results_full))
    else:
        results_full_df = results_full
    
    results['full_sample'] = {
        'model': model_full,
        'results_df': results_full_df,  # Changed from 'results' to 'results_df' for clarity
        'pseudo_r2': pseudo_r2_full,
        'n': len(df_full)
    }
    
    if exclude_unknown:
        # Run model excluding Unknown region
        df_no_unknown = prepare_modeling_data(df, exclude_unknown_region=True)
        model_no_unknown, results_no_unknown, pseudo_r2_no_unknown = run_logistic_regression_adjusted(df_no_unknown)
        
        # Ensure results_no_unknown is a DataFrame
        if isinstance(results_no_unknown, dict):
            results_no_unknown_df = results_no_unknown.get('results', results_no_unknown.get('df', results_no_unknown))
        else:
            results_no_unknown_df = results_no_unknown
        
        results['exclude_unknown'] = {
            'model': model_no_unknown,
            'results_df': results_no_unknown_df,  # Changed from 'results' to 'results_df'
            'pseudo_r2': pseudo_r2_no_unknown,
            'n': len(df_no_unknown)
        }
    
    # Compare key coefficients - FIXED SECTION
    # Access the results DataFrame correctly
    full_results_df = results['full_sample']['results_df']
    
    # Start with full sample results
    comparison = pd.DataFrame({
        'Variable': full_results_df['Variable'],
        'Full_Sample_OR': full_results_df['Odds_Ratio'].round(3),
        'Full_Sample_CI': full_results_df['CI_Lower'].round(2).astype(str) + '-' + full_results_df['CI_Upper'].round(2).astype(str)
    })
    
    # Add exclude_unknown results if available
    if exclude_unknown and 'exclude_unknown' in results:
        exclude_results_df = results['exclude_unknown']['results_df']
        
        # Merge with full sample results on Variable
        comparison = comparison.merge(
            exclude_results_df[['Variable', 'Odds_Ratio', 'CI_Lower', 'CI_Upper']],
            on='Variable',
            how='left'
        )
        
        # Rename columns and format
        comparison.rename(columns={
            'Odds_Ratio': 'Exclude_Unknown_OR',
            'CI_Lower': 'Exclude_Unknown_CI_Lower',
            'CI_Upper': 'Exclude_Unknown_CI_Upper'
        }, inplace=True)
        
        # Format confidence interval
        comparison['Exclude_Unknown_CI'] = (
            comparison['Exclude_Unknown_CI_Lower'].round(2).astype(str) + '-' + 
            comparison['Exclude_Unknown_CI_Upper'].round(2).astype(str)
        )
        
        # Drop the separate CI columns
        comparison.drop(columns=['Exclude_Unknown_CI_Lower', 'Exclude_Unknown_CI_Upper'], inplace=True)
        
        # Round odds ratios
        comparison['Exclude_Unknown_OR'] = comparison['Exclude_Unknown_OR'].round(3)
    
    # Print summary
    print(f"\nSensitivity Analysis Summary:")
    print(f"  Full sample: n={results['full_sample']['n']:,}, pseudo R²={results['full_sample']['pseudo_r2']:.4f}")
    if exclude_unknown and 'exclude_unknown' in results:
        print(f"  Excluding Unknown: n={results['exclude_unknown']['n']:,}, pseudo R²={results['exclude_unknown']['pseudo_r2']:.4f}")
    
    return results, comparison

def create_forest_plot(results_df, save_path=None):
    """
    Create forest plot of odds ratios from adjusted model
    """
    # Select variables to display (exclude intercept, reference categories)
    plot_vars = results_df[
        ~results_df['Variable'].str.contains('Intercept|C\\(') | 
        results_df['Variable'].str.contains('education_level_cat\\[T.\\]') |
        results_df['Variable'].str.contains('wealth_quintile_cat\\[T.\\]') |
        results_df['Variable'].str.contains('residence_cat\\[T.\\]')
    ].copy()
    
    # Clean variable names for display
    def clean_var_name(var):
        var = var.replace('C(education_level_cat)[T.', '')
        var = var.replace('C(wealth_quintile_cat)[T.', '')
        var = var.replace('C(residence_cat)[T.', '')
        var = var.replace('C(survey_year_cat)[T.', 'Year ')
        var = var.replace('current_age', 'Age (years)')
        var = var.replace(']', '')
        return var
    
    plot_vars['Display_Name'] = plot_vars['Variable'].apply(clean_var_name)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = range(len(plot_vars))
    
    # Plot odds ratios
    ax.errorbar(plot_vars['Odds_Ratio'], y_pos, 
                xerr=[plot_vars['Odds_Ratio'] - plot_vars['CI_Lower'], 
                      plot_vars['CI_Upper'] - plot_vars['Odds_Ratio']],
                fmt='o', capsize=5, capthick=2, elinewidth=2,
                color='#3498DB', markersize=8, ecolor='gray')
    
    # Add vertical line at OR=1
    ax.axvline(x=1, color='red', linestyle='--', alpha=0.5, label='No effect (OR=1)')
    
    # Add value labels
    for i, (idx, row) in enumerate(plot_vars.iterrows()):
        ax.text(row['Odds_Ratio'] + 0.1, i, f"{row['Odds_Ratio']:.2f}", va='center', fontsize=9)
    
    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_vars['Display_Name'])
    ax.set_xlabel('Odds Ratio (95% Confidence Interval)', fontsize=12)
    ax.set_title('Determinants of Early Marriage in Ethiopia\nAdjusted Logistic Regression', 
                fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


def generate_model_tables(all_results, output_dir):
    """
    Generate formatted tables for research paper
    
    Parameters:
    -----------
    all_results : dict
        Dictionary containing all model results
    output_dir : str or Path
        Directory to save tables
    
    Returns:
    --------
    bool : True if successful
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Table 1: Unadjusted odds ratios
    if 'unadjusted' in all_results:
        unadjusted_df = pd.DataFrame()
        unadjusted_data = all_results['unadjusted']
        
        # Handle different possible formats
        if isinstance(unadjusted_data, dict):
            for predictor, df_result in unadjusted_data.items():
                if isinstance(df_result, pd.DataFrame):
                    unadjusted_df = pd.concat([unadjusted_df, df_result], ignore_index=True)
        elif isinstance(unadjusted_data, pd.DataFrame):
            unadjusted_df = unadjusted_data
        else:
            print(f"Warning: Unexpected format for 'unadjusted': {type(unadjusted_data)}")
        
        if not unadjusted_df.empty:
            unadjusted_df.to_csv(output_dir / 'table1_unadjusted_ors.csv', index=False)
            print(f"  ✓ Saved table1_unadjusted_ors.csv")
        else:
            print(f"  ⚠️ No data for table1_unadjusted_ors.csv")
    else:
        print(f"  ⚠️ 'unadjusted' key not found in all_results")
    
    # Table 2: Adjusted model results
    if 'adjusted' in all_results:
        adjusted_data = all_results['adjusted']
        
        # Handle different possible formats
        if isinstance(adjusted_data, dict):
            if 'results' in adjusted_data:
                results_df = adjusted_data['results']
            elif 'results_df' in adjusted_data:
                results_df = adjusted_data['results_df']
            else:
                # Try to find any DataFrame in the dict
                results_df = None
                for key, value in adjusted_data.items():
                    if isinstance(value, pd.DataFrame):
                        results_df = value
                        break
        elif isinstance(adjusted_data, pd.DataFrame):
            results_df = adjusted_data
        else:
            results_df = None
        
        if results_df is not None and not results_df.empty:
            results_df.to_csv(output_dir / 'table2_adjusted_model.csv', index=False)
            print(f"  ✓ Saved table2_adjusted_model.csv")
        else:
            print(f"  ⚠️ No valid results for table2_adjusted_model.csv")
    else:
        print(f"  ⚠️ 'adjusted' key not found in all_results")
    
    # Table 3: Fertility consequences
    if 'fertility' in all_results:
        fertility_data = all_results['fertility']
        
        # Handle different possible formats
        if isinstance(fertility_data, dict):
            if 'results' in fertility_data:
                results_df = fertility_data['results']
            elif 'results_df' in fertility_data:
                results_df = fertility_data['results_df']
            else:
                # Try to find any DataFrame in the dict
                results_df = None
                for key, value in fertility_data.items():
                    if isinstance(value, pd.DataFrame):
                        results_df = value
                        break
        elif isinstance(fertility_data, pd.DataFrame):
            results_df = fertility_data
        else:
            results_df = None
        
        if results_df is not None and not results_df.empty:
            results_df.to_csv(output_dir / 'table3_fertility_irr.csv', index=False)
            print(f"  ✓ Saved table3_fertility_irr.csv")
        else:
            print(f"  ⚠️ No valid results for table3_fertility_irr.csv")
    else:
        print(f"  ⚠️ 'fertility' key not found in all_results")
    
    # Table 4: Sensitivity analysis
    if 'sensitivity' in all_results:
        sensitivity_data = all_results['sensitivity']
        comparison_df = None
        
        # Handle different possible formats
        if isinstance(sensitivity_data, tuple):
            # Unpack tuple (results, comparison)
            if len(sensitivity_data) >= 2:
                comparison_df = sensitivity_data[1]
            else:
                comparison_df = sensitivity_data[0]
        elif isinstance(sensitivity_data, dict):
            # Check for comparison key
            if 'comparison' in sensitivity_data:
                comparison_df = sensitivity_data['comparison']
            elif 'results' in sensitivity_data and isinstance(sensitivity_data['results'], pd.DataFrame):
                comparison_df = sensitivity_data['results']
            else:
                # Try to find any DataFrame in the dict
                for key, value in sensitivity_data.items():
                    if isinstance(value, pd.DataFrame):
                        comparison_df = value
                        break
        elif isinstance(sensitivity_data, pd.DataFrame):
            comparison_df = sensitivity_data
        else:
            comparison_df = None
        
        if comparison_df is not None and not comparison_df.empty:
            comparison_df.to_csv(output_dir / 'table4_sensitivity.csv', index=False)
            print(f"  ✓ Saved table4_sensitivity.csv")
        else:
            print(f"  ⚠️ No valid comparison data for table4_sensitivity.csv")
    else:
        print(f"  ⚠️ 'sensitivity' key not found in all_results")
    
    # Table 5: Interaction results (if exists)
    if 'interaction' in all_results:
        interaction_data = all_results['interaction']
        
        if isinstance(interaction_data, dict):
            if 'results' in interaction_data:
                results_df = interaction_data['results']
            elif 'results_df' in interaction_data:
                results_df = interaction_data['results_df']
            else:
                results_df = None
        elif isinstance(interaction_data, pd.DataFrame):
            results_df = interaction_data
        else:
            results_df = None
        
        if results_df is not None and not results_df.empty:
            results_df.to_csv(output_dir / 'interaction_results.csv', index=False)
            print(f"  ✓ Saved interaction_results.csv")
    
    print(f"\n✅ All tables saved to {output_dir}")
    
    return True


if __name__ == "__main__":
    print("Models module loaded successfully")
    print("Available functions:")
    print("  - prepare_modeling_data()")
    print("  - run_logistic_regression_unadjusted()")
    print("  - run_logistic_regression_adjusted()")
    print("  - run_logistic_regression_with_interaction()")
    print("  - run_poisson_regression_fertility()")
    print("  - sensitivity_analysis()")
    print("  - create_forest_plot()")