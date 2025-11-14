import numpy as np
import pandas as pd
import scipy.stats as stats
from itertools import combinations
from statsmodels.stats.multitest import multipletests

def calculate_difference(group1, group2, central_tendency = 'median'):
    # Observed test statistic (e.g., difference in means)
    if central_tendency == 'mean':
        observed_statistic = np.nanmean(group1) - np.nanmean(group2)
    elif central_tendency =='median':
        observed_statistic = np.nanmedian(group1) - np.nanmedian(group2)
    return observed_statistic
def perm_test(group1, group2, num_permutations = 10000, central_tendency = 'median', return_stats = False):
    """Permutation test for independent samples.

    This function computes the p-value for a two-sample permutation test
    based on the difference in means between two independent groups.

    Args:
        group1 (array-like): Data for group 1.
        group2 (array-like): Data for group 2.

    Returns:
        float: p-value for the permutation test.
    """
    observed_statistic= calculate_difference(group1, group2, central_tendency)
    # Create an array to store the permuted test statistics
    permuted_statistics = np.zeros(num_permutations)
    # Combine the data from both groups
    combined_data = np.concatenate((group1, group2))
    # Perform the permutation test
    for i in range(num_permutations):
        # Randomly shuffle the combined data
        np.random.shuffle(combined_data)
    
        # Split the shuffled data back into two groups
        permuted_group1 = combined_data[:len(group1)]
        permuted_group2 = combined_data[len(group1):]
        
        # Calculate the test statistic for this permutation
        permuted_statistic = calculate_difference(permuted_group1, permuted_group2, central_tendency)
        # Store the permuted test statistic
        permuted_statistics[i] = permuted_statistic

    # Calculate the p-value by comparing the observed statistic to the permuted distribution
    p_value = (np.abs(permuted_statistics) >= np.abs(observed_statistic)).mean()

    if return_stats is True:
        return permuted_statistics, observed_statistic, p_value
    
    return p_value


def perm_test_paired(group1, group2):
    """Permutation test for paired samples.

    This function computes the p-value for a paired permutation test
    based on the difference in means between two paired groups.

    Args:
        group1 (array-like): Data for group 1.
        group2 (array-like): Data for group 2.

    Returns:
        float: p-value for the paired permutation test.
    """
    # Observed test statistic (e.g., difference in means)

    observed_statistic = np.nanmean(group2-group1)

    # Number of permutations to perform
    num_permutations = 10000

    # Create an array to store the permuted test statistics
    permuted_statistics = np.zeros(num_permutations)

    # Combine the differences
    pooled_differences = group2-group1
    
    # Perform the permutation test
    for i in range(num_permutations):
        # shuffle differences
        permuted_differences = pooled_differences * np.random.choice([-1, 1], size=len(pooled_differences))
        
        # Recalculate mean difference for the permuted dataset
        permuted_mean_difference = np.nanmean(permuted_differences)
    
        # Store the permuted mean difference
        permuted_statistics[i] = permuted_mean_difference

    # Calculate the p-value by comparing the observed statistic to the permuted distribution
    p_value = (np.abs(permuted_statistics) >= np.abs(observed_statistic)).mean()

    return p_value

# Function to perform hierarchical permutation test with animal bootstrapping
def hierarchical_permutation_test_old(data, mouse_or_date, independent_variable, neuron_property,perm_type='ind', num_permutations=1000, central_tendency = 'median'):
    observed_statistic = calculate_statistic(data, independent_variable, neuron_property, perm_type, central_tendency=central_tendency)  # Replace with your actual calculation
    """Hierarchical permutation test with animal bootstrapping.

    This function performs a hierarchical permutation test with animal bootstrapping.
    It calculates a statistic of interest for the observed data and then generates
    permuted datasets by bootstrapping animals. The p-value is computed by comparing
    the observed statistic to the distribution of permuted statistics.

    Args:
        data (pandas.DataFrame): Input DataFrame containing the data.
        mouse_or_date (str): Identifier for mouse or date.
        dependent_variable (str): Dependent variable.
        neuron_property (str): Neuron property.
        perm_type (str, optional): Type of permutation test ('ind' for independent or 'paired' for paired). Defaults to 'ind'.
        num_permutations (int, optional): Number of permutations. Defaults to 1000.

    Returns:
        float: p-value for the hierarchical permutation test.
    """
    # Create an empty array to store permuted statistics
    permuted_statistics = np.zeros(num_permutations)

    # Iterate through each permutation
    for i in range(num_permutations):
        # Bootstrap animals (resample entire animals with replacement)
        bootstrap_animals_or_dates = np.random.choice(data[mouse_or_date].unique(), size=len(data[mouse_or_date].unique()), replace=True)
        data2 = data[data[mouse_or_date].isin(bootstrap_animals_or_dates)]
        if 'mouse' in mouse_or_date.lower():
            bootstrapped_data = data[data[mouse_or_date].isin(bootstrap_animals_or_dates)]
            # min_cells_per_mouse = min(data[data[mouse_or_date].isin(bootstrap_animals_or_dates)].groupby(['Mouse',independent_variable])[neuron_property].count())
            # bootstrapped_data = pd.concat([group_.sample(min_cells_per_mouse, replace=False) for _, group_ in data2.groupby(['Mouse',independent_variable])])
        else:
            min_cells_per_date = min(data[data[mouse_or_date].isin(bootstrap_animals_or_dates)].groupby([mouse_or_date,independent_variable])[neuron_property].count())
            bootstrapped_data = pd.concat([group_.sample(min_cells_per_date, replace=False) for _, group_ in data2.groupby([mouse_or_date,independent_variable])])

        if perm_type =='ind':
            # Permute values within each bootstrapped animal
            for animal in bootstrapped_data[mouse_or_date].unique():
                animal_values = bootstrapped_data.loc[bootstrapped_data[mouse_or_date] == animal, neuron_property].values
                np.random.shuffle(animal_values)
                bootstrapped_data.loc[bootstrapped_data[mouse_or_date] == animal, neuron_property] = animal_values
            # Calculate the permuted statistic
            permuted_statistic = calculate_statistic(bootstrapped_data, independent_variable, neuron_property, perm_type=perm_type, central_tendency=central_tendency)
        elif perm_type =='paired':
            permuted_statistic = calculate_statistic(bootstrapped_data, independent_variable, neuron_property, perm_type=perm_type, paired_shuffle=True)
        # Store the permuted statistic
        permuted_statistics[i] = permuted_statistic
    # Calculate the p-value
    p_value = np.mean(np.abs(permuted_statistics) >= np.abs(observed_statistic))
    return p_value



def hierarchical_permutation_test_gpt1(
    data, mouse_or_date, independent_variable, neuron_property,
    perm_type='ind', num_permutations=1000, central_tendency='median',
    random_state=None
):
    print('hierarchical permutation test with bootstrapping')
    rng = np.random.default_rng(random_state)

    # keep only mice/dates that have ≥1 neuron in each condition
    counts = (data.groupby([mouse_or_date, independent_variable])[neuron_property]
              .size().unstack(fill_value=0))
    # require exactly two conditions
    if counts.shape[1] != 2:
        raise ValueError(f"Expected exactly 2 levels in {independent_variable}, found {counts.shape[1]}.")
    complete_units = counts[(counts > 0).all(axis=1)].index
    if len(complete_units) < 3:
        return np.nan

    data = data[data[mouse_or_date].isin(complete_units)].copy()

    def agg(x):
        return np.nanmean(x) if central_tendency == 'mean' else np.nanmedian(x)

    # ---- observed statistic (mouse/date-level) ----
    if perm_type == 'ind':
        pm = (data.groupby([mouse_or_date, independent_variable])[neuron_property]
                .apply(agg).unstack())
        # difference in a fixed order (sign only matters if you care about direction)
        observed_statistic = np.nanmean(pm.iloc[:, 0] - pm.iloc[:, 1])
    elif perm_type == 'paired':
        pm = (data.groupby([mouse_or_date, independent_variable])[neuron_property]
                .apply(agg).unstack())
        diffs = pm.iloc[:, 0] - pm.iloc[:, 1]
        observed_statistic = np.nanmean(diffs)
    else:
        raise ValueError("perm_type must be 'ind' or 'paired'.")

    permuted_statistics = np.zeros(num_permutations, dtype=float)
    complete_units = np.array(complete_units)

    for i in range(num_permutations):
        # bootstrap mice/dates with replacement (preserve duplicates)
        sampled = rng.choice(complete_units, size=len(complete_units), replace=True)
        boot = pd.concat([data.loc[data[mouse_or_date] == u] for u in sampled], ignore_index=True)

        if perm_type == 'ind':
            # shuffle CONDITION LABELS within each mouse/date (NOT the values)
            for u in boot[mouse_or_date].unique():
                idx = boot[mouse_or_date] == u
                labels = boot.loc[idx, independent_variable].values
                boot.loc[idx, independent_variable] = rng.permutation(labels)

            pm = (boot.groupby([mouse_or_date, independent_variable])[neuron_property]
                    .apply(agg).unstack())
            perm_stat = np.nanmean(pm.iloc[:, 0] - pm.iloc[:, 1])

        else:  # paired
            pm = (boot.groupby([mouse_or_date, independent_variable])[neuron_property]
                    .apply(agg).unstack())
            pm = pm.dropna()
            if pm.empty:
                perm_stat = np.nan
            else:
                diffs = pm.iloc[:, 0] - pm.iloc[:, 1]
                # sign-flip the per-mouse differences
                signs = rng.choice([-1, 1], size=len(diffs))
                perm_stat = np.nanmean(diffs * signs)

        permuted_statistics[i] = perm_stat

    permuted_statistics = permuted_statistics[~np.isnan(permuted_statistics)]
    if permuted_statistics.size == 0:
        return np.nan

    p_value = (np.abs(permuted_statistics) >= np.abs(observed_statistic)).mean()
    return p_value

def _agg_mouse_long(df, mouse_or_date, independent_variable, neuron_property, central_tendency):
    """Return long df aggregated at mouse/date × condition."""
    if central_tendency == 'mean':
        agg = df.groupby([mouse_or_date, independent_variable], observed=True)[neuron_property].mean()
    else:
        agg = df.groupby([mouse_or_date, independent_variable], observed=True)[neuron_property].median()
    out = agg.reset_index()
    # ensure deterministic order by mouse/date then condition
    return out.sort_values([mouse_or_date, independent_variable]).reset_index(drop=True)

def _ensure_two_levels(df, mouse_or_date, independent_variable, neuron_property):
    # FIX: drop unused categorical levels so extra zero-count columns don’t trigger the 2-level check
    if hasattr(df[independent_variable], "cat"):
        df[independent_variable] = df[independent_variable].cat.remove_unused_categories()

    counts = (df.groupby([mouse_or_date, independent_variable], observed=True)[neuron_property]
                .size().unstack(fill_value=0))
    counts = counts.loc[:, counts.sum(0) > 0]  # FIX: drop all-zero columns
    if counts.shape[1] != 2:
        raise ValueError(f"Need exactly 2 non-empty levels in {independent_variable}, got {counts.shape[1]}.")


    complete = counts[(counts > 0).all(axis=1)].index
    return df[df[mouse_or_date].isin(complete)].copy(), list(counts.columns)

def hierarchical_permutation_test(
    data, mouse_or_date, independent_variable, neuron_property,
    perm_type='ind', num_permutations=1000, central_tendency='median', random_state=None
):
    rng = np.random.default_rng(random_state)

    # keep only mice/dates that have ≥1 neuron in each condition
    data, cond_levels = _ensure_two_levels(data, mouse_or_date, independent_variable, neuron_property)
    if data[mouse_or_date].nunique() < 3:
        return np.nan  # too few clusters

    # ---------- Observed statistic (mouse-level) ----------
    obs_long = _agg_mouse_long(data, mouse_or_date, independent_variable, neuron_property, central_tendency)

    if perm_type == 'ind':
        # your function will compute the difference of the two condition-distributions across mice
        observed_statistic = calculate_statistic(
            obs_long, group=independent_variable, neuron_property=neuron_property,
            perm_type='ind', central_tendency=central_tendency
        )
    elif perm_type == 'paired':
        # sort so the two condition arrays align by mouse/date inside calculate_statistic
        obs_long = obs_long.sort_values([mouse_or_date, independent_variable]).reset_index(drop=True)
        observed_statistic = calculate_statistic(
            obs_long, group=independent_variable, neuron_property=neuron_property,
            perm_type='paired', paired_shuffle=False, central_tendency=central_tendency
        )
    else:
        raise ValueError("perm_type must be 'ind' or 'paired'.")

    # ---------- Permutation null with true cluster bootstrap ----------
    permuted_statistics = np.zeros(num_permutations, dtype=float)
    units = data[mouse_or_date].unique()

    for i in range(num_permutations):
        # bootstrap mice/dates with replacement (duplicates preserved)
        sampled = rng.choice(units, size=len(units), replace=True)
        boot = pd.concat([data.loc[data[mouse_or_date] == u] for u in sampled], ignore_index=True)

        if perm_type == 'ind':
            # permute CONDITION LABELS within each mouse/date (NOT the values)
            for u in boot[mouse_or_date].unique():
                idx = boot[mouse_or_date] == u
                boot.loc[idx, independent_variable] = rng.permutation(
                    boot.loc[idx, independent_variable].values
                )
            boot_long = _agg_mouse_long(boot, mouse_or_date, independent_variable, neuron_property, central_tendency)
            perm_stat = calculate_statistic(
                boot_long, group=independent_variable, neuron_property=neuron_property,
                perm_type='ind', central_tendency=central_tendency
            )

        else:  # paired
            # aggregate to mouse/date × condition and align rows by mouse/date
            boot_long = _agg_mouse_long(boot, mouse_or_date, independent_variable, neuron_property, central_tendency)
            boot_long = boot_long.sort_values([mouse_or_date, independent_variable]).reset_index(drop=True)
            # sign-flip null via your calculate_statistic's paired_shuffle=True
            perm_stat = calculate_statistic(
                boot_long, group=independent_variable, neuron_property=neuron_property,
                perm_type='paired', paired_shuffle=True, central_tendency=central_tendency
            )

        permuted_statistics[i] = perm_stat

    # guard against rare NaNs (shouldn't really happen after prefiltering)
    permuted_statistics = permuted_statistics[~np.isnan(permuted_statistics)]
    if permuted_statistics.size == 0:
        return np.nan

    p_value = (np.abs(permuted_statistics) >= np.abs(observed_statistic)).mean()
    return p_value

def get_group_pairwise_stats(df,variable1,variable1_order, neuron_property,perm_t=False, perm_type='ind', central_tendency='median', hierarchical=False, sort_keys=False):
    """
    Perform pairwise post-hoc tests between groups based on a given variable.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    variable1 : str
        The name of the column representing the independent variable.
    variable1_order : list
        The order of groups for the independent variable.
    neuron_property : str
        The name of the column representing the dependent variable (property of neurons).
    perm_t : bool, optional
        Whether to perform permutation tests instead of traditional t-tests (default is False).
    perm_type : {'ind', 'paired'}, optional
        Type of permutation test to perform. 'ind' for independent samples permutation test, 'paired' for paired samples permutation test (default is 'ind').
    Returns:
    --------
    p_val_names : list
        List of names for pairwise comparisons.
    adjusted_p_values : list
        List of adjusted p-values for pairwise comparisons after applying the Benjamini-Hochberg correction.

    Notes:
    ------
    This function performs one-way ANOVA and subsequent pairwise post-hoc tests between groups based on the given independent variable.
    It calculates either traditional t-tests or permutation tests depending on the specified parameters.
    """
    df_posthoc = df.copy() #lets not drop the na because sometimes i have irrelevant columns with na that mess things up
    # Perform pairwise t-tests with Benjamini-Hochberg correction
    groups = variable1_order
    p_values = []
    p_val_names = []

    for group1, group2 in combinations(groups, 2):
        if sort_keys is not None:
            df_posthoc = df_posthoc.sort_values(by=sort_keys)
        group1_data = df_posthoc[df_posthoc[f'{variable1}'] == group1][neuron_property].reset_index(drop=True)
        group2_data = df_posthoc[df_posthoc[f'{variable1}'] == group2][neuron_property].reset_index(drop=True)
        print(f'comparing {group1} vs {group2}, n={len(group1_data)} vs n={len(group2_data)}')
        if sort_keys is not None:
            print('neuron indices per group:')
            print(np.concatenate(df_posthoc[df_posthoc[f'{variable1}'] == group1][sort_keys].values[:6]))
            print(np.concatenate(df_posthoc[df_posthoc[f'{variable1}'] == group2][sort_keys].values[:6]))
        if hierarchical is True:
            # perform hierarchical permutation test here using group1_data and group2_data
            # minimal change: build pairwise slice with hierarchy column
            df_pair = df_posthoc[df_posthoc[variable1].isin([group1, group2])][
                ['Mouse Name', variable1, neuron_property]
            ].copy()

            p_value = hierarchical_permutation_test(
                data=df_pair,
                mouse_or_date='Mouse Name',
                independent_variable=variable1,
                neuron_property=neuron_property,
                perm_type=perm_type,
                num_permutations=1000,
                central_tendency=central_tendency,
            )
        elif perm_type == 'paired':
            p_value = perm_test_paired(group1_data, group2_data)
            print('paired test')
            print(f'group1 mean: {np.nanmean(group1_data)}, group2 mean: {np.nanmean(group2_data)}')
            print(f'p_value: {p_value}')
            print('\n')
        elif perm_t is True:
            p_value = perm_test(group1_data, group2_data, central_tendency=central_tendency)
        else:
            _, p_value = stats.ttest_ind(group1_data, group2_data)
        p_values.append(p_value)
        p_val_names.append(group1 + '_' + group2)

    # Apply Benjamini-Hochberg correction
    adjusted_p_values = multipletests(p_values, method='fdr_bh')[1]

    return p_val_names, adjusted_p_values

def get_group_pairwise_stars(df_, dependent_variable,dependent_variable_order, neuron_property, perm_t=True, 
                            perm_type='ind', central_tendency='median', hierarchical=False, return_pvals=False, sort_keys=None):
    """
    Perform one-way ANOVA and generate significance stars for pairwise comparisons.

    Parameters:
    -----------
    df_ : pandas.DataFrame
        The DataFrame containing the data.
    dependent_variable : str
        The name of the column representing the independent variable.
    dependent_variable_order : list
        The order of groups for the independent variable.
    neuron_property : str
        The name of the column representing the dependent variable (property of neurons).
    perm_t : bool, optional
        Whether to perform permutation tests instead of traditional t-tests (default is True).
    perm_type : {'ind', 'paired'}, optional
        Type of permutation test to perform. 'ind' for independent samples permutation test, 'paired' for paired samples permutation test (default is 'ind').
    
    Returns:
    --------
    p_val_names : list
        List of names for pairwise comparisons.
    all_stars : list
        List of significance stars indicating the level of significance for each pairwise comparison.

    Notes:
    ------
    This function performs one-way ANOVA and generates significance stars based on the adjusted p-values for pairwise comparisons.
    The level of significance is indicated by the number of stars: *** for p < 0.001, ** for p < 0.01, * for p < 0.05, and 'n.s.' for not significant.
    """
    
    all_stars = []
    p_val_names, adjusted_p_values= get_group_pairwise_stats(df_,dependent_variable,dependent_variable_order, 
                                                            neuron_property,perm_t=perm_t, perm_type=perm_type, central_tendency=central_tendency,
                                                            hierarchical=hierarchical, sort_keys=sort_keys)
    for name, p_value in zip(p_val_names, adjusted_p_values):
        print(name, p_value)
        if p_value <1e-3:
            stars = '***'
        elif p_value <1e-2:
            stars = '**'
        elif p_value <0.05:
            stars='*'
        else:
            stars='n.s.'
            
        all_stars.append(stars)
    if return_pvals is True:
        return p_val_names,all_stars, adjusted_p_values
    return p_val_names, all_stars

# Example function for the statistic of interest
def calculate_statistic(data, group, neuron_property, perm_type='ind', paired_shuffle=False, central_tendency='median'):
    """Calculate the statistic of interest.

    This function calculates the statistic of interest based on the input data.

    Args:
        data (pandas.DataFrame): Input DataFrame containing the data.
        group (str): Group identifier.
        neuron_property (str): Neuron property.
        perm_type (str, optional): Type of permutation test ('ind' for independent or 'paired' for paired). Defaults to 'ind'.
        paired_shuffle (bool, optional): Whether to perform paired shuffling. Defaults to False.

    Returns:
        float: Calculated statistic.
    """
    groups = data[group].unique()
    
    if len(groups) <2:
        print(groups)
        print(data['Mouse'].unique())
    if perm_type =='ind':
        if central_tendency =='mean':
            mean_group_a = data[data[group] == groups[0]][neuron_property].mean()
            mean_group_b = data[data[group] == groups[1]][neuron_property].mean()
        elif central_tendency == 'median':
            mean_group_a = data[data[group] == groups[0]][neuron_property].median()
            mean_group_b = data[data[group] == groups[1]][neuron_property].median()
        return mean_group_a - mean_group_b
    elif perm_type =='paired':
        if data[data[group] == groups[0]][neuron_property].size != data[data[group] == groups[1]][neuron_property].size:
            print('sizes are not the same, you should not used a paired permutation test here')
            print(data[data[group] == groups[0]][neuron_property].size,data[data[group] == groups[1]][neuron_property].size)
        pooled_differences = data[data[group] == groups[0]][neuron_property].values-data[data[group] == groups[1]][neuron_property].values
        if paired_shuffle is True:
            permuted_differences = pooled_differences * np.random.choice([-1, 1], size=len(pooled_differences))
            # Recalculate mean difference for the permuted dataset
            return np.nanmean(permuted_differences)
        else:
            return np.nanmean(pooled_differences)
        
def get_comparison_test_stars(df_, independent_variable, neuron_property, print_pval=False, 
                    perm_t=True, perm_type='ind', hierarchical=False, num_permutations=10000, 
                    mouse_or_date='Mouse_Name', central_tendency='median', return_pval=False):
    """Perform t-test and return significance stars.

    This function conducts a t-test between two groups defined by a dependent variable,
    computes the p-value, and assigns significance stars based on the p-value.

    Args:
        df_ (pandas.DataFrame): Input DataFrame containing the data.
        dependent_variable (str): Dependent variable defining the groups.
        neuron_property (str): Neuron property to compare between groups.
        print_pval (bool, optional): Whether to print the p-value. Defaults to False.
        perm_t (bool, optional): Whether to perform a permutation test. Defaults to True.
        perm_type (str, optional): Type of permutation test ('ind' for independent or 'paired' for paired). Defaults to 'ind'.
        hierarchical (bool, optional): Whether to perform hierarchical permutation test. Defaults to False.
        num_permutations (int, optional): Number of permutations. Defaults to 1000.

    Returns:
        str: Significance stars indicating the level of significance.
    """
    variables = df_[independent_variable].unique()
    group_1 =df_[df_[independent_variable]==variables[0]][neuron_property].dropna().values
    group_2 =df_[df_[independent_variable]==variables[1]][neuron_property].dropna().values
    
    if hierarchical is True:
        p_value = hierarchical_permutation_test(df_,mouse_or_date=mouse_or_date, 
                                        independent_variable=independent_variable, 
                                        neuron_property=neuron_property,
                                        perm_type=perm_type,num_permutations=num_permutations,
                                        central_tendency=central_tendency)
        
    elif perm_type=='paired':
        p_value = perm_test_paired(group_1, group_2)
        print('paired test')
    elif perm_t is True:
        p_value = perm_test(group_1, group_2)
    elif perm_type=='ind':
        _, p_value = stats.ttest_ind(group_1, group_2, equal_var=False)
    else:
        print('perm_type must be either ind or paired')
        return np.nan
    if p_value <1e-3:
        stars = '***'
    elif p_value <1e-2:
        stars = '**'
    elif p_value <0.05:
        stars='*'
    else:
        stars='n.s.'
        
    if print_pval is True:
        print(p_value)
    if return_pval is True:
        return p_value, stars
    return stars

