# Trial-by-trial Inter-areal Interactions in Visual Cortex

A comprehensive analysis pipeline for studying trial-by-trial inter-areal interactions in visual cortex in the presence or absence of visual stimulation. This project examines how neural activity in one brain area can predict activity in another area on single trials, with a focus on mouse V1 layers (L2/3 and L4) and macaque visual areas (V1 and V4), using ridge regression and generalized linear models.

## Project Overview

This project investigates trial-by-trial inter-areal interactions in visual cortex by predicting neuronal activity in one area based on activity in another, distinguishing between stimulus-driven and non-stimulus-driven shared variability. Key findings include:

- **Inter-laminar predictions**: Layer 4 activity predicts layer 2/3 activity (and vice versa) in mouse V1
- **Inter-cortical predictions**: V1 activity predicts V4 activity in macaques, with directional asymmetry
- **Stimulus independence**: Neuronal responses can be predicted even in the absence of visual stimulation
- **Bimodal distributions**: Some neurons are primarily visual, others show predictable spontaneous activity
- **Multiple factors**: Predictability depends on neuronal properties, receptive field overlap, and timing

### Data Sources
- **Mouse data**: Calcium imaging from Stringer et al. (2019) - V1 layers 2/3 and 4 responses to gratings/natural images
- **Macaque data**: 1024-channel electrophysiology from Chen et al. (2022) - V1 and V4 responses to various stimuli

## Data Access

This analysis uses two publicly available datasets:

### Mouse Data (Stringer et al., 2019)
- **Source**: https://figshare.com/articles/dataset/Recordings_of_ten_thousand_neurons_in_visual_cortex_in_response_to_2_800_natural_images/6845348
- **Content**: Calcium imaging from ~10,000 neurons in mouse V1 layers 2/3 and 4
- **Stimuli**: Drifting gratings, natural images, and spontaneous activity
- **Format**: MATLAB .mat files

### Macaque Data (Chen et al., 2022)  
- **Source**: https://gin.g-node.org/NIN/V1_V4_1024_electrode_resting_state_data
- **Content**: 1024-channel electrophysiology from macaque V1 and V4
- **Stimuli**: Checkerboard patterns, moving bars, gray screen, lights-off conditions
- **Format**: Neo/NIX files

**Note**: You will need to download these datasets separately and configure the paths in your analysis scripts.

## Project Structure

```
inter_areal_predictability/
├── fig_scripts/                    # Analysis and plotting pipeline
│   ├── fig2_code.py               # Figure 2 analysis: Lower level activity predicts higher level
│   ├── fig2_plot.ipynb            # Figure 2 plotting notebook
│   ├── fig3_code.py               # Figure 3 analysis: Asymmetrical inter-cortical predictions  
│   ├── fig3_plot.ipynb            # Figure 3 plotting notebook
│   ├── fig4_code.py               # Figure 4 analysis: Stimulus-dependent predictability
│   ├── fig4_plot.ipynb            # Figure 4 plotting notebook
│   ├── fig5_code.py               # Figure 5 analysis: Spontaneous activity predictions
│   ├── fig5_plot.ipynb            # Figure 5 plotting notebook
│   ├── fig6_code.py               # Figure 6 analysis: Neural properties impact on predictability
│   ├── fig6_plot.ipynb            # Figure 6 plotting notebook
│   ├── fig7_code.py               # Figure 7 analysis: Stimulus vs non-stimulus driven components
│   ├── fig7_plot.ipynb            # Figure 7 plotting notebook
│   ├── beh_contrib_code.py        # Behavioral contribution analysis
│   ├── beh_contrib_plot.ipynb     # Behavioral analysis plotting
│   ├── LFPs_timelags_code.py      # LFP time lag analysis
│   ├── LFPs_timelags_plot.ipynb   # Time lag plotting notebook
│   ├── timescale_difs_code.py     # Timescale differences analysis
│   ├── timescale_difs_plot.ipynb  # Timescale plotting notebook
│   ├── subsample_monkey_indices.py # Monkey data subsampling utilities
│   └── set_home_directory.py       # Project path configuration
├── utils/                          # Core analysis functions
│   ├── ridge_regression_functions.py     # Ridge regression pipeline
│   ├── glm_prediction_functions.py       # GLM prediction pipeline (Poisson GLMs)
│   ├── macaque_data_functions.py         # Macaque data loading and processing
│   ├── mouse_data_functions.py           # Mouse data loading and processing
│   ├── neuron_properties_functions.py    # Neural property analysis and statistics
│   ├── stats_functions.py                # Statistical testing (permutation tests, etc.)
│   ├── fig_2_functions.py                # Figure 2 specific utilities
│   ├── fig_3_functions.py                # Figure 3 specific utilities
│   ├── fig_4_functions.py                # Figure 4 specific utilities
│   ├── fig_5_functions.py                # Figure 5 specific utilities
│   ├── fig_6_functions.py                # Figure 6 specific utilities
│   ├── fig_7_functions.py                # Figure 7 specific utilities
│   └── beh_contrib_functions.py          # Behavioral analysis utilities
├── requirements.txt       # Python dependencies (note: may be outdated)
└── README.md             # This documentation
```

## Key Features

### Data Processing
- **Multi-species support**: Handles both mouse and macaque datasets
- **Flexible epoch extraction**: Configurable stimulus and spontaneous periods
- **Data preprocessing**: Standardization, filtering, and quality control
- **Multi-format support**: Neo, HDF5, MATLAB, and CSV data formats

### Predictive Modeling
- **Ridge Regression**: Multi-output ridge regression with cross-validation
- **Poisson GLMs**: Single-neuron Poisson generalized linear models
- **Cross-validation**: 10-fold CV with temporal frame reduction
- **Parallel processing**: Joblib-based parallelization across neurons
- **Hyperparameter optimization**: Automated alpha selection

### Statistical Analysis
- **Hierarchical testing**: Mouse/date-level permutation tests
- **Effect size calculation**: Explained variance and correlation metrics
- **Multiple comparisons**: Bonferroni and FDR corrections
- **Bootstrap confidence intervals**: Robust uncertainty estimation

### Visualization
- **Interactive notebooks**: Jupyter notebooks for each analysis figure
- **Publication-ready plots**: High-quality matplotlib/seaborn figures
- **Statistical annotations**: Automatic significance testing and plotting
- **Scalebars and formatting**: Professional scientific figure formatting

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd inter_areal_predictability
```

### 2. Set Up Environment

**Note**: The `requirements.txt` and `.yaml` files in this repository may be outdated. Install dependencies as needed based on import errors.

#### Key Dependencies
- **Core**: numpy, scipy, pandas, matplotlib, seaborn
- **Machine Learning**: scikit-learn, statsmodels, joblib
- **Data I/O**: neo, h5py, quantities
- **Visualization**: jupyter, matplotlib-scalebar

## Usage

### Basic Workflow

Each figure in the paper has a corresponding analysis and plotting pipeline:

1. **Run analysis**: Execute `fig_scripts/figX_code.py` to perform computations and generate data
2. **Generate plots**: Open and run `fig_scripts/figX_plot.ipynb` to create publication figures
3. **Utilities**: Each figure uses specific functions from `utils/figX_functions.py`

### Example: Running Predictability Analysis

```python
from fig_scripts.set_home_directory import get_project_root_homedir_in_sys_path
from utils.ridge_regression_functions import get_best_alpha_evars
from utils.mouse_data_functions import mt_retriever

# Set up project paths
project_root, main_dir = get_project_root_homedir_in_sys_path("inter_areal_predictability")

# Load mouse data
mt_ret = mt_retriever(main_dir, 'ori32')
mt_ret.set_raw_responses()

# Run predictability analysis
predictions, evars = get_best_alpha_evars(
    layer_to_use=source_area_responses,
    layer_to_predict=target_area_responses,
    n_splits=10,
    prediction_type='ridge'
)
```

### Example: Poisson GLM Analysis

```python
from utils.glm_prediction_functions import get_glm_predictions_evars_parallel

# Run Poisson GLM prediction
predictions, evars = get_glm_predictions_evars_parallel(
    layer_used=predictors,
    layer_to_predict=targets,
    alpha=1e2,
    n_splits=10
)
```

## Data Requirements

### Mouse Data (Stringer et al.)
- **Format**: MATLAB .mat files
- **Content**: Neural responses to orientation gratings and natural images
- **Structure**: Stimulus responses, spontaneous activity, cell metadata

### Macaque Data
- **Format**: Neo/NIX files, HDF5, MATLAB
- **Content**: Multi-electrode recordings from V1/V4
- **Paradigms**: Various visual stimulation protocols
- **Metadata**: Electrode positions, stimulus timing, trial information

## Analysis Pipeline

### 1. Data Preprocessing
```python
# Extract epochs and preprocess
epochs = get_epoch_times(resp_array, date, stim_on=0, stim_off=400)
responses = preprocess_responses(epochs, standardize=True)
```

### 2. Predictive Modeling
```python
# Ridge regression with CV
best_alpha, evars = get_best_alpha_evars(
    source_responses, target_responses,
    alphas=np.logspace(-2, 4, 20),
    n_splits=10
)
```

### 3. Statistical Testing
```python
# Hierarchical permutation test
p_value = hierarchical_permutation_test(
    data=results_df,
    mouse_or_date='mouse_id',
    independent_variable='brain_area',
    neuron_property='explained_variance'
)
```

## Key Functions

### Ridge Regression Pipeline
- `get_predictions_evars_parallel()`: Main prediction function
- `worker_function()`: Parallelized CV worker
- `get_best_alpha_evars()`: Hyperparameter optimization

### GLM Pipeline  
- `get_glm_predictions_evars_parallel()`: Poisson GLM predictions
- `glm_worker_function()`: Single-neuron GLM fitting
- `cross_validate_glm()`: GLM cross-validation

### Data Processing
- `get_epoch_times()`: Extract stimulus/spontaneous epochs
- `mt_retriever`: Mouse data loading class
- `preprocess_responses()`: Neural response preprocessing

### Statistical Analysis
- `hierarchical_permutation_test()`: Multi-level statistical testing
- `perm_test()`: Basic permutation testing
- `calculate_difference()`: Effect size calculation

## Output Files

Results are organized in `results/` directory:
- **Figure data**: Preprocessed data for each figure
- **Statistics**: P-values, effect sizes, confidence intervals
- **Predictions**: Model predictions and performance metrics
- **Plots**: Publication-ready figures in PDF/PNG format

## Figure-Specific Analysis Pipeline

Each figure represents a key finding from the paper:

- **Figure 2**: Basic inter-areal predictability (L4→L2/3 in mice, V1→V4 in macaques)
- **Figure 3**: Directional asymmetry in predictions (V1→V4 stronger than V4→V1)  
- **Figure 4**: Stimulus-type dependence of predictability
- **Figure 5**: Predictions during spontaneous activity (no visual input)
- **Figure 6**: Impact of neural properties and receptive field overlap
- **Figure 7**: Separating stimulus-driven vs. non-stimulus-driven components

### Running Individual Analyses

```bash
# Example: Run Figure 2 analysis
python fig_scripts/fig2_code.py

# Then open and run the corresponding notebook
jupyter notebook fig_scripts/fig2_plot.ipynb
```

## Contributing

When adding new analyses:
1. Create functions in appropriate `utils/` modules
2. Add analysis scripts to `fig_scripts/`
3. Include corresponding Jupyter notebooks for visualization
4. Update documentation and tests


## Citation

If you use this code in your research, please cite the associated publication:

```bibtex
@article{hidalgo2025trial,
  title={Trial-by-trial inter-areal interactions in visual cortex in the presence or absence of visual stimulation},
  author={Hidalgo, Dianna and Dellaferrera, Giorgia and Xiao, Will and Papadopouli, Maria and Smirnakis, Stelios and Kreiman, Gabriel},
  journal={eLife},
  volume={},
  pages={e105119},
  year={2025},
  publisher={eLife Sciences Publications Limited},
  doi={10.7554/eLife.105119}
}
```

**Paper**: [eLife Reviewed Preprint](https://elifesciences.org/reviewed-preprints/105119v1)  
**DOI**: https://doi.org/10.7554/eLife.105119

## Contact

For questions about the code or collaboration opportunities, please contact:
- **Dianna Hidalgo**: diannahidalgo@g.harvard.edu (Harvard Medical School)
- **Gabriel Kreiman**: gabriel.kreiman@tch.harvard.edu (Children's Hospital, Harvard Medical School)

## Troubleshooting

### Common Issues
1. **Path issues**: Ensure `set_home_directory.py` correctly finds the project root
2. **Data access**: Download datasets from the original sources (see Data Requirements section)  
3. **Memory issues**: Use `n_jobs=1` for debugging or reduce data size
4. **Convergence**: Increase `max_iter` for GLM models if needed

### Performance Tips

- Use parallel processing (`n_jobs=-1`) for large datasets
- Consider subsampling for initial exploration
- Cache intermediate results for repeated analyses
- Use `frames_reduced` parameter for temporal independence in CV
