# Fast Classify Exoplanet

An interactive Streamlit web application for classifying exoplanet candidates from TESS Object of Interest (TOI) data.

This tool provides a simple, step-by-step user interface to walk through the entire machine learning workflow: from loading and visualizing data to training a model, running inference, and interpreting the results.

## Key Features

This demo shows the application running at 2× speed.
[![Demo (2× speed)](assets/demo_thumb.png)](assets/output_2x.mp4?raw=1)

*   **Interactive Web Interface**: A user-friendly, browser-based application for the entire classification process.
*   **Flexible Data Input**: Upload your own CSV data or use the sample TOI files included in the project.
*   **Data Exploration**: Interactively visualize data distributions and feature relationships with 1D and 2D plots.
*   **Guided Workflow**: Follow a clear, numbered sequence of steps:
    1.  Data Loading
    2.  Header Information Review
    3.  Train/Test Splitting
    4.  Data Visualization
    5.  Feature Selection
    6.  Data Filtering
    7.  Binary Target Creation
    8.  XGBoost Model Training with Cross-Validation
    9.  Inference on Test Data
    10. Model Interpretation with SHAP Plots
*   **Customizable Training**: Tune XGBoost hyperparameters directly in the UI before running the training.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/tungtylee/fast-classify-exoplanet.git
    cd fast-classify-exoplanet
    ```
2.  Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage (Web App)

To start the interactive web application, run the following command from the project's root directory:

```bash
streamlit run main.py
```

A new tab should open in your web browser. Simply follow the numbered steps in the application to proceed through the workflow.

## Advanced Usage (Command-Line)

For advanced users who prefer to work with command-line tools or wish to understand the underlying data processing scripts in more detail, comprehensive documentation is available.

Please refer to the **[data_TOI/README_TOI.md](data_TOI/README_TOI.md)** file for a complete guide to the individual Python scripts, their parameters, and example workflows.

## Data

The data used in this project is from the NASA TESS Object of Interest (TOI) catalog. Sample data files are included in the `data_TOI/data/` directory.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
