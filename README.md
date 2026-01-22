# Machine Learning Projects

A comprehensive collection of end-to-end machine learning projects demonstrating the complete ML lifecycle: from problem definition through data collection, model training, evaluation, and predictions.

## Project Structure

Each project is a self-contained Jupyter notebook that walks through all 6 phases of the ML lifecycle:

1. **Phase 1: Decide on the Question** - Define the problem and business context
2. **Phase 2: Collect and Prepare Data** - Load, explore, and preprocess data
3. **Phase 3: Choose a Training Method** - Select appropriate algorithm with justification
4. **Phase 4: Train the Model** - Fit the model to training data
5. **Phase 5: Evaluate the Model** - Assess performance with relevant metrics
6. **Phase 6: Predict and Visualize** - Make predictions and interpret results

## Naming Convention

Files follow this pattern: `XX-category-algorithm-problem.ipynb`

- `XX` = Sequential project number (01, 02, 03, etc.)
- `category` = ML category (regression, classification, clustering, nlp, etc.)
- `algorithm` = Specific algorithm used
- `problem` = Brief description of the prediction task

### Examples
- `01-regression-linear-regression-predict-diabetes-from-bmi.ipynb`
- `02-classification-logistic-regression-iris-flower-classification.ipynb`
- `03-clustering-kmeans-customer-segmentation.ipynb`

## Projects

### Regression
- **01-regression-linear-regression-predict-diabetes-from-bmi** - Linear regression modeling diabetes disease progression from BMI using the scikit-learn diabetes dataset (442 patients, 10 features).

## Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab
- Required packages: `pandas`, `numpy`, `scikit-learn`, `plotly`, `matplotlib`

### Installation

```bash
git clone https://github.com/yourusername/machine-learning-projects.git
cd machine-learning-projects
pip install -r requirements.txt
```

### Running a Project

```bash
jupyter notebook 01-regression-linear-regression-predict-diabetes-from-bmi.ipynb
```

## Features

✨ **Each notebook includes:**
- Clear problem statement and clinical/business motivation
- Exploratory Data Analysis with interactive visualizations
- Feature engineering and data preprocessing
- Model training with proper train-test splitting
- Comprehensive evaluation metrics with interpretations
- Interactive regression/prediction visualizations using Plotly
- Professional summary with next steps for improvement

## Technologies

- **Data Processing**: Pandas, NumPy
- **Machine Learning**: scikit-learn
- **Visualization**: Plotly Express, Matplotlib
- **Notebooks**: Jupyter

## Portfolio Quality

These projects are designed to be portfolio-ready, demonstrating:
- Complete ML workflows from problem to prediction
- Professional code organization and commenting
- Clear data storytelling and visualization
- Proper evaluation and interpretation of results
- Scalable structure for expanding to more projects

## Contributing

Feel free to fork this repo and add your own ML projects following the same structure and naming convention!

## License

MIT License - feel free to use these projects for learning and portfolio building.

---

**Author**: Chrispine Oluoch  
**Last Updated**: January 2026
