# **Automated Regression Workflow for Interpretable Deflection Prediction in Bio-Inspired Laminated Composite Plates**

This repository contains the code and data associated with the research paper:

"**Automated Regression Workflow for Interpretable Deflection Prediction in Bio-Inspired Laminated Composite Plates**"
Shakti P. Padhy, Shubham Saurabh, Raj Kiran, Nhon Nguyen-Thanh

_This paper is currently under revision for publication in Frontiers of Structural and Civil Engineering._

## **Keywords**

Automated Regression Workflow; Interpretable Machine Learning; Bio-inspired laminated composites; Finite element analysis

## **Project Overview**

This project focuses on developing an Automated Regression Workflow (ARW) in Python to streamline the process of building, optimizing, and interpreting machine learning models for predicting the deflection of bio-inspired laminated composite plates. The workflow automates hyperparameter optimization, model training, and performance evaluation, making it highly efficient and reproducible.

## **Methodology**

The core of this project is the **Automated Regression Workflow (ARW)**, developed using Python. This workflow automates approximately 300 lines of detailed Python code for complete development, hyperparameter optimization, and evaluation of ML regression models into a concise 10-20 lines of user-defined code.

### **ARW Workflow in Brief:**

The Automated Regression Workflow (ARW) is implemented through the ```mlregworkflow.py``` script. Here's a brief overview of its key steps and how it's applied in this work:

1. **Data Preprocessing:**
    - The raw dataset (```Bioinspired composite_loading-25.csv```) is loaded using pandas.
    - Categorical features (```Loading Type```, ```Constraint```, ```Configuration```) are identified and converted into numerical format using one-hot encoding (```pd.get_dummies```).
    - Numerical features (```Number of Layer```, ```E1/E2```, ```a/h```) are separated from the target variable (```Deflection```).
    - All features (```X```) and the target (```y```) are then scaled using ```StandardScaler``` to normalize their ranges.
2. **Data Splitting:**
    - The preprocessed data is split into training and testing sets using ```train_test_split``` (80% for training, 20% for testing) to ensure unbiased model evaluation.
3. **Model Definition:**
    - Six different machine learning regression models are defined: Linear Regression (Linear), Support Vector Regression (SVR), Gradient Boosting Regression (GBR), Random Forest Regression (RFR), eXtreme Gradient Boosting Regression (XGBR), and Neural Network Regression (NNR).
    - For each model (except Linear Regression), a hyperparameter search space is defined using ```skopt.space.Real```, ```skopt.space.Integer```, and ```skopt.space.Categorical```. This includes parameters like ```C```, ```epsilon```, ```kernel``` for SVR; ```n_estimators```, ```loss```, ```learning_rate```, ```alpha```, ```max_depth``` for GBR; and ```hidden_layers```, ```units```, ```activation```, ```learning_rate``` for Neural Networks.
4. **Model Optimization and Evaluation:**
    - The ```optimize_model``` function (from ```mlregworkflow.py```) is the central component for this step.
    - It uses **Bayesian Optimization (BO)** via ```skopt.gp_minimize``` to systematically search for the optimal hyperparameters for each model, aiming to maximize the $R^2$ score.
    - A robust **5-fold cross-validation** (```KFold```) is integrated into the optimization process to ensure reliable and generalized model performance. For the Neural Network, a manual cross-validation loop (```cross_val_nn```) is implemented due to its specific training requirements.
    - After optimization, each model is trained on the full training set with its best parameters and evaluated on both the training and unseen testing data.
    - Performance metrics ($R^2$, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE)) are computed for both sets.
5. **Automated Visualizations and Results Output:**
    - For each model, diagnostic plots (predicted vs. true values) are automatically generated and saved to a ```plots``` directory.
    - A ```results.csv``` file is generated, summarizing the performance of all models, including their optimized hyperparameters and evaluation metrics.

### **Key Features of ARW:**

- **Bayesian Optimization (BO):** Utilizes ```Scikit-Optimize``` for systematic identification of optimal hyperparameters, maximizing the coefficient of determination ($R^2$).
- **Cross-Validation:** Integrates a robust 5-fold cross-validation process for reliable and generalized model performance.
- **Train-Test Split:** Employs an 80-20 train-test data split for unbiased assessment of predictive accuracy.
- **Automated Visualizations:** Generates diagnostic plots (predicted vs. true values) for training and testing datasets, stored in a ```plots``` directory.
- **Performance Metrics Output:** Outputs a ```results.csv``` file containing model names, optimized hyperparameters, and performance metrics ($R^2$, MAE, RMSE).

### **Predictive Regression Models**

Six different regression models were employed to predict deflection values:

- Linear Regression (Linear)
- Support Vector Regression (SVR)
- Gradient Boosting Regression (GBR)
- Random Forest Regression (RFR)
- eXtreme Gradient Boosting Regression (XGBR)
- Neural Network Regression (NNR)

Hyperparameter optimization was performed for all models (except Linear Regression) using the ARW.

### **Interpretability Analysis**

To understand the influence of individual features on deflection predictions, SHAP (SHapley Additive exPlanations) via the ```SHAP``` library and LIME (Local Interpretable Model-Agnostic Explanations) via the ```lime``` library were performed on the best-performing model.

## **Data**

The study utilizes a comprehensive dataset of 672 datapoints generated through finite element simulations. The input features include:

- **Layup Configurations:** Helicoidal Recursive (HR1, HR2, HR3), Helicoidal Exponential (HE1, HE2, HE3), Helicoidal Semicircular (HS1, HS2, HS3), Linear Helicoidal (LH1, LH2, LH3), Fibonacci Helicoidal (FH), and Quasi-Isotropic (QI).
- **Elasticity Ratio:** $E_1/E_2$ (10 and 40)
- **Loading Types:** Uniformly Distributed Load (UDL) and Sinusoidal Distributed Load (SSL)
- **Boundary Conditions:** Simply Supported (SSSS) or Clamped (CCCC)
- **Length-to-Thickness Ratio:** a/h (10 and 100)
- **Number of Layers:** 12, 16, and 20

Categorical variables (configurations, loading types, boundary conditions) were one-hot encoded.

## **Results**

- **Model Performance:** All models (except Linear Regression) achieved high fidelity ($R^2\\approx0.996–0.998$; $MAE\\approx0.010–0.015$; $RMSE\\approx0.014–0.021$).
- **Superior Model:** The **eXtreme Gradient Boosting Regression (XGBR)** model demonstrated superior performance, achieving the highest $R^2$ (0.999) and lowest MAE (0.010) and RMSE (0.013) on the test dataset.
- **Key Influential Factors (SHAP & LIME):**
  - **Boundary Conditions (Constraint_CCCC):** Most influential factor. Clamped boundary conditions significantly reduce predicted deflection.
  - **Ratio of Elastic Moduli (*$E_1/E_2$*):** Second most critical factor. Larger ratios (stiffer material distribution) lead to reduced deflection.
  - **Aspect Ratio (*a/h*) and Loading Type (Loading Type_SSL):** Showed moderate influence.
  - **Configuration Encodings and Number of Layers:** Exhibited minimal global impact on predictions.

These findings provide crucial insights for optimizing the design parameters of bio-inspired composite structures.

## **Installation and Usage**

To use the Automated Regression Workflow, follow these steps:

1. **Clone the repository:**  
    ```
    git clone https://github.com/Shakti-95/ARW-Interpretable-ML-Composite-Deflection.git
    cd ARW-Interpretable-ML-Composite-Deflection
    ```

2. Install dependencies:  
    It is recommended to use a virtual environment.  
    ```
    python -m venv venv  
    source venv/bin/activate # On Windows, use \`venv\\Scripts\\activate\`  
    pip install -r requirements.txt
    ```

3. Prepare your data:  
    Ensure your dataset is in a suitable format (e.g., CSV). The provided data should be used as input.

4. Run the workflow:  
    Place the ``mlregworkflow.py`` script (which contains the ``run_workflow`` function) in your working directory.  
    ```
    # Example usage in your main script (e.g., `main.py`)
    from mlregworkflow import run_workflow
    import pandas as pd
    from sklearn.svm import SVR
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from xgboost import XGBRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.linear_model import LinearRegression

    # Load your dataset
    # Replace 'your_data.csv' with the actual path to your dataset
    data = pd.read_csv('your_data.csv')

    # Define your features (X) and target (y)
    # Adjust column names as per your dataset
    X = data.drop('deflection_output_column', axis=1)
    y = data['deflection_output_column']
    
    # Define the models and their hyperparameter search spaces
    models_to_run = {
        "SVR": {
            "model": SVR(),
            "params": {
                "kernel": ["linear", "rbf"],
                "epsilon": [0.01, 1.0],
                "C": [0.1, 10.0]
            }
        },
        "GBR": {
            "model": GradientBoostingRegressor(),
            "params": {
                "n_estimators": [3, 10],
                "loss": ["squared_error", "absolute_error", "huber", "quantile"],
                "learning_rate": [0.005, 0.9],
                "alpha": [0.005, 0.9, prior="log-uniform"],
                "max_depth": [1, 7]
            }
        },
        "RFR": {
            "model": RandomForestRegressor(),
            "params": {
                "n_estimators": [3, 10],
                "max_depth": [3, 10]
            }
        },
        "XGBR": {
            "model": XGBRegressor(use_label_encoder=False, eval_metric='rmse'), # Add eval_metric for newer XGBoost versions
            "params": {
                "eta": [0.005, 1],
                "n_estimators": [1, 5], # Changed to match paper's search space
                "max_depth": [1, 7],     # Changed to match paper's search space
                "subsample": [0.1, 0.9],
                "colsample_bytree": [0.005, 0.9]
            }
        },
        "NeuralNetwork": {
            "model": MLPRegressor(max_iter=100, batch_size=32), # Fixed epochs and batch size as per paper
            "params": {
                "hidden_layer_sizes": [(32,), (64,), (128,), (32, 32), (64, 64), (128, 128)], # Example for hidden_layers and units
                "activation": ["relu", "tanh"],
                "learning_rate_init": [1e-6, 1e-2] # Corresponds to learning_rate in paper
            }
        },
        "Linear": { # Linear Regression does not require hyperparameter tuning
            "model": LinearRegression(),
            "params": {}
        }
    }
    
    # Run the automated workflow
    run_workflow(X, y, models_to_run)
    ```
    
## **Repository Structure (Proposed)**

```
.  
├── data/  
│ └── Bioinspired composite_loading-25.csv # The finite element generated dataset  
├── plots/ # Automatically generated plots by ARW  
├── models/ # To store trained neural network models (.h5 files)  
├── mlregworkflow.py # The Automated Regression Workflow script  
├── Bioinspired ML.ipynb # Jupyter Notebook with overall project code and analysis  
├── results.csv # Automatically generated results by ARW  
├── requirements.txt # Python dependencies  
└── README.md # This file  
└── LICENSE # MIT License file
```  

## **Contributing**

We welcome contributions to this project. Please feel free to fork the repository, make changes, and submit pull requests.

## **License**

MIT License

## **Contact**

For any questions or inquiries, please contact:

- Dr. Shakti Prasad Padhy: <padhy.shaktiprasad@gmail.com>, <shaktippadhy@tamu.edu>
- Dr. Raj Kiran: <raj@iitmandi.ac.in>
- Dr. Nhon Nguyen-Thanh: <nguyenthanhnhon@tdtu.edu.vn>
