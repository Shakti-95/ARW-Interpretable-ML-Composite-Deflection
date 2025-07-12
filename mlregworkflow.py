import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from skopt import gp_minimize
from sklearn.model_selection import train_test_split,cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor

# Global scalers for the entire workflow
x_scaler = StandardScaler()
y_scaler = StandardScaler()

# Function to compute evaluation metrics for regression models
def compute_metrics(y_true, y_pred):
    """
    Compute and return R², MAE, and RMSE for the given true and predicted values.

    Parameters:
    - y_true: Array of true target values.
    - y_pred: Array of predicted target values.

    Returns:
    - r2: R-squared score.
    - mae: Mean Absolute Error.
    - rmse: Root Mean Squared Error.
    """
    r2 = round(r2_score(y_true, y_pred),3)
    mae = round(mean_absolute_error(y_true, y_pred),3)
    rmse = round(np.sqrt(mean_squared_error(y_true, y_pred)),3)
    
    return r2, mae, rmse

# Function to create and compile a Keras neural network
def create_nn(hidden_layers=1, units=64, activation='relu', learning_rate=0.001):
    """
    Create and compile a Keras neural network model.

    Parameters:
    - hidden_layers: Number of hidden layers in the network.
    - units: Number of units (neurons) per hidden layer.
    - activation: Activation function for the hidden layers.
    - learning_rate: Learning rate for the Adam optimizer.

    Returns:
    - model: Compiled Keras model.
    """
    input_dim = x_scaler.n_features_in_
    output_dim = y_scaler.n_features_in_
    
    model = Sequential()
    # Input layer
    model.add(Dense(units, activation=activation, input_dim=input_dim))
    # Hidden layers
    for _ in range(hidden_layers - 1):
        model.add(Dense(units, activation=activation))
    # Output layer with linear activation
    model.add(Dense(output_dim, activation='linear'))
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

# Function to manually handle neural network cross-validation
def cross_val_nn(X, y, build_fn, params, cv=5, epochs=100, batch_size=32):
    """
    Perform manual cross-validation for a Keras neural network.

    Parameters:
    - X: Feature matrix.
    - y: Target matrix.
    - build_fn: Function to build the neural network model.
    - params: Hyperparameters for building the neural network.
    - cv: Number of folds for cross-validation.
    - epochs: Number of training epochs.
    - batch_size: Size of each training batch.

    Returns:
    - mean_score: Mean R² score across all folds.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Build and train the model
        model = build_fn(**params)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        # Evaluate on the validation set
        y_pred_val = model.predict(X_val)
        scores.append(r2_score(y_val, y_pred_val))

    return np.mean(scores)

# Function to optimize model hyperparameters using Bayesian optimization
def optimize_model(model, param_space, X_train, y_train, X_test, y_test, is_nn=False, model_save_folder="models"):
    """
    Optimize a model's hyperparameters using Bayesian optimization (gp_minimize) 
    and evaluate its performance on the training and testing data.

    Parameters:
    - model: Scikit-learn or Keras model to optimize.
    - param_space: Hyperparameter search space for optimization.
    - X_train, y_train: Training data and target values.
    - X_test, y_test: Testing data and target values.
    - is_nn: Boolean flag to indicate if the model is a neural network.
    - model_save_folder: Directory to save the neural network model.

    Returns:
    - best_params: Dictionary of the best hyperparameters found.
    - train_metrics: Tuple (R², MAE, RMSE) for training data.
    - test_metrics: Tuple (R², MAE, RMSE) for testing data.
    - y_pred_train: Predicted values for training data.
    - y_pred_test: Predicted values for testing data.
    """
    # Scale the data using the global scalers
    X_train_scaled = x_scaler.fit_transform(X_train)
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1) if len(y_train.shape) == 1 else y_train)
    X_test_scaled = x_scaler.transform(X_test)
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1) if len(y_test.shape) == 1 else y_test)
    
    if not param_space:  # No hyperparameters to optimize
        # Directly train and evaluate the model
        model.fit(X_train_scaled, y_train_scaled)
        y_pred_train = y_scaler.inverse_transform(model.predict(X_train_scaled).reshape(-1, 1)).ravel()
        y_pred_test = y_scaler.inverse_transform(model.predict(X_test_scaled).reshape(-1, 1)).ravel()

        # Compute evaluation metrics
        train_metrics = compute_metrics(y_train, y_pred_train)
        test_metrics = compute_metrics(y_test, y_pred_test)

        return {}, train_metrics, test_metrics, y_pred_train, y_pred_test

    # Convert the parameter space into dimensions for Bayesian optimization
    dimensions = [value for value in param_space.values()]
    param_names = list(param_space.keys())
    
    # Define the objective function for optimization with cross-validation
    def objective(params):
        if is_nn:  
            # Neural network-specific cross-validation
            nn_params = {
                'hidden_layers': params[0],
                'units': params[1],
                'activation': params[2],
                'learning_rate': params[3]
            }
            return -cross_val_nn(X_train_scaled, y_train_scaled, create_nn, nn_params, cv=5, epochs=100, batch_size=32)
        else:  # Scikit-learn model handling
            model.set_params(**dict(zip(param_names, params)))
            cv_scores = cross_val_score(model, X_train_scaled, y_train_scaled, cv=5, scoring="r2", n_jobs=-1)
            return -np.mean(cv_scores)  # Negative R² for minimization

    # Perform Bayesian optimization
    results = gp_minimize(objective, dimensions=dimensions, n_calls=100, n_jobs=-1, random_state=42)

    # Extract the best parameters and train the model
    if is_nn:
        best_params = {
            'hidden_layers': results.x[0],
            'units': results.x[1],
            'activation': results.x[2],
            'learning_rate': results.x[3],
        }
        model = create_nn(**best_params)
        model.fit(X_train_scaled, y_train_scaled, epochs=100, batch_size=32, verbose=0)
        model.save(os.path.join(model_save_folder, f"nn_model_best.h5"))  # Save the model
    else:
        best_params = dict(zip(param_names, results.x))
        model.set_params(**best_params)
        model.fit(X_train_scaled, y_train_scaled)
    
    # Predictions
    y_pred_train = y_scaler.inverse_transform(model.predict(X_train_scaled).reshape(-1, 1))
    y_pred_test = y_scaler.inverse_transform(model.predict(X_test_scaled).reshape(-1, 1))
    
    # Compute evaluation metrics
    train_metrics = compute_metrics(y_train, y_pred_train)
    test_metrics = compute_metrics(y_test, y_pred_test)

    return best_params, train_metrics, test_metrics, y_pred_train, y_pred_test

# Function to create y-y plots for visualizing model predictions
def plot_yy(y_train, y_pred_train, y_test, y_pred_test, model_name, train_metrics, test_metrics, save_folder="plots"):
    """
    Create and display y-y plots (predicted vs actual values) for training and testing data.

    Parameters:
    - y_train: True target values for training data.
    - y_pred_train: Predicted values for training data.
    - y_test: True target values for testing data.
    - y_pred_test: Predicted values for testing data.
    - model_name: Name of the model (used in plot title).
    - train_metrics: Tuple containing RMSE, MAE, and R² for training data.
    - test_metrics: Tuple containing RMSE, MAE, and R² for testing data.
    - save_folder: Folder to save the plots (default is 'plots').
    """
    # Ensure the save directory exists
    os.makedirs(save_folder, exist_ok=True)

    plt.figure(figsize=(14, 6))
    # Create subplots for training and testing data
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    
    # Scatter plot for training data
    ax1.scatter(y_train, y_pred_train, c='b', alpha=0.7)
    ax1.plot(
        [min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())],
        [min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())],
        linestyle="--", color="red", label="Ideal"
    )
    ax1.set_xlabel("True Values", fontsize=18)
    ax1.set_ylabel("Predicted Values", fontsize=18)
    ax1.set_title(f"{model_name} Y-Y Plot - Train"
                  + '\nRMSE: ' + str(round(train_metrics[2], 3))
                  + '\nMAE: ' + str(round(train_metrics[1], 3))
                  + '\nR2: ' + str(round(train_metrics[0], 3)), fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    
    # Scatter plot for testing data
    ax2.scatter(y_test, y_pred_test, c='g', alpha=0.7)
    ax2.plot(
        [min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())],
        [min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())],
        linestyle="--", color="red", label="Ideal"
    )
    ax2.set_xlabel("True Values", fontsize=18)
    ax2.set_ylabel("Predicted Values", fontsize=18)
    ax2.set_title(f"{model_name} Y-Y Plot - Test"
                  + '\nRMSE: ' + str(round(test_metrics[2], 3))
                  + '\nMAE: ' + str(round(test_metrics[1], 3))
                  + '\nR2: ' + str(round(test_metrics[0], 3)), fontsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=16)

    # Adjust layout and save the figure
    plt.tight_layout()
    file_path = os.path.join(save_folder, f"{model_name}_yy_plot.png")
    plt.savefig(file_path)  # Save the plot

    # Show the plot to the user
    plt.show()  

    # Print confirmation of save
    print(f"Plot saved for {model_name}: {file_path}")

# Main workflow function to train and evaluate multiple models
def run_workflow(X, y, models, param_spaces):
    """
    Run the complete machine learning workflow, including scaling, optimization, evaluation, and plotting.

    Parameters:
    - X: Feature matrix.
    - y: Target values.
    - models: Dictionary of models to train.
    - param_spaces: Dictionary of hyperparameter spaces for each model.

    Returns:
    - results_df: DataFrame containing metrics and best parameters for all models.
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Store results for each model
    results = []

    # Iterate through each model
    for name, model in models.items():
        is_nn = name == "NeuralNetwork"  # Check if the model is a neural network
        # Use the model's parameter space if defined, otherwise use an empty dictionary
        param_space = param_spaces.get(name, {})

        # Optimize and evaluate the model
        best_params, train_metrics, test_metrics, y_pred_train, y_pred_test = optimize_model(
            model, param_space, X_train, y_train, X_test, y_test, is_nn
        )

        # Append the results for this model
        results.append({
            "Model": name,
            "Best Params": best_params,
            "Train R2": train_metrics[0], "Train MAE": train_metrics[1], "Train RMSE": train_metrics[2],
            "Test R2": test_metrics[0], "Test MAE": test_metrics[1], "Test RMSE": test_metrics[2],
        })

        # Generate y-y plot for the model
        plot_yy(y_train, y_pred_train, y_test, y_pred_test, name, train_metrics, test_metrics)

    # Convert results to a DataFrame for easier analysis
    results_df = pd.DataFrame(results)

    # Save the DataFrame to a CSV file
    results_df.to_csv("results.csv", index=False)

    # Confirm save
    print("Results saved to results.csv")

    return results_df
