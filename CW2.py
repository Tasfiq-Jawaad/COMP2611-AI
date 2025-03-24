# COMP2611-Artificial Intelligence-Coursework#2 - Descision Trees

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.tree import export_text
import warnings
import os

# STUDENT NAME: Mohammad Tasfiq Jawaad
# STUDENT EMAIL:  sc23mtj@leeds.ac.uk
    
def print_tree_structure(model, header_list):
    tree_rules = export_text(model, feature_names=header_list[:-1])
    print(tree_rules)
    
# Task 1 [8 marks]: 
def load_data(file_path, delimiter=','):
    num_rows, data, header_list=None, None, None
    if not os.path.isfile(file_path):
        warnings.warn(f"Task 1: Warning - CSV file '{file_path}' does not exist.")
        return None, None, None
    # Insert your code here for task 1

    try:
        # Read CSV using pandas
        df = pd.read_csv(file_path, delimiter=delimiter)
        
        # Extract header list
        header_list = list(df.columns)
        
        # Convert dataframe to numpy array
        data = df.to_numpy()
        
        # Get number of rows (samples)
        num_rows = data.shape[0]
        
    except Exception as e:
        warnings.warn(f"Task 1: Error while reading the CSV file - {e}")
        return None, None, None
    
    return num_rows, data, header_list

# Task 2[8 marks]: 
def filter_data(data):
    if data is None:
        warnings.warn("Task 2: Warning - Input data is None.")
        return None

    # Convert data to NumPy array if it's not already
    data = np.array(data)

    # Keep rows that do NOT contain -99
    filtered_data = data[~np.any(data == -99, axis=1)]

    return filtered_data


# Task 3 [8 marks]: 
def statistics_data(data):
    if data is None or len(data) == 0:
        warnings.warn("Task 3: Warning - No data available for statistics.")
        return None

    # Step 1: Filter the data (same approach as Task 2)
    data = np.array(data, dtype=np.float64)
    data = data[~np.any(data == -99, axis=1)]  # Remove rows containing -99

    # Step 2: Exclude the last column (label column)
    feature_data = data[:, :-1]  

    # Step 3: Compute mean and standard deviation
    mean = np.mean(feature_data, axis=0)
    std_dev = np.std(feature_data, axis=0)

    # Step 4: Compute coefficient of variation
    # If mean is 0, return np.inf to avoid division by zero
    coefficient_of_variation = np.where(mean == 0, np.inf, std_dev / mean)

    return coefficient_of_variation


# Task 4 [8 marks]: 
def split_data(data, test_size=0.3, random_state=1):
    np.random.seed(random_state)  # Ensure reproducibility

    # Convert data to numpy array
    data = np.array(data, dtype=np.float64)

    # Extract features (X) and labels (y)
    X = data[:, :-1]  # All columns except the last one (features)
    y = data[:, -1]   # Last column (labels)

    # Split the dataset while maintaining label ratios (stratification)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return x_train, x_test, y_train, y_test

# Task 5 [8 marks]: 
def train_decision_tree(x_train, y_train, ccp_alpha=0):
    # Initialize the DecisionTreeClassifier with specified parameters
    model = DecisionTreeClassifier(random_state=1, ccp_alpha=ccp_alpha)
    
    # Train (fit) the model using the training data
    model.fit(x_train, y_train)
    
    return model

# Task 6 [8 marks]: 
def make_predictions(model, X_test):
    # Use the trained model to make predictions
    y_test_predicted = model.predict(X_test)
    
    return y_test_predicted


# Task 7 [8 marks]: 
def evaluate_model(model, x, y): 
    # Get predictions
    y_pred = model.predict(x)
    
    # Calculate accuracy and recall
    accuracy = accuracy_score(y, y_pred)
    recall = recall_score(y, y_pred)
    
    return accuracy, recall


# Task 8 [8 marks]: 
def optimal_ccp_alpha(x_train, y_train, x_test, y_test):
    # Train an initial unpruned model with ccp_alpha=0
    model = DecisionTreeClassifier(ccp_alpha=0, random_state=1)
    model.fit(x_train, y_train)
    
    # Get baseline accuracy
    base_accuracy = accuracy_score(y_test, model.predict(x_test))
    
    # If the model achieves perfect accuracy, return 0.0
    if base_accuracy == 1.0:
        return 0.0
    
    # Define acceptable accuracy range (±1%)
    min_accuracy = base_accuracy - 0.01
    max_accuracy = base_accuracy + 0.01
    
    # Start searching for optimal ccp_alpha
    optimal_alpha = 0.0
    for alpha in np.arange(0.001, 1.001, 0.001):  # Step from 0.001 to 1.0
        model = DecisionTreeClassifier(ccp_alpha=alpha, random_state=1)
        model.fit(x_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(x_test))
        
        # If accuracy drops below threshold, stop and return last valid ccp_alpha
        if accuracy < min_accuracy:
            break
        
        # Update optimal_alpha to the last valid alpha
        optimal_alpha = alpha

    return optimal_alpha


# Task 9 [8 marks]: 
def tree_depths(model):
    depth = model.get_depth()  # Get the depth of the decision tree
    return depth


 # Task 10 [8 marks]: 
def important_feature(x_train, y_train, header_list):
    best_feature = None
    ccp_alpha = 0.0
    while ccp_alpha <= 1.0:
        # Train the model with the current ccp_alpha value
        model = DecisionTreeClassifier(random_state=1, ccp_alpha=ccp_alpha)
        model.fit(x_train, y_train)

        # Check if the model has depth 1
        if model.get_depth() == 1:
            # Get the feature used for the split at depth 1
            best_feature = header_list[model.tree_.feature[0]]
            break
        
        # Increment ccp_alpha and check again
        ccp_alpha += 0.01
    
    return best_feature

    
# Task 11 [10 marks]: 
def optimal_ccp_alpha_single_feature(x_train, y_train, x_test, y_test, header_list):
    # Step 1: Identify the most important feature using the method from Task 10
    important_feature_name = important_feature(x_train, y_train, header_list)

    # Step 2: Create new datasets containing only the most important feature
    important_feature_index = header_list.index(important_feature_name)
    x_train_single_feature = x_train[:, important_feature_index].reshape(-1, 1)
    x_test_single_feature = x_test[:, important_feature_index].reshape(-1, 1)

    # Step 3: Use the optimal_ccp_alpha method from Task 8 with the new datasets
    optimal_ccp_alpha_value = optimal_ccp_alpha(x_train_single_feature, y_train, x_test_single_feature, y_test)

    # Return the optimal ccp_alpha value found
    return optimal_ccp_alpha_value


# Task 12 [10 marks]: 
def optimal_depth_two_features(x_train, y_train, x_test, y_test, header_list):
    # Step 1: Identify the most important feature using the method from Task 10
    important_feature_name = important_feature(x_train, y_train, header_list)

    # Step 2: Remove the most important feature from consideration
    important_feature_index = header_list.index(important_feature_name)
    header_list_without_first = header_list[:important_feature_index] + header_list[important_feature_index+1:]
    x_train_without_first = np.delete(x_train, important_feature_index, axis=1)
    x_test_without_first = np.delete(x_test, important_feature_index, axis=1)

    # Step 3: Identify the second most important feature using the updated list of features
    second_important_feature_name = important_feature(x_train_without_first, y_train, header_list_without_first)

    # Step 4: Create a new dataset with only the two important features
    second_important_feature_index = header_list_without_first.index(second_important_feature_name)
    
    # Adjust for the first feature that was removed: Add 1 to the index of the second feature
    if second_important_feature_index >= important_feature_index:
        second_important_feature_index += 1

    selected_features_indices = [important_feature_index, second_important_feature_index]
    x_train_two_features = x_train[:, selected_features_indices]
    x_test_two_features = x_test[:, selected_features_indices]

    # Step 5: Use the optimal_ccp_alpha method from Task 8 with the new dataset containing two features
    optimal_ccp_alpha_value = optimal_ccp_alpha(x_train_two_features, y_train, x_test_two_features, y_test)

    # Step 6: Train the decision tree with the optimal ccp_alpha
    model = train_decision_tree(x_train_two_features, y_train, ccp_alpha=optimal_ccp_alpha_value)

    # Step 7: Return the depth of the trained decision tree
    optimal_depth = tree_depths(model)

    return optimal_depth



# Example usage (Main section):
if __name__ == "__main__":
    # Load data
    file_path = "DT.csv"
    num_rows, data, header_list = load_data(file_path)
    print(f"Data is read. Number of Rows: {num_rows}"); 
    print("-" * 50)

    # Filter data
    data_filtered = filter_data(data)
    num_rows_filtered=data_filtered.shape[0]
    print(f"Data is filtered. Number of Rows: {num_rows_filtered}"); 
    print("-" * 50)

    # Data Statistics
    coefficient_of_variation = statistics_data(data_filtered)
    print("Coefficient of Variation for each feature:")
    for header, coef_var in zip(header_list[:-1], coefficient_of_variation):
        print(f"{header}: {coef_var}")
    print("-" * 50)
    # Split data
    x_train, x_test, y_train, y_test = split_data(data_filtered)
    print(f"Train set size: {len(x_train)}")
    print(f"Test set size: {len(x_test)}")
    print("-" * 50)
    
    # Train initial Decision Tree
    model = train_decision_tree(x_train, y_train)
    print("Initial Decision Tree Structure:")
    print_tree_structure(model, header_list)
    print("-" * 50)
    
    # Evaluate initial model
    acc_test, recall_test = evaluate_model(model, x_test, y_test)
    print(f"Initial Decision Tree - Test Accuracy: {acc_test:.2%}, Recall: {recall_test:.2%}")
    print("-" * 50)
    # Train Pruned Decision Tree
    model_pruned = train_decision_tree(x_train, y_train, ccp_alpha=0.002)
    print("Pruned Decision Tree Structure:")
    print_tree_structure(model_pruned, header_list)
    print("-" * 50)
    # Evaluate pruned model
    acc_test_pruned, recall_test_pruned = evaluate_model(model_pruned, x_test, y_test)
    print(f"Pruned Decision Tree - Test Accuracy: {acc_test_pruned:.2%}, Recall: {recall_test_pruned:.2%}")
    print("-" * 50)
    # Find optimal ccp_alpha
    optimal_alpha = optimal_ccp_alpha(x_train, y_train, x_test, y_test)
    print(f"Optimal ccp_alpha for pruning: {optimal_alpha:.4f}")
    print("-" * 50)
    # Train Pruned and Optimized Decision Tree
    model_optimized = train_decision_tree(x_train, y_train, ccp_alpha=optimal_alpha)
    print("Optimized Decision Tree Structure:")
    print_tree_structure(model_optimized, header_list)
    print("-" * 50)
    
    # Get tree depths
    depth_initial = tree_depths(model)
    depth_pruned = tree_depths(model_pruned)
    depth_optimized = tree_depths(model_optimized)
    print(f"Initial Decision Tree Depth: {depth_initial}")
    print(f"Pruned Decision Tree Depth: {depth_pruned}")
    print(f"Optimized Decision Tree Depth: {depth_optimized}")
    print("-" * 50)
    
    # Feature importance
    important_feature_name = important_feature(x_train, y_train,header_list)
    print(f"Important Feature for Fraudulent Transaction Prediction: {important_feature_name}")
    print("-" * 50)
    
    # Test optimal ccp_alpha with single feature
    optimal_alpha_single = optimal_ccp_alpha_single_feature(x_train, y_train, x_test, y_test, header_list)
    print(f"Optimal ccp_alpha using single most important feature: {optimal_alpha_single:.4f}")
    print("-" * 50)
    
    # Test optimal depth with two features
    optimal_depth_two = optimal_depth_two_features(x_train, y_train, x_test, y_test, header_list)
    print(f"Optimal tree depth using two most important features: {optimal_depth_two}")
    print("-" * 50)        
# References: 
# Here please provide recognition to any source if you have used or got code snippets from
# Please tell the lines that are relavant to that reference.
# For example: 
# Line 80-87 is inspired by a code at https://stackoverflow.com/questions/48414212/how-to-calculate-accuracy-from-decision-trees


