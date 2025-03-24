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
        df = pd.read_csv(file_path, delimiter=delimiter)
        
        header_list = list(df.columns)
        
        # Convert dataframe to numpy array
        data = df.to_numpy()
        
        num_rows = data.shape[0]
        
    except Exception as exc:
        warnings.warn(f"Error while reading the CSV file - {exc}")
        return None, None, None
    
    return num_rows, data, header_list

# Task 2[8 marks]: 
def filter_data(data):
    filtered_data=[None]*1
    # Insert your code here for task 2
    data = np.array(data)

    # Keep rows that do NOT contain -99
    return data[~np.any(data == -99, axis=1)]



# Task 3 [8 marks]: 
def statistics_data(data):
    coefficient_of_variation=None
    # Insert your code here for task 3

    # Filter the data (remove rows with -99)
    filtered_data = data[~np.any(data == -99, axis=1)]

    features = filtered_data[:, :-1]

    # Compute the mean and standard deviation for each feature
    means = np.mean(features, axis=0, dtype=np.float64)
    std_devs = np.std(features, axis=0, dtype=np.float64)

    # Compute coefficient of variation (std / mean)
    return np.where(means == 0, np.inf, std_devs / means)


# Task 4 [8 marks]: 
def split_data(data, test_size=0.3, random_state=1):
    x_train, x_test, y_train, y_test=None, None, None, None
    np.random.seed(1)
    # Insert your code here for task 4

    # Split features and labels
    X = data[:, :-1] 
    y = data[:, -1]  

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return x_train, x_test, y_train, y_test

# Task 5 [8 marks]: 
def train_decision_tree(x_train, y_train, ccp_alpha=0):
    model=None
    # Insert your code here for task 5

    model = DecisionTreeClassifier(random_state=1, ccp_alpha=ccp_alpha)

    model.fit(x_train, y_train)

    return model

# Task 6 [8 marks]: 
def make_predictions(model, X_test):
    y_test_predicted=None
    # Insert your code here for task 6

    y_test_predicted = model.predict(X_test)

    return y_test_predicted



# Task 7 [8 marks]: 
def evaluate_model(model, x, y):
    accuracy, recall = None, None
    # Insert your code here for task 7

    y_pred = model.predict(x)

    accuracy = accuracy_score(y, y_pred)

    recall = recall_score(y, y_pred)

    return accuracy, recall



# Task 8 [8 marks]: 
def optimal_ccp_alpha(x_train, y_train, x_test, y_test):
    optimal_ccp_alpha = None

    # Insert your code here for task 8

    base_model = DecisionTreeClassifier(random_state=1, ccp_alpha=0)
    base_model.fit(x_train, y_train)
    
    base_accuracy = accuracy_score(y_test, base_model.predict(x_test))

    if base_accuracy == 1.0:
        return 0.0

    lower_bound = base_accuracy - 0.01

    last_valid_ccp_alpha = 0.0

    for ccp_alpha in np.arange(0.001, 1.001, 0.001):
        model = DecisionTreeClassifier(random_state=1, ccp_alpha=ccp_alpha)
        model.fit(x_train, y_train)
        
        accuracy = accuracy_score(y_test, model.predict(x_test))
        
        if accuracy < lower_bound:
            return last_valid_ccp_alpha
        
        last_valid_ccp_alpha = ccp_alpha

    return last_valid_ccp_alpha



# Task 9 [8 marks]: 
def tree_depths(model):
    depth = None
    # Get the depth of the unpruned tree
    # Insert your code here for task 9
    
    depth = model.get_depth()
    
    return depth



 # Task 10 [8 marks]: 
def important_feature(x_train, y_train, header_list):
    best_feature = None
    # Insert your code here for task 10

    last_valid_feature = None
    ccp_alpha = 0.0

    for _ in range(100):
        model = DecisionTreeClassifier(random_state=1, ccp_alpha=ccp_alpha)
        model.fit(x_train, y_train)
        depth = model.get_depth()

        if depth == 1:
            best_feature = header_list[model.tree_.feature[0]]
            return best_feature
        elif depth > 1:
            last_valid_feature = header_list[model.tree_.feature[0]]
        
        ccp_alpha += 0.01
        if ccp_alpha > 1.0:
            break

    return last_valid_feature


    
# Task 11 [10 marks]: 
def optimal_ccp_alpha_single_feature(x_train, y_train, x_test, y_test, header_list):
    feature_name = important_feature(x_train, y_train, header_list)

    feature_index = header_list.index(feature_name)
    x_train_single_feature = x_train[:, feature_index].reshape(-1, 1)
    x_test_single_feature = x_test[:, feature_index].reshape(-1, 1)

    return optimal_ccp_alpha(x_train_single_feature, y_train, x_test_single_feature, y_test)



# Task 12 [10 marks]: 
def optimal_depth_two_features(x_train, y_train, x_test, y_test, header_list):
    optimal_depth = None
    # Insert your code here for task 12
    
    most_important_feature = important_feature(x_train, y_train, header_list)

    most_important_feature_index = header_list.index(most_important_feature)
    header_list_without_first = header_list[:most_important_feature_index] + header_list[most_important_feature_index + 1:]
    x_train_without_first = np.delete(x_train, most_important_feature_index, axis=1)

    second_important_feature = important_feature(x_train_without_first, y_train, header_list_without_first)
    
    second_important_feature_index = header_list.index(second_important_feature)

    selected_features_indices = [most_important_feature_index, second_important_feature_index]
    x_train_two_features = x_train[:, selected_features_indices]
    x_test_two_features = x_test[:, selected_features_indices]

    optimal_ccp_alpha_value = optimal_ccp_alpha(x_train_two_features, y_train, x_test_two_features, y_test)

    model = train_decision_tree(x_train_two_features, y_train, ccp_alpha=optimal_ccp_alpha_value)

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


