import torch
import torch.nn as nn
import kagglehub
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from corner import corner

# Set up device and random seeds
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

def visualize_dataset(df, dataset_name, target_cols, feature_cols, max_features=8):
    """
    Create corner plot and basic visualizations for the dataset.
    
    Args:
        df (pd.DataFrame): The dataset
        dataset_name (str): Name of the dataset
        target_cols (list): Target column names
        feature_cols (list): Feature column names  
        max_features (int): Maximum number of features to include in corner plot
    """
    
    # Limit features for corner plot readability
    selected_features = feature_cols[:max_features] if len(feature_cols) > max_features else feature_cols
    
    # Create subset for visualization
    viz_cols = selected_features + (target_cols if isinstance(target_cols, list) else [target_cols])
    viz_data = df[viz_cols].copy()
    
    # Handle missing values
    viz_data = viz_data.dropna()
    
    # Limit sample size for performance (corner plots can be slow with large datasets)
    if len(viz_data) > 2000:
        viz_data = viz_data.sample(n=2000, random_state=seed)
    
    print(f"\n=== Dataset Visualization: {dataset_name.upper()} ===")
    print(f"Showing {len(selected_features)} features out of {len(feature_cols)} total features")
    print(f"Sample size for visualization: {len(viz_data)}")
    
    # Basic statistics
    print(f"\nDataset shape: {df.shape}")
    print(f"Target column(s): {target_cols}")
    print(f"Feature columns: {len(feature_cols)}")
    
    # Create corner plot
    plt.figure(figsize=(12, 10))
    
    try:
        # Create corner plot with target as color if it's classification
        if dataset_name in ["avocado", "diabetes"] or (isinstance(target_cols, list) and "GradeClass" in target_cols):
            # For classification datasets, color by target
            target_col = target_cols[1] if isinstance(target_cols, list) and "GradeClass" in target_cols else target_cols
            if isinstance(target_col, list):
                target_col = target_col[0]
            
            # Get unique classes and create color map
            unique_classes = sorted(viz_data[target_col].unique())
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))
            
            fig = corner(viz_data[selected_features].values, 
                        labels=selected_features,
                        hist_kwargs={'alpha': 0.7},
                        scatter_kwargs={'alpha': 0.6, 's': 2})
            
            # Add target distribution subplot
            fig.suptitle(f'{dataset_name.title()} Dataset - Corner Plot\n'
                        f'Features: {len(selected_features)}, Samples: {len(viz_data)}', 
                        fontsize=14, y=0.98)
            
        else:
            # For regression datasets
            fig = corner(viz_data[selected_features].values,
                        labels=selected_features,
                        hist_kwargs={'alpha': 0.7, 'bins': 30},
                        scatter_kwargs={'alpha': 0.6, 's': 2})
            
            fig.suptitle(f'{dataset_name.title()} Dataset - Corner Plot\n'
                        f'Features: {len(selected_features)}, Samples: {len(viz_data)}',
                        fontsize=14, y=0.98)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Corner plot failed: {e}")
        print("Creating alternative visualization...")
        
        # Close the unused matplotlib figure
        plt.close()
        
        # Select fewer features for seaborn pairplot
        pair_features = selected_features[:6] if len(selected_features) > 6 else selected_features
        pair_data = viz_data[pair_features].copy()
        
        # Add target for coloring if classification
        if dataset_name in ["avocado", "diabetes"]:
            target_col = target_cols if isinstance(target_cols, str) else target_cols[0]
            pair_data['target'] = viz_data[target_col]
            g = sns.pairplot(pair_data, hue='target', plot_kws={'alpha': 0.6, 's': 20})
        else:
            g = sns.pairplot(pair_data, plot_kws={'alpha': 0.6, 's': 20})
        
        g.fig.suptitle(f'{dataset_name.title()} Dataset - Pairplot', y=1.02)
        plt.tight_layout()
        plt.show()
    
    # Target distribution
    plt.figure(figsize=(12, 4))
    
    if isinstance(target_cols, list):
        # Multiple targets
        for i, target in enumerate(target_cols):
            plt.subplot(1, len(target_cols), i+1)
            if target in df.columns:
                if df[target].dtype in ['object', 'category'] or df[target].nunique() < 10:
                    # Categorical target
                    df[target].value_counts().plot(kind='bar', alpha=0.7)
                    plt.title(f'{target} Distribution')
                    plt.xticks(rotation=45)
                else:
                    # Continuous target
                    plt.hist(df[target].dropna(), bins=30, alpha=0.7, edgecolor='black')
                    plt.title(f'{target} Distribution')
                    plt.xlabel(target)
                    plt.ylabel('Frequency')
    else:
        # Single target
        if df[target_cols].dtype in ['object', 'category'] or df[target_cols].nunique() < 10:
            # Categorical target
            df[target_cols].value_counts().plot(kind='bar', alpha=0.7)
            plt.title(f'{target_cols} Distribution')
            plt.xticks(rotation=45)
        else:
            # Continuous target
            plt.hist(df[target_cols].dropna(), bins=30, alpha=0.7, edgecolor='black')
            plt.title(f'{target_cols} Distribution')
            plt.xlabel(target_cols)
            plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    # Feature correlation heatmap
    if len(selected_features) > 1:
        plt.figure(figsize=(10, 8))
        correlation_matrix = viz_data[selected_features].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title(f'{dataset_name.title()} - Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()

def load_dataset(dataset_name, validation=True, visualize=False, ripeness_class="ripe"):
    """
    Load and preprocess datasets for neural network training.
    
    Args:
        dataset_name (str): Name of dataset - "particle", "weather", "grades", "avocado", or "diabetes"
        validation (bool): Whether to create a validation set
        visualize (bool): Whether to create visualization plots
        ripeness_class (str): For avocado dataset - "ripe", "firm_ripe", or "all" (default: "all")
    
    Returns:
        dict: Dictionary containing processed tensors and metadata
    """
    
    # Load chosen datasets from KaggleHub
    if dataset_name == "particle":
        path = kagglehub.dataset_download("fedesoriano/cern-electron-collision-data")
        df = pd.read_csv(path + "/dielectron.csv")
        target = "M"  # Mass of the particle
        drops = ["Run", "Event", "M"]  # Columns to drop
    elif dataset_name == "weather":
        path = kagglehub.dataset_download("budincsevity/szeged-weather")
        df = pd.read_csv(path + "/weatherHistory.csv")
        target = "Apparent Temperature (C)"  # Apparent temperature
        drops = ["Daily Summary", "Precip Type", "Formatted Date", "Summary", "Apparent Temperature (C)"]  # Columns to drop
    elif dataset_name == "grades":
        path = kagglehub.dataset_download("rabieelkharoua/students-performance-dataset")
        df = pd.read_csv(path + "/Student_performance_data _.csv")
        target = ["GPA", "GradeClass"]  # GPA of the students
        drops = ["GPA", "GradeClass"]  # Columns to drop
    elif dataset_name == "avocado":
        path = kagglehub.dataset_download("amldvvs/avocado-ripeness-classification-dataset")
        df = pd.read_csv(path + "/avocado_ripeness_dataset.csv")
        
        # Handle ripeness class selection
        if ripeness_class == "all":
            target = "ripeness"  # Multiclass classification
            drops = ["color_category", "ripeness"]
        elif ripeness_class in ["ripe", "firm_ripe"]:
            # Binary classification: selected class vs all others
            df["binary_ripeness"] = (df["ripeness"] == ripeness_class).astype(int)
            target = "binary_ripeness"
            drops = ["color_category", "ripeness", "binary_ripeness"]
            print(f"Avocado binary classification: '{ripeness_class}' vs others")
            print(f"Class distribution - {ripeness_class}: {df['binary_ripeness'].sum()}, others: {(df['binary_ripeness'] == 0).sum()}")
        else:
            raise ValueError("ripeness_class must be 'ripe', 'firm_ripe', or 'all'")
            
    elif dataset_name == "diabetes":
        path = kagglehub.dataset_download("alexteboul/diabetes-health-indicators-dataset")
        df = pd.read_csv(path + "/diabetes_012_health_indicators_BRFSS2015.csv")
        target = "Diabetes_012"  # Diabetes class
        drops = ["Diabetes_012"]  # Columns to drop
    else:
        raise ValueError("Invalid dataset choice. Choose from: particle, weather, grades, avocado, diabetes.")

    # Remove rows with NaN values in ANY column (features or targets)
    print(f"Original dataset shape: {df.shape}")
    df_clean = df.dropna()
    print(f"After removing NaN values: {df_clean.shape}")
    print(f"Removed {df.shape[0] - df_clean.shape[0]} rows with NaN values")
    df = df_clean

    # Get feature column names before preprocessing
    feature_cols = [col for col in df.columns if col not in drops]
    
    # Visualization before preprocessing
    if visualize:
        visualize_dataset(df, dataset_name, target, feature_cols)

    # Preprocess the dataset
    if dataset_name == "avocado" and ripeness_class == "all":
        df["ripeness"] = df["ripeness"].astype("category").cat.codes  # Convert categorical to numerical
        print(f"Avocado multiclass mapping: {dict(enumerate(df['ripeness'].astype('category').cat.categories))}")

    # Split the dataset
    if validation:
        train_df, val_df = train_test_split(df, test_size=0.1, random_state=seed)
        df = train_df  # Use the training set for further processing
    
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=seed)

    # Prepare the target variables
    if type(target) is list:
        train_targets = train_df[target[0]].values
        val_targets = val_df[target[0]].values if validation else None
        test_targets = test_df[target[0]].values
        train_targets2 = train_df[target[1]].values if len(target) > 1 else None
        val_targets2 = val_df[target[1]].values if validation and len(target) > 1 else None
        test_targets2 = test_df[target[1]].values if len(target) > 1 else None
    else:
        train_targets2 = None
        train_targets = train_df[target].values
        val_targets = val_df[target].values if validation else None
        test_targets = test_df[target].values
    
    # Prepare feature matrices
    X = train_df.drop(columns=drops).values
    if validation:
        X_val = val_df.drop(columns=drops).values
    X_test = test_df.drop(columns=drops).values
    
    # Check for any remaining NaN values after splitting and warn
    if np.isnan(X).any():
        print("Warning: NaN values found in training features after splitting")
    if np.isnan(train_targets).any():
        print("Warning: NaN values found in training targets after splitting")
    if validation and np.isnan(X_val).any():
        print("Warning: NaN values found in validation features after splitting")
    if validation and val_targets is not None and np.isnan(val_targets).any():
        print("Warning: NaN values found in validation targets after splitting")
    
    # Normalize features using training set statistics
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    if validation:
        X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Normalize targets for regression tasks
    target_scaler = None
    if isinstance(target, str) and dataset_name in ["particle", "weather"]:
        target_scaler = StandardScaler()
        train_targets = target_scaler.fit_transform(train_targets.reshape(-1, 1)).flatten()
        if validation:
            val_targets = target_scaler.transform(val_targets.reshape(-1, 1)).flatten()
        test_targets = target_scaler.transform(test_targets.reshape(-1, 1)).flatten()
    elif isinstance(target, list) and dataset_name == "grades":
        target_scaler = StandardScaler()
        train_targets = target_scaler.fit_transform(train_targets.reshape(-1, 1)).flatten()
        if validation:
            val_targets = target_scaler.transform(val_targets.reshape(-1, 1)).flatten()
        test_targets = target_scaler.transform(test_targets.reshape(-1, 1)).flatten()
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    if validation:
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    train_targets_tensor = torch.tensor(train_targets, dtype=torch.float32).to(device)
    if validation:
        val_targets_tensor = torch.tensor(val_targets, dtype=torch.float32).to(device)
    test_targets_tensor = torch.tensor(test_targets, dtype=torch.float32).to(device)
    if train_targets2 is not None:
        train_targets2_tensor = torch.tensor(train_targets2, dtype=torch.float32).to(device)
        if validation:
            val_targets2_tensor = torch.tensor(val_targets2, dtype=torch.float32).to(device)
        test_targets2_tensor = torch.tensor(test_targets2, dtype=torch.float32).to(device)
    
    # Return data dictionary
    data_dict = {
        'X_train': X_train_tensor,
        'X_test': X_test_tensor,
        'train_targets': train_targets_tensor,
        'test_targets': test_targets_tensor,
        'input_size': X.shape[1],
        'output_size': 1,
        'dataset_name': dataset_name,
        'target_type': target,
        'feature_names': feature_cols,
        'feature_scaler': scaler,
        'target_scaler': target_scaler
    }
    
    if validation:
        data_dict['X_val'] = X_val_tensor
        data_dict['val_targets'] = val_targets_tensor
    
    if train_targets2 is not None:
        data_dict['train_targets2'] = train_targets2_tensor
        data_dict['test_targets2'] = test_targets2_tensor
        if validation:
            data_dict['val_targets2'] = val_targets2_tensor
    
    return data_dict

print(f"Using device: {device}")


# Add this cell after the data loading cell:

def train_model(model, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None,
                task_type='regression', epochs=100, learning_rate=0.001, batch_size=32, 
                early_stopping_patience=10, print_every=10):
    """
    Train a PyTorch neural network model.
    
    Args:
        model: PyTorch neural network model
        X_train, y_train: Training data and targets
        X_val, y_val: Validation data and targets (optional)
        X_test, y_test: Test data and targets (optional)
        task_type: 'regression' or 'classification'
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        early_stopping_patience: Stop training if validation doesn't improve for this many epochs
        print_every: Print progress every N epochs
    
    Returns:
        dict: Training history with losses and metrics
    """
    
    # Move model to device
    model = model.to(device)
    
    # Choose loss function based on task type
    if task_type == 'regression':
        criterion = nn.MSELoss()
    elif task_type == 'classification':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("task_type must be 'regression' or 'classification'")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metric': [],
        'val_metric': []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"Training {task_type} model for {epochs} epochs...")
    print(f"Device: {device}")
    print("-" * 50)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_targets = []
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # Handle output dimensions
            if task_type == 'classification':
                if outputs.dim() > 1 and outputs.size(1) == 1:
                    outputs = outputs.squeeze(1)
                batch_y = batch_y.long()
            else:  # regression
                if outputs.dim() > 1 and outputs.size(1) == 1:
                    outputs = outputs.squeeze(1)
            
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Store predictions for metrics
            if task_type == 'classification':
                train_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy() if outputs.dim() > 1 else (outputs > 0.5).float().cpu().numpy())
            else:
                train_predictions.extend(outputs.detach().cpu().numpy())
            train_targets.extend(batch_y.cpu().numpy())
        
        # Calculate training metrics
        train_loss /= len(train_loader)
        if task_type == 'classification':
            train_accuracy = np.mean(np.array(train_predictions) == np.array(train_targets))
            train_metric = train_accuracy
        else:
            train_rmse = np.sqrt(np.mean((np.array(train_predictions) - np.array(train_targets))**2))
            train_metric = train_rmse
        
        history['train_loss'].append(train_loss)
        history['train_metric'].append(train_metric)
        
        # Validation phase
        val_loss = 0.0
        val_metric = 0.0
        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                
                # Handle output dimensions
                if task_type == 'classification':
                    if val_outputs.dim() > 1 and val_outputs.size(1) == 1:
                        val_outputs = val_outputs.squeeze(1)
                    val_y_long = y_val.long()
                else:
                    if val_outputs.dim() > 1 and val_outputs.size(1) == 1:
                        val_outputs = val_outputs.squeeze(1)
                    val_y_long = y_val
                
                val_loss = criterion(val_outputs, val_y_long).item()
                
                # Calculate validation metrics
                if task_type == 'classification':
                    val_predictions = torch.argmax(val_outputs, dim=1) if val_outputs.dim() > 1 else (val_outputs > 0.5).float()
                    val_accuracy = (val_predictions == y_val).float().mean().item()
                    val_metric = val_accuracy
                else:
                    val_rmse = torch.sqrt(torch.mean((val_outputs - y_val)**2)).item()
                    val_metric = val_rmse
            
            history['val_loss'].append(val_loss)
            history['val_metric'].append(val_metric)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Print progress
        if (epoch + 1) % print_every == 0:
            if X_val is not None:
                metric_name = 'Accuracy' if task_type == 'classification' else 'RMSE'
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train {metric_name}: {train_metric:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val {metric_name}: {val_metric:.4f}")
            else:
                metric_name = 'Accuracy' if task_type == 'classification' else 'RMSE'
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train {metric_name}: {train_metric:.4f}")
    
    # Load best model if validation was used
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model from validation")
    
    # Test evaluation
    if X_test is not None and y_test is not None:
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            
            # Handle output dimensions
            if task_type == 'classification':
                if test_outputs.dim() > 1 and test_outputs.size(1) == 1:
                    test_outputs = test_outputs.squeeze(1)
                test_y_long = y_test.long()
            else:
                if test_outputs.dim() > 1 and test_outputs.size(1) == 1:
                    test_outputs = test_outputs.squeeze(1)
                test_y_long = y_test
            
            test_loss = criterion(test_outputs, test_y_long).item()
            
            if task_type == 'classification':
                test_predictions = torch.argmax(test_outputs, dim=1) if test_outputs.dim() > 1 else (test_outputs > 0.5).float()
                test_accuracy = (test_predictions == y_test).float().mean().item()
                print(f"\nTest Results - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
            else:
                test_rmse = torch.sqrt(torch.mean((test_outputs - y_test)**2)).item()
                print(f"\nTest Results - Loss: {test_loss:.4f}, RMSE: {test_rmse:.4f}")
    
    return history, model
