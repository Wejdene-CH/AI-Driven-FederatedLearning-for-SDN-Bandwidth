import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import ipaddress
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class TrafficClassifier(nn.Module):
    def __init__(self, input_features):
        super(TrafficClassifier, self).__init__()
        # Fully connected layers
        self.fc1 = nn.Linear(input_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # Neural network forward pass
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def preprocess_data(df):
    """
    Preprocess the dataframe by converting IP addresses and categorical variables
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe to preprocess
    
    Returns:
    --------
    pandas.DataFrame: Preprocessed dataframe
    dict: Label encoders for categorical columns
    """
    # Drop unnecessary columns
    df = df.drop(columns=['No.'])
    
    # Feature engineering for ports
    df['SourcePortRange'] = df['srcPort'] // 1000
    df['DestPortRange'] = df['DestPort'] // 1000
    
    # Convert IP addresses to numeric
    ip_columns = ['Source', 'Destination']
    for col in ip_columns:
        try:
            df[col] = df[col].apply(lambda x: int(ipaddress.ip_address(str(x))) if pd.notna(x) else np.nan)
        except Exception as e:
            print(f"Warning: Could not convert IP addresses in column {col}. Error: {e}")
    
    # Encode categorical columns
    categorical_columns = ['Protocol', 'Info']
    label_encoders = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        # Handle NaN values by converting to string first
        df[col] = df[col].fillna('Unknown')
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    return df, label_encoders

def prepare_data(csv_file, test_size=0.2, random_state=42):
    """
    Load and preprocess data from CSV file with robust NaN handling
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file
    test_size : float, optional
        Proportion of the dataset to include in the test split
    random_state : int, optional
        Controls the shuffling applied to the data before applying the split
    
    Returns:
    --------
    tuple: (X_train, X_test, y_train, y_test) as torch tensors
    """
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    print("Original DataFrame info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    
    # Check and handle NaN values
    print("\nMissing values before preprocessing:")
    print(df.isnull().sum())
    
    # Preprocess the data
    df, label_encoders = preprocess_data(df)
    
    # Fill NaN values
    # For numeric columns, fill with median
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    
    # Create a binary classification target based on 'Info' column
    # This is an example - adjust based on your specific classification goal
    df['target'] = (df['Info'] != 0).astype(int)
    
    # Separate features and target
    feature_columns = [col for col in df.columns if col not in ['target', 'Info']]
    X = df[feature_columns].values
    y = df['target'].values
    
    # Additional check after NaN handling
    print("\nChecking for NaN or infinite values after filling:")
    print("NaN in X:", np.isnan(X).any())
    print("Infinite in X:", np.isinf(X).any())
    print("NaN in y:", np.isnan(y).any())
    print("Infinite in y:", np.isinf(y).any())
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Convert to PyTorch tensors 
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)
    y_test = torch.FloatTensor(y_test).unsqueeze(1)
    
    return X_train, X_test, y_train, y_test, scaler, label_encoders

def train_with_federated_model(federated_model_path, local_csv_file, updated_weights_path):
    """
    Train a local model using weights from a federated model
    
    Parameters:
    -----------
    federated_model_path : str
        Path to the pre-trained federated model weights
    local_csv_file : str
        Path to the local training CSV file
    updated_weights_path : str
        Path to save the updated model weights
    """
    # Prepare data from CSV
    X_train, X_test, y_train, y_test, scaler, label_encoders = prepare_data(local_csv_file)
    
    # Initialize the model with the correct input features
    model = TrafficClassifier(X_train.size(1))
    
    # Load federated model weights if available
    try:
        pretrained_dict = torch.load(federated_model_path, map_location=torch.device('cpu'), weights_only=True)
        model_dict = model.state_dict()
        
        # Filter out keys that do not match between the pretrained and model dictionaries
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        
        # Update model's state_dict with the pretrained weights
        model_dict.update(pretrained_dict)
        
        # Load the updated state_dict into the model
        model.load_state_dict(model_dict)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load pre-trained weights. Initializing with random weights. Error: {e}")
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop with gradient clipping and loss tracking
    model.train()
    num_epochs = 10
    for epoch in range(num_epochs):
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Check for nan loss
        if torch.isnan(loss):
            print(f"Detected NaN loss at epoch {epoch+1}")
            print("Model outputs:", outputs)
            print("Target values:", y_train)
            break
        
        # Backward pass and optimize
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        
        # Calculate accuracy
        predicted = (torch.sigmoid(test_outputs) > 0.5).float()
        accuracy = (predicted == y_test).float().mean()
        
        print(f"Test Loss: {test_loss.item():.4f}")
        print(f"Test Accuracy: {accuracy.item():.4f}")
    
    # Save the updated model weights along with preprocessing information
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'label_encoders': label_encoders
    }, updated_weights_path)
    print(f"Updated model weights saved to {updated_weights_path}")

def load_saved_model(model_path, input_features):
    """
    Load a saved model with its preprocessing information
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model file
    input_features : int
        Number of input features
    
    Returns:
    --------
    tuple: (model, scaler, label_encoders)
    """
    # Load the saved checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Reinitialize the model
    model = TrafficClassifier(input_features)
    
    # Load the model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint['scaler'], checkpoint['label_encoders']

def predict_sample(model, sample, scaler, label_encoders):
    """
    Make a prediction for a single sample
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained neural network model
    sample : numpy.ndarray or list
        Input features for a single sample
    scaler : StandardScaler
        Scaler used for preprocessing
    label_encoders : dict
        Dictionary of label encoders used for categorical variables
    
    Returns:
    --------
    float: Predicted probability of the positive class
    """
    # Ensure the sample is a numpy array
    sample = np.array(sample).reshape(1, -1)
    
    # Scale the sample
    sample_scaled = scaler.transform(sample)
    
    # Convert to PyTorch tensor
    sample_tensor = torch.FloatTensor(sample_scaled)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        output = torch.sigmoid(model(sample_tensor))
        prediction_prob = output.numpy()[0][0]
    
    return prediction_prob

if __name__ == "__main__":
    # Example usage
    federated_model_path = "model.pth"
    local_csv_file = "s3pcap.csv"
    updated_weights_path = "updated_model.pth"
    
    # Train the model
    train_with_federated_model(federated_model_path, local_csv_file, updated_weights_path)
    
    # Example of loading and using the model
    try:
        # Prepare data to get the correct number of features
        X_train, X_test, y_train, y_test, scaler, label_encoders = prepare_data(local_csv_file)
        
        # Load the saved model
        loaded_model, loaded_scaler, loaded_label_encoders = load_saved_model(
            updated_weights_path, 
            input_features=X_train.size(1)
        )
        
        # Use an actual sample from the test data
        sample = X_test[0].numpy()  # Take first test sample
        prediction = predict_sample(loaded_model, sample, loaded_scaler, loaded_label_encoders)
        print(f"Sample Prediction Probability: {prediction:.4f}")
    
    except Exception as e:
        print(f"Error in model loading or prediction: {e}")