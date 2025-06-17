import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

# Configuration
DATASET_PATH = "dataset.json"
MODELS_DIR = "saved_models"
BATCH_SIZE = 64
EPOCHS = 50  
LEARNING_RATE = 1e-4
TRAIN_SPLIT = 0.6  # 60% for training
VAL_SPLIT = 0.2    # 20% for validation  
TEST_SPLIT = 0.2   # 20% for testing
INPUT_SIZE = 1000  
HIDDEN_SIZE = 128

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)


class FingerprintDataset(Dataset):
    """Custom Dataset class for website fingerprinting data."""
    
    def __init__(self, traces, labels, scaler=None, fit_scaler=True):
        """
        Args:
            traces: List or array of trace data
            labels: List or array of corresponding labels
            scaler: StandardScaler object for normalization
            fit_scaler: Whether to fit the scaler on this data
        """
        self.traces = np.array(traces, dtype=np.float32)
        self.labels = np.array(labels, dtype=np.int64)
        
        # Apply standardization
        if scaler is None:
            self.scaler = StandardScaler()
            self.traces = self.scaler.fit_transform(self.traces)
        else:
            self.scaler = scaler
            if fit_scaler:
                self.traces = self.scaler.fit_transform(self.traces)
            else:
                self.traces = self.scaler.transform(self.traces)
    
    def __len__(self):
        return len(self.traces)
    
    def __getitem__(self, idx):
        return torch.tensor(self.traces[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


def load_dataset(dataset_path):
    """Load dataset from JSON file and return traces, labels, and website names.
    
    Args:
        dataset_path: Path to the JSON dataset file
        
    Returns:
        traces: List of trace data arrays
        labels: List of corresponding labels (website indices)
        website_names: List of unique website names
        website_to_idx: Dictionary mapping website names to indices
    """
    print(f"Loading dataset from {dataset_path}...")
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} total samples")
    
    # Extract unique websites and create mapping
    websites = list(set(item['website'] for item in data))
    websites.sort()  # Sort for consistent ordering
    website_to_idx = {website: idx for idx, website in enumerate(websites)}
    
    print(f"Found {len(websites)} websites:")
    for idx, website in enumerate(websites):
        count = sum(1 for item in data if item['website'] == website)
        print(f"  {idx}: {website} ({count} samples)")
    
    # Prepare traces and labels
    traces = []
    labels = []
    
    for item in data:
        trace_data = item['trace_data']
        
        # Ensure trace data is the right size
        if len(trace_data) > INPUT_SIZE:
            trace_data = trace_data[:INPUT_SIZE]  # Truncate if too long
        elif len(trace_data) < INPUT_SIZE:
            # Pad with zeros if too short
            trace_data = trace_data + [0] * (INPUT_SIZE - len(trace_data))
        
        traces.append(trace_data)
        labels.append(website_to_idx[item['website']])
    
    return np.array(traces), np.array(labels), websites, website_to_idx


def create_data_splits(traces, labels, train_split=0.6, val_split=0.2, test_split=0.2, random_state=42):
    """Split data into train, validation, and test sets with stratification.
    
    Args:
        traces: Array of trace data
        labels: Array of corresponding labels
        train_split: Fraction for training set
        val_split: Fraction for validation set
        test_split: Fraction for test set
        random_state: Random seed for reproducibility
        
    Returns:
        train_indices, val_indices, test_indices: Arrays of indices for each split
    """
    print(f"\nSplitting dataset: {train_split*100:.0f}% train, {val_split*100:.0f}% val, {test_split*100:.0f}% test")
    
    # First split: separate train+val from test
    train_val_split = train_split + val_split
    splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=random_state)
    train_val_indices, test_indices = next(splitter1.split(traces, labels))
    
    # Second split: separate train from val
    val_size_from_train_val = val_split / train_val_split
    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size_from_train_val, random_state=random_state)
    train_indices, val_indices = next(splitter2.split(traces[train_val_indices], labels[train_val_indices]))
    
    # Convert back to original indices
    train_indices = train_val_indices[train_indices]
    val_indices = train_val_indices[val_indices]
    
    print(f"Train set: {len(train_indices)} samples")
    print(f"Validation set: {len(val_indices)} samples") 
    print(f"Test set: {len(test_indices)} samples")
    
    # Verify class distribution
    print("\nClass distribution:")
    unique_labels = np.unique(labels)
    for label in unique_labels:
        train_count = np.sum(labels[train_indices] == label)
        val_count = np.sum(labels[val_indices] == label)
        test_count = np.sum(labels[test_indices] == label)
        total_count = np.sum(labels == label)
        print(f"  Class {label}: Train={train_count}, Val={val_count}, Test={test_count}, Total={total_count}")
    
    return train_indices, val_indices, test_indices


def create_data_loaders(traces, labels, train_indices, val_indices, test_indices, batch_size=64):
    """Create PyTorch DataLoaders for train, validation, and test sets.
    
    Args:
        traces: Array of trace data
        labels: Array of corresponding labels
        train_indices, val_indices, test_indices: Arrays of indices for each split
        batch_size: Batch size for DataLoaders
        
    Returns:
        train_loader, val_loader, test_loader: PyTorch DataLoaders
        scaler: Fitted StandardScaler object
    """
    print(f"\nCreating DataLoaders with batch size {batch_size}...")
    
    # Create datasets - fit scaler only on training data
    train_dataset = FingerprintDataset(traces[train_indices], labels[train_indices], scaler=None, fit_scaler=True)
    val_dataset = FingerprintDataset(traces[val_indices], labels[val_indices], scaler=train_dataset.scaler, fit_scaler=False)
    test_dataset = FingerprintDataset(traces[test_indices], labels[test_indices], scaler=train_dataset.scaler, fit_scaler=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Validation loader: {len(val_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader, train_dataset.scaler


class FingerprintClassifier(nn.Module):
    """Basic neural network model for website fingerprinting classification."""
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(FingerprintClassifier, self).__init__()
        
        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        conv_output_size = input_size // 8  # After two 2x pooling operations
        self.fc_input_size = conv_output_size * 64
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Reshape for 1D convolution: (batch_size, 1, input_size)
        x = x.unsqueeze(1)
        
        # Convolutional layers
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten for fully connected layers
        x = x.view(-1, self.fc_input_size)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
        
class ComplexFingerprintClassifier(nn.Module):
    """A more complex neural network model for website fingerprinting classification."""
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(ComplexFingerprintClassifier, self).__init__()
        
        # 1D Convolutional layers with batch normalization
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        conv_output_size = input_size // 8  # After three 2x pooling operations
        self.fc_input_size = conv_output_size * 128
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, hidden_size*2)
        self.bn4 = nn.BatchNorm1d(hidden_size*2)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(hidden_size*2, hidden_size)
        self.bn5 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Reshape for 1D convolution: (batch_size, 1, input_size)
        x = x.unsqueeze(1)
        
        # Convolutional layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Flatten for fully connected layers
        x = x.view(-1, self.fc_input_size)
        
        # Fully connected layers
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x



def train(model, train_loader, val_loader, criterion, optimizer, epochs, model_save_path, model_name="Model"):
    """Train a PyTorch model with validation and early stopping.
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        epochs: Number of epochs to train
        model_save_path: Path to save the best model
        model_name: Name of the model for logging
    Returns:
        best_val_loss: Best validation loss achieved
        training_history: Dictionary with training history
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"🔧 Training {model_name} on device: {device}")
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience = 3  # Early stopping patience
    patience_counter = 0
    
    print(f"📊 Training Configuration:")
    print(f"   - Epochs: {epochs}")
    print(f"   - Early stopping patience: {patience}")
    print(f"   - Device: {device}")
    print(f"   - Optimizer: {type(optimizer).__name__}")
    print(f"   - Learning rate: {optimizer.param_groups[0]['lr']}")
    print("-" * 50)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (traces, labels) in enumerate(train_loader):
            traces, labels = traces.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(traces)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_accuracy = correct / total
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_accuracy)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for traces, labels in val_loader:
                traces, labels = traces.to(device), labels.to(device)
                outputs = model(traces)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_accuracy = val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_accuracy)
        
        # Print progress
        print(f'Epoch {epoch+1:3d}/{epochs} | '
              f'Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_accuracy:.4f} | '
              f'Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_accuracy:.4f}', end='')
        
        # Check for best model (based on validation loss)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f' ✅ Best model saved!')
        else:
            patience_counter += 1
            print(f' (patience: {patience_counter}/{patience})')
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"\n⏹️  Early stopping triggered after {epoch + 1} epochs")
            print(f"   Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
            break
    
    # Training completed
    if patience_counter < patience:
        print(f"\n✅ Training completed after {epochs} epochs")
    print(f"🏆 Best model: Epoch {best_epoch}, Validation Loss: {best_val_loss:.4f}")
    
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch
    }
    
    return best_val_loss, training_history



def evaluate(model, test_loader, website_names, model_name="Model"):
    """Evaluate a PyTorch model on the test set and show detailed classification metrics.
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for testing data
        website_names: List of website names for classification report
        model_name: Name of the model for logging
    Returns:
        test_accuracy: Overall test accuracy
        classification_metrics: Dictionary with detailed metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"\n📊 Evaluating {model_name} on test set...")
    
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for traces, labels in test_loader:
            traces, labels = traces.to(device), labels.to(device)
            outputs = model(traces)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_accuracy = correct / total
    
    # Print detailed classification report
    print(f"\n📈 {model_name} Test Results:")
    print(f"   Overall Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"   Correct Predictions: {correct}/{total}")
    print("-" * 60)
    
    # Generate classification report with detailed metrics
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    # Per-class metrics
    print("Per-class Performance:")
    print(f"{'Class':<30} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 70)
    
    for i, website in enumerate(website_names):
        print(f"{website:<30} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {support[i]:<10}")
    
    # Overall metrics
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)
    
    print("-" * 70)
    print(f"{'Macro Average':<30} {macro_precision:<10.4f} {macro_recall:<10.4f} {macro_f1:<10.4f} {np.sum(support):<10}")
    print(f"{'Weighted Average':<30} {weighted_precision:<10.4f} {weighted_recall:<10.4f} {weighted_f1:<10.4f} {np.sum(support):<10}")
    
    # Detailed classification report
    print(f"\n📋 Detailed Classification Report for {model_name}:")
    print(classification_report(
        all_labels, 
        all_preds, 
        target_names=website_names,
        zero_division=0,
        digits=4
    ))
    
    # Store metrics for comparison
    classification_metrics = {
        'accuracy': test_accuracy,
        'precision_macro': macro_precision,
        'recall_macro': macro_recall,
        'f1_macro': macro_f1,
        'precision_weighted': weighted_precision,
        'recall_weighted': weighted_recall,
        'f1_weighted': weighted_f1,
        'per_class_precision': precision,
        'per_class_recall': recall,
        'per_class_f1': f1,
        'per_class_support': support
    }
    
    return test_accuracy, classification_metrics


def main():
    """ Implement the main function to train and evaluate the models.
    1. Load the dataset from the JSON file, probably using a custom Dataset class
    2. Split the dataset into training and testing sets
    3. Create data loader for training and testing
    4. Define the models to train
    5. Train and evaluate each model
    6. Print comparison of results
    """
    print("🚀 Starting Website Fingerprinting Model Training")
    print("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset file not found: {DATASET_PATH}")
        print("Please run collect.py first to generate the dataset.")
        return
    
    try:
        # 1. Load the dataset from the JSON file
        traces, labels, website_names, website_to_idx = load_dataset(DATASET_PATH)
        print(f"✅ Dataset loaded successfully")
        print(f"   - Total samples: {len(traces)}")
        print(f"   - Number of classes: {len(website_names)}")
        print(f"   - Input size: {traces.shape[1]}")
        
        # 2. Split the dataset into training, validation, and testing sets
        train_indices, val_indices, test_indices = create_data_splits(
            traces, labels, 
            train_split=TRAIN_SPLIT, 
            val_split=VAL_SPLIT, 
            test_split=TEST_SPLIT
        )
        
        # 3. Create data loaders for training, validation, and testing
        train_loader, val_loader, test_loader, scaler = create_data_loaders(
            traces, labels, 
            train_indices, val_indices, test_indices, 
            batch_size=BATCH_SIZE
        )
        
        print("✅ Data preparation completed successfully!")
        print("\n📊 Dataset Statistics:")
        print(f"   - Training samples: {len(train_indices)} ({len(train_indices)/len(traces)*100:.1f}%)")
        print(f"   - Validation samples: {len(val_indices)} ({len(val_indices)/len(traces)*100:.1f}%)")
        print(f"   - Test samples: {len(test_indices)} ({len(test_indices)/len(traces)*100:.1f}%)")
        print(f"   - Batch size: {BATCH_SIZE}")
        print(f"   - Number of batches per epoch: {len(train_loader)}")
        
        print("\n🎯 Ready for model training!")
        print("Data loaders created and standardized.")
        print("Models defined and ready to train.")
        
        # Show sample from each class
        print("\n📋 Sample distribution verification:")
        for i, website in enumerate(website_names):
            train_count = np.sum(labels[train_indices] == i)
            val_count = np.sum(labels[val_indices] == i)
            test_count = np.sum(labels[test_indices] == i)
            print(f"   {website}: Train={train_count}, Val={val_count}, Test={test_count}")
        
        # 4. Define the models to train
        num_classes = len(website_names)
        print(f"\n🧠 Initializing models for {num_classes} classes...")
        
        model1 = FingerprintClassifier(INPUT_SIZE, HIDDEN_SIZE, num_classes)
        model2 = ComplexFingerprintClassifier(INPUT_SIZE, HIDDEN_SIZE, num_classes)
        
        model1_params = sum(p.numel() for p in model1.parameters())
        model2_params = sum(p.numel() for p in model2.parameters())
        
        print(f"   - Basic Model: {model1_params:,} parameters")
        print(f"   - Complex Model: {model2_params:,} parameters")
        
        # Model save paths
        model1_path = os.path.join(MODELS_DIR, "basic_fingerprint_model.pth")
        model2_path = os.path.join(MODELS_DIR, "complex_fingerprint_model.pth")
        
        # 5. Train and evaluate each model
        print("\n" + "="*60)
        print("🚀 STARTING MODEL TRAINING")
        print("="*60)
        
        # Define loss function and optimizers
        criterion = nn.CrossEntropyLoss()
        
        results = {}
        
        # Train Model 1 (Basic)
        print(f"\n🎯 Training Model 1: Basic FingerprintClassifier")
        print("-" * 50)
        
        optimizer1 = optim.Adam(model1.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        
        best_val_loss1, history1 = train(
            model=model1,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer1,
            epochs=EPOCHS,
            model_save_path=model1_path,
            model_name="Basic Model"
        )
        
        # Load best model for evaluation
        model1.load_state_dict(torch.load(model1_path))
        test_acc1, metrics1 = evaluate(model1, test_loader, website_names, "Basic Model")
        
        results['basic_model'] = {
            'best_val_loss': best_val_loss1,
            'test_accuracy': test_acc1,
            'metrics': metrics1,
            'training_history': history1
        }
        
        # Train Model 2 (Complex)
        print(f"\n🎯 Training Model 2: Complex FingerprintClassifier")
        print("-" * 50)
        
        optimizer2 = optim.Adam(model2.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        
        best_val_loss2, history2 = train(
            model=model2,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer2,
            epochs=EPOCHS,
            model_save_path=model2_path,
            model_name="Complex Model"
        )
        
        # Load best model for evaluation
        model2.load_state_dict(torch.load(model2_path))
        test_acc2, metrics2 = evaluate(model2, test_loader, website_names, "Complex Model")
        
        results['complex_model'] = {
            'best_val_loss': best_val_loss2,
            'test_accuracy': test_acc2,
            'metrics': metrics2,
            'training_history': history2
        }
        
        # 6. Print comparison of results
        print("\n" + "="*60)
        print("🏆 FINAL MODEL COMPARISON")
        print("="*60)
        
        print(f"\n📊 Training Summary:")
        print(f"{'Model':<20} {'Best Val Loss':<15} {'Test Accuracy':<15} {'Parameters':<15}")
        print("-" * 65)
        print(f"{'Basic Model':<20} {best_val_loss1:<15.4f} {test_acc1:<15.4f} {model1_params:<15,}")
        print(f"{'Complex Model':<20} {best_val_loss2:<15.4f} {test_acc2:<15.4f} {model2_params:<15,}")
        
        print(f"\n📈 Detailed Performance Metrics:")
        print(f"{'Model':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 56)
        print(f"{'Basic Model':<20} {metrics1['precision_macro']:<12.4f} {metrics1['recall_macro']:<12.4f} {metrics1['f1_macro']:<12.4f}")
        print(f"{'Complex Model':<20} {metrics2['precision_macro']:<12.4f} {metrics2['recall_macro']:<12.4f} {metrics2['f1_macro']:<12.4f}")
        
        # Determine best model
        if test_acc1 > test_acc2:
            best_model_name = "Basic Model"
            best_accuracy = test_acc1
        else:
            best_model_name = "Complex Model"  
            best_accuracy = test_acc2
            
        print(f"\n🥇 Best Performing Model: {best_model_name}")
        print(f"   Test Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        
        # Save results summary
        results_summary = {
            'basic_model_accuracy': test_acc1,
            'complex_model_accuracy': test_acc2,
            'basic_model_metrics': {
                'precision': metrics1['precision_macro'],
                'recall': metrics1['recall_macro'],
                'f1': metrics1['f1_macro']
            },
            'complex_model_metrics': {
                'precision': metrics2['precision_macro'],
                'recall': metrics2['recall_macro'],
                'f1': metrics2['f1_macro']
            },
            'best_model': best_model_name,
            'website_names': website_names
        }
        
        # Save results to JSON
        results_path = os.path.join(MODELS_DIR, "training_results.json")
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\n💾 Models saved:")
        print(f"   - Basic Model: {model1_path}")
        print(f"   - Complex Model: {model2_path}")
        print(f"   - Results Summary: {results_path}")
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error during data preparation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
