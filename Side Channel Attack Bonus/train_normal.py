import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict

# Configuration
DATASET_PATH = "merged_dataset.json"
MODELS_DIR = "saved_models"
BATCH_SIZE = 64
EPOCHS = 50  
LEARNING_RATE = 1e-4
K_FOLDS = 5  # Number of folds for cross validation
INPUT_SIZE = 1000  
HIDDEN_SIZE = 128

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)


class FingerprintDataset(Dataset):
    """Custom Dataset class for website fingerprinting data."""
    
    def __init__(self, traces, labels, scaler=None, fit_scaler=True):
        """
        Args:
            traces: List or array of trace data (already normalized)
            labels: List or array of corresponding labels
            scaler: Not used - data is pre-normalized
            fit_scaler: Not used - data is pre-normalized
        """
        # Data is already normalized by the merger script, so we skip normalization
        self.traces = np.array(traces, dtype=np.float32)
        self.labels = np.array(labels, dtype=np.int64)
        self.scaler = None  # No scaler needed since data is pre-normalized
    
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
    
    # Extract unique websites using the existing website_index mapping
    website_index_to_name = {}
    for item in data:
        website_index_to_name[item['website_index']] = item['website']
    
    # Sort by website index to maintain consistent ordering
    sorted_indices = sorted(website_index_to_name.keys())
    websites = [website_index_to_name[idx] for idx in sorted_indices]
    website_to_idx = {website: idx for idx, website in enumerate(websites)}
    
    print(f"Found {len(websites)} websites:")
    for idx, website in enumerate(websites):
        count = sum(1 for item in data if item['website_index'] == idx)
        print(f"  {idx}: {website} ({count} samples)")
    
    # Prepare traces and labels using existing website_index
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
        # Use the existing website_index from the dataset
        labels.append(item['website_index'])
    
    return np.array(traces), np.array(labels), websites, website_to_idx


def create_kfold_splits(traces, labels, k_folds=5, random_state=42):
    """Create k-fold cross validation splits.
    
    Args:
        traces: Array of trace data
        labels: Array of corresponding labels
        k_folds: Number of folds for cross validation
        random_state: Random seed for reproducibility
        
    Returns:
        List of (train_indices, val_indices) tuples for each fold
    """
    print(f"\nCreating {k_folds}-fold cross validation splits...")
    
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    fold_splits = []
    
    for fold, (train_indices, val_indices) in enumerate(skf.split(traces, labels)):
        print(f"Fold {fold + 1}: Train={len(train_indices)}, Val={len(val_indices)}")
        
        # Print class distribution for this fold
        unique_labels = np.unique(labels)
        print(f"  Class distribution:")
        for label in unique_labels:
            train_count = np.sum(labels[train_indices] == label)
            val_count = np.sum(labels[val_indices] == label)
            print(f"    Class {label}: Train={train_count}, Val={val_count}")
        
        fold_splits.append((train_indices, val_indices))
    
    return fold_splits


def create_fold_data_loaders(traces, labels, train_indices, val_indices, batch_size=64):
    """Create PyTorch DataLoaders for a specific fold.
    
    Args:
        traces: Array of trace data (already normalized)
        labels: Array of corresponding labels
        train_indices, val_indices: Arrays of indices for train and validation
        batch_size: Batch size for DataLoaders
        
    Returns:
        train_loader, val_loader: PyTorch DataLoaders
    """
    # Create datasets - no normalization needed since data is pre-normalized
    train_dataset = FingerprintDataset(traces[train_indices], labels[train_indices])
    val_dataset = FingerprintDataset(traces[val_indices], labels[val_indices])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader


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
    patience = 5  # Early stopping patience
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
    """Main function to train and evaluate models using 5-fold cross-validation.
    1. Load the dataset from the JSON file
    2. Create 5-fold cross-validation splits
    3. Train each model on all folds
    4. Evaluate and compare results
    """
    print("🚀 Starting Website Fingerprinting Model Training with 5-Fold Cross-Validation")
    print("=" * 80)
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset file not found: {DATASET_PATH}")
        print("Please run merger.py first to generate the dataset.")
        return
    
    try:
        # 1. Load the dataset from the JSON file
        traces, labels, website_names, website_to_idx = load_dataset(DATASET_PATH)
        print(f"✅ Dataset loaded successfully")
        print(f"   - Total samples: {len(traces)}")
        print(f"   - Number of classes: {len(website_names)}")
        print(f"   - Input size: {traces.shape[1]}")
        
        # 2. Create 5-fold cross-validation splits
        fold_splits = create_kfold_splits(traces, labels, k_folds=K_FOLDS)
        
        # 3. Define the models to train
        num_classes = len(website_names)
        print(f"\n🧠 Initializing models for {num_classes} classes...")
        
        # Model types and their parameters for comparison
        model_configs = [
            ("Basic", FingerprintClassifier),
            ("Complex", ComplexFingerprintClassifier)
        ]
        
        # Results storage for cross-validation
        all_results = {}
        
        # Define loss function
        criterion = nn.CrossEntropyLoss()
        
        # Train each model type using 5-fold cross-validation
        for model_name, model_class in model_configs:
            print(f"\n" + "="*80)
            print(f"🎯 Training {model_name} Model with 5-Fold Cross-Validation")
            print("="*80)
            
            # Initialize model to get parameter count
            sample_model = model_class(INPUT_SIZE, HIDDEN_SIZE, num_classes)
            model_params = sum(p.numel() for p in sample_model.parameters())
            print(f"   - {model_name} Model: {model_params:,} parameters")
            
            fold_results = []
            
            # Train on each fold
            for fold_idx, (train_indices, val_indices) in enumerate(fold_splits):
                print(f"\n📊 Fold {fold_idx + 1}/{K_FOLDS}")
                print("-" * 50)
                
                # Create data loaders for this fold
                train_loader, val_loader = create_fold_data_loaders(
                    traces, labels, train_indices, val_indices, batch_size=BATCH_SIZE
                )
                
                # Initialize model for this fold
                model = model_class(INPUT_SIZE, HIDDEN_SIZE, num_classes)
                
                # Use AdamW optimizer with weight decay
                optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
                
                # Model save path for this fold
                model_save_path = os.path.join(MODELS_DIR, f"{model_name.lower()}_model_fold_{fold_idx + 1}.pth")
                
                # Train the model
                best_val_loss, training_history = train(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    epochs=EPOCHS,
                    model_save_path=model_save_path,
                    model_name=f"{model_name} Model (Fold {fold_idx + 1})"
                )
                
                # Load best model for evaluation on validation set
                model.load_state_dict(torch.load(model_save_path))
                val_accuracy, val_metrics = evaluate(model, val_loader, website_names, 
                                                   f"{model_name} Model (Fold {fold_idx + 1})")
                
                # Store fold results
                fold_result = {
                    'fold': fold_idx + 1,
                    'best_val_loss': best_val_loss,
                    'val_accuracy': val_accuracy,
                    'val_metrics': val_metrics,
                    'training_history': training_history,
                    'model_path': model_save_path
                }
                fold_results.append(fold_result)
                
                print(f"✅ Fold {fold_idx + 1} completed - Val Accuracy: {val_accuracy:.4f}")
            
            # Calculate average performance across all folds
            avg_val_accuracy = np.mean([result['val_accuracy'] for result in fold_results])
            avg_val_loss = np.mean([result['best_val_loss'] for result in fold_results])
            avg_precision = np.mean([result['val_metrics']['precision_macro'] for result in fold_results])
            avg_recall = np.mean([result['val_metrics']['recall_macro'] for result in fold_results])
            avg_f1 = np.mean([result['val_metrics']['f1_macro'] for result in fold_results])
            
            # Calculate standard deviations
            std_val_accuracy = np.std([result['val_accuracy'] for result in fold_results])
            std_val_loss = np.std([result['best_val_loss'] for result in fold_results])
            std_precision = np.std([result['val_metrics']['precision_macro'] for result in fold_results])
            std_recall = np.std([result['val_metrics']['recall_macro'] for result in fold_results])
            std_f1 = np.std([result['val_metrics']['f1_macro'] for result in fold_results])
            
            # Store overall results for this model
            all_results[model_name.lower() + '_model'] = {
                'model_name': model_name,
                'model_params': model_params,
                'fold_results': fold_results,
                'avg_val_accuracy': avg_val_accuracy,
                'std_val_accuracy': std_val_accuracy,
                'avg_val_loss': avg_val_loss,
                'std_val_loss': std_val_loss,
                'avg_precision': avg_precision,
                'std_precision': std_precision,
                'avg_recall': avg_recall,
                'std_recall': std_recall,
                'avg_f1': avg_f1,
                'std_f1': std_f1
            }
            
            # Print summary for this model
            print(f"\n📊 {model_name} Model Cross-Validation Summary:")
            print(f"   - Average Val Accuracy: {avg_val_accuracy:.4f} ± {std_val_accuracy:.4f}")
            print(f"   - Average Val Loss: {avg_val_loss:.4f} ± {std_val_loss:.4f}")
            print(f"   - Average Precision: {avg_precision:.4f} ± {std_precision:.4f}")
            print(f"   - Average Recall: {avg_recall:.4f} ± {std_recall:.4f}")
            print(f"   - Average F1-Score: {avg_f1:.4f} ± {std_f1:.4f}")
        
        # 4. Print final comparison of all models
        print("\n" + "="*80)
        print("🏆 FINAL MODEL COMPARISON (5-Fold Cross-Validation)")
        print("="*80)
        
        print(f"\n📊 Cross-Validation Results Summary:")
        print(f"{'Model':<15} {'Avg Accuracy':<15} {'Std Accuracy':<15} {'Avg F1-Score':<15} {'Parameters':<15}")
        print("-" * 75)
        
        for model_key, results in all_results.items():
            print(f"{results['model_name']:<15} "
                  f"{results['avg_val_accuracy']:<15.4f} "
                  f"{results['std_val_accuracy']:<15.4f} "
                  f"{results['avg_f1']:<15.4f} "
                  f"{results['model_params']:<15,}")
        
        print(f"\n📈 Detailed Performance Metrics:")
        print(f"{'Model':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Val Loss':<12}")
        print("-" * 63)
        
        for model_key, results in all_results.items():
            print(f"{results['model_name']:<15} "
                  f"{results['avg_precision']:.4f}±{results['std_precision']:.3f} "
                  f"{results['avg_recall']:.4f}±{results['std_recall']:.3f} "
                  f"{results['avg_f1']:.4f}±{results['std_f1']:.3f} "
                  f"{results['avg_val_loss']:.4f}±{results['std_val_loss']:.3f}")
        
        # Determine best model based on average validation accuracy
        best_model_key = max(all_results.keys(), key=lambda k: all_results[k]['avg_val_accuracy'])
        best_model_results = all_results[best_model_key]
        
        print(f"\n🥇 Best Performing Model: {best_model_results['model_name']}")
        print(f"   Average Validation Accuracy: {best_model_results['avg_val_accuracy']:.4f} ± {best_model_results['std_val_accuracy']:.4f}")
        print(f"   Average F1-Score: {best_model_results['avg_f1']:.4f} ± {best_model_results['std_f1']:.4f}")
        
        # Save comprehensive results
        results_summary = {
            'cross_validation_results': all_results,
            'best_model': best_model_results['model_name'],
            'best_model_accuracy': best_model_results['avg_val_accuracy'],
            'best_model_std': best_model_results['std_val_accuracy'],
            'k_folds': K_FOLDS,
            'website_names': website_names,
            'total_samples': len(traces),
            'num_classes': num_classes,
            'training_config': {
                'epochs': EPOCHS,
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'optimizer': 'AdamW',
                'early_stopping_patience': 5
            }
        }
        
        # Save results to JSON
        results_path = os.path.join(MODELS_DIR, "cross_validation_results.json")
        with open(results_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            # Clean results for JSON serialization
            json_results = json.loads(json.dumps(results_summary, default=convert_numpy))
            json.dump(json_results, f, indent=2)
        
        print(f"\n💾 Results saved:")
        print(f"   - Cross-validation results: {results_path}")
        print(f"   - Model files saved in: {MODELS_DIR}")
        
        print("\n" + "="*80)
        print("✅ 5-FOLD CROSS-VALIDATION TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
