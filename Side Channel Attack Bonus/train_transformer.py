import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

# Configuration
DATASET_PATH = "merged_dataset.json"
MODELS_DIR = "saved_models"
BATCH_SIZE = 64
EPOCHS = 300  # Extended for deeper convergence
LEARNING_RATE = 5e-5  # Reduced learning rate for stability
WARMUP_EPOCHS = 15  # Extended warmup
TRAIN_SPLIT = 0.6  # 60% for training
VAL_SPLIT = 0.2    # 20% for validation
TEST_SPLIT = 0.2   # 20% for testing
INPUT_SIZE = 1000
HIDDEN_SIZE = 1024  # Reduced from 2048 to prevent overfitting
PATIENCE = 25  # Increased for stability

# GPU optimizations
torch.backends.cudnn.benchmark = True  # Optimize cudnn for consistent input sizes
torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        self.scale = self.head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        batch_size, channels, seq_len = x.size()
        
        q = self.q_linear(x.transpose(1, 2)).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x.transpose(1, 2)).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x.transpose(1, 2)).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = self.softmax(attn_scores)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_linear(attn_output).transpose(1, 2)


class AttentionCNNClassifier(nn.Module):
    """Deep CNN with multi-head attention and corrected residual connections."""
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(AttentionCNNClassifier, self).__init__()
        
        # Initial adjustment for residual
        self.initial_conv = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=1, stride=1)
        
        # Convolutional layers with dropout for regularization
        self.conv1 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv_dropout1 = nn.Dropout(0.1)
        
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv_dropout2 = nn.Dropout(0.15)
        
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(512)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv_dropout3 = nn.Dropout(0.2)
        
        self.conv4 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(1024)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv_dropout4 = nn.Dropout(0.25)
        
        self.conv5 = nn.Conv1d(in_channels=1024, out_channels=2048, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(2048)
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv_dropout5 = nn.Dropout(0.3)
        
        # Attention mechanism
        self.attention = MultiHeadAttention(embed_dim=2048, num_heads=8)
        
        # Global context
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers with increased dropout for regularization
        self.fc_input_size = 2048  # Matches the output channels after global pooling
        self.fc1 = nn.Linear(self.fc_input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(0.6)  # Increased from 0.4
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout2 = nn.Dropout(0.5)  # Increased from 0.3
        
        # Residual connection adjustment
        self.residual_adjust = nn.Conv1d(in_channels=64, out_channels=2048, kernel_size=1, stride=1)
        
        # Activation functions
        self.relu = nn.ReLU()
        
        # Initialize weights for stability
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights for better stability."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial processing for residual
        residual = self.initial_conv(x.unsqueeze(1))  # Shape: (batch_size, 64, 1000)
        x = self.relu(residual)
        
        # Convolutional layers with dropout
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.conv_dropout1(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.conv_dropout2(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.conv_dropout3(x)
        
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = self.conv_dropout4(x)
        
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.pool5(x)  # Shape: (batch_size, 2048, seq_len/32)
        x = self.conv_dropout5(x)
        
        # Adjust residual to match dimensions
        residual = self.pool1(self.pool2(self.pool3(self.pool4(self.pool5(residual)))))  # Match pooling
        if residual.shape[2] != x.shape[2]:
            residual = nn.functional.interpolate(residual, size=x.shape[2:], mode='linear', align_corners=False)
        residual = self.residual_adjust(residual)  # Adjust channels to 2048
        x = x + residual[:, :2048, :]  # Residual connection
        
        # Attention mechanism
        x = self.attention(x)  # Shape: (batch_size, 2048, seq_len/32)
        x = x.sum(dim=2)  # (batch_size, 2048)
        
        # Global context
        x = self.global_pool(x.unsqueeze(-1)).squeeze(-1)  # (batch_size, 2048)
        
        # Fully connected layers
        x = self.relu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.dropout2(self.fc2(x))
        
        return x


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, model_save_path, patience, warmup_epochs):
    """Train a PyTorch model with warmup and early stopping using validation set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    model.to(device)
    
    # Enable mixed precision training for better GPU utilization
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    best_accuracy = 0.0
    best_val_loss = float('inf')  # Track best validation loss
    best_epoch = 0
    current_patience = 0
    
    for epoch in range(epochs):
        # Learning rate warmup
        if epoch < warmup_epochs:
            lr = LEARNING_RATE * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', leave=False)
        for traces, labels in train_pbar:
            traces, labels = traces.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Use mixed precision training if available
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(traces)
                    loss = criterion(outputs, labels)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"NaN loss detected at epoch {epoch+1}, skipping batch")
                    continue
                    
                scaler.scale(loss).backward()
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(traces)
                loss = criterion(outputs, labels)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"NaN loss detected at epoch {epoch+1}, skipping batch")
                    continue
                    
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            running_loss += loss.item() * traces.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            current_accuracy = correct / total
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_accuracy:.4f}'
            })
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct / total
        
        # Check for NaN in epoch metrics
        if torch.isnan(torch.tensor(epoch_loss)):
            print(f"NaN epoch loss detected at epoch {epoch+1}")
            epoch_loss = float('inf')  # Set to infinity to avoid saving this model
        
        # Evaluation
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]', leave=False)
            for traces, labels in val_pbar:
                traces, labels = traces.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = model(traces)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(traces)
                    loss = criterion(outputs, labels)
                
                # Check for NaN loss in validation
                if torch.isnan(loss):
                    print(f"NaN loss detected in validation at epoch {epoch+1}, skipping batch")
                    continue
                
                running_loss += loss.item() * traces.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                current_accuracy = correct / total
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_accuracy:.4f}'
                })
        
        val_loss = running_loss / len(val_loader.dataset)
        val_accuracy = correct / total
        
        # Check for NaN in validation metrics
        if torch.isnan(torch.tensor(val_loss)):
            print(f"NaN validation loss detected at epoch {epoch+1}")
            val_loss = float('inf')  # Set to infinity
        
        # Update learning rate
        scheduler.step()
        
        # Print status
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
        
        # Early stopping based on validation loss (better for overfitting)
        if val_loss < best_val_loss and val_accuracy > 0.70:  # Lowered threshold since we're improving
            best_val_loss = val_loss
            best_accuracy = val_accuracy
            best_epoch = epoch
            current_patience = 0
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved with validation loss: {best_val_loss:.4f}, accuracy: {best_accuracy:.4f}')
        else:
            current_patience += 1
            if current_patience >= patience:
                print(f'Early stopping triggered after epoch {epoch+1}')
                break
    
    # Load the best model
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    print(f'Best accuracy: {best_accuracy:.4f} at epoch {best_epoch+1}')
    return best_accuracy

def evaluate(model, test_loader, website_names):
    """Evaluate a PyTorch model on the test set and show classification report with website names."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        eval_pbar = tqdm(test_loader, desc='Evaluating', leave=False)
        for traces, labels in eval_pbar:
            traces = traces.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(traces)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            eval_pbar.set_postfix({'Batch': f'{len(all_preds)}/{len(test_loader.dataset)}'})
    
    # Print classification report with website names instead of indices
    print("\nClassification Report:")
    classification_rep = classification_report(
        all_labels, 
        all_preds, 
        target_names=website_names,
        zero_division=1,
        output_dict=True
    )
    print(classification_report(
        all_labels, 
        all_preds, 
        target_names=website_names,
        zero_division=1
    ))
    
    return all_preds, all_labels, classification_rep


class FingerprintDataset(Dataset):
    """Custom Dataset for website fingerprinting data with augmented normalization."""
    
    def __init__(self, data):
        self.data = data
        self.traces = []
        self.labels = []
        
        for item in data:
            # Ensure trace_data is exactly INPUT_SIZE (truncate or pad)
            trace = item['trace_data'][:INPUT_SIZE]
            if len(trace) < INPUT_SIZE:
                trace.extend([0] * (INPUT_SIZE - len(trace)))
            # Normalize and augment with stronger data augmentation
            trace = np.array(trace, dtype=np.float32)
            trace = (trace - np.mean(trace)) / (np.std(trace) + 1e-8)  # Z-score normalization
            
            # Enhanced data augmentation for better generalization
            noise = np.random.normal(0, 0.02, INPUT_SIZE)  # Increased noise
            shift = np.random.randint(-40, 41)  # Increased shift range
            scale = np.random.uniform(0.9, 1.1)  # Add scaling augmentation
            
            # Apply augmentations
            trace = trace * scale + noise  # Scale and add noise
            trace = np.roll(trace, shift)  # Apply shift
            
            # Additional augmentation: random dropout of some values
            if np.random.random() < 0.1:  # 10% chance
                dropout_mask = np.random.random(INPUT_SIZE) > 0.05  # Drop 5% of values
                trace = trace * dropout_mask
            
            self.traces.append(trace)
            self.labels.append(item['website_index'])
        
        self.traces = np.array(self.traces)
        self.labels = np.array(self.labels, dtype=np.int64)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        trace = torch.tensor(self.traces[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return trace, label

def main():
    """Main function to train and evaluate the model."""
    # 1. Load the dataset from the JSON file
    with open(DATASET_PATH, 'r') as f:
        dataset_json = json.load(f)
    
    # Create dataset
    dataset = FingerprintDataset(dataset_json)
    
    # Get unique website names and number of classes
    website_names = list(set(item['website'] for item in dataset_json))
    num_classes = len(website_names)
    
    # 2. Split the dataset into train/validation/test sets (60/20/20)
    labels = dataset.labels
    
    # First split: 60% train, 40% temp (which will be split into 20% val, 20% test)
    sss_train = StratifiedShuffleSplit(n_splits=1, train_size=TRAIN_SPLIT, random_state=42)
    train_indices, temp_indices = next(sss_train.split(dataset.traces, labels))
    
    # Second split: Split the 40% temp into 20% val and 20% test (50/50 of the temp)
    temp_labels = labels[temp_indices]
    sss_val_test = StratifiedShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
    val_indices_temp, test_indices_temp = next(sss_val_test.split(dataset.traces[temp_indices], temp_labels))
    
    # Map back to original indices
    val_indices = temp_indices[val_indices_temp]
    test_indices = temp_indices[test_indices_temp]
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # 3. Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=4, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=torch.cuda.is_available())
    
    # 4. Define the model
    model = AttentionCNNClassifier(INPUT_SIZE, HIDDEN_SIZE, num_classes)
    
    # 5. Training setup with improved stability and regularization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Add label smoothing for stability
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-3, eps=1e-8)  # Increased weight decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS)
    model_save_path = os.path.join(MODELS_DIR, 'AttentionCNNClassifier_best.pth')
    
    # 6. Check if model already exists, if yes, skip training
    if os.path.exists(model_save_path):
        print(f'\nModel already exists at {model_save_path}. Skipping training...')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        best_val_accuracy = None  # We don't have validation accuracy from previous training
    else:
        # Train the model using validation set for early stopping
        print('\nTraining AttentionCNNClassifier...')
        best_val_accuracy = train(model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCHS, model_save_path, PATIENCE, WARMUP_EPOCHS)
        
        # Load the best model for final evaluation on test set
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_save_path, map_location=device))
    
    # 7. Evaluate the model on test set
    print('\nEvaluating AttentionCNNClassifier on Test Set...')
    all_preds, all_labels, classification_rep = evaluate(model, test_loader, website_names)
    
    # Calculate final test accuracy
    test_accuracy = sum(pred == label for pred, label in zip(all_preds, all_labels)) / len(all_labels)
    
    # 8. Prepare results for JSON export
    results = {
        "classification_report": classification_rep,
        "final_results": {
            "best_validation_accuracy": best_val_accuracy if best_val_accuracy is not None else "N/A (model was pre-trained)",
            "final_test_accuracy": test_accuracy
        },
        "model_info": {
            "model_name": "AttentionCNNClassifier",
            "test_accuracy": test_accuracy,
            "dataset_split": {
                "train": len(train_dataset),
                "validation": len(val_dataset),
                "test": len(test_dataset)
            },
            "hyperparameters": {
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "hidden_size": HIDDEN_SIZE,
                "input_size": INPUT_SIZE,
                "patience": PATIENCE
            }
        },
        "website_names": website_names
    }
    
    # 9. Save results to JSON file
    results_file = os.path.join(MODELS_DIR, 'training_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    # 10. Print results
    print("\nFinal Results:")
    if best_val_accuracy is not None:
        print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    
    # 11. Print model info
    print("\nModel Info:")
    print(f"AttentionCNNClassifier: Test Accuracy = {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
