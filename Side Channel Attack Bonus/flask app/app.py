from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import json
import os
import time
import random
from sklearn.preprocessing import StandardScaler
import pickle
from scipy.signal import savgol_filter, find_peaks
import scipy.stats
# Optional imports for advanced preprocessing
try:
    from scipy.signal import savgol_filter, find_peaks
    import scipy.stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

app = Flask(__name__)

# Configuration
MODELS_DIR = "saved_models"
CROSS_VALIDATION_RESULTS_PATH = os.path.join(MODELS_DIR, "cross_validation_results.json")

# Model parameters (should match training configuration)
INPUT_SIZE = 1000
HIDDEN_SIZE = 128

# Website mapping (should match training data)
WEBSITES = [
    "https://cse.buet.ac.bd/moodle/",
    "https://google.com", 
    "https://prothomalo.com"
]

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

def load_models():
    """Load the trained models from cross-validation results"""
    try:
        num_classes = len(WEBSITES)
        
        # Load cross-validation results
        with open(CROSS_VALIDATION_RESULTS_PATH, 'r') as f:
            cv_results = json.load(f)
        
        # Find best fold for each model type
        basic_results = cv_results['cross_validation_results']['basic_model']
        complex_results = cv_results['cross_validation_results']['complex_model']
        
        # Find best basic model fold (highest val_accuracy)
        best_basic_fold = max(basic_results['fold_results'], key=lambda x: x['val_accuracy'])
        best_basic_model_path = os.path.join(MODELS_DIR, f"basic_model_fold_{best_basic_fold['fold']}.pth")
        
        # Find best complex model fold (highest val_accuracy)
        best_complex_fold = max(complex_results['fold_results'], key=lambda x: x['val_accuracy'])
        best_complex_model_path = os.path.join(MODELS_DIR, f"complex_model_fold_{best_complex_fold['fold']}.pth")
        
        # Load best basic model
        basic_model = FingerprintClassifier(INPUT_SIZE, HIDDEN_SIZE, num_classes)
        basic_model.load_state_dict(torch.load(best_basic_model_path, map_location='cpu', weights_only=True))
        basic_model.eval()
        
        # Load best complex model
        complex_model = ComplexFingerprintClassifier(INPUT_SIZE, HIDDEN_SIZE, num_classes)
        complex_model.load_state_dict(torch.load(best_complex_model_path, map_location='cpu', weights_only=True))
        complex_model.eval()
        
        # Determine which model is overall better
        best_basic_accuracy = best_basic_fold['val_accuracy']
        best_complex_accuracy = best_complex_fold['val_accuracy']
        
        if best_basic_accuracy > best_complex_accuracy:
            selected_model = basic_model
            model_name = "Basic Model"
            accuracy = best_basic_accuracy
            selected_fold = best_basic_fold['fold']
        else:
            selected_model = complex_model
            model_name = "Complex Model"
            accuracy = best_complex_accuracy
            selected_fold = best_complex_fold['fold']
        
        # Prepare results summary
        results_summary = {
            'cv_results': cv_results,
            'best_basic_fold': best_basic_fold,
            'best_complex_fold': best_complex_fold,
            'selected_model': model_name,
            'selected_fold': selected_fold,
            'selected_accuracy': accuracy,
            'basic_avg_accuracy': basic_results['avg_val_accuracy'],
            'complex_avg_accuracy': complex_results['avg_val_accuracy']
        }
        
        return selected_model, basic_model, complex_model, results_summary
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None, None

# Global variables for models
selected_model, basic_model, complex_model, model_results = load_models()

# Global StandardScaler for consistent data preprocessing
global_scaler = StandardScaler()

def simulate_network_trace():
    """Simulate network trace data for demonstration"""
    # Generate realistic-looking network trace data
    # In a real scenario, this would come from actual network monitoring
    
    # Simulate different patterns for different websites
    patterns = {
        0: [100, 200, 150, 300, 250, 400, 350, 500],  # Moodle pattern
        1: [50, 100, 75, 150, 125, 200, 175, 250],    # Google pattern  
        2: [150, 300, 225, 450, 375, 600, 525, 750]   # Prothom Alo pattern
    }
    
    # Randomly select a pattern or create unknown pattern
    if random.random() < 0.8:  # 80% chance of known pattern
        pattern_key = random.choice([0, 1, 2])
        base_pattern = patterns[pattern_key]
    else:  # 20% chance of unknown pattern
        base_pattern = [random.randint(10, 100) for _ in range(8)]
    
    # Generate full trace by repeating and adding noise
    trace = []
    for i in range(INPUT_SIZE):
        base_value = base_pattern[i % len(base_pattern)]
        noise = random.randint(-20, 20)
        trace.append(max(0, base_value + noise))
    
    return trace

def standardize_trace_data(trace_data):
    """
    Standardize trace data using StandardScaler to match training data preprocessing.
    
    CRITICAL: This function ensures data consistency between training and inference.
    The training data was normalized using sklearn's StandardScaler in merger.py,
    so we must apply the same standardization here to ensure the model receives
    data in the expected format. Without this standardization, the model will
    perform poorly due to input distribution mismatch.
    
    Args:
        trace_data: Raw trace data as list or array
        
    Returns:
        Standardized trace data as numpy array (z-score normalized)
    """
    try:
        trace_array = np.array(trace_data, dtype=np.float32)
        
        # Ensure correct size first
        if len(trace_array) > INPUT_SIZE:
            # Use middle section (most stable part of measurement)
            start_idx = (len(trace_array) - INPUT_SIZE) // 2
            trace_array = trace_array[start_idx:start_idx + INPUT_SIZE]
        elif len(trace_array) < INPUT_SIZE:
            # Pad with zeros to match training data preprocessing
            trace_array = np.pad(trace_array, (0, INPUT_SIZE - len(trace_array)), 'constant', constant_values=0)
        
        # Apply StandardScaler normalization (same as training data)
        # Reshape for sklearn: (n_samples, n_features) -> (n_features, 1)
        trace_reshaped = trace_array.reshape(-1, 1)
        
        # Fit and transform using StandardScaler (z-score normalization)
        # This matches the normalization done in merger.py during training data preparation
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(trace_reshaped)
        
        # Flatten back to 1D array
        return standardized_data.flatten()
        
    except Exception as e:
        print(f"Error in standardization: {e}")
        # Fallback to basic preprocessing
        trace_array = np.array(trace_data, dtype=np.float32)
        if len(trace_array) > INPUT_SIZE:
            trace_array = trace_array[:INPUT_SIZE]
        elif len(trace_array) < INPUT_SIZE:
            trace_array = np.pad(trace_array, (0, INPUT_SIZE - len(trace_array)), 'constant')
        
        # Basic z-score normalization as fallback
        trace_array = (trace_array - np.mean(trace_array)) / (np.std(trace_array) + 1e-8)
        return trace_array

def preprocess_realtime_data(trace_data):
    """Advanced preprocessing for real-time cache sweep data"""
    try:
        trace_array = np.array(trace_data, dtype=np.float32)
        
        # Remove outliers using IQR method
        Q1 = np.percentile(trace_array, 25)
        Q3 = np.percentile(trace_array, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Clip outliers instead of removing (to maintain array size)
        # Apply smoothing filter to reduce noise
        if HAS_SCIPY and len(trace_array) > 10:
            try:
                trace_array = savgol_filter(trace_array, window_length=min(11, len(trace_array)//2*2+1), polyorder=3)
            except:
                pass  # If filtering fails, use original data
        
        # Now apply standardization to match training data format
        # This is the crucial step for consistency with training data
        trace_array = standardize_trace_data(trace_array)
        
        return trace_array
        
    except Exception as e:
        # Fallback to basic preprocessing with standardization
        print(f"Advanced preprocessing failed: {e}, using fallback...")
        trace_array = standardize_trace_data(trace_data)
        return trace_array

def extract_statistical_features(trace_data):
    """Extract statistical features that might be more robust to noise"""
    features = {}
    
    # Basic statistics
    features['mean'] = float(np.mean(trace_data))
    features['std'] = float(np.std(trace_data))
    features['median'] = float(np.median(trace_data))
    
    if HAS_SCIPY:
        features['skewness'] = float(scipy.stats.skew(trace_data))
        features['kurtosis'] = float(scipy.stats.kurtosis(trace_data))
    else:
        features['skewness'] = 0.0
        features['kurtosis'] = 0.0
    
    # Frequency domain features (FFT)
    fft = np.fft.fft(trace_data)
    fft_magnitude = np.abs(fft[:len(fft)//2])
    features['dominant_frequency'] = int(np.argmax(fft_magnitude))
    features['spectral_centroid'] = float(np.sum(fft_magnitude * np.arange(len(fft_magnitude))) / np.sum(fft_magnitude))
    
    # Time domain patterns
    if HAS_SCIPY:
        features['peak_count'] = len(find_peaks(trace_data)[0])
    else:
        features['peak_count'] = 0
    
    features['zero_crossings'] = int(np.sum(np.diff(np.sign(trace_data - np.mean(trace_data))) != 0))
    
    # Autocorrelation features
    autocorr = np.correlate(trace_data, trace_data, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    features['autocorr_peak'] = int(np.argmax(autocorr[1:]) + 1) if len(autocorr) > 1 else 0
    
    return features

def predict_website(trace_data):
    """Predict website from trace data with robust preprocessing"""
    if selected_model is None:
        return {"error": "Model not loaded"}

    try:
        # Step 1: Apply advanced preprocessing (includes standardization)
        trace_array = preprocess_realtime_data(trace_data)
        
        # Step 1.5: Verify standardization was applied correctly
        verification = verify_standardized_data(trace_array, "prediction_input")
        
        # Step 2: Extract statistical features for additional analysis
        # Note: Use original trace_data for feature extraction before standardization
        features = extract_statistical_features(trace_data)
        
        # Step 3: Convert standardized data to tensor
        trace_tensor = torch.tensor(trace_array, dtype=torch.float32).unsqueeze(0)
        
        # Step 4: Make prediction with the standardized data
        with torch.no_grad():
            outputs = selected_model(trace_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Apply confidence threshold for unknown detection
        CONFIDENCE_THRESHOLD = 0.6  # Adjust based on your needs
        
        # Map to website name
        if predicted_class < len(WEBSITES) and confidence > CONFIDENCE_THRESHOLD:
            predicted_website = WEBSITES[predicted_class]
            is_known = True
            certainty = "High" if confidence > 0.8 else "Medium"
        else:
            predicted_website = "Unknown Website (Low Confidence)"
            is_known = False
            certainty = "Low"
        
        # Get all probabilities for display
        all_probs = []
        for i, website in enumerate(WEBSITES):
            all_probs.append({
                'website': website,
                'probability': float(probabilities[0][i].item()),
                'percentage': float(probabilities[0][i].item() * 100)
            })
        
        return {
            'predicted_website': predicted_website,
            'confidence': float(confidence),
            'confidence_percentage': float(confidence * 100),
            'certainty': certainty,
            'is_known': bool(is_known),
            'all_probabilities': all_probs,
            'preprocessing_info': {
                'data_standardized': True,
                'trace_length': int(len(trace_data)),
                'processed_length': int(len(trace_array)),
                'features_extracted': int(len(features)),
                'standardization_verification': verification
            }
        }
        
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html', 
                         websites=WEBSITES,
                         model_results=model_results)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests with cache sweep data"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        if 'traces' not in data:
            # If no traces provided, generate simulated data for demo
            print("⚠️ No cache sweep data provided, generating simulated data...")
            traces = simulate_network_trace()
            metadata = {"source": "simulated", "note": "No cache sweep data received"}
        else:
            traces = data['traces']
            metadata = data.get('metadata', {})
        
        # Validate trace data
        if not traces or len(traces) == 0:
            return jsonify({"error": "Empty trace data"}), 400
        
        print(f"📊 Received cache sweep data:")
        print(f"   - Trace length: {len(traces)}")
        print(f"   - Metadata: {metadata}")
        print(f"   - Sample values: {traces[:10]}...")
        
        # Make prediction using cache sweep data (includes standardization)
        result = predict_website(traces)
        
        # Add metadata to result
        result['trace_info'] = {
            'trace_sample': traces[:50] if len(traces) > 50 else traces  # First 50 points for visualization
        }
        
        if 'preprocessing_info' in result:
            print(f"🔧 Data preprocessing completed:")
            print(f"   - Standardization applied: {result['preprocessing_info']['data_standardized']}")
            print(f"   - Original length: {result['preprocessing_info']['trace_length']}")
            print(f"   - Processed length: {result['preprocessing_info']['processed_length']}")
        
        print(f"🎯 Prediction result: {result['predicted_website']} (confidence: {result['confidence']:.3f})")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error in prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/models/info')
def models_info():
    """Get information about loaded models and cross-validation results"""
    if model_results is None:
        return jsonify({"error": "Models not loaded"}), 500
    
    return jsonify({
        'cross_validation_summary': {
            'k_folds': model_results['cv_results']['k_folds'],
            'total_samples': model_results['cv_results']['total_samples'],
            'websites': model_results['cv_results']['website_names']
        },
        'basic_model': {
            'avg_accuracy': model_results['basic_avg_accuracy'],
            'best_fold': model_results['best_basic_fold']['fold'],
            'best_fold_accuracy': model_results['best_basic_fold']['val_accuracy'],
            'best_fold_metrics': model_results['best_basic_fold']['val_metrics']
        },
        'complex_model': {
            'avg_accuracy': model_results['complex_avg_accuracy'],
            'best_fold': model_results['best_complex_fold']['fold'],
            'best_fold_accuracy': model_results['best_complex_fold']['val_accuracy'],
            'best_fold_metrics': model_results['best_complex_fold']['val_metrics']
        },
        'selected_model': {
            'type': model_results['selected_model'],
            'fold': model_results['selected_fold'],
            'accuracy': model_results['selected_accuracy']
        },
        'overall_best': model_results['cv_results']['best_model'],
        'websites': WEBSITES
    })

def verify_standardized_data(data, trace_id="unknown"):
    """
    Verify that the standardized data has the expected properties.
    Standardized data should have approximately mean=0 and std=1.
    
    Args:
        data: Standardized data array
        trace_id: Identifier for logging purposes
    
    Returns:
        Dictionary with verification results
    """
    try:
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        # Acceptable ranges for standardized data
        mean_acceptable = abs(mean_val) < 0.1  # Should be close to 0
        std_acceptable = 0.8 < std_val < 1.2   # Should be close to 1
        
        verification_result = {
            'mean': float(mean_val),
            'std': float(std_val),
            'mean_acceptable': bool(mean_acceptable),
            'std_acceptable': bool(std_acceptable),
            'properly_standardized': bool(mean_acceptable and std_acceptable)
        }
        
        if not verification_result['properly_standardized']:
            print(f"⚠️  Standardization verification failed for {trace_id}:")
            print(f"    Mean: {mean_val:.4f} (expected: ~0)")
            print(f"    Std:  {std_val:.4f} (expected: ~1)")
        
        return verification_result
        
    except Exception as e:
        print(f"❌ Error verifying standardized data for {trace_id}: {e}")
        return {'error': str(e)}

if __name__ == '__main__':
    if selected_model is not None:
        app.run(debug=False, host='0.0.0.0', port=5001)
