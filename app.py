from flask import Flask, render_template, request, jsonify
import os
import pickle
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score

# Define KELM class (must be identical to the one used during training)
class KELM:
    """
    Kernel Extreme Learning Machine (KELM)
    
    Parameters:
    -----------
    C : float
        Regularization parameter
    kernel : str
        Kernel type ('rbf', 'linear', 'poly')
    sigma : float
        RBF kernel parameter (gamma = 1/(2*sigma^2))
    """
    
    def __init__(self, C=100, kernel='rbf', sigma=15):
        self.C = C
        self.kernel = kernel
        self.sigma = sigma
        self.gamma = 1.0 / (2 * sigma ** 2)  # RBF gamma parameter
        self.X_train = None
        self.beta = None
        
    def rbf_kernel(self, X1, X2):
        """
        RBF (Gaussian) kernel: K(x, y) = exp(-gamma * ||x - y||^2)
        """
        dist_sq = cdist(X1, X2, 'sqeuclidean')
        return np.exp(-self.gamma * dist_sq)
    
    def fit(self, X, y):
        """
        Train KELM model
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        self.X_train = X
        n_samples = X.shape[0]
        
        # Convert labels to one-hot encoding
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        T = np.zeros((n_samples, n_classes))
        for i, class_label in enumerate(self.classes_):
            T[y == class_label, i] = 1
        
        # Compute kernel matrix
        K = self.rbf_kernel(X, X)
        
        # Compute output weights: beta = (K + I/C)^-1 * T
        identity = np.eye(n_samples)
        self.beta = np.linalg.solve(K + identity / self.C, T)
        
        return self
    
    def predict(self, X):
        """
        Predict class labels
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
            
        Returns:
        --------
        y_pred : array-like, shape (n_samples,)
            Predicted class labels
        """
        # Compute kernel matrix between test and train data
        K_test = self.rbf_kernel(X, self.X_train)
        
        # Compute outputs
        outputs = K_test @ self.beta
        
        # Return class with maximum output
        y_pred = self.classes_[np.argmax(outputs, axis=1)]
        
        return y_pred
    
    def predict_proba(self, X):
        """
        Predict class probabilities (confidence scores)
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
            
        Returns:
        --------
        probabilities : array-like, shape (n_samples, n_classes)
            Probability estimates for each class
        """
        # Compute kernel matrix between test and train data
        K_test = self.rbf_kernel(X, self.X_train)
        
        # Compute outputs
        outputs = K_test @ self.beta
        
        # Apply softmax to convert outputs to probabilities
        exp_outputs = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
        probabilities = exp_outputs / np.sum(exp_outputs, axis=1, keepdims=True)
        
        return probabilities
    
    def score(self, X, y):
        """
        Calculate accuracy score
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables untuk model
loaded_model = None
loaded_label_encoder = None
loaded_metadata = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the trained KELM model"""
    global loaded_model, loaded_label_encoder, loaded_metadata
    
    try:
        # Load model (sesuaikan dengan nama file model Anda)
        model_files = [f for f in os.listdir('.') if f.startswith('kelm_best_model') and f.endswith('.pkl')]
        if model_files:
            model_filename = model_files[0]
            with open(model_filename, 'rb') as f:
                loaded_model = pickle.load(f)
            print(f"✓ Model loaded: {model_filename}")
        
        # Load label encoder
        if os.path.exists('label_encoder.pkl'):
            with open('label_encoder.pkl', 'rb') as f:
                loaded_label_encoder = pickle.load(f)
            print("✓ Label encoder loaded")
        
        # Load metadata
        if os.path.exists('model_metadata.pkl'):
            with open('model_metadata.pkl', 'rb') as f:
                loaded_metadata = pickle.load(f)
            print("✓ Metadata loaded")
            
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def extract_features(image_path):
    """
    Extract features from image using ResNet50 pretrained model
    Uses 'avg_pool' layer activation from ResNet50
    """
    try:
        from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
        from tensorflow.keras.preprocessing import image
        
        # Load ResNet50 pretrained model (without top classification layer)
        # Output from 'avg_pool' layer gives 2048 features
        model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        
        # Load and preprocess image
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Extract features from avg_pool layer
        features = model.predict(img_array, verbose=0)
        features = features.flatten()  # Shape: (2048,)
        
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'}), 400
    
    if loaded_model is None or loaded_label_encoder is None:
        return jsonify({'error': 'Model not loaded. Please ensure model files exist.'}), 500
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract features
        features = extract_features(filepath)
        if features is None:
            return jsonify({'error': 'Failed to extract features from image'}), 500
        
        # Reshape features untuk prediksi
        features = features.reshape(1, -1)
        
        # Predict with confidence
        prediction_encoded = loaded_model.predict(features)
        prediction_label = loaded_label_encoder.inverse_transform(prediction_encoded)[0]
        
        # Get confidence scores
        probabilities = loaded_model.predict_proba(features)[0]
        max_confidence = np.max(probabilities) * 100
        
        # Get all class probabilities
        class_confidences = {}
        for i, class_name in enumerate(loaded_label_encoder.classes_):
            class_confidences[class_name] = float(probabilities[i] * 100)
        
        # Read image untuk di-encode ke base64
        with open(filepath, 'rb') as f:
            img_data = f.read()
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        
        # Clean up uploaded file
        os.remove(filepath)
        
        # Return result
        return jsonify({
            'success': True,
            'prediction': prediction_label,
            'image': f'data:image/jpeg;base64,{img_base64}',
            'confidence': float(max_confidence),
            'all_confidences': class_confidences
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/model-info')
def model_info():
    if loaded_metadata:
        return jsonify({
            'success': True,
            'info': {
                'model_name': loaded_metadata['model_name'],
                'kernel': loaded_metadata['kernel'],
                'accuracy': f"{loaded_metadata['mean_accuracy']:.2f}%",
                'best_fold_accuracy': f"{loaded_metadata['best_accuracy']:.2f}%",
                'n_classes': loaded_metadata['n_classes'],
                'classes': loaded_metadata['classes'],
                'date_created': loaded_metadata['date_created']
            }
        })
    else:
        return jsonify({'error': 'Model metadata not available'}), 404

if __name__ == '__main__':
    print("="*70)
    print("FRUIT CLASSIFICATION SYSTEM - KELM")
    print("="*70)
    
    # Load model saat aplikasi start
    if load_model():
        print("\n✓ All models loaded successfully!")
        print("\nStarting Flask application...")
        print("Access the app at: http://127.0.0.1:5000")
        print("="*70)
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n✗ Failed to load models!")
        print("Please ensure the following files exist:")
        print("  - kelm_best_model_fold*.pkl")
        print("  - label_encoder.pkl")
        print("  - model_metadata.pkl")
        print("="*70)
