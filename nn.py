import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

class CallDurationPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.categorical_columns = ['service_type', 'caller_location', 'callee_location']
        self.train_metrics = {}
        self.test_metrics = {}
        
    def create_features(self, df):
        """Create advanced features for the neural network - extract time features from timestamp"""
        X = df.copy()
        
        # Ensure timestamp is datetime
        if 'timestamp' in X.columns:
            X['timestamp'] = pd.to_datetime(X['timestamp'])
            
            # Extract ALL time-based features from timestamp
            X['hour'] = X['timestamp'].dt.hour
            X['day_index'] = X['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
            X['day_name'] = X['timestamp'].dt.day_name()
            X['is_weekend'] = (X['timestamp'].dt.dayofweek >= 5).astype(int)
            X['month'] = X['timestamp'].dt.month
        
        # Create cyclical time features
        X['hour_sin'] = np.sin(2 * np.pi * X['hour'] / 24)
        X['hour_cos'] = np.cos(2 * np.pi * X['hour'] / 24)
        X['day_of_week_sin'] = np.sin(2 * np.pi * X['day_index'] / 7)
        X['day_of_week_cos'] = np.cos(2 * np.pi * X['day_index'] / 7)
        X['month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
        X['month_cos'] = np.cos(2 * np.pi * X['month'] / 12)
        
        # Create interaction features
        X['rel_score_call_count'] = X['relationship_score'] * np.log1p(X['pair_call_count'])
        X['rel_score_time_since'] = X['relationship_score'] * X['time_since_last_call_hours']
        X['weekend_evening'] = ((X['is_weekend']) & (X['hour'] >= 18)).astype(int)
        X['call_frequency'] = 1 / (X['time_since_last_call_hours'] + 1)
        X['rel_score_bin'] = pd.cut(X['relationship_score'], bins=5, labels=False)
        
        return X
    
    def prepare_features(self, X):
        """Prepare feature matrix with one-hot encoding"""
        feature_columns = [
            'relationship_score', 'pair_call_count', 'time_since_last_call_hours',
            'past_avg_duration', 'hour', 'hour_sin', 'hour_cos', 'day_of_week_sin', 
            'day_of_week_cos', 'month_sin', 'month_cos', 'rel_score_call_count',
            'rel_score_time_since', 'weekend_evening', 'call_frequency', 'rel_score_bin'
        ]
        
        X_features = X[feature_columns].copy()
        
        # Add one-hot encoded categorical variables
        categorical_cols = [col for col in self.categorical_columns if col in X.columns]
        for col in categorical_cols:
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=False)
            X_features = pd.concat([X_features, dummies], axis=1)
        
        # Add time-based categorical variables extracted from timestamp
        if 'day_name' in X.columns:
            day_dummies = pd.get_dummies(X['day_name'], prefix='day_name', drop_first=False)
            X_features = pd.concat([X_features, day_dummies], axis=1)
        
        if 'is_weekend' in X.columns:
            X_features['is_weekend'] = X['is_weekend']
        
        # Ensure all columns are numeric
        X_features = X_features.apply(pd.to_numeric, errors='coerce').fillna(0)
        non_numeric_cols = X_features.select_dtypes(include=['object', 'category']).columns
        if len(non_numeric_cols) > 0:
            X_features = X_features.select_dtypes(exclude=['object', 'category'])
        
        return X_features
    
    def train(self, df, test_size=0.2, random_state=42, verbose=True):
        """Train the neural network model - SEPARATE from testing"""
        if verbose:
            print("Creating features...")
        
        X_enhanced = self.create_features(df)
        X_features = self.prepare_features(X_enhanced)
        y = df['duration_sec']
        
        # Store feature columns for prediction
        self.feature_columns = X_features.columns.tolist()
        
        if verbose:
            print(f"Generated {len(self.feature_columns)} features")
            print(f"Training set size: {len(X_features)} samples")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32, 16),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=10
        )
        
        if verbose:
            print("Training Neural Network...")
        
        self.model.fit(X_train_scaled, y_train)
        
        # Store training and test data for evaluation
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        self.X_train_scaled, self.X_test_scaled = X_train_scaled, X_test_scaled
        
        if verbose:
            print(" Model trained successfully!")
        
        return self
    def calculate_mape(self, y_true, y_pred):
        """Calculate Mean Absolute Percentage Error (MAPE)"""
        # Avoid division by zero by adding small epsilon
        epsilon = 1e-10
        return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    def evaluate(self, verbose=True):
        """Evaluate model performance on training and test sets"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Make predictions
        y_train_pred = self.model.predict(self.X_train_scaled)
        y_test_pred = self.model.predict(self.X_test_scaled)
        
        # Calculate comprehensive metrics including MAPE
        self.train_metrics = {
            'r2': r2_score(self.y_train, y_train_pred),
            'rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
            'mae': mean_absolute_error(self.y_train, y_train_pred),
            'mape': self.calculate_mape(self.y_train, y_train_pred)  # Add MAPE
        }
        
        self.test_metrics = {
            'r2': r2_score(self.y_test, y_test_pred),
            'rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
            'mae': mean_absolute_error(self.y_test, y_test_pred),
            'mape': self.calculate_mape(self.y_test, y_test_pred)  # Add MAPE
        }
        
        if verbose:
            self._print_evaluation_metrics()
        
        return self.train_metrics, self.test_metrics
    
    def _print_evaluation_metrics(self):
        """Print detailed evaluation metrics"""
        print("\n" + "="*60)
        print("MODEL EVALUATION METRICS")
        print("="*60)
        
        print(f"\n TRAINING PERFORMANCE:")
        print(f"   R² Score:    {self.train_metrics['r2']:.4f} ({self.train_metrics['r2']*100:.1f}%)")
        print(f"   RMSE:        {self.train_metrics['rmse']:.2f}s ({self.train_metrics['rmse']/60:.1f}min)")
        print(f"   MAE:         {self.train_metrics['mae']:.2f}s ({self.train_metrics['mae']/60:.1f}min)")
        print(f"   MAPE:        {self.train_metrics['mape']:.2f}%")          
        print(f"\nTESTING PERFORMANCE:")
        print(f"   R² Score:    {self.test_metrics['r2']:.4f} ({self.test_metrics['r2']*100:.1f}%)")
        print(f"   RMSE:        {self.test_metrics['rmse']:.2f}s ({self.test_metrics['rmse']/60:.1f}min)")
        print(f"   MAE:         {self.test_metrics['mae']:.2f}s ({self.test_metrics['mae']/60:.1f}min)")
        print(f"   MAPE:        {self.test_metrics['mape']:.2f}%")  
       
        
        print(f"\n ACCURACY ANALYSIS:")
        residuals = self.y_test - self.model.predict(self.X_test_scaled)
        within_30s = (np.abs(residuals) <= 30).sum() / len(residuals) * 100
        within_60s = (np.abs(residuals) <= 60).sum() / len(residuals) * 100
        
        # Add MAPE interpretation
        
        print(f"   Predictions within 30s: {within_30s:.1f}%")
        print(f"   Predictions within 60s: {within_60s:.1f}%")
       
        
     
    def plot_feature_importance(self, top_n=15):
        """Plot feature importance using permutation importance with grouped categorical features"""
        from sklearn.inspection import permutation_importance
        
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Calculate permutation importance
        result = permutation_importance(
            self.model, self.X_test_scaled, self.y_test, 
            n_repeats=10, random_state=42
        )
        
        # Get original feature importance
        importance_df = pd.DataFrame({
            'feature': self.X_test.columns,
            'importance': result.importances_mean,
            'std': result.importances_std
        })
        
        # Group categorical features
        grouped_importance = {}
        
        for feature, importance, std in zip(importance_df['feature'], 
                                        importance_df['importance'], 
                                        importance_df['std']):
            # Group service_type features
            if feature.startswith('service_type_'):
                group_name = 'service_type'
            # Group caller_location features  
            elif feature.startswith('caller_location_'):
                group_name = 'caller_location'
            # Group callee_location features
            elif feature.startswith('callee_location_'):
                group_name = 'callee_location'
            # Group day_name features (rename to days_of_week)
            elif feature.startswith('day_name_'):
                group_name = 'days_of_week'
            # Keep numerical features as is
            else:
                group_name = feature
            
            # Aggregate importance for grouped features
            if group_name in grouped_importance:
                grouped_importance[group_name]['importance'] += importance
                grouped_importance[group_name]['std'] = np.sqrt(grouped_importance[group_name]['std']**2 + std**2)
            else:
                grouped_importance[group_name] = {
                    'importance': importance,
                    'std': std,
                    'type': 'categorical' if group_name in ['service_type', 'caller_location', 'callee_location', 'days_of_week'] else 'numerical'
                }
        
        # Create new DataFrame with grouped features
        grouped_df = pd.DataFrame([
            {'feature': feature, 'importance': data['importance'], 'std': data['std'], 'type': data['type']}
            for feature, data in grouped_importance.items()
        ]).sort_values('importance', ascending=True).tail(top_n)
        
        # Plot with correct colors
        plt.figure(figsize=(10, 8))
        colors = ['#1f77b4' if typ == 'numerical' else 'skyblue' for typ in grouped_df['type']]
        bars = plt.barh(grouped_df['feature'], grouped_df['importance'], 
                        xerr=grouped_df['std'], capsize=5, alpha=0.7, color=colors)
        
        # Add legend for feature types
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#1f77b4', alpha=0.7, label='Numerical Features'),  # Default matplotlib blue
            Patch(facecolor='skyblue', alpha=0.7, label='Categorical Features')  # Sky blue
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.xlabel('Permutation Importance')
        plt.title(f'Top {top_n} Most Important Features (Grouped Categorical)')
        plt.tight_layout()
        plt.show()
        
        # Print feature importance summary
        print("\nFEATURE IMPORTANCE SUMMARY:")
        print("=" * 40)
        for _, row in grouped_df.sort_values('importance', ascending=False).iterrows():
            feature_type = "Categorical" if row['type'] == 'categorical' else "Numerical"
            print(f"{row['feature']:25} {row['importance']:8.4f} ({feature_type})")
        
        return grouped_df
        
    def plot_residuals(self):
        """Plot residual analysis"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        y_test_pred = self.model.predict(self.X_test_scaled)
        residuals = self.y_test - y_test_pred
        
        # Create separate figures for each plot
        
        # Plot 2: Residual distribution with 20-second intervals
        plt.figure(figsize=(10, 6))
        min_residual = residuals.min()
        max_residual = residuals.max()
        bin_edges = np.arange(
            np.floor(min_residual / 20) * 20,
            np.ceil(max_residual / 20) * 20 + 20,
            20
        )

        plt.hist(residuals, bins=bin_edges, alpha=0.7, edgecolor='black')
        plt.xlabel('Residuals (seconds)')
        plt.ylabel('Frequency')
        plt.title('Residual Distribution')
        plt.grid(True, alpha=0.3)
        
        # Customize x-axis ticks
        x_ticks = np.arange(
            np.floor(min_residual / 20) * 20,
            np.ceil(max_residual / 20) * 20 + 20,
            40  # Show every other tick to avoid crowding
        )
        plt.xticks(x_ticks)
        
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.scatter(self.y_test, y_test_pred, alpha=0.6, s=10)
        plt.plot(
            [self.y_test.min(), self.y_test.max()],
            [self.y_test.min(), self.y_test.max()],
            'r--', lw=2, label='Ideal Fit'
        )
        plt.xlabel('Actual Duration (seconds)')
        plt.ylabel('Predicted Duration (seconds)')
        plt.title('Actual vs Predicted Duration')
        plt.legend()


        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        
    def plot_training_history(self):
        """Plot neural network training history"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        if hasattr(self.model, 'loss_curve_'):
            plt.figure(figsize=(10, 5))
            plt.plot(self.model.loss_curve_)
            plt.title('Neural Network Training Loss')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            plt.show()
            print(f"Final training loss: {self.model.loss_curve_[-1]:.4f}")
        else:
            print("Training history not available for this model type")
    
    def predict(self, new_data):
        """Predict call duration for new data"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        if isinstance(new_data, dict):
            new_data = self._prepare_manual_input(new_data)
        
        X_enhanced = self.create_features(new_data)
        X_features = self.prepare_features(X_enhanced)
        
        for col in self.feature_columns:
            if col not in X_features.columns:
                X_features[col] = 0
        
        X_features = X_features.reindex(columns=self.feature_columns, fill_value=0)
        X_scaled = self.scaler.transform(X_features)
        
        return self.model.predict(X_scaled)
    
    def _prepare_manual_input(self, input_dict):
        """Convert manual input dictionary to DataFrame"""
        defaults = {
            'timestamp': pd.Timestamp.now(),
            'past_avg_duration': 300
        }
        
        full_input = {**defaults, **input_dict}
        return pd.DataFrame([full_input])
    
    def predict_manual(self, relationship_score, pair_call_count, time_since_last_call_hours,
                      service_type, caller_location, callee_location, timestamp=None,
                      past_avg_duration=300):
        """Convenience method for manual prediction - timestamp is now required"""
        
        if timestamp is None:
            timestamp = pd.Timestamp.now()
        
        manual_input = {
            'relationship_score': relationship_score,
            'pair_call_count': pair_call_count,
            'time_since_last_call_hours': time_since_last_call_hours,
            'service_type': service_type,
            'caller_location': caller_location,
            'callee_location': callee_location,
            'timestamp': timestamp,
            'past_avg_duration': past_avg_duration
        }
        
        prediction = self.predict(manual_input)
        return prediction[0]
    
    def save(self, filepath):
        """Save model and all components"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'categorical_columns': self.categorical_columns,
            'train_metrics': self.train_metrics,
            'test_metrics': self.test_metrics
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model and all components"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        self.categorical_columns = data.get('categorical_columns', self.categorical_columns)
        self.train_metrics = data.get('train_metrics', {})
        self.test_metrics = data.get('test_metrics', {})
        print(f" Model loaded from {filepath}")
        return self