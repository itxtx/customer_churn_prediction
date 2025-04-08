




class ChurnPredictor:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.numeric_features = []
        self.categorical_features = []
        self.target = None
        self.models = {}
        self.pipeline = None
        self.feature_importances = None
        
    def prepare_data(self, 
                    data: pd.DataFrame,
                    target_column: str,
                    numeric_features: List[str],
                    categorical_features: List[str]) -> None:
        """Previous implementation remains the same"""
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.target = target_column
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data.drop(target_column, axis=1),
            data[target_column],
            test_size=0.2,
            random_state=self.random_state
        )

    def perform_eda(self) -> Dict[str, Any]:
        """Enhanced EDA with additional analyses"""
        eda_results = {
            'missing_values': self.X_train.isnull().sum(),
            'numeric_stats': self.X_train[self.numeric_features].describe(),
            'categorical_stats': {
                feature: self.X_train[feature].value_counts() 
                for feature in self.categorical_features
            },
            'correlations': self.X_train[self.numeric_features].corr()
        }
        
        # Target distribution analysis
        eda_results['target_distribution'] = self.y_train.value_counts()
        eda_results['imbalance_ratio'] = (
            self.y_train.value_counts()[0] / self.y_train.value_counts()[1]
        )
        
        # Additional numeric feature analysis
        eda_results['numeric_skewness'] = self.X_train[self.numeric_features].skew()
        eda_results['numeric_kurtosis'] = self.X_train[self.numeric_features].kurtosis()
        
     
    def create_ml_pipeline(self, model) -> Pipeline:
        """Create a scikit-learn pipeline for preprocessing and modeling.
        
        Args:
            model: A scikit-learn model instance (e.g., RandomForestClassifier)
            
        Returns:
            A scikit-learn Pipeline instance
        """
        try:
            # Define feature types
            numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 
                                 'AvgMonthlyCharges', 'TotalServices']
            
            categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                                   'PhoneService', 'MultipleLines', 'InternetService', 
                                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                                   'TechSupport', 'StreamingTV', 'StreamingMovies', 
                                   'Contract', 'PaperlessBilling', 'PaymentMethod']
            
            # Create preprocessing pipelines
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            # Combine preprocessing steps
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ], remainder='drop'  # Drop other columns not specified
            )
            
            # Create the full pipeline with the model
            full_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            logger.info(f"Created ML pipeline with {type(model).__name__} model")
            return full_pipeline
            
        except Exception as e:
            logger.error(f"Error creating ML pipeline: {e}")
            raise
    
    def train_model(self, model, test_size: float = 0.2, random_state: int = 42) -> Tuple[Pipeline, Dict]:
        """Train a machine learning model on the customer data.
        
        Args:
            model: A scikit-learn model instance
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (trained pipeline, performance metrics)
        """
        try:
            # Get all data
            customers = self.get_all_customers()
            if not customers:
                logger.warning("No data available for model training")
                return None, {}
                
            df = pd.DataFrame(customers)
            
            # Prepare data
            # Convert target to binary
            df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
            
            # Handle missing values in TotalCharges
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            
            # Split features and target
            X = df.drop(['customerID', 'Churn'], axis=1, errors='ignore')
            y = df['Churn']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Create and train pipeline
            pipeline = self.create_ml_pipeline(model)
            pipeline.fit(X_train, y_train)
            
            # Evaluate
            y_pred = pipeline.predict(X_test)
            
            # Get metrics
            report = classification_report(y_test, y_pred, output_dict=True)
            
            metrics = {
                'accuracy': report['accuracy'],
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1': report['weighted avg']['f1-score'],
                'class_report': report
            }
            
            logger.info(f"Model training completed. Accuracy: {metrics['accuracy']:.4f}")
            return pipeline, metrics
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            return None, {}
    
    def save_model(self, pipeline: Pipeline, filename: str) -> bool:
        """Save the trained model to a file.
        
        Args:
            pipeline: Trained scikit-learn pipeline
            filename: Path to save the model
            
        Returns:
            Boolean indicating success
        """
        try:
            joblib.dump(pipeline, filename)
            logger.info(f"Model saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filename: str) -> Optional[Pipeline]:
        """Load a trained model from a file.
        
        Args:
            filename: Path to the saved model
            
        Returns:
            Loaded scikit-learn pipeline or None if error
        """
        try:
            pipeline = joblib.load(filename)
            logger.info(f"Model loaded from {filename}")
            return pipeline
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def predict_churn(self, pipeline: Pipeline, customer_data: Union[Dict, List[Dict]]) -> Dict[str, Any]:
        """Predict churn for new customer data.
        
        Args:
            pipeline: Trained scikit-learn pipeline
            customer_data: Dictionary or list of dictionaries with customer features
            
        Returns:
            Dictionary with predictions and probabilities
        """
        try:
            # Convert to DataFrame if it's a single customer (dict)
            if isinstance(customer_data, dict):
                df = pd.DataFrame([customer_data])
            else:
                df = pd.DataFrame(customer_data)
            
            # Make predictions
            predictions = pipeline.predict(df)
            
            # Get probabilities if the model supports it
            try:
                probabilities = pipeline.predict_proba(df)[:, 1]  # Probability of class 1 (churn)
            except (AttributeError, IndexError):
                probabilities = None
            
            # Prepare result
            result = {
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist() if probabilities is not None else None,
                'churn_predictions': ['Yes' if p == 1 else 'No' for p in predictions]
            }
            
            # Add customer IDs if present
            if 'customerID' in df.columns:
                result['customerIDs'] = df['customerID'].tolist()
            
            logger.info(f"Made predictions for {len(df)} customers")
            return result
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return {'error': str(e)}