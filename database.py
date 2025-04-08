import sqlite3
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple, Any
import logging
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Database:
    def __init__(self, db_path: str):
        """Initialize database connection.
        
        Args:
            db_path: Path to the SQLite database file
        """
        try:
            self.conn = sqlite3.connect(db_path)
            self.cursor = self.conn.cursor()
            logger.info(f"Connected to database: {db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            raise

    def __del__(self):
        """Close the database connection when the object is destroyed."""
        if hasattr(self, 'conn'):
            self.conn.close()
            logger.info("Database connection closed")

    def create_customers_table(self):
        """Create the customers table with all required columns."""
        try:
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS customers (
                    customerID TEXT PRIMARY KEY,
                    gender TEXT,
                    SeniorCitizen INTEGER,
                    Partner TEXT,
                    Dependents TEXT,
                    tenure INTEGER,
                    PhoneService TEXT,
                    MultipleLines TEXT,
                    InternetService TEXT,
                    OnlineSecurity TEXT,
                    OnlineBackup TEXT,
                    DeviceProtection TEXT,
                    TechSupport TEXT,
                    StreamingTV TEXT,
                    StreamingMovies TEXT,
                    Contract TEXT,
                    PaperlessBilling TEXT,
                    PaymentMethod TEXT,
                    MonthlyCharges REAL,
                    TotalCharges REAL,
                    Churn TEXT,
                    
                    -- Derived features
                    AvgMonthlyCharges REAL,
                    TotalServices INTEGER,
                    ContractCategory INTEGER,
                    SeniorFamily INTEGER,
                    ValueCustomer INTEGER
                )
            ''')
            self.conn.commit()
            logger.info("Customers table created or already exists")
        except sqlite3.Error as e:
            logger.error(f"Error creating customers table: {e}")
            self.conn.rollback()
            raise

    def get_median_values(self) -> Dict[str, float]:
        """Get median values for numerical columns to handle missing values.
        
        Returns:
            Dictionary with column names as keys and median values as values
        """
        try:
            # Get existing data to calculate medians
            self.cursor.execute('''
                SELECT tenure, MonthlyCharges, TotalCharges
                FROM customers
            ''')
            results = self.cursor.fetchall()
            
            # If we have data, calculate medians
            if results:
                tenure_values = [row[0] for row in results if row[0] is not None]
                monthly_charges_values = [row[1] for row in results if row[1] is not None]
                total_charges_values = [row[2] for row in results if row[2] is not None]
                
                # Calculate medians
                tenure_median = np.median(tenure_values) if tenure_values else 0
                monthly_median = np.median(monthly_charges_values) if monthly_charges_values else 0
                total_median = np.median(total_charges_values) if total_charges_values else 0
            else:
                # Default values if no data exists
                tenure_median = 0
                monthly_median = 0
                total_median = 0
                
            return {
                'tenure': tenure_median,
                'MonthlyCharges': monthly_median,
                'TotalCharges': total_median
            }
        except sqlite3.Error as e:
            logger.error(f"Error calculating median values: {e}")
            return {'tenure': 0, 'MonthlyCharges': 0, 'TotalCharges': 0}

    def cap_outliers(self, value: float, column: str) -> float:
        """Cap outliers using the IQR method.
        
        Args:
            value: The value to check and potentially cap
            column: Column name for which to calculate bounds
            
        Returns:
            Capped value
        """
        try:
            # Get column values to calculate quartiles
            self.cursor.execute(f"SELECT {column} FROM customers WHERE {column} IS NOT NULL")
            values = [row[0] for row in self.cursor.fetchall()]
            
            # If we have enough data, calculate bounds
            if len(values) > 4:  # Need enough data for meaningful quartiles
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Cap the value
                if value > upper_bound:
                    return upper_bound
                elif value < lower_bound:
                    return lower_bound
            
            # Return original value if we don't have enough data or value is within bounds
            return value
        except (sqlite3.Error, ValueError) as e:
            logger.error(f"Error capping outliers for {column}: {e}")
            return value  # Return original value on error

    def calculate_derived_features(self, customer_data: dict) -> dict:
        """Calculate derived features for the customer data.
        
        Args:
            customer_data: Dictionary containing customer data
            
        Returns:
            Dictionary with additional derived features
        """
        # Make a copy of the original data
        enriched_data = customer_data.copy()
        
        # 1. Average monthly charges
        tenure = enriched_data.get('tenure', 0)
        if tenure is None or tenure == '':
            tenure = 0
        try:
            tenure = float(tenure)
        except (ValueError, TypeError):
            tenure = 0
            
        total_charges = enriched_data.get('TotalCharges', 0)
        if total_charges is None or total_charges == '':
            total_charges = 0
        try:
            total_charges = float(total_charges)
        except (ValueError, TypeError):
            total_charges = 0
            
        # Add 1 to avoid division by zero
        enriched_data['AvgMonthlyCharges'] = total_charges / (tenure + 1)
        
        # 2. Total services subscribed
        services = [
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        total_services = 0
        for service in services:
            value = enriched_data.get(service, 'No')
            if value == 'Yes':
                total_services += 1
            elif service == 'InternetService' and value in ['DSL', 'Fiber optic']:
                total_services += 1
        
        enriched_data['TotalServices'] = total_services
        
        # 3. Contract category
        contract_mapping = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
        contract = enriched_data.get('Contract', 'Month-to-month')
        enriched_data['ContractCategory'] = contract_mapping.get(contract, 0)
        
        # 4. Senior citizen family indicator
        senior_citizen = int(enriched_data.get('SeniorCitizen', 0))
        partner = enriched_data.get('Partner', 'No')
        enriched_data['SeniorFamily'] = 1 if senior_citizen == 1 and partner == 'Yes' else 0
        
        # 5. Value customer indicator (requires comparison with overall data)
        # This will be a simple proxy without the medians for now
        monthly_charges = enriched_data.get('MonthlyCharges', 0)
        if monthly_charges is None or monthly_charges == '':
            monthly_charges = 0
        try:
            monthly_charges = float(monthly_charges)
        except (ValueError, TypeError):
            monthly_charges = 0
            
        # Simple heuristic: customers with tenure > 12 months and charges > $50 are "value customers"
        enriched_data['ValueCustomer'] = 1 if tenure > 12 and monthly_charges > 50 else 0
        
        return enriched_data

    def insert_customer(self, customer_data: dict) -> bool:
        """Insert a new customer with data cleaning and feature engineering.
        
        Args:
            customer_data: Dictionary containing customer data
            
        Returns:
            Boolean indicating success
        """
        try:
            # Handle missing values using medians from existing data
            medians = self.get_median_values()
            
            # Fill missing values
            for col in ['TotalCharges', 'MonthlyCharges', 'tenure']:
                if col in customer_data and (customer_data[col] is None or customer_data[col] == ''):
                    customer_data[col] = medians[col]
            
            # Convert string numbers to float (TotalCharges is often provided as string)
            for col in ['TotalCharges', 'MonthlyCharges', 'tenure']:
                if col in customer_data and isinstance(customer_data[col], str):
                    try:
                        customer_data[col] = float(customer_data[col])
                    except ValueError:
                        customer_data[col] = medians[col]
            
            # Handle outliers
            for col in ['TotalCharges', 'MonthlyCharges', 'tenure']:
                if col in customer_data and customer_data[col] is not None:
                    customer_data[col] = self.cap_outliers(float(customer_data[col]), col)
            
            # Calculate derived features
            enriched_data = self.calculate_derived_features(customer_data)
            
            # Prepare column and value strings for SQL
            columns = ', '.join(enriched_data.keys())
            placeholders = ', '.join(['?' for _ in enriched_data])
            
            # Execute insert
            self.cursor.execute(
                f"INSERT INTO customers ({columns}) VALUES ({placeholders})",
                list(enriched_data.values())
            )
            
            self.conn.commit()
            logger.info(f"Customer {enriched_data.get('customerID', 'unknown')} inserted successfully")
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Error inserting customer: {e}")
            self.conn.rollback()
            return False

    def bulk_insert_customers(self, customers_data: List[Dict]) -> Tuple[int, int]:
        """Insert multiple customers at once.
        
        Args:
            customers_data: List of dictionaries containing customer data
            
        Returns:
            Tuple of (success_count, fail_count)
        """
        success_count = 0
        fail_count = 0
        
        for customer in customers_data:
            if self.insert_customer(customer):
                success_count += 1
            else:
                fail_count += 1
                
        logger.info(f"Bulk insert completed. Success: {success_count}, Failed: {fail_count}")
        return success_count, fail_count

    def get_customer(self, customer_id: str) -> Optional[Dict]:
        """Retrieve a customer by ID.
        
        Args:
            customer_id: The customer ID to look up
            
        Returns:
            Dictionary containing customer data or None if not found
        """
        try:
            self.cursor.execute("SELECT * FROM customers WHERE customerID = ?", (customer_id,))
            result = self.cursor.fetchone()
            
            if result:
                # Convert to dictionary
                columns = [desc[0] for desc in self.cursor.description]
                return dict(zip(columns, result))
            return None
            
        except sqlite3.Error as e:
            logger.error(f"Error retrieving customer {customer_id}: {e}")
            return None

    def update_customer(self, customer_id: str, update_data: Dict) -> bool:
        """Update a customer's information.
        
        Args:
            customer_id: The customer ID to update
            update_data: Dictionary containing fields to update
            
        Returns:
            Boolean indicating success
        """
        try:
            # Get current customer data
            current_data = self.get_customer(customer_id)
            if not current_data:
                logger.warning(f"Customer {customer_id} not found for update")
                return False
                
            # Merge current data with updates
            merged_data = {**current_data, **update_data}
            
            # Handle data cleaning and feature engineering
            enriched_data = self.calculate_derived_features(merged_data)
            
            # Remove customerID from the update fields
            if 'customerID' in enriched_data:
                del enriched_data['customerID']
                
            # Prepare update statement
            set_clause = ', '.join([f"{key} = ?" for key in enriched_data.keys()])
            values = list(enriched_data.values()) + [customer_id]
            
            # Execute update
            self.cursor.execute(
                f"UPDATE customers SET {set_clause} WHERE customerID = ?",
                values
            )
            
            self.conn.commit()
            logger.info(f"Customer {customer_id} updated successfully")
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Error updating customer {customer_id}: {e}")
            self.conn.rollback()
            return False

    def delete_customer(self, customer_id: str) -> bool:
        """Delete a customer from the database.
        
        Args:
            customer_id: The customer ID to delete
            
        Returns:
            Boolean indicating success
        """
        try:
            self.cursor.execute("DELETE FROM customers WHERE customerID = ?", (customer_id,))
            self.conn.commit()
            
            if self.cursor.rowcount > 0:
                logger.info(f"Customer {customer_id} deleted successfully")
                return True
            else:
                logger.warning(f"Customer {customer_id} not found for deletion")
                return False
                
        except sqlite3.Error as e:
            logger.error(f"Error deleting customer {customer_id}: {e}")
            self.conn.rollback()
            return False

    def get_all_customers(self) -> List[Dict]:
        """Retrieve all customers from the database.
        
        Returns:
            List of dictionaries containing customer data
        """
        try:
            self.cursor.execute("SELECT * FROM customers")
            results = self.cursor.fetchall()
            
            if results:
                # Convert to list of dictionaries
                columns = [desc[0] for desc in self.cursor.description]
                return [dict(zip(columns, row)) for row in results]
            return []
            
        except sqlite3.Error as e:
            logger.error(f"Error retrieving all customers: {e}")
            return []

    def get_churn_statistics(self) -> Dict:
        """Get statistics about customer churn.
        
        Returns:
            Dictionary with churn statistics
        """
        try:
            # Calculate churn rate
            self.cursor.execute("""
                SELECT 
                    COUNT(*) as total_customers,
                    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned_customers
                FROM customers
            """)
            
            result = self.cursor.fetchone()
            if result and result[0] > 0:
                total = result[0]
                churned = result[1]
                churn_rate = (churned / total) * 100
            else:
                total = 0
                churned = 0
                churn_rate = 0
                
            # Get churn by contract type
            self.cursor.execute("""
                SELECT 
                    Contract,
                    COUNT(*) as total,
                    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned
                FROM customers
                GROUP BY Contract
            """)
            
            contract_results = self.cursor.fetchall()
            contract_churn = {
                row[0]: {
                    'total': row[1],
                    'churned': row[2],
                    'rate': (row[2] / row[1] * 100) if row[1] > 0 else 0
                } for row in contract_results
            }
            
            return {
                'total_customers': total,
                'churned_customers': churned,
                'churn_rate': churn_rate,
                'churn_by_contract': contract_churn
            }
            
        except sqlite3.Error as e:
            logger.error(f"Error getting churn statistics: {e}")
            return {
                'total_customers': 0,
                'churned_customers': 0,
                'churn_rate': 0,
                'churn_by_contract': {}
            }

    def import_from_csv(self, csv_path: str) -> Tuple[int, int]:
        """Import customer data from a CSV file.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            Tuple of (success_count, fail_count)
        """
        try:
            # Read CSV file
            df = pd.read_csv(csv_path)
            
            # Convert dataframe to list of dictionaries
            customers_data = df.to_dict('records')
            
            # Use bulk insert function
            return self.bulk_insert_customers(customers_data)
            
        except Exception as e:
            logger.error(f"Error importing from CSV {csv_path}: {e}")
            return 0, 0

    def get_customers_by_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """Get customers using a custom SQL query.
        
        Args:
            query: SQL query string
            params: Parameters for the query
            
        Returns:
            List of dictionaries containing customer data
        """
        try:
            self.cursor.execute(query, params)
            results = self.cursor.fetchall()
            
            if results:
                # Convert to list of dictionaries
                columns = [desc[0] for desc in self.cursor.description]
                return [dict(zip(columns, row)) for row in results]
            return []
            
        except sqlite3.Error as e:
            logger.error(f"Error executing custom query: {e}")
            return []

    def export_to_csv(self, csv_path: str) -> bool:
        """Export all customer data to a CSV file.
        
        Args:
            csv_path: Path to save the CSV file
            
        Returns:
            Boolean indicating success
        """
        try:
            # Get all customers
            customers = self.get_all_customers()
            
            if customers:
                # Convert to dataframe and save to CSV
                df = pd.DataFrame(customers)
                df.to_csv(csv_path, index=False)
                logger.info(f"Exported {len(customers)} customers to {csv_path}")
                return True
            else:
                logger.warning("No customers to export")
                return False
                
        except Exception as e:
            logger.error(f"Error exporting to CSV {csv_path}: {e}")
            return False
            
    def target_encode_kfold(self, column: str, target: str, n_folds: int = 5) -> pd.DataFrame:
        """Apply target encoding using k-fold to prevent data leakage.
        
        Args:
            column: The categorical column to encode
            target: The target variable column
            n_folds: Number of folds for cross-validation
            
        Returns:
            DataFrame with the original data and the new encoded column
        """
        try:
            # Get all data as DataFrame
            customers = self.get_all_customers()
            if not customers:
                logger.warning("No data available for target encoding")
                return pd.DataFrame()
                
            df = pd.DataFrame(customers)
            
            # Create a copy of the dataframe
            df_copy = df.copy()
            
            # For target column, convert 'Yes'/'No' to 1/0 if needed
            if target in df_copy.columns and df_copy[target].dtype == 'object':
                if set(df_copy[target].unique()) == {'Yes', 'No'}:
                    df_copy[target] = df_copy[target].map({'Yes': 1, 'No': 0})
            
            # Create a new column for the encoded feature
            encoded_column = f"{column}_target_encoded"
            df_copy[encoded_column] = np.nan
            
            # Set up KFold
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            
            # For each fold
            for train_idx, test_idx in kfold.split(df_copy):
                # Get the means from the training fold
                means = df_copy.iloc[train_idx].groupby(column)[target].mean()
                
                # Map the means to the test fold
                df_copy.loc[test_idx, encoded_column] = df_copy.iloc[test_idx][column].map(means)
            
            # Handle missing values (categories that didn't appear in a particular fold)
            global_mean = df_copy[target].mean()
            df_copy[encoded_column].fillna(global_mean, inplace=True)
            
            logger.info(f"Successfully applied target encoding to {column}")
            return df_copy
            
        except Exception as e:
            logger.error(f"Error in target encoding: {e}")
            return pd.DataFrame()
    
    def apply_feature_scaling(self, scaler_type: str = 'standard') -> pd.DataFrame:
        """Apply feature scaling to numerical columns.
        
        Args:
            scaler_type: Type of scaler ('standard' or 'minmax')
            
        Returns:
            DataFrame with scaled features
        """
        try:
            # Get all data as DataFrame
            customers = self.get_all_customers()
            if not customers:
                logger.warning("No data available for feature scaling")
                return pd.DataFrame()
                
            df = pd.DataFrame(customers)
            
            # Identify numerical columns for scaling (exclude target 'Churn')
            numeric_features = [
                col for col in df.select_dtypes(include=['int64', 'float64']).columns 
                if col != 'Churn'
            ]
            
            if not numeric_features:
                logger.warning("No numerical features found for scaling")
                return df
            
            # Apply appropriate scaler
            if scaler_type.lower() == 'standard':
                scaler = StandardScaler()
                df[numeric_features] = scaler.fit_transform(df[numeric_features])
                logger.info(f"Applied StandardScaler to {len(numeric_features)} features")
            elif scaler_type.lower() == 'minmax':
                scaler = MinMaxScaler()
                df[numeric_features] = scaler.fit_transform(df[numeric_features])
                logger.info(f"Applied MinMaxScaler to {len(numeric_features)} features")
            else:
                logger.warning(f"Unknown scaler type: {scaler_type}. Using StandardScaler.")
                scaler = StandardScaler()
                df[numeric_features] = scaler.fit_transform(df[numeric_features])
            
            return df
            
        except Exception as e:
            logger.error(f"Error in feature scaling: {e}")
            return pd.DataFrame()
    
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


# Example usage
if __name__ == "__main__":
    # Initialize database
    db = Database("customer_churn.db")
    
    # Create tables
    db.create_customers_table()
    
    # Example customer data
    customer1 = {
        "customerID": "7590-VHVEG",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
        "Churn": "No"
    }
    
    customer2 = {
        "customerID": "5575-GNVDE",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 34,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 56.95,
        "TotalCharges": 1889.5,
        "Churn": "No"
    }
    
    # Insert customers
    db.insert_customer(customer1)
    db.insert_customer(customer2)
    
    # Get all customers
    all_customers = db.get_all_customers()
    print(f"Total customers: {len(all_customers)}")
    
    # Get churn statistics
    churn_stats = db.get_churn_statistics()
    print(f"Churn rate: {churn_stats['churn_rate']:.2f}%")