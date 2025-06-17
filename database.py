
import sqlite3
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
import yaml
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Database:
    """Handles all database operations for customer data."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize database connection.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        self.db_path = config['database']['path']
        self.table_name = config['database']['table_name']
        
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            logger.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database: {e}")
            raise
    
    def __del__(self):
        """Close database connection when object is destroyed."""
        if hasattr(self, 'conn'):
            self.conn.close()
    
    def create_customers_table(self) -> bool:
        """
        Create the customers table if it doesn't exist.
        
        Returns:
            Boolean indicating success
        """
        try:
            self.cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.table_name} (
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            self.conn.commit()
            logger.info(f"Table '{self.table_name}' created successfully")
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Error creating table: {e}")
            self.conn.rollback()
            return False
    
    def insert_customer(self, customer_data: Dict[str, Any]) -> bool:
        """
        Insert a single customer into the database.
        
        Args:
            customer_data: Dictionary containing customer information
            
        Returns:
            Boolean indicating success
        """
        try:
            # Add timestamp
            customer_data['created_at'] = datetime.now()
            customer_data['updated_at'] = datetime.now()
            
            columns = ', '.join(customer_data.keys())
            placeholders = ', '.join(['?' for _ in customer_data])
            query = f"INSERT INTO {self.table_name} ({columns}) VALUES ({placeholders})"
            
            self.cursor.execute(query, list(customer_data.values()))
            self.conn.commit()
            
            logger.info(f"Customer {customer_data.get('customerID', 'Unknown')} inserted successfully")
            return True
            
        except sqlite3.IntegrityError as e:
            logger.warning(f"Customer already exists: {e}")
            return False
        except sqlite3.Error as e:
            logger.error(f"Error inserting customer: {e}")
            self.conn.rollback()
            return False
    
    def bulk_insert_customers(self, customers_data: List[Dict[str, Any]]) -> Tuple[int, int]:
        """
        Insert multiple customers into the database.
        
        Args:
            customers_data: List of dictionaries containing customer information
            
        Returns:
            Tuple of (success_count, fail_count)
        """
        success_count = 0
        fail_count = 0
        
        for customer in customers_data:
            # Add timestamps
            customer['created_at'] = datetime.now()
            customer['updated_at'] = datetime.now()
            
            if self.insert_customer(customer):
                success_count += 1
            else:
                fail_count += 1
        
        logger.info(f"Bulk insert complete: {success_count} succeeded, {fail_count} failed")
        return success_count, fail_count
    
    def get_customer(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single customer by ID.
        
        Args:
            customer_id: The customer ID to retrieve
            
        Returns:
            Dictionary containing customer data or None if not found
        """
        try:
            query = f"SELECT * FROM {self.table_name} WHERE customerID = ?"
            self.cursor.execute(query, (customer_id,))
            result = self.cursor.fetchone()
            
            if result:
                columns = [desc[0] for desc in self.cursor.description]
                return dict(zip(columns, result))
            
            logger.warning(f"Customer {customer_id} not found")
            return None
            
        except sqlite3.Error as e:
            logger.error(f"Error retrieving customer {customer_id}: {e}")
            return None
    
    def update_customer(self, customer_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update customer information.
        
        Args:
            customer_id: The customer ID to update
            update_data: Dictionary of fields to update
            
        Returns:
            Boolean indicating success
        """
        try:
            # Add updated timestamp
            update_data['updated_at'] = datetime.now()
            
            # Build update query
            set_clause = ', '.join([f"{key} = ?" for key in update_data.keys()])
            query = f"UPDATE {self.table_name} SET {set_clause} WHERE customerID = ?"
            
            values = list(update_data.values()) + [customer_id]
            self.cursor.execute(query, values)
            self.conn.commit()
            
            if self.cursor.rowcount > 0:
                logger.info(f"Customer {customer_id} updated successfully")
                return True
            else:
                logger.warning(f"Customer {customer_id} not found for update")
                return False
                
        except sqlite3.Error as e:
            logger.error(f"Error updating customer {customer_id}: {e}")
            self.conn.rollback()
            return False
    
    def delete_customer(self, customer_id: str) -> bool:
        """
        Delete a customer from the database.
        
        Args:
            customer_id: The customer ID to delete
            
        Returns:
            Boolean indicating success
        """
        try:
            query = f"DELETE FROM {self.table_name} WHERE customerID = ?"
            self.cursor.execute(query, (customer_id,))
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
    
    def get_all_customers(self) -> List[Dict[str, Any]]:
        """
        Retrieve all customers from the database.
        
        Returns:
            List of dictionaries containing customer data
        """
        try:
            query = f"SELECT * FROM {self.table_name}"
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            
            if results:
                columns = [desc[0] for desc in self.cursor.description]
                return [dict(zip(columns, row)) for row in results]
            
            return []
            
        except sqlite3.Error as e:
            logger.error(f"Error retrieving all customers: {e}")
            return []
    
    def get_customers_by_criteria(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Retrieve customers matching specific criteria.
        
        Args:
            criteria: Dictionary of field-value pairs to filter by
            
        Returns:
            List of dictionaries containing customer data
        """
        try:
            # Build WHERE clause
            where_conditions = [f"{key} = ?" for key in criteria.keys()]
            where_clause = " AND ".join(where_conditions)
            
            query = f"SELECT * FROM {self.table_name} WHERE {where_clause}"
            self.cursor.execute(query, list(criteria.values()))
            results = self.cursor.fetchall()
            
            if results:
                columns = [desc[0] for desc in self.cursor.description]
                return [dict(zip(columns, row)) for row in results]
            
            return []
            
        except sqlite3.Error as e:
            logger.error(f"Error retrieving customers by criteria: {e}")
            return []
    
    def get_churn_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about customer churn.
        
        Returns:
            Dictionary with churn statistics
        """
        try:
            # Overall statistics
            query = f"""
                SELECT 
                    COUNT(*) as total_customers,
                    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned_customers,
                    AVG(tenure) as avg_tenure,
                    AVG(MonthlyCharges) as avg_monthly_charges
                FROM {self.table_name}
            """
            self.cursor.execute(query)
            result = self.cursor.fetchone()
            
            if result and result[0] > 0:
                total = result[0]
                churned = result[1]
                churn_rate = (churned / total) * 100
                avg_tenure = result[2]
                avg_charges = result[3]
            else:
                return {}
            
            # Churn by contract type
            query = f"""
                SELECT 
                    Contract,
                    COUNT(*) as total,
                    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned
                FROM {self.table_name}
                GROUP BY Contract
            """
            self.cursor.execute(query)
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
                'avg_tenure': avg_tenure,
                'avg_monthly_charges': avg_charges,
                'churn_by_contract': contract_churn
            }
            
        except sqlite3.Error as e:
            logger.error(f"Error getting churn statistics: {e}")
            return {}
    
    def import_from_csv(self, csv_path: str) -> Tuple[int, int]:
        """
        Import customer data from a CSV file.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            Tuple of (success_count, fail_count)
        """
        try:
            # Read CSV file
            df = pd.read_csv(csv_path)
            logger.info(f"Read {len(df)} records from {csv_path}")
            
            # Convert dataframe to list of dictionaries
            customers_data = df.to_dict('records')
            
            # Use bulk insert function
            return self.bulk_insert_customers(customers_data)
            
        except Exception as e:
            logger.error(f"Error importing from CSV {csv_path}: {e}")
            return 0, 0
    
    def export_to_csv(self, csv_path: str, criteria: Optional[Dict[str, Any]] = None) -> bool:
        """
        Export customer data to a CSV file.
        
        Args:
            csv_path: Path to save the CSV file
            criteria: Optional filtering criteria
            
        Returns:
            Boolean indicating success
        """
        try:
            if criteria:
                customers = self.get_customers_by_criteria(criteria)
            else:
                customers = self.get_all_customers()
            
            if customers:
                df = pd.DataFrame(customers)
                df.to_csv(csv_path, index=False)
                logger.info(f"Exported {len(customers)} customers to {csv_path}")
                return True
            else:
                logger.warning("No customers to export")
                return False
                
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False
    
    def execute_custom_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """
        Execute a custom SQL query (for advanced use cases).
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of results as dictionaries
        """
        try:
            self.cursor.execute(query, params)
            results = self.cursor.fetchall()
            
            if results:
                columns = [desc[0] for desc in self.cursor.description]
                return [dict(zip(columns, row)) for row in results]
            
            return []
            
        except sqlite3.Error as e:
            logger.error(f"Error executing custom query: {e}")
            return []