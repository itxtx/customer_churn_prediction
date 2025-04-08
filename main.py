
from database import Database
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV

def main():
    """Demonstrate the ML pipeline functionality in the Database class"""
    # Initialize the database
    db = Database("customer_churn.db")
    
    # Create the customers table if it doesn't exist
    db.create_customers_table()

    # Import data from CSV
    db.import_from_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")



    # Load the best model
    loaded_pipeline = db.load_model("churn_model_rf.pkl")


    


if __name__ == "__main__":
    main()