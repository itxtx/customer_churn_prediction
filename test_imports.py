#!/usr/bin/env python3
"""
Test script to verify all imports work correctly.
Run this from the project root directory.
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

print("Testing imports...")

# Test data_processing imports
try:
    from src.data_processing import DataProcessor
    print("✓ Successfully imported DataProcessor")
except ImportError as e:
    print(f"✗ Failed to import from data_processing: {e}")

# Test train imports
try:
    from src.train import ModelTrainer
    print("✓ Successfully imported ModelTrainer")
except ImportError as e:
    print(f"✗ Failed to import from train: {e}")

# Test predict imports
try:
    from src.predict import ChurnPredictor
    print("✓ Successfully imported ChurnPredictor")
except ImportError as e:
    print(f"✗ Failed to import from predict: {e}")

# Test database imports
try:
    from src.database import Database
    print("✓ Successfully imported Database")
except ImportError as e:
    print(f"✗ Failed to import from database: {e}")

# Test main imports
try:
    from src.main import train_models, make_predictions
    print("✓ Successfully imported main functions")
except ImportError as e:
    print(f"✗ Failed to import from main: {e}")

print("\nAll import tests completed!") 