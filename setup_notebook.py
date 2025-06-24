#!/usr/bin/env python3
"""
Setup script to help with notebook imports.
Run this script to verify that all imports work correctly.
"""

import sys
import os

# Add project root to Python path
current_dir = os.getcwd()
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

print(f"Current directory: {current_dir}")
print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries

# Test imports
try:
    from src.data_processing import DataProcessor
    print("✓ Successfully imported DataProcessor")
except ImportError as e:
    print(f"✗ Failed to import DataProcessor: {e}")

try:
    from src.train import ModelTrainer
    print("✓ Successfully imported ModelTrainer")
except ImportError as e:
    print(f"✗ Failed to import ModelTrainer: {e}")

try:
    from src.predict import ChurnPredictor
    print("✓ Successfully imported ChurnPredictor")
except ImportError as e:
    print(f"✗ Failed to import ChurnPredictor: {e}")

try:
    from src.database import Database
    print("✓ Successfully imported Database")
except ImportError as e:
    print(f"✗ Failed to import Database: {e}")

print("\nIf all imports succeeded, you can use the same path setup in your notebook.") 