#!/usr/bin/env python3
"""
Simple runner script for the Flask app.
This handles the import issues by setting up the Python path correctly.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Now import and run the Flask app
from app import app

if __name__ == "__main__":
    print("Starting Line of Sight Analysis Flask App...")
    print("App will be available at: http://0.0.0.0:5001")
    app.run(host='0.0.0.0', port=5001, debug=True) 