#!/usr/bin/env python3
"""
Batch processing script for all cities.
Run this to process all 14 cities and create separate CSV files for each.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from batch_city_processor import process_all_cities, check_opentopography_usage

def main():
    print("=== Line of Sight Analysis - Batch Processing ===")
    print("Processing all cities and creating separate CSV files for each location.")
    print()
    
    # Check API usage first
    print("Checking OpenTopography API usage...")
    check_opentopography_usage()
    print()
    
    # Confirm with user
    print("This will process the following cities:")
    cities = [
        ("Fayetteville", "NC"), ("Selma", "NC"), ("Rocky Mount", "NC"),
        ("Morehead City", "NC"), ("Salisbury", "NC"), ("Pembroke", "NC"),
        ("Wilmington", "NC"), ("Spartanburg", "SC"), ("Columbia", "SC"),
        ("Denmark", "SC"), ("Kingstree", "SC"), ("Florence", "SC"),
        ("Savannah", "GA"), ("Norfolk", "VA")
    ]
    
    for city, state in cities:
        print(f"  • {city}, {state}")
    
    print()
    print("Estimated time: ~2-3 hours (with 5-minute delays between cities)")
    print("Each city will create a separate CSV file in the web_data directory.")
    print()
    
    response = input("Do you want to proceed? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Batch processing cancelled.")
        return
    
    print()
    print("Starting batch processing...")
    print("This will take several hours. You can monitor progress in batch_processor.log")
    print()
    
    # Run the batch processing
    successful, failed = process_all_cities(
        output_dir='web_data',
        delay_between_cities=300  # 5 minutes between cities
    )
    
    print()
    print("=== BATCH PROCESSING COMPLETE ===")
    print(f"Successfully processed: {len(successful)} cities")
    print(f"Failed: {len(failed)} cities")
    
    if successful:
        print("\nSuccessful cities:")
        for city_info in successful:
            print(f"  ✓ {city_info['city']}, {city_info['state']} ({city_info['provider']})")
    
    if failed:
        print("\nFailed cities:")
        for city_info in failed:
            print(f"  ✗ {city_info['city']}, {city_info['state']}: {city_info['error']}")

if __name__ == "__main__":
    main() 