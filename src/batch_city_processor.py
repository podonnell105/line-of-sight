import os
import time
import logging
from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from city_los_processor import process_city_los_analysis
from elevation_providers import OpenTopographyProvider, GoogleElevationProvider

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_processor.log'),
        logging.StreamHandler()
    ]
)

# Define all cities to process
CITIES_TO_PROCESS = [
    # North Carolina
   ("Fayetteville", "NC"),
    ("Selma", "NC"),
    ("Rocky Mount", "NC"),
    ("Morehead City", "NC"),
    ("Salisbury", "NC"),
    ("Pembroke", "NC"),
    ("Wilmington", "NC"),
    
    # South Carolina
    ("Spartanburg", "SC"),
    ("Columbia", "SC"),
    ("Denmark", "SC"),
    ("Kingstree", "SC"),
    ("Florence", "SC"),
    
    # Georgia
    ("Savannah", "GA"),
    
    # Virginia
    ("Norfolk", "VA")
]

def process_all_cities(output_dir='web_data', delay_between_cities=300):
    """
    Process all cities in the list, creating separate CSV files for each.
    
    Args:
        output_dir: Directory to save results
        delay_between_cities: Delay in seconds between processing cities (default 5 minutes)
    
    Returns:
        List of successfully processed cities and their output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    successful_cities = []
    failed_cities = []
    
    logging.info(f"Starting batch processing of {len(CITIES_TO_PROCESS)} cities")
    logging.info(f"Delay between cities: {delay_between_cities} seconds")
    
    for i, (city_name, state_abbr) in enumerate(CITIES_TO_PROCESS, 1):
        logging.info(f"Processing city {i}/{len(CITIES_TO_PROCESS)}: {city_name}, {state_abbr}")
        
        try:
            # Try OpenTopography first (more accurate)
            logging.info(f"Attempting to process {city_name}, {state_abbr} with OpenTopography...")
            output_file = process_city_los_analysis(
                city_name=city_name,
                state_abbr=state_abbr,
                output_dir=output_dir,
                elevation_provider=OpenTopographyProvider()
            )
            
            if output_file and os.path.exists(output_file):
                successful_cities.append({
                    'city': city_name,
                    'state': state_abbr,
                    'output_file': output_file,
                    'provider': 'OpenTopography'
                })
                logging.info(f"Successfully processed {city_name}, {state_abbr} with OpenTopography")
            else:
                # Fallback to Google Elevation API
                logging.warning(f"OpenTopography failed for {city_name}, {state_abbr}, trying Google Elevation API...")
                output_file = process_city_los_analysis(
                    city_name=city_name,
                    state_abbr=state_abbr,
                    output_dir=output_dir,
                    elevation_provider=GoogleElevationProvider()
                )
                
                if output_file and os.path.exists(output_file):
                    successful_cities.append({
                        'city': city_name,
                        'state': state_abbr,
                        'output_file': output_file,
                        'provider': 'Google'
                    })
                    logging.info(f"Successfully processed {city_name}, {state_abbr} with Google Elevation API")
                else:
                    failed_cities.append({
                        'city': city_name,
                        'state': state_abbr,
                        'error': 'Both elevation providers failed'
                    })
                    logging.error(f"Failed to process {city_name}, {state_abbr} with both providers")
                    
        except Exception as e:
            failed_cities.append({
                'city': city_name,
                'state': state_abbr,
                'error': str(e)
            })
            logging.error(f"Exception processing {city_name}, {state_abbr}: {str(e)}")
        
        # Add delay between cities to respect API rate limits
        if i < len(CITIES_TO_PROCESS):
            logging.info(f"Waiting {delay_between_cities} seconds before next city...")
            time.sleep(delay_between_cities)
    
    # Log summary
    logging.info(f"Batch processing complete!")
    logging.info(f"Successfully processed: {len(successful_cities)} cities")
    logging.info(f"Failed: {len(failed_cities)} cities")
    
    # Print summary
    print(f"\n=== BATCH PROCESSING SUMMARY ===")
    print(f"Total cities: {len(CITIES_TO_PROCESS)}")
    print(f"Successful: {len(successful_cities)}")
    print(f"Failed: {len(failed_cities)}")
    
    if successful_cities:
        print(f"\n=== SUCCESSFUL CITIES ===")
        for city_info in successful_cities:
            print(f"✓ {city_info['city']}, {city_info['state']} ({city_info['provider']})")
            print(f"  Output: {city_info['output_file']}")
    
    if failed_cities:
        print(f"\n=== FAILED CITIES ===")
        for city_info in failed_cities:
            print(f"✗ {city_info['city']}, {city_info['state']}")
            print(f"  Error: {city_info['error']}")
    
    return successful_cities, failed_cities

def check_opentopography_usage():
    """
    Check OpenTopography API usage and limits.
    Note: OpenTopography doesn't provide usage statistics via API,
    but we can estimate based on typical limits.
    """
    logging.info("Checking OpenTopography API usage...")
    
    # OpenTopography typical limits:
    # - Free tier: 1000 requests/day
    # - Each city requires ~1-3 elevation requests
    # - We have 14 cities, so ~14-42 requests total
    
    estimated_requests = len(CITIES_TO_PROCESS) * 2  # Conservative estimate
    daily_limit = 1000
    
    logging.info(f"Estimated OpenTopography requests needed: {estimated_requests}")
    logging.info(f"Daily limit: {daily_limit}")
    
    if estimated_requests <= daily_limit:
        logging.info("✓ Estimated usage is within daily limits")
        return True
    else:
        logging.warning("⚠ Estimated usage may exceed daily limits")
        logging.info("Will use Google Elevation API as fallback")
        return False

if __name__ == "__main__":
    # Check API usage before starting
    check_opentopography_usage()
    
    # Process all cities
    successful, failed = process_all_cities(
        output_dir='web_data',
        delay_between_cities=300  # 5 minutes between cities
    )
    
    # Save summary to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = f'batch_processing_summary_{timestamp}.txt'
    
    with open(summary_file, 'w') as f:
        f.write(f"Batch Processing Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total cities: {len(CITIES_TO_PROCESS)}\n")
        f.write(f"Successful: {len(successful)}\n")
        f.write(f"Failed: {len(failed)}\n\n")
        
        if successful:
            f.write("SUCCESSFUL CITIES:\n")
            f.write("-" * 20 + "\n")
            for city_info in successful:
                f.write(f"✓ {city_info['city']}, {city_info['state']} ({city_info['provider']})\n")
                f.write(f"  Output: {city_info['output_file']}\n\n")
        
        if failed:
            f.write("FAILED CITIES:\n")
            f.write("-" * 15 + "\n")
            for city_info in failed:
                f.write(f"✗ {city_info['city']}, {city_info['state']}\n")
                f.write(f"  Error: {city_info['error']}\n\n")
    
    print(f"\nSummary saved to: {summary_file}") 