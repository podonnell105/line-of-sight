# Line of Sight Analysis Tool

This tool analyzes line of sight (LOS) between addresses and rail lines using building footprints and LiDAR data. It is designed for efficient, large-scale, batch processing of US states, focusing on addresses near rail corridors.

## Features

- **Line of Sight Analysis**: Determines visibility between addresses and rail lines considering:
  - Building obstructions
  - Vegetation (trees and shrubs)
  - Elevation changes using LiDAR data
- **Web Interface**: Interactive web application for:
  - State selection
  - Address upload
  - Batch processing
- **Data Sources**:
  - Rail lines from Federal Railroad Administration (FRA)
  - Addresses from OpenStreetMap 
  - Building footprints from OpenStreetMap
  - Elevation data from Google Elevation API
- **Output**:
  - CSV files with LOS scores
  - Interactive maps with visualization
  - Detailed statistics and analysis

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd line-of-sight
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Google Maps API Key**
   - Create a `.env` file in the project root:
     ```
     GOOGLE_MAPS_API_KEY=your_api_key_here
     ```

5. **Run the web application**
   ```bash
   python src/app.py
   ```
   - The application will be available at `http://localhost:5001`

## How It Works

1. **Data Collection**
   - Fetches rail lines from FRA API
   - Retrieves addresses from OpenStreetMap
   - Downloads building footprints from OpenStreetMap
   - Gets elevation data from Google Elevation API

2. **Processing Pipeline**
   - Buffers rail lines to create corridors
   - Tiles the area for efficient processing
   - Filters addresses within specified distance
   - Performs LOS analysis considering:
     - Building obstructions
     - Vegetation (trees/shrubs)
     - Elevation changes

3. **Analysis Methods**
   - **Basic LOS**: Considers only building obstructions
   - **Advanced LOS**: Includes vegetation and elevation data
   - **LiDAR-based**: Uses detailed elevation data for precise analysis

4. **Output Generation**
   - CSV files with LOS scores

## Project Structure

- `src/`
  - `app.py`: Web application and main interface
  - `address_los_score.py`: Basic LOS analysis
  - `address_los_score_lidar.py`: Advanced LOS with LiDAR
  - `addresses_near_rail.py`: Address filtering near rail lines
  - `fetch_lidar.py`: LiDAR data retrieval and processing
  - `fetch_census_addresses.py`: Address data collection
  - `process_rail.py`: Rail line processing utilities

## Usage

1. **Web Interface**
   - Select a state
   - Upload address data (GeoJSON)
   - Run analysis
   - View results and download reports

2. **Command Line**
   ```bash
   # Basic LOS analysis
   python src/address_los_score.py

   # Advanced LOS with LiDAR
   python src/address_los_score_lidar.py

   # Find addresses near rail
   python src/addresses_near_rail.py
   ```

## Data Sources

- **Rail Lines**: FRA MainLine MapServer
- **Addresses**: OpenStreetMap and Census data
- **Buildings**: OpenStreetMap
- **Elevation**: Google Elevation API

## Notes

- The project is optimized for memory efficiency
- Processing is done in chunks to handle large datasets
- Results are written incrementally to avoid memory overload
- Multiple elevation data sources are supported for redundancy

## Troubleshooting

- If you encounter API rate limits, the script includes delays between requests
- Ensure your Python environment has all required packages
- Check the logs in the project directory for detailed error information
- For large states, consider increasing the delay between API requests


