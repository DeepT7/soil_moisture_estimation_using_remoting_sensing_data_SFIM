import os
import glob
import requests
import pandas as pd
import geopandas as gpd
import ee
from datetime import datetime, timedelta

# Define directories for CSV files and shapefiles.
csv_dir = 'daily_smap'
shp_base_dir = 'shapefiles'

# New folder for downloaded images.
download_base_dir = 'downloaded_images'
os.makedirs(download_base_dir, exist_ok=True)

# Get list of all CSV files in the csv_dir.
csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))

def load_shapefile(file_path):
    gdf = gpd.read_file(file_path)
    geom = gdf.geometry.unary_union
    return ee.Geometry(geom.__geo_interface__)

def download_gee_image(image, ee_geom, scale, output_filepath):
    """
    Download a GEE image using getDownloadURL.
    :param image: ee.Image to download.
    :param ee_geom: ee.Geometry used as region.
    :param scale: Scale (in meters) for the download.
    :param output_filepath: Local filepath to save the image.
    """
    params = {
        'scale': scale,
        'region': ee_geom.coordinates().getInfo(),
        'filePerBand': False,
        'format': 'GeoTIFF'
    }
    try:
        download_url = image.getDownloadURL(params)
        print(f"Downloading from URL: {download_url}")
        response = requests.get(download_url, stream=True)
        if response.status_code == 200:
            with open(output_filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            print(f"Downloaded image saved to {output_filepath}")
        else:
            print(f"Failed to download image, status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading image: {e}")

def download_images_for_region(region_name, ee_geom, unique_dates, scale=1000):
    """
    For each unique date, download one image from each collection for a given region.
    Images are saved in a subfolder named after the region.
    :param region_name: Name of the region.
    :param ee_geom: Earth Engine geometry of the region.
    :param unique_dates: List or array of unique date objects.
    :param scale: Download scale in meters.
    """
    region_folder = os.path.join(download_base_dir, region_name)
    os.makedirs(region_folder, exist_ok=True)
    
    for day in unique_dates:
        start_date = datetime.combine(day, datetime.min.time())
        end_date = start_date + timedelta(days=1)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        print(f"Downloading images for region '{region_name}' on {start_str}")
        
        # Filter each image collection by date and geometry.
        collection_d = ee.ImageCollection('NASA/VIIRS/002/VNP21A1D') \
            .filterDate(start_str, end_str) \
            .filterBounds(ee_geom)
        collection_n = ee.ImageCollection('NASA/VIIRS/002/VNP21A1N') \
            .filterDate(start_str, end_str) \
            .filterBounds(ee_geom)
        
        # Get the first image from each collection (if available).
        image_d = collection_d.first()
        image_n = collection_n.first()
        
        if image_d:
            output_file_d = os.path.join(region_folder, f"{start_str}_VNP21A1D.tif")
            download_gee_image(image_d, ee_geom, scale, output_file_d)
        else:
            print(f"  No VNP21A1D image found for {start_str} in region {region_name}")
        
        if image_n:
            output_file_n = os.path.join(region_folder, f"{start_str}_VNP21A1N.tif")
            download_gee_image(image_n, ee_geom, scale, output_file_n)
        else:
            print(f"  No VNP21A1N image found for {start_str} in region {region_name}")

# Main loop: Process each CSV and its corresponding shapefile.
for csv_path in csv_files[:1]:
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    print(f"\nProcessing region: {base_name}")

    # Construct the corresponding shapefile folder path.
    shp_folder = os.path.join(shp_base_dir, base_name)
    shp_files = glob.glob(os.path.join(shp_folder, '*.shp'))
    if not shp_files:
        print(f" - No shapefile found for {base_name}. Skipping.")
        continue
    shp_path = shp_files[0]  # Use the first found shapefile.
    
    # Read CSV and filter rows with valid coordinates.
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['smap_lon', 'smap_lat'])
    if df.empty:
        print(f" - No valid coordinate rows found in {base_name}.")
        continue

    try:
        df['time'] = pd.to_datetime(df['time'])
    except Exception as e:
        print(f" - Error parsing dates in {base_name}: {e}")
        continue

    unique_dates = df['time'].dt.date.unique()
    print(f" - Found {len(unique_dates)} days with coordinates.")
    
    ee_geom = load_shapefile(shp_path)

    # Optionally: Process images in GEE (e.g., count images per day)
    for day in unique_dates[:5]:
        start_date = datetime.combine(day, datetime.min.time())
        end_date = start_date + timedelta(days=1)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        collection_d = ee.ImageCollection('NASA/VIIRS/002/VNP21A1D') \
            .filterDate(start_str, end_str) \
            .filterBounds(ee_geom)
        collection_n = ee.ImageCollection('NASA/VIIRS/002/VNP21A1N') \
            .filterDate(start_str, end_str) \
            .filterBounds(ee_geom)
        
        count_d = collection_d.size().getInfo()
        count_n = collection_n.size().getInfo()
        print(f"   Date: {start_str} | VNP21A1D: {count_d} images, VNP21A1N: {count_n} images")
    
    # Now, call the download function for the region.
    download_images_for_region(base_name, ee_geom, unique_dates, scale=1000)

print("All processing and downloads complete.")

