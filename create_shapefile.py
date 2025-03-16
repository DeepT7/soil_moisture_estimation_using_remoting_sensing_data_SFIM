import os
import geopandas as gpd
from shapely.geometry import Polygon
from pyproj.crs import CRS 

output_dir = "buildings_labels/ID127"
os.makedirs(output_dir, exist_ok=True)

