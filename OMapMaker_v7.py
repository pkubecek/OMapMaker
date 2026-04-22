import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import xml.etree.ElementTree as ET
from pysheds.grid import Grid as PyshedsGrid
import tempfile
from svgpath2mpl import parse_path
from ast import literal_eval
import laspy
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.enums import Resampling
import numpy as np
from scipy.interpolate import griddata, splprep, splev
import matplotlib.pyplot as plt
from scipy.ndimage import (binary_dilation, binary_erosion, gaussian_filter, label,
                           find_objects, minimum_filter, maximum_filter,
                           binary_opening, binary_closing)
from matplotlib.collections import LineCollection
from matplotlib.transforms import Affine2D
from matplotlib.patches import PathPatch, Polygon as MplPolygon
import geopandas as gpd
from shapely.geometry import (LineString, MultiLineString, Polygon, MultiPolygon,
                               box, shape, Point, MultiPoint)
from shapely.ops import unary_union
from shapely import affinity
import fiona
import osmnx as ox
from pyproj import Transformer, CRS
import pandas as pd
from PIL import Image, ImageTk
from skimage import measure
import os
import re
import threading
import urllib.request
import urllib.error
import zipfile
import io
import math
import time

CURRENT_CRS = "EPSG:5514"
OOM_EXPORT_LAYERS: dict = {}

def _oom_isom_code(sym_key):
    raw = sym_key[3:] if sym_key.startswith("sym") else sym_key
    m = re.match(r'^(\d+)', raw)
    return m.group(1) if m else None

def oom_collect(sym_key, gdf):
    if gdf is None or gdf.empty:
        return
    code = _oom_isom_code(sym_key)
    if code is None:
        return
    if code not in OOM_EXPORT_LAYERS:
        OOM_EXPORT_LAYERS[code] = {"Point": [], "Line": [], "Polygon": []}

    for geom_type, bucket_key in [("Point", "Point"), ("Line", "Line"), ("Polygon", "Polygon")]:
        type_mask = gdf.geometry.geom_type.isin(
            ["Point", "MultiPoint"] if geom_type == "Point" else
            ["LineString", "MultiLineString"] if geom_type == "Line" else
            ["Polygon", "MultiPolygon"]
        )
        subset = gdf.loc[type_mask, ["geometry"]].copy()
        if not subset.empty:
            OOM_EXPORT_LAYERS[code][bucket_key].append(subset)

def export_oom_gpkg(output_path):
    if not OOM_EXPORT_LAYERS:
        print("GPKG export: no layers to export.")
        return

    SUFFIX = {"Point": "_point", "Line": "_line", "Polygon": "_poly"}
    written = 0

    for code in sorted(OOM_EXPORT_LAYERS.keys(), key=lambda x: int(x)):
        buckets = OOM_EXPORT_LAYERS[code]
        non_empty = {k: v for k, v in buckets.items() if v}
        if not non_empty:
            continue

        use_suffix = len(non_empty) > 1

        for geom_type, frames in non_empty.items():
            try:
                merged = gpd.GeoDataFrame(
                    pd.concat(frames, ignore_index=True),
                    crs=CURRENT_CRS
                )
                merged = merged[merged.geometry.notna() & ~merged.geometry.is_empty]
                if merged.empty:
                    continue

                if geom_type == "Polygon":
                    merged.geometry = merged.geometry.buffer(0)
                    merged = merged[merged.geometry.is_valid & ~merged.geometry.is_empty]
                    if merged.empty:
                        continue

                merged = merged.drop_duplicates(subset=["geometry"])

                layer_name = "isom_{}{}".format(
                    code, SUFFIX[geom_type] if use_suffix else ""
                )
                merged = merged[["geometry"]].copy()
                merged["Layer"] = layer_name
                merged.to_file(output_path, layer=layer_name, driver="GPKG")
                print("  -> {}: {} prvků [{}]".format(layer_name, len(merged), geom_type))
                written += 1
            except Exception as e:
                print("  -> Error při exportu isom_{} [{}]: {}".format(code, geom_type, e))

    if written > 0:
        print("GPKG export finished: {} layers -> {}".format(written, output_path))
    else:
        print("GPKG export: no layer succesfully written.")


def load_symbol_library(xml_file):
    print(f"--- Loading symbols from: {xml_file} ---")
    library = {}

    if not os.path.exists(xml_file):
        print(" Error: File symbols.xml not found.")
        return {}

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for symbol in root.findall('symbol'):
            sid = symbol.get('id')
            stype = symbol.get('type')
            if not sid:
                continue

            style_elem = symbol.find('style')
            props = style_elem.attrib.copy() if style_elem is not None else {}
            ticks_elem = symbol.find('style_ticks')
            if ticks_elem is not None:
                for k, v in ticks_elem.attrib.items():
                    props[f"tick_{k}"] = v

            clean_props = {}
            for k, v in props.items():
                val_str = str(v).strip()
                if val_str.startswith('('):
                    try:
                        clean_props[k] = literal_eval(val_str)
                        continue
                    except Exception:
                        pass
                val_fixed = val_str.replace(',', '.')
                try:
                    clean_props[k] = float(val_fixed)
                    continue
                except ValueError:
                    pass
                clean_props[k] = val_str

            path_obj = None
            path_d = None
            path_elem = symbol.find('path')
            if path_elem is not None:
                path_d = path_elem.get('d')
                if path_d:
                    try:
                        path_obj = parse_path(path_d)
                        ext = path_obj.get_extents()
                        center = (ext.xmin + ext.xmax) / 2, (ext.ymin + ext.ymax) / 2
                        path_obj.vertices -= center
                        path_obj.vertices[:, 1] *= -1
                    except Exception as e:
                        print(f"Error parsování path u {sid}: {e}")

            library[sid] = {
                "type": stype,
                "props": clean_props,
                "path": path_obj,
                "path_d": path_d
            }

        print(f" Loaded {len(library)} symbols.")
        return library

    except Exception as e:
        print(f" KRITICKÁ Error XML: {e}")
        return {}


def load_dmr_grid(dmr_path, target_crs_code, pixel_size=0.5, sigma_smooth=6.5):
    print(f"Loading DTM")
    print(f"  ->  CRS: {target_crs_code}")
    status_label.config(text=f"Loading DTM ({pixel_size}m pixel)...")
    root.update_idletasks()

    MAX_POINTS_DMR = 2_500_000
    xs, ys, zs = [], [], []
    transformer = None

    with laspy.open(dmr_path) as fh:
        try:
            source_crs = fh.header.parse_crs()
            if source_crs is None:
                raise ValueError("No CRS in header")
        except Exception:
            print("  -> No CRS found in header, using EPSG:5514 (S-JTSK).")
            source_crs = CRS.from_epsg(5514)

        try:
            target_crs_obj = CRS.from_string(target_crs_code)
            if source_crs != target_crs_obj:
                print(f"  -> Transformuji body: {source_crs.to_string()} -> {target_crs_code}")
                transformer = Transformer.from_crs(source_crs, target_crs_obj, always_xy=True)
        except Exception as e:
            print(f"Error při přípravě transformace: {e}")

        total_points = fh.header.point_count
        fraction = min(1.0, MAX_POINTS_DMR / total_points) if total_points > 0 else 1.0
        if fraction < 1.0:
            print(f"  -> File has {total_points:,} points, loading ~{fraction*100:.0f}% (random sample per-chunk).")

        for chunk in fh.chunk_iterator(1_000_000):
            clas = np.array(chunk.classification)
            ground_mask = (clas == 2) | (clas == 8)
            if not np.any(ground_mask):
                continue
            chunk_x = np.array(chunk.x[ground_mask])
            chunk_y = np.array(chunk.y[ground_mask])
            chunk_z = np.array(chunk.z[ground_mask])
            if fraction < 1.0:
                rnd = np.random.rand(len(chunk_x)) < fraction
                chunk_x, chunk_y, chunk_z = chunk_x[rnd], chunk_y[rnd], chunk_z[rnd]
            if len(chunk_x) == 0:
                continue
            if transformer:
                chunk_x, chunk_y = transformer.transform(chunk_x, chunk_y)
            xs.append(chunk_x)
            ys.append(chunk_y)
            zs.append(chunk_z)

    if not xs:
        raise ValueError("No points classified as ground.")

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    z = np.concatenate(zs)

    print(f"  -> Loaded {len(x):,} DTM points.")

    buffer_dist = 1* pixel_size
    min_x, max_x = x.min() - buffer_dist, x.max() + buffer_dist
    min_y, max_y = y.min() - buffer_dist, y.max() + buffer_dist
    extent = (min_x, max_x, min_y, max_y)

    grid_x, grid_y = np.mgrid[extent[0]:extent[1]:pixel_size,
                               extent[2]:extent[3]:pixel_size]

    print(f"Interpolating DTM (area: {grid_x.shape})...")
    status_label.config(text="Interpolating DTM...")
    root.update_idletasks()

    points = np.vstack((x, y)).T

    shift_x = np.mean(points[:, 0])
    shift_y = np.mean(points[:, 1])
    points_shifted = points - np.array([shift_x, shift_y])
    grid_x_shifted = grid_x - shift_x
    grid_y_shifted = grid_y - shift_y

    print("  -> Metoda Cubic...")
    dmr_grid = griddata(points_shifted, z, (grid_x_shifted, grid_y_shifted), method='cubic')
    mask_nan = np.isnan(dmr_grid)
    if np.any(mask_nan):
        dmr_grid_nearest = griddata(points_shifted, z, (grid_x_shifted, grid_y_shifted), method='nearest')
        dmr_grid[mask_nan] = dmr_grid_nearest[mask_nan]
    dmr_grid = gaussian_filter(dmr_grid, sigma=sigma_smooth)
    
    return dmr_grid, grid_x, grid_y, extent, points, z
    


def load_dmp_grid(dmp_path, grid_x, grid_y, extent, target_crs_code):
    print(f"Loading DSM: {dmp_path}")
    status_label.config(text="Loading DSM (Random sample)...")
    root.update_idletasks()
    MAX_POINTS_DMP = 2_500_000
    file_ext = os.path.splitext(dmp_path)[1].lower()

    if file_ext in ['.las', '.laz']:
        xs, ys, zs = [], [], []
        transformer = None

        with laspy.open(dmp_path) as fh:
            try:
                source_crs = fh.header.parse_crs()
                if source_crs is None:
                    source_crs = CRS.from_epsg(5514)
            except Exception:
                source_crs = CRS.from_epsg(5514)

            try:
                target_crs_obj = CRS.from_string(target_crs_code)
                if source_crs != target_crs_obj:
                    transformer = Transformer.from_crs(source_crs, target_crs_obj, always_xy=True)
            except Exception as e:
                print(f"Error transformace DMP: {e}")

            total_points = fh.header.point_count
            fraction = min(1.0, MAX_POINTS_DMP / total_points) if total_points > 0 else 1.0

            for chunk in fh.chunk_iterator(1_000_000):
                cx = np.array(chunk.x)
                cy = np.array(chunk.y)
                cz = np.array(chunk.z)
                cc = np.array(chunk.classification)

                valid_mask = (cc != 7)
                if fraction < 1.0:
                    random_mask = np.random.rand(len(cx)) < fraction
                    final_mask = valid_mask & random_mask
                else:
                    final_mask = valid_mask

                if np.any(final_mask):
                    chunk_x = cx[final_mask]
                    chunk_y = cy[final_mask]
                    chunk_z = cz[final_mask]
                    if transformer:
                        chunk_x, chunk_y = transformer.transform(chunk_x, chunk_y)
                    xs.append(chunk_x)
                    ys.append(chunk_y)
                    zs.append(chunk_z)

        if not xs:
            raise ValueError("V DMP nejsou platná data.")

        x = np.concatenate(xs)
        y = np.concatenate(ys)
        z = np.concatenate(zs)
        points = np.vstack((x, y)).T

        print("Interpolating DSM...")
        status_label.config(text="Interpolating DSM...")
        root.update_idletasks()

        shift_x = np.mean(points[:, 0])
        shift_y = np.mean(points[:, 1])
        points_shifted = points - np.array([shift_x, shift_y])
        grid_x_shifted = grid_x - shift_x
        grid_y_shifted = grid_y - shift_y

        dmp_grid = griddata(points_shifted, z, (grid_x_shifted, grid_y_shifted), method='linear')
        if np.isnan(dmp_grid).all():
            print("  -> Linear method failed, using nearest.")
            dmp_grid = griddata(points_shifted, z, (grid_x_shifted, grid_y_shifted), method='nearest')
    ## Prepared for raster datasets
    elif file_ext in ['.tif', '.tiff']:
        print("GeoTIFF  CRS error!")
        with rasterio.open(dmp_path) as src:
            total_pixels = src.width * src.height
            if total_pixels > MAX_POINTS_DMP:
                scale = (MAX_POINTS_DMP / total_pixels) ** 0.5
                nw, nh = int(src.width * scale), int(src.height * scale)
                print(f"  -> Zmenšuji TIF na {nw}x{nh} (Bilinear)...")
                data = src.read(1, out_shape=(nh, nw), resampling=Resampling.bilinear)
                transform = src.transform * src.transform.scale(src.width / nw, src.height / nh)
            else:
                data = src.read(1)
                transform = src.transform

            rows, cols = np.indices(data.shape)
            xs, ys = rasterio.transform.xy(transform, rows.flatten(), cols.flatten())
            z = data.flatten()

            if src.nodata is not None:
                mask = (z != src.nodata)
                x, y, z = np.array(xs)[mask], np.array(ys)[mask], z[mask]
            else:
                x, y = np.array(xs), np.array(ys)

            points = np.vstack((x, y)).T
            shift_x = np.mean(points[:, 0])
            shift_y = np.mean(points[:, 1])
            points_shifted = points - np.array([shift_x, shift_y])
            grid_x_shifted = grid_x - shift_x
            grid_y_shifted = grid_y - shift_y

            dmp_grid = griddata(points_shifted, z, (grid_x_shifted, grid_y_shifted), method='linear')
            if np.isnan(dmp_grid).all():
                dmp_grid = griddata(points_shifted, z, (grid_x_shifted, grid_y_shifted), method='nearest')
    else:
        raise ValueError(f"Format {file_ext} is not supported")

    return dmp_grid


def generate_and_plot_contours(ax, padded_grid, levels, style_id, zorder, transform_info, clip_geom):
    style = SYMBOL_LIBRARY.get(style_id)
    if not style:
        return
 
    min_x, min_y, px_step_x, px_step_y, pad = transform_info
    lines = []
 
    for level in levels:
        contours = measure.find_contours(padded_grid, level)
        for contour in contours:
            x_coords = min_x + (contour[:, 0] - pad) * px_step_x
            y_coords = min_y + (contour[:, 1] - pad) * px_step_y
            if len(x_coords) > 2:
                lines.append(LineString(np.column_stack((x_coords, y_coords))))
 
    if not lines:
        return
 
    gdf = gpd.GeoDataFrame(geometry=lines, crs=CURRENT_CRS)
    if clip_geom is not None:
        gdf = gpd.clip(gdf, clip_geom)
    if not gdf.empty:
        plot_masked(sym_key=style_id, zorder=zorder, mask=None, gdf=gdf, ax=ax, to_mask=False)


def add_contour_lines(ax, grid_x, grid_y, dmr_grid_unclipped, clip_polygon=None):
    print("Drawing contours...")
    status_label.config(text="Drawing contours...")
    root.update_idletasks()
 
    valid_data_mask = (dmr_grid_unclipped > 0) & (~np.isnan(dmr_grid_unclipped))
    safe_mask = valid_data_mask
    dmr_grid_plot = np.where(safe_mask, dmr_grid_unclipped, np.nan)
 
    pad_width = 100
    dmr_padded = np.pad(dmr_grid_plot, pad_width=pad_width, mode='edge')
 
    min_x, max_x = grid_x.min(), grid_x.max()
    min_y, max_y = grid_y.min(), grid_y.max()
    px_step_x = (max_x - min_x) / (grid_x.shape[0] - 1)
    px_step_y = (max_y - min_y) / (grid_y.shape[1] - 1)
 
    transform_info = (min_x, min_y, px_step_x, px_step_y, pad_width)
 
    # Ořez vrstevnic: pokud je clip_polygon, použijeme ho; jinak rozsah gridu
    original_extent_box = box(min_x, min_y, max_x, max_y)
    clip_geom = clip_polygon if clip_polygon is not None else original_extent_box
 
    min_z = np.nanmin(dmr_grid_plot)
    max_z = np.nanmax(dmr_grid_plot)
 
    major_levels = np.arange(np.floor(min_z / 25) * 25, np.ceil(max_z / 25) * 25 + 1, 25)
    base_levels = np.arange(np.floor(min_z / 5) * 5, np.ceil(max_z / 5) * 5 + 1, 5)
    base_levels = np.setdiff1d(base_levels, major_levels)
 
    generate_and_plot_contours(ax, dmr_padded, base_levels, 'sym101', 25, transform_info, clip_geom)
    generate_and_plot_contours(ax, dmr_padded, major_levels, 'sym102', 25, transform_info, clip_geom)
 
    filled_mean = np.nanmean(dmr_grid_unclipped)
    dmr_grid_calc = np.nan_to_num(dmr_grid_unclipped, nan=filled_mean)
 
    gy, gx = np.gradient(dmr_grid_calc)
    gxx, _ = np.gradient(gx)
    _, gyy = np.gradient(gy)
    curvature = np.abs(gxx + gyy)
    slope = np.hypot(gx, gy)
 
    curvature_threshold = np.percentile(curvature[safe_mask], 30)
    curvature_mask = (curvature > curvature_threshold) & safe_mask
    gentle_slope_mask = (slope < np.percentile(slope[safe_mask], 40)) & safe_mask
    combined_mask = curvature_mask & gentle_slope_mask
    dilated_mask = binary_dilation(combined_mask, iterations=2)
    dmr_grid_minor = np.where(dilated_mask, dmr_grid_plot, np.nan)
    dmr_padded_minor = np.pad(dmr_grid_minor, pad_width=pad_width, mode='edge')
 
    minor_levels = np.arange(np.floor(min_z / 2.5) * 2.5, np.ceil(max_z / 2.5) * 2.5 + 1, 2.5)
    minor_levels = np.setdiff1d(minor_levels, np.union1d(major_levels, base_levels))
 
    generate_and_plot_contours(ax, dmr_padded_minor, minor_levels, 'sym103', 25, transform_info, clip_geom)
 
    print("Contours ready")
 


def vectorize_rocks(grid_x, grid_y, dmr_grid, transform, slope_threshold_deg=54):
    print("Drawing cliffs...")
    status_label.config(text="Drawing cliffs...")
    root.update_idletasks()

    pixel_size_x = abs(transform.a)
    pixel_size_y = abs(transform.e)

    dy, dx = np.gradient(dmr_grid, pixel_size_y, pixel_size_x)
    slope = np.rad2deg(np.arctan(np.hypot(dx, dy)))

    valid_data_mask = (dmr_grid > 0) & (~np.isnan(dmr_grid))
    safe_mask = binary_erosion(valid_data_mask, iterations=7)
    rock_mask_raw = (slope > slope_threshold_deg) & safe_mask

    rock_area = rock_mask_raw.astype(np.int32).T
    rock_area = np.flipud(rock_area)

    pixel_area = pixel_size_x * pixel_size_y
    min_area = 10 * pixel_area

    if not np.any(rock_area):
        return gpd.GeoDataFrame(columns=['class_name', 'geometry'], crs=CURRENT_CRS)

    try:
        results_generator = rasterio.features.shapes(rock_area, mask=(rock_area != 0), transform=transform)
        features = []
        for geom, value in results_generator:
            features.append({'geometry': shape(geom), 'class_name': 'Skala'})

        if not features:
            return gpd.GeoDataFrame(columns=['class_name', 'geometry'], crs=CURRENT_CRS)

        gdf = gpd.GeoDataFrame(features, crs=CURRENT_CRS)
        gdf = gdf[gdf.geometry.area >= min_area]
        print(" Cliffs ready")

        gdf.geometry = gdf.geometry.buffer(0.4).simplify(0.3)
        dissolved_gdf = gdf.dissolve(by='class_name').reset_index()
        return dissolved_gdf

    except Exception as e:
        print(f"Error skal: {e}")
        return gpd.GeoDataFrame(columns=['class_name', 'geometry'], crs=CURRENT_CRS)


def _fill_depressions_numpy(dem, iterations=50):
    """
    implementace Fill Sinks (Priority-Flood lite).
    Funguje bez externích knihoven – pouze numpy + scipy.
    """
    filled = dem.copy()
    border_val = np.nanmin(dem) - 1.0
    filled[0, :]  = border_val
    filled[-1, :] = border_val
    filled[:, 0]  = border_val
    filled[:, -1] = border_val

    for i in range(iterations):
        neighbor_min = minimum_filter(filled, size=3, mode='nearest')
        new_filled = np.maximum(dem, neighbor_min)
        new_filled[0, :]  = border_val
        new_filled[-1, :] = border_val
        new_filled[:, 0]  = border_val
        new_filled[:, -1] = border_val

        if np.allclose(new_filled, filled, atol=1e-6):
            break
        filled = new_filled

    return filled


def add_depressions(ax, grid_x, grid_y, dmr_grid, pixel_size=0.5, min_diameter=2, max_diameter=5, min_depth=0.7):
    print(f"Finding depressions...")
    status_label.config(text="Finding depressions...")
    root.update_idletasks()

    sym_key = 'sym111'
    symbol_info = SYMBOL_LIBRARY.get(sym_key)
    if not symbol_info or symbol_info['type'] != 'point':
        print(f"Error: Symbol '{sym_key}' nebyl nalezen v knihovně.")
        return

    symbol_props = symbol_info['props'].copy()
    symbol_path = symbol_info['path']

    svg_path_string = symbol_props.get('path_d')
    if svg_path_string:
        try:
            marker_path = parse_path(svg_path_string)
            marker_path.vertices -= marker_path.vertices.mean(axis=0)
            symbol_path = marker_path
        except Exception as e:
            print(f" Error SVG with symbol {sym_key}: {e}")

    symbol_props.setdefault('zorder', 20)
    scale_factor = 1.0
    _strip_custom_keys(symbol_props)

    valid_mask = (dmr_grid > 0) & (~np.isnan(dmr_grid))
    safe_mask = binary_erosion(valid_mask, iterations=3)

    fill_input = dmr_grid.copy()
    fill_mean = np.nanmean(fill_input[valid_mask])
    fill_input = np.where(np.isnan(fill_input) | ~valid_mask, fill_mean, fill_input)

    #FILL SINKS
    try:
        from pysheds.grid import Grid as PyshedsGrid
        import tempfile
        from rasterio.transform import from_bounds

        min_x, max_x = grid_x.min(), grid_x.max()
        min_y, max_y = grid_y.min(), grid_y.max()
        tform = from_bounds(min_x, min_y, max_x, max_y,
                            fill_input.shape[0], fill_input.shape[1])

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = tmp.name

        dem_for_write = fill_input.T

        with rasterio.open(
            tmp_path, 'w',
            driver='GTiff',
            height=dem_for_write.shape[0],
            width=dem_for_write.shape[1],
            count=1,
            dtype=dem_for_write.dtype,
            crs=CURRENT_CRS,
            transform=tform,
            nodata=-9999
        ) as dst:
            dst.write(dem_for_write, 1)

        grid = PyshedsGrid.from_raster(tmp_path)
        dem_rd = grid.read_raster(tmp_path)
        pit_filled = grid.fill_pits(dem_rd)
        dem_filled = grid.fill_depressions(pit_filled)

        depth_grid = np.array(dem_filled) - np.array(dem_rd)
        depth_grid = depth_grid.T

        os.unlink(tmp_path)
        print(f"  -> Fill sinks completed, max. depth: {depth_grid.max():.2f} m")

    except ImportError:
        print("  -> pysheds is not accesible, using numpy Fill Sinks...")
        filled = _fill_depressions_numpy(fill_input.T.astype(np.float64))
        depth_grid = (filled - fill_input.T).T
        print(f"  -> Fill Sinks completed, max. depth: {depth_grid.max():.2f} m")

    except Exception as e:
        print(f"  -> Error in Fill Sinks ({e})")
        filled = _fill_depressions_numpy(fill_input.T.astype(np.float64))
        depth_grid = (filled - fill_input.T).T

    depression_mask = (depth_grid > min_depth) & safe_mask

    labeled, num_features = label(depression_mask)
    slices = find_objects(labeled)

    count_final = 0
    pts = []

    for slc in slices:
        region = labeled[slc]
        region_mask = (region > 0)
        ny, nx = region_mask.shape
        diameter_meters = max(ny, nx) * pixel_size

        if not (min_diameter <= diameter_meters <= max_diameter):
            continue

        cy, cx = np.argwhere(region_mask).mean(axis=0)
        i0 = int(slc[0].start + cy)
        j0 = int(slc[1].start + cx)
        if i0 >= dmr_grid.shape[0] or j0 >= dmr_grid.shape[1]:
            continue

        x0 = grid_x[i0, j0]
        y0 = grid_y[i0, j0]

        transform_patch = Affine2D().rotate_deg(180).scale(scale_factor).translate(x0, y0) + ax.transData
        patch = PathPatch(symbol_path, transform=transform_patch, **symbol_props)
        ax.add_patch(patch)

        pts.append(Point(x0, y0))
        count_final += 1

    print(f"  -> {count_final} depressions drawn.")

    if pts:
        oom_collect('sym111', gpd.GeoDataFrame(geometry=pts, crs=CURRENT_CRS))

def add_knoll_symbols(ax, grid_x, grid_y, dmr_grid, pixel_size=0.5, min_diameter=1.5, max_diameter=10, min_height=0.5):
    status_label.config(text="Finding knolls...")
    root.update_idletasks()

    sym_key = 'sym109'
    symbol_info = SYMBOL_LIBRARY.get(sym_key)
    if not symbol_info or symbol_info['type'] != 'point':
        return

    symbol_props = symbol_info['props'].copy()
    symbol_path = symbol_info['path']

    svg_path_string = symbol_props.get('path_d')
    if svg_path_string:
        try:
            marker_path = parse_path(svg_path_string)
            marker_path.vertices -= marker_path.vertices.mean(axis=0)
            symbol_path = marker_path
        except Exception:
            pass

    symbol_props.setdefault('zorder', 20)
    scale_factor = 1.0
    _strip_custom_keys(symbol_props)

    valid_mask = (dmr_grid > 0) & (~np.isnan(dmr_grid))
    safe_mask = binary_erosion(valid_mask, iterations=3)

    inverted_dem = -dmr_grid.copy()
    fill_input = inverted_dem.copy()
    fill_mean = np.nanmean(fill_input[valid_mask])
    fill_input = np.where(np.isnan(fill_input) | ~valid_mask, fill_mean, fill_input)

    try:
        from pysheds.grid import Grid as PyshedsGrid
        import tempfile
        from rasterio.transform import from_bounds

        min_x, max_x = grid_x.min(), grid_x.max()
        min_y, max_y = grid_y.min(), grid_y.max()
        tform = from_bounds(min_x, min_y, max_x, max_y,
                            fill_input.shape[0], fill_input.shape[1])

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = tmp.name

        dem_for_write = fill_input.T

        with rasterio.open(
            tmp_path, 'w',
            driver='GTiff',
            height=dem_for_write.shape[0],
            width=dem_for_write.shape[1],
            count=1,
            dtype=dem_for_write.dtype,
            crs=CURRENT_CRS,
            transform=tform,
            nodata=-9999
        ) as dst:
            dst.write(dem_for_write, 1)

        grid = PyshedsGrid.from_raster(tmp_path)
        dem_rd = grid.read_raster(tmp_path)
        pit_filled = grid.fill_pits(dem_rd)
        dem_filled = grid.fill_depressions(pit_filled)

        height_grid = np.array(dem_filled) - np.array(dem_rd)
        height_grid = height_grid.T

        os.unlink(tmp_path)

    except ImportError:
        filled = _fill_depressions_numpy(fill_input.T.astype(np.float64))
        height_grid = (filled - fill_input.T).T

    except Exception:
        filled = _fill_depressions_numpy(fill_input.T.astype(np.float64))
        height_grid = (filled - fill_input.T).T

    knoll_mask = (height_grid > min_height) & safe_mask

    labeled, num_features = label(knoll_mask)
    slices = find_objects(labeled)

    count_final = 0
    pts = []

    for slc in slices:
        region = labeled[slc]
        region_mask = (region > 0)
        ny, nx = region_mask.shape
        diameter_meters = max(ny, nx) * pixel_size

        if not (min_diameter <= diameter_meters <= max_diameter):
            continue

        cy, cx = np.argwhere(region_mask).mean(axis=0)
        i0 = int(slc[0].start + cy)
        j0 = int(slc[1].start + cx)
        if i0 >= dmr_grid.shape[0] or j0 >= dmr_grid.shape[1]:
            continue

        x0 = float(grid_x[i0, j0])
        y0 = float(grid_y[i0, j0])

        transform_patch = Affine2D().scale(scale_factor).translate(x0, y0) + ax.transData
        patch = PathPatch(
            symbol_path,
            transform=transform_patch,
            facecolor=symbol_props.get('facecolor', '#d15c00'),
            edgecolor=symbol_props.get('edgecolor', 'none'),
            linewidth=0,
            zorder=symbol_props.get('zorder', 20)
        )
        ax.add_patch(patch)

        pts.append(Point(x0, y0))
        count_final += 1

    if pts:
        oom_collect('sym109', gpd.GeoDataFrame(geometry=pts, crs=CURRENT_CRS))


CUSTOM_PLOT_KEYS = frozenset(['dotsize', 'dotdistance', 'dotcolor', 'marker_shape',
                              'facecolor_alt', 'hatchdistance', 'hatchcolor',
                              'hatchwidth', 'hatchstyle', 'd', 'path_d'])

def _strip_custom_keys(props):
    for k in CUSTOM_PLOT_KEYS:
        props.pop(k, None)

def in2m(inch, SCALE=10_000):
    return inch * 0.0254 * SCALE

def pt2m(pt, SCALE=10_000):
    return pt * 0.0003527 * SCALE


def plot_dashed_hatch(ax, gdf, style_props, zorder):
    hatch_distance = style_props.pop('hatchdistance')
    hatch_color = style_props.pop('hatchcolor')
    hatch_width = style_props.pop('hatchwidth')
    hatch_style = style_props.pop('hatchstyle')

    gdf.plot(ax=ax, zorder=zorder, **style_props)

    if gdf.empty:
        return

    try:
        all_geoms = unary_union(gdf.geometry)
    except Exception:
        all_geoms = gdf.geometry.buffer(0).unary_union

    if all_geoms.is_empty:
        return

    minx, miny, maxx, maxy = all_geoms.bounds

    if pt2m(hatch_distance, SCALE):
        y_coords = np.arange(
            np.floor(miny / pt2m(hatch_distance, SCALE)) * pt2m(hatch_distance, SCALE),
            maxy,
            pt2m(hatch_distance, SCALE)
        )
        h_lines = [LineString([(minx, y), (maxx, y)]) for y in y_coords]
        if not h_lines:
            return

        multi_lines = MultiLineString(h_lines)
        clipped_lines = multi_lines.intersection(all_geoms)

        if not clipped_lines.is_empty:
            plot_series = gpd.GeoSeries([clipped_lines])
            plot_series.plot(ax=ax, color=hatch_color, linewidth=hatch_width,
                             linestyle=hatch_style, zorder=zorder - 0.1)
            if ax.collections:
                ax.collections[-1].set_linestyle(hatch_style)


def plot_dotted_hatch(ax, gdf, style_props, zorder):
    dot_distance = style_props.pop('dotdistance')
    dot_size = style_props.pop('dotsize')
    dot_color = style_props.pop('dotcolor')

    gdf.plot(ax=ax, zorder=zorder, **style_props)

    if gdf.empty:
        return

    try:
        all_geoms = gdf.geometry.union_all()
    except Exception:
        all_geoms = gdf.geometry.buffer(0).union_all()

    if all_geoms.is_empty:
        return

    minx, miny, maxx, maxy = all_geoms.bounds

    if pt2m(dot_distance, SCALE):
        x_coords = np.arange(
            np.floor(minx / pt2m(dot_distance, SCALE)) * pt2m(dot_distance, SCALE),
            maxx,
            pt2m(dot_distance, SCALE)
        )
        y_coords = np.arange(
            np.floor(miny / pt2m(dot_distance, SCALE)) * pt2m(dot_distance, SCALE),
            maxy,
            pt2m(dot_distance, SCALE)
        )

        if len(x_coords) == 0 or len(y_coords) == 0:
            return

        xx, yy = np.meshgrid(x_coords, y_coords)
        flat_x = xx.flatten()
        flat_y = yy.flatten()

        from shapely import prepare, contains_xy
        prepare(all_geoms)
        inside = contains_xy(all_geoms, flat_x, flat_y)
        x_final = flat_x[inside]
        y_final = flat_y[inside]

        if len(x_final) > 0:
            ax.scatter(x_final, y_final, marker='.', color=dot_color, s=dot_size,
                       zorder=zorder + 0.1, edgecolors='none')


def add_magnetic_north_lines(ax, extent, scale, rotation=0, spacing_mm=30, zorder=20):
    spacing_meters = (spacing_mm / 1000.0) * scale
    minx, maxx, miny, maxy = extent

    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2
    diagonal = np.hypot(maxx - minx, maxy - miny)
    num_lines = int(diagonal / spacing_meters) + 2

    lines = []
    for i in range(-num_lines, num_lines + 1):
        x = center_x + (i * spacing_meters)
        line = LineString([(x, center_y - diagonal), (x, center_y + diagonal)])
        if rotation != 0:
            line = affinity.rotate(line, -rotation, origin=(center_x, center_y))
        lines.append(line)

    map_box = box(minx, miny, maxx, maxy)
    multi = MultiLineString(lines)
    clipped = multi.intersection(map_box)
    if clipped.is_empty:
        return
    visible_lines = list(clipped.geoms) if hasattr(clipped, 'geoms') else [clipped]

    try:
        gdf_north = gpd.GeoDataFrame(geometry=visible_lines, crs=CURRENT_CRS)
        plot_masked(sym_key='sym601', zorder=zorder, mask=None, gdf=gdf_north, ax=ax, to_mask=False)
        if 'sym601' not in SYMBOL_LIBRARY:
            gdf_north.plot(ax=ax, color='#21d1ff', linewidth=0.35, zorder=zorder)
    except Exception as e:
        print(f"Error while drawing north lines: {e}")


def get_col(df, col_name):
    if col_name in df.columns:
        return df[col_name]
    return pd.Series([''] * len(df), index=df.index)

"""
def smooth_line(line, s, k):
    if not isinstance(line, LineString) or len(line.coords) < (k + 1):
        return line

    x, y = line.xy
    is_closed = (x[0] == x[-1]) and (y[0] == y[-1])

    if is_closed:
        x = x[:-1]
        y = y[:-1]
    if len(x) < (k + 1):
        return line

    try:
        tck, u = splprep([x, y], s=s, k=k, per=is_closed)
        u_new = np.linspace(u.min(), u.max(), len(x) * 5)
        x_new, y_new = splev(u_new, tck)
        coords = np.vstack((x_new, y_new)).T
        if len(coords) < (k + 1):
            return line
        if is_closed:
            coords[-1] = coords[0]
        return LineString(coords)
    except Exception as e:
        print(f"  -> B-spline selhalo: {e}. Vracím původní linii.")
        return line
    """

def vectorize_vegetation(classified_raster_raw, class_names, transform, dmr_path, save_gpkg=False):
    print("Classifying vegetation...")
    status_label.config(text="Classifying vegetation...")
    root.update_idletasks()

    print("  -> Preparating raster...")
    cleaned_raster = classified_raster_raw.copy()
    struct = np.ones((3, 3), dtype=bool)

    for c in np.unique(cleaned_raster):
        if c == 0:
            continue
        mask = (cleaned_raster == c)
        mask = binary_opening(mask, structure=struct)
        mask = binary_closing(mask, structure=struct)
        cleaned_raster[cleaned_raster == c] = 0
        cleaned_raster[mask] = c

    classified_raster = np.flipud(cleaned_raster.T)

    pixel_area = abs(transform.a * transform.e)
    min_area = 50 * pixel_area
    mask = (classified_raster != 0)

    try:
        results_generator = rasterio.features.shapes(
            classified_raster, mask=mask, transform=transform
        )
        features = []
        for geom, value in results_generator:
            class_id = int(value)
            if class_id == 0:
                continue
            features.append({
                'geometry': shape(geom),
                'class_id': class_id,
                'class_name': class_names.get(class_id, 'Neznama')
            })

        if not features:
            print("No vegetation polygons found.")
            return gpd.GeoDataFrame(columns=['class_name', 'class_id', 'geometry'], crs=CURRENT_CRS)

        print(f"Found {len(features)} polygons.")
        gdf = gpd.GeoDataFrame(features, crs=CURRENT_CRS)
        original_crs = gdf.crs

        print("  -> Filtering small polygons...")
        gdf = gdf[gdf.geometry.area >= min_area]

        if gdf.empty:
            return gpd.GeoDataFrame(columns=['class_name', 'class_id', 'geometry'], crs=original_crs)

        print("  -> Simplyfying polygons...")
        status_label.config(text="Simplyfying polygons...")
        root.update_idletasks()


        print("  -> Disolving polygons...")
        gdf.geometry = gdf.geometry.simplify(0.5, preserve_topology=True)
        dissolved_gdf = gdf.dissolve(by='class_name', aggfunc='first').reset_index()
        gdf.geometry = gdf.geometry.buffer(0.7).buffer(0)
        dissolved_gdf = gpd.GeoDataFrame(dissolved_gdf, geometry='geometry', crs=original_crs)

        if save_gpkg:
            output_file = os.path.splitext(dmr_path)[0] + "_vegetace.gpkg"
            dissolved_gdf.to_file(output_file, driver="GPKG")

        return dissolved_gdf

    except Exception as e:
        print(f" An error occured while vectorizing vegetation: {e}")
        status_label.config(text=f" An error occured while vectorizing vegetation: {e}", foreground="red")
        return gpd.GeoDataFrame(columns=['class_name', 'class_id', 'geometry'], crs="EPSG:5514")


def plot_masked(sym_key, zorder, mask, gdf, ax, to_mask=True, dmr_grid=None, grid_x=None, grid_y=None):
    if gdf is None or gdf.empty:
        return None

    if to_mask:
        if mask is None:
            return None
        if isinstance(mask, (pd.Series, gpd.GeoSeries)):
            mask = mask.reindex(gdf.index).fillna(False)
        subset = gdf[mask].copy()
        if subset.empty:
            return None
    else:
        subset = gdf.copy()

    sym_data = SYMBOL_LIBRARY.get(sym_key, {})
    sym_type = sym_data.get('type')
    sym_path = sym_data.get('path')
    sym_props = sym_data.get('props', {}).copy()

    if 'solid_capstyle' in sym_props:
        sym_props['capstyle'] = sym_props.pop('solid_capstyle')

    # symbol 510: plot + tick marks
    if sym_key == 'sym510':
        subset.plot(ax=ax, zorder=zorder, **sym_props)
        tick_len = 10
        tick_segments = []

        for geom in subset.geometry:
            if geom is None or geom.is_empty:
                continue
            parts = [geom] if geom.geom_type == 'LineString' else geom.geoms

            for line in parts:
                coords = np.array(line.coords)
                if len(coords) < 2:
                    continue
                vectors = np.diff(coords, axis=0)
                norms = np.hypot(vectors[:, 0], vectors[:, 1])
                valid = norms > 0
                vectors = vectors[valid]
                norms = norms[valid]
                coords_clean = np.vstack([coords[0], coords[1:][valid]])
                if len(vectors) == 0:
                    continue

                tangents = vectors / norms[:, None]
                for i in range(len(coords_clean)):
                    x, y = coords_clean[i]
                    if i == 0:
                        t = tangents[0]
                    elif i == len(coords_clean) - 1:
                        t = tangents[-1]
                    else:
                        t = (tangents[i - 1] + tangents[i])
                        nm = np.hypot(t[0], t[1])
                        t = t / nm if nm != 0 else tangents[i - 1]

                    nx, ny = -t[1], t[0]
                    p1 = (x - nx * tick_len / 2, y - ny * tick_len / 2)
                    p2 = (x + nx * tick_len / 2, y + ny * tick_len / 2)
                    tick_segments.append([p1, p2])

        if tick_segments:
            lc = LineCollection(tick_segments,
                                colors=sym_props.get('color', 'black'),
                                linewidths=sym_props.get('linewidth', 1.0),
                                zorder=zorder)
            ax.add_collection(lc)
        oom_collect(sym_key, subset)
        return

    # symbol 201: plot + tick marks (pointed down)
    cliff_ids = ['sym104', 'sym201', 'sym202']
    is_cliff = sym_key in cliff_ids or 'tick_length' in sym_props

    if is_cliff and dmr_grid is not None and grid_x is not None and grid_y is not None:
        tick_len = float(sym_props.pop('tick_length', 4))
        tick_space = float(sym_props.pop('tick_spacing', 4))
        tick_width = float(sym_props.pop('tick_linewidth', 0.3))
        tick_color = sym_props.pop('tick_color', sym_props.get('color', 'black'))

        subset.plot(ax=ax, zorder=zorder, **sym_props)
        grad_x, grad_y = np.gradient(dmr_grid)
        min_x, max_x = grid_x.min(), grid_x.max()
        min_y, max_y = grid_y.min(), grid_y.max()
        shape_x, shape_y = grid_x.shape
        px_size_x = (max_x - min_x) / (shape_x - 1)
        px_size_y = (max_y - min_y) / (shape_y - 1)

        ticks_segments = []
        epsilon = 0.1

        for geom in subset.geometry:
            if geom is None or geom.is_empty:
                continue
            parts = [geom] if geom.geom_type == 'LineString' else geom.geoms

            for line in parts:
                line_len = line.length
                if line_len < tick_space:
                    continue
                distances = np.arange(tick_space / 2, line_len, tick_space)

                for dist in distances:
                    pt = line.interpolate(dist)
                    pt_ahead = line.interpolate(min(dist + epsilon, line_len))
                    dx = pt_ahead.x - pt.x
                    dy = pt_ahead.y - pt.y
                    tan_len = np.hypot(dx, dy)
                    if tan_len == 0:
                        continue

                    tx, ty = dx / tan_len, dy / tan_len
                    n1x, n1y = ty, -tx
                    n2x, n2y = -ty, tx

                    ix = int((pt.x - min_x) / px_size_x)
                    iy = int((pt.y - min_y) / px_size_y)

                    if 0 <= ix < shape_x and 0 <= iy < shape_y:
                        gx = grad_x[ix, iy]
                        gy = grad_y[ix, iy]

                        if gx == 0 and gy == 0:
                            final_nx, final_ny = n1x, n1y
                        elif (n1x * gx) + (n1y * gy) < 0:
                            final_nx, final_ny = n1x, n1y
                        else:
                            final_nx, final_ny = n2x, n2y

                        ticks_segments.append([
                            (pt.x, pt.y),
                            (pt.x + final_nx * tick_len, pt.y + final_ny * tick_len)
                        ])

        if ticks_segments:
            lc = LineCollection(ticks_segments, colors=tick_color,
                                linewidths=tick_width, zorder=zorder)
            ax.add_collection(lc)
        oom_collect(sym_key, subset)
        return

    # Hatch / dot fill
    if 'hatchdistance' in sym_props:
        plot_dashed_hatch(ax, subset.dissolve(), sym_props, zorder=zorder)
        oom_collect(sym_key, subset)
        return
    elif 'dotdistance' in sym_props:
        plot_dotted_hatch(ax, subset.dissolve(), sym_props, zorder=zorder)
        oom_collect(sym_key, subset)
        return

    # Point symbols (SVG path) 
    if sym_type == 'point' and sym_path is not None:
        _strip_custom_keys(sym_props)
        for geom in subset.geometry:
            points_to_plot = []
            if geom.geom_type == 'Point':
                points_to_plot.append((geom.x, geom.y))
            elif geom.geom_type == 'MultiPoint':
                points_to_plot.extend([(p.x, p.y) for p in geom.geoms])
            for x, y in points_to_plot:
                transform = Affine2D().translate(x, y) + ax.transData
                patch = PathPatch(sym_path, transform=transform, zorder=zorder, **sym_props)
                ax.add_patch(patch)
        oom_collect(sym_key, subset)
        return

    # Else
    _strip_custom_keys(sym_props)
    try:
        subset.plot(ax=ax, zorder=zorder, **sym_props)
        oom_collect(sym_key, subset)
    except Exception as e:
        print(f"Error while drawing symbol {sym_key}: {e}")


def clip_vecor_layers(gdf, extent):
    if gdf is None or gdf.empty:
        return gdf
    try:
        clip_box_geom = box(extent[0], extent[2], extent[1], extent[3])
        return gpd.clip(gdf, clip_box_geom)
    except Exception as e:
        print(f"Error při ořezu: {e}")
        return gdf


def add_vector_layers(ax, gdf, extent, zabaged_gdfs, dmr_grid_linear_viz_features, grid_x, grid_y, visibility, isom_gdfs):
    if visibility is None:
        visibility = {k: True for k in ["water", "roads", "buildings", "fences", "man_made", "vegetation"]}

    gdf = clip_vecor_layers(gdf, extent)
    for zabaged_key in zabaged_gdfs:
        zabaged_gdfs[zabaged_key] = clip_vecor_layers(zabaged_gdfs[zabaged_key], extent)

    if (gdf is None or gdf.empty) and not zabaged_gdfs and not isom_gdfs:
        return

    _cols = {c: get_col(gdf, c).fillna('') for c in [
        "access", "amenity", "barrier", "bridge", "building", "covered", "emergency",
        "geological", "highway", "historic", "intermittent", "landuse", "leisure",
        "man_made", "military", "natural", "parking", "place", "power", "railway",
        "surface", "tracktype", "tunnel", "water", "waterway", "wetland", "aerialway"
    ]}
    access = _cols["access"]
    amenity = _cols["amenity"]
    barrier = _cols["barrier"]
    bridge = _cols["bridge"]
    building = _cols["building"]
    covered = _cols["covered"]
    emergency = _cols["emergency"]
    geological = _cols["geological"]
    highway = _cols["highway"]
    historic = _cols["historic"]
    intermittent = _cols["intermittent"]
    landuse = _cols["landuse"]
    leisure = _cols["leisure"]
    man_made = _cols["man_made"]
    military = _cols["military"]
    natural = _cols["natural"]
    parking = _cols["parking"]
    place = _cols["place"]
    power = _cols["power"]
    railway = _cols["railway"]
    surface = _cols["surface"]
    trail_visibility = get_col(gdf, "trail_visibility")
    tracktype = _cols["tracktype"]
    tunnel = _cols["tunnel"]
    water = _cols["water"]
    waterway = _cols["waterway"]
    wetland = _cols["wetland"]
    aerialway = _cols["aerialway"]

    gdf_centroids = gdf.copy()
    gdf_centroids['geometry'] = gdf_centroids.geometry.centroid
    gdf_points = gdf[gdf.geometry.geom_type.isin(["Point"])].copy()
    gdf_lines = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    gdf_polygons = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    # TERRAIN SHAPES
    if visibility.get("contours", True):
        # 104 - Zemní Sráz
        cgdf = isom_gdfs.get("104")
        if cgdf is not None:
            plot_masked(sym_key="sym104", zorder=21, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "StupenSraz" in zabaged_gdfs:
            plot_masked(sym_key="sym104", zorder=21, mask=None, gdf=zabaged_gdfs["StupenSraz"], ax=ax,
                        to_mask=False, dmr_grid=dmr_grid_linear_viz_features, grid_x=grid_x, grid_y=grid_y)
        else:
            plot_masked(sym_key="sym104", zorder=21, mask=(man_made == "embankment"), gdf=gdf_lines, ax=ax,
                        dmr_grid=dmr_grid_linear_viz_features, grid_x=grid_x, grid_y=grid_y)

        # 105 - Zemní val
        cgdf = isom_gdfs.get("105")
        if cgdf is not None:
            plot_masked(sym_key="sym105-1a", zorder=21, mask=None, gdf=cgdf, ax=ax, to_mask=False)
            plot_masked(sym_key="sym105-1b", zorder=21, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "HradbaValBastaOpevneni" in zabaged_gdfs:
            plot_masked(sym_key="sym105-1a", zorder=30, mask=None, gdf=zabaged_gdfs["HradbaValBastaOpevneni"], ax=ax, to_mask=False)
            plot_masked(sym_key="sym105-1b", zorder=30, mask=None, gdf=zabaged_gdfs["HradbaValBastaOpevneni"], ax=ax, to_mask=False)

        # 107 - Rýha / Výmol
        cgdf = isom_gdfs.get("107")
        if cgdf is not None:
            plot_masked(sym_key="sym107", zorder=20, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "RokleVymol" in zabaged_gdfs:
            plot_masked(sym_key="sym107", zorder=20, mask=None, gdf=zabaged_gdfs["RokleVymol"], ax=ax, to_mask=False)

        # 108 - Malá erozní rýha
        cgdf = isom_gdfs.get("108")
        if cgdf is not None:
            plot_masked(sym_key="sym108", zorder=21, mask=None, gdf=cgdf, ax=ax, to_mask=False)

        # 109 - Kupka
        cgdf = isom_gdfs.get("109")
        if cgdf is not None:
            plot_masked(sym_key="sym109", zorder=21, mask=None, gdf=cgdf, ax=ax, to_mask=False)

        # 111 - Malá Prohlubeň
        cgdf = isom_gdfs.get("111")
        if cgdf is not None:
            plot_masked(sym_key="sym111", zorder=21, mask=None, gdf=cgdf, ax=ax, to_mask=False)

        # 112 - Jáma
        cgdf = isom_gdfs.get("112")
        if cgdf is not None:
            plot_masked(sym_key="sym112", zorder=21, mask=None, gdf=cgdf, ax=ax, to_mask=False)

    if visibility.get("rocks", True):
        ''' Not accurate
        # 202 - Skalní sráz
        sym = "sym201"
        cgdf = isom_gdfs.get("201")
        if cgdf is not None:
            plot_masked(sym_key="sym202", zorder=56, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            mask_cliff_high = (natural == "cliff")
            plot_masked(sym_key=sym, zorder=56, mask=mask_cliff_high, gdf=gdf_lines, ax=ax, dmr_grid=dmr_grid_linear_viz_features, grid_x=grid_x, grid_y=grid_y)  
        '''

        # 203-1 - Jeskyně
        cgdf = isom_gdfs.get("203.1")
        if cgdf is not None:
            plot_masked(sym_key="sym203-1", zorder=56, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        if "VstupDoJeskyne" in zabaged_gdfs:
            plot_masked(sym_key="sym203-1", zorder=56, mask=None, gdf=zabaged_gdfs["VstupDoJeskyne"], ax=ax, to_mask=False)
        else:
            mask_cave = (natural == "cave_entrance") | (man_made == "adit")
            plot_masked(sym_key="sym203-1", zorder=56, mask=mask_cave, gdf=gdf_centroids, ax=ax)

        # 203-2 - Nebezpečná jáma
        cgdf = isom_gdfs.get("203.1")
        if cgdf is not None:
            plot_masked(sym_key="sym203-2", zorder=56, mask=None, gdf=cgdf, ax=ax, to_mask=False)

        # 204 - Malý balvan
        cgdf = isom_gdfs.get("204")
        if cgdf is not None:
            plot_masked(sym_key="sym204", zorder=56, mask=None, gdf=cgdf, ax=ax, to_mask=False)

        # 205 - Balvan
        cgdf = isom_gdfs.get("205")
        if cgdf is not None:
            plot_masked(sym_key="sym205", zorder=56, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "OsamelyBalvanSkalaSkalniSuk" in zabaged_gdfs:
            plot_masked(sym_key="sym205", zorder=56, mask=None, gdf=zabaged_gdfs["OsamelyBalvanSkalaSkalniSuk"], ax=ax, to_mask=False)
        else:
            mask_boulder = (natural.isin(["stone", "rock"])) | (geological == "glacial_erratic")
            plot_masked(sym_key="sym205", zorder=56, mask=mask_boulder, gdf=gdf_centroids, ax=ax)

        # 207 - Skupina balvanů
        cgdf = isom_gdfs.get("207")
        if cgdf is not None:
            plot_masked(sym_key="sym207", zorder=56, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "SkupinaBalvanu_b" in zabaged_gdfs:
            plot_masked(sym_key="sym207", zorder=56, mask=None, gdf=zabaged_gdfs["SkupinaBalvanu_b"], ax=ax, to_mask=False)

        # 208 - Kamenité pole
        cgdf = isom_gdfs.get("208")
        if cgdf is not None:
            plot_masked(sym_key="sym208", zorder=18, mask=None, gdf=cgdf, ax=ax, to_mask=False)

        # 209
        cgdf = isom_gdfs.get("209")
        if cgdf is not None:
            plot_masked(sym_key="sym209", zorder=18, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        '''
        else:
            plot_masked(sym_key="sym209", zorder=18, mask=(natural == "scree"), gdf=gdf_polygons, ax=ax)
        '''
        # 210 - Suťoviště
        cgdf = isom_gdfs.get("210")
        if cgdf is not None:
            plot_masked(sym_key="sym210", zorder=18, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            plot_masked(sym_key="sym210", zorder=18, mask=(natural == "blockfield"), gdf=gdf_polygons, ax=ax)

        # 211, 212 - mapovány na sym210
        for code in ["211", "212"]:
            cgdf = isom_gdfs.get(code)
            if cgdf is not None:
                plot_masked(sym_key="sym210", zorder=18, mask=None, gdf=cgdf, ax=ax, to_mask=False)

        # 213 - Písek
        cgdf = isom_gdfs.get("213")
        if cgdf is not None:
            plot_masked(sym_key="sym213", zorder=15, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            plot_masked(sym_key="sym213", zorder=15, mask=(natural.isin(["sand", "dune"])), gdf=gdf_polygons, ax=ax)

        # 214 - Skalní podloží
        cgdf = isom_gdfs.get("214")
        if cgdf is not None:
            plot_masked(sym_key="sym214", zorder=17, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            plot_masked(sym_key="sym214", zorder=17, mask=(natural == "bare_rock"), gdf=gdf_polygons, ax=ax)

        # 215 - Příkop
        mask_ditch = (barrier == "ditch") | (military == "trench")
        cgdf = isom_gdfs.get("215")
        if cgdf is not None:
            plot_masked(sym_key="sym215a", zorder=21, mask=None, gdf=cgdf, ax=ax, to_mask=False)
            plot_masked(sym_key="sym215b", zorder=21, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            plot_masked(sym_key="sym215a", zorder=21, mask=mask_ditch, gdf=gdf_lines, ax=ax)
            plot_masked(sym_key="sym215b", zorder=21, mask=mask_ditch, gdf=gdf_lines, ax=ax)

    # WATER
    if visibility.get("water", True):

        # 301 - Vodní plocha
        cgdf = isom_gdfs.get("301")
        if cgdf is not None:
            plot_masked(sym_key="sym301", zorder=27, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif any(k in zabaged_gdfs for k in ["VodniPlocha", "PozemniNadrz"]):
            for zk in ["VodniPlocha", "PozemniNadrz"]:
                if zk in zabaged_gdfs:
                    plot_masked(sym_key="sym301", zorder=27, mask=None, gdf=zabaged_gdfs[zk], ax=ax, to_mask=False)
        else:
            mask_water_deep = (natural.isin(["lake", "water", "canal"])) | \
                              (water.isin(["lake", "river", "basin", "bay", "reservoir"])) | \
                              (landuse == "basin") | (leisure == "swimming_pool")
            plot_masked(sym_key="sym301", zorder=27, mask=mask_water_deep, gdf=gdf_polygons, ax=ax)

        # 302 - Mělká voda
        cgdf = isom_gdfs.get("302")
        if cgdf is not None:
            plot_masked(sym_key="sym302", zorder=27, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            plot_masked(sym_key="sym302", zorder=27, mask=(water == "stream"), gdf=gdf_polygons, ax=ax)

        # 303 - Jáma s vodou
        cgdf = isom_gdfs.get("303")
        if cgdf is not None:
            plot_masked(sym_key="sym303", zorder=27, mask=None, gdf=cgdf, ax=ax, to_mask=False)

        # 304 - Překonatelný vodní tok
        cgdf = isom_gdfs.get("304")
        if cgdf is not None:
            plot_masked(sym_key="sym304", zorder=26, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "VodniTok" in zabaged_gdfs:
            mask = (get_col(zabaged_gdfs["VodniTok"], "typtoku_p").isin(["povrchový splavný"])) & \
                   (get_col(zabaged_gdfs["VodniTok"], "vydattok_p").isin(["stálý"]))
            plot_masked(sym_key="sym304", zorder=26, mask=mask, gdf=zabaged_gdfs["VodniTok"], ax=ax)
        else:
            mask_river = ((waterway == "river") | (waterway == "canal")) & \
                         (~tunnel.isin(["culvert", "yes", "pipe", "covered", "cave"])) & \
                         (~intermittent.isin(["yes", "dry"]))
            plot_masked(sym_key="sym304", zorder=26, mask=mask_river, gdf=gdf_lines, ax=ax)

        # 305 - Malý překonatelný vodí tok
        cgdf = isom_gdfs.get("305")
        if cgdf is not None:
            plot_masked(sym_key="sym305", zorder=26, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "VodniTok" in zabaged_gdfs:
            mask = (get_col(zabaged_gdfs["VodniTok"], "typtoku_p").isin(["povrchový nesplavný"])) & \
                   (get_col(zabaged_gdfs["VodniTok"], "vydattok_p").isin(["stálý"]))
            plot_masked(sym_key="sym305", zorder=26, mask=mask, gdf=zabaged_gdfs["VodniTok"], ax=ax)
        else:
            mask_stream = ((waterway == "stream") | (waterway == "ditch")) & \
                          (~tunnel.isin(["culvert", "yes", "pipe", "covered", "cave"])) & \
                          (~intermittent.isin(["yes", "dry"]))
            plot_masked(sym_key="sym305", zorder=26, mask=mask_stream, gdf=gdf_lines, ax=ax)

        # 306 - Malý občasný vodní příkop
        cgdf = isom_gdfs.get("306")
        if cgdf is not None:
            plot_masked(sym_key="sym306", zorder=26, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "VodniTok" in zabaged_gdfs:
            mask = (get_col(zabaged_gdfs["VodniTok"], "typtoku_p").isin(["povrchový splavný", "povrchový nesplavný"])) & \
                   (get_col(zabaged_gdfs["VodniTok"], "vydattok_p").isin(["občasný"]))
            plot_masked(sym_key="sym306", zorder=26, mask=mask, gdf=zabaged_gdfs["VodniTok"], ax=ax)
        else:
            mask_drain = ((waterway == "drain") | (waterway.isin(["stream", "ditch"])) & \
                          (intermittent.isin(["yes", "dry"]))) & \
                         (~tunnel.isin(["culvert", "yes", "pipe", "covered", "cave"]))
            plot_masked(sym_key="sym306", zorder=26, mask=mask_drain, gdf=gdf_lines, ax=ax)

        # 307 - Neprůchodná bažina
        cgdf = isom_gdfs.get("307")
        if cgdf is not None:
            plot_masked(sym_key="sym307", zorder=25, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "Raseliniste" in zabaged_gdfs:
            plot_masked(sym_key="sym307", zorder=25, mask=None, gdf=zabaged_gdfs["Raseliniste"], ax=ax, to_mask=False)
        else:
            plot_masked(sym_key="sym307", zorder=25, mask=(wetland == "reedbed"), gdf=gdf_polygons, ax=ax)

        # 308 - Bažina
        cgdf = isom_gdfs.get("308")
        if cgdf is not None:
            plot_masked(sym_key="sym308", zorder=25, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "BazinaMocal" in zabaged_gdfs:
            plot_masked(sym_key="sym308", zorder=25, mask=None, gdf=zabaged_gdfs["BazinaMocal"], ax=ax, to_mask=False)
        else:
            mask_wetland2 = (natural == "wetland") & (~wetland.isin(["marsh", "wet_meadow", "reedbed"]))
            plot_masked(sym_key="sym308", zorder=25, mask=mask_wetland2, gdf=gdf_polygons, ax=ax)

        # 309 - Úzká bažina
        cgdf = isom_gdfs.get("309")
        if cgdf is not None:
            plot_masked(sym_key="sym309", zorder=25, mask=None, gdf=cgdf, ax=ax, to_mask=False)

        # 310 - Nezřetelná bažina
        cgdf = isom_gdfs.get("310")
        if cgdf is not None:
            plot_masked(sym_key="sym308", zorder=25, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            mask_wetland3 = (wetland == "marsh") | (water == "wet_meadow")
            plot_masked(sym_key="sym308", zorder=25, mask=mask_wetland3, gdf=gdf_polygons, ax=ax)

        # 311 - Studna / Nádrž
        cgdf = isom_gdfs.get("311")
        if cgdf is not None:
            plot_masked(sym_key="sym311", zorder=52, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            mask_well = (man_made == "water_well") | (amenity == "fountain") | \
                        (natural == "geysyer") | \
                        (((emergency == "water_tank") | (man_made == "storage_tank")) &
                         (~covered.isin(["yes", "roof", "shelter"])))
            plot_masked(sym_key="sym311", zorder=52, mask=mask_well, gdf=gdf_centroids, ax=ax)

        # 312 - Pramen
        cgdf = isom_gdfs.get("312")
        if cgdf is not None:
            plot_masked(sym_key="sym312", zorder=52, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "ZdrojPodzemnichVod" in zabaged_gdfs:
            plot_masked(sym_key="sym312", zorder=52, mask=None, gdf=zabaged_gdfs["ZdrojPodzemnichVod"], ax=ax, to_mask=False)
        else:
            plot_masked(sym_key="sym312", zorder=52, mask=((natural == "spring") & (covered != "yes")), gdf=gdf_centroids, ax=ax)

        # 313 - Výrazný vodní objekt
        cgdf = isom_gdfs.get("313")
        if cgdf is not None:
            plot_masked(sym_key="sym312", zorder=52, mask=None, gdf=cgdf, ax=ax, to_mask=False)

    # ======================================================================
    # VEGETATION
    # ======================================================================
    if visibility.get("vegetation", True):

        # 401 - Otevřený prostor
        cgdf = isom_gdfs.get("401")
        if cgdf is not None:
            plot_masked(sym_key="sym401", zorder=1.0, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "TrvalyTravniPorost" in zabaged_gdfs:
            plot_masked(sym_key="sym401", zorder=1.0, mask=None, gdf=zabaged_gdfs["TrvalyTravniPorost"], ax=ax, to_mask=False)
        else:
            mask_grass = (landuse.isin(["grassland", "grass", "meadow"])) | \
                         (natural.isin(["grassland", "fell", "heath"]))
            plot_masked(sym_key="sym401", zorder=1.0, mask=mask_grass, gdf=gdf_polygons, ax=ax)

        # 402 - Park
        cgdf = isom_gdfs.get("402")
        if cgdf is not None:
            plot_masked(sym_key="sym402", zorder=1.0, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "OkrasnaZahradaPark" in zabaged_gdfs:
            plot_masked(sym_key="sym402", zorder=1.0, mask=None, gdf=zabaged_gdfs["OkrasnaZahradaPark"], ax=ax, to_mask=False)
        else:
            plot_masked(sym_key="sym402", zorder=1.0, mask=(leisure == "park"), gdf=gdf_polygons, ax=ax)

        # 403 - Divoký otevřený prostor
        cgdf = isom_gdfs.get("403")
        if cgdf is not None:
            plot_masked(sym_key="sym403", zorder=1.0, mask=None, gdf=cgdf, ax=ax, to_mask=False)

        # 404 - Divoký otevřený prostor se stromy
        cgdf = isom_gdfs.get("404")
        if cgdf is not None:
            plot_masked(sym_key="sym404", zorder=1.0, mask=None, gdf=cgdf, ax=ax, to_mask=False)

        # 408 - Alej / Živý plot
        cgdf = isom_gdfs.get("408")
        if cgdf is not None:
            plot_masked(sym_key="sym408l", zorder=19, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "LiniovaVegetace" in zabaged_gdfs:
            mask = get_col(zabaged_gdfs["LiniovaVegetace"], "typveg_p").isin(["živý plot"])
            plot_masked(sym_key="sym408l", zorder=19, mask=mask, gdf=zabaged_gdfs["LiniovaVegetace"], ax=ax)
        else:
            plot_masked(sym_key="sym408l", zorder=99, mask=(natural == "tree_row"), gdf=gdf_polygons, ax=ax)

        # 412 - Orná půda
        cgdf = isom_gdfs.get("412")
        if cgdf is not None:
            plot_masked(sym_key="sym412a", zorder=1.9, mask=None, gdf=cgdf, ax=ax, to_mask=False)
            plot_masked(sym_key="sym412b", zorder=15, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "OrnaPudaAOstatniDaleNespecifikovanePlochy" in zabaged_gdfs:
            mask = get_col(zabaged_gdfs["OrnaPudaAOstatniDaleNespecifikovanePlochy"], "typ_pudy_p").isin(["orná půda"])
            plot_masked(sym_key="sym412a", zorder=1.9, mask=mask, gdf=zabaged_gdfs["OrnaPudaAOstatniDaleNespecifikovanePlochy"], ax=ax)
            plot_masked(sym_key="sym412b", zorder=15, mask=mask, gdf=zabaged_gdfs["OrnaPudaAOstatniDaleNespecifikovanePlochy"], ax=ax)
        else:
            mask_field = (landuse == "farmland")
            plot_masked(sym_key="sym412a", zorder=1.9, mask=mask_field, gdf=gdf_polygons, ax=ax)
            plot_masked(sym_key="sym412b", zorder=15, mask=mask_field, gdf=gdf_polygons, ax=ax)

        # 413 - Sad
        cgdf = isom_gdfs.get("413")
        if cgdf is not None:
            plot_masked(sym_key="sym413", zorder=1.9, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            plot_masked(sym_key="sym413", zorder=1.9, mask=(landuse == "orchard"), gdf=gdf_polygons, ax=ax)

        # 414 - Vinice / Chmelnice
        cgdf = isom_gdfs.get("414")
        if cgdf is not None:
            plot_masked(sym_key="sym414", zorder=1.9, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif any(k in zabaged_gdfs for k in ["Vinice", "Chmelnice"]):
            for zk in ["Vinice", "Chmelnice"]:
                if zk in zabaged_gdfs:
                    plot_masked(sym_key="sym414", zorder=1.9, mask=None, gdf=zabaged_gdfs[zk], ax=ax, to_mask=False)
        else:
            plot_masked(sym_key="sym414", zorder=1.9, mask=(landuse.isin(["plant_nursery", "vineyard"])), gdf=gdf_polygons, ax=ax)

        # 415 - Hranice obdělávané půdy
        cgdf = isom_gdfs.get("415")
        if cgdf is not None:
            plot_masked(sym_key="sym216l", zorder=15, mask=None, gdf=cgdf, ax=ax, to_mask=False)

        # 416 - Hranice vegetace
        cgdf = isom_gdfs.get("416")
        if cgdf is not None:
            plot_masked(sym_key="sym416p", zorder=1.8, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "LesniPudaSeStromyKategorizovana" in zabaged_gdfs:
            mask = (get_col(zabaged_gdfs["LesniPudaSeStromyKategorizovana"], "druh_k").isin(["J"])) & \
                   (get_col(zabaged_gdfs["LesniPudaSeStromyKategorizovana"], "vyska_k").isin(["3"]))
            plot_masked(sym_key="sym416p", zorder=1.8, mask=mask, gdf=zabaged_gdfs["LesniPudaSeStromyKategorizovana"], ax=ax)

        # 417 - Výrazný strom
        cgdf = isom_gdfs.get("417")
        if cgdf is not None:
            plot_masked(sym_key="sym417a", zorder=54, mask=None, gdf=cgdf, ax=ax, to_mask=False)
            plot_masked(sym_key="sym417b", zorder=54, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "VyznamnyNeboOsamelyStromLesik" in zabaged_gdfs:
            plot_masked(sym_key="sym417a", zorder=54, mask=None, gdf=zabaged_gdfs["VyznamnyNeboOsamelyStromLesik"], ax=ax, to_mask=False)
            plot_masked(sym_key="sym417b", zorder=54, mask=None, gdf=zabaged_gdfs["VyznamnyNeboOsamelyStromLesik"], ax=ax, to_mask=False)
        else:
            mask_tree = (natural == "tree")
            if not gdf_centroids[mask_tree].empty:
                plot_masked(sym_key="sym417a", zorder=54, mask=mask_tree, gdf=gdf_centroids, ax=ax)
                plot_masked(sym_key="sym417b", zorder=55, mask=mask_tree, gdf=gdf_centroids, ax=ax)

        # 418 - Výrazný keř
        cgdf = isom_gdfs.get("418")
        if cgdf is not None:
            plot_masked(sym_key="sym418a", zorder=54, mask=None, gdf=cgdf, ax=ax, to_mask=False)
            plot_masked(sym_key="sym418b", zorder=54, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            mask_shrub = (natural == "shrub")
            plot_masked(sym_key="sym418a", zorder=54, mask=mask_shrub, gdf=gdf_centroids, ax=ax)
            plot_masked(sym_key="sym418b", zorder=55, mask=mask_shrub, gdf=gdf_centroids, ax=ax)

        # 419 - Výrazný vegetační objekt
        cgdf = isom_gdfs.get("419")
        if cgdf is not None:
            plot_masked(sym_key="sym419", zorder=54, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            plot_masked(sym_key="sym419", zorder=54, mask=(natural == "tree_stump"), gdf=gdf_centroids, ax=ax)

    # ROADS and railway
    if visibility.get("roads", True):
        # 501 - Parkoviště
        cgdf = isom_gdfs.get("501")
        if cgdf is not None:
            plot_masked(sym_key="sym501", zorder=49, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif any(k in zabaged_gdfs for k in ["ParkovisteOdpocivka", "ArealUceloveZastavby"]):
            if "ParkovisteOdpocivka" in zabaged_gdfs:
                plot_masked(sym_key="sym501", zorder=49, mask=None, gdf=zabaged_gdfs["ParkovisteOdpocivka"], ax=ax, to_mask=False)
            if "ArealUceloveZastavby" in zabaged_gdfs:
                mask = get_col(zabaged_gdfs["ArealUceloveZastavby"], "typzast_k").isin(["408"])
                plot_masked(sym_key="sym501", zorder=49, mask=mask, gdf=zabaged_gdfs["ArealUceloveZastavby"], ax=ax)
        else:
            mask_parking = ((amenity == "parking") & (~parking.isin(["garage", "underground"]))) | \
                           (place == "square") | \
                           (highway.isin(["service", "pedestrian", "footway"])) | \
                           (man_made == "bunker_silo")
            plot_masked(sym_key="sym501", zorder=49, mask=mask_parking, gdf=gdf_polygons, ax=ax)

        # 502D - Dálnice
        cgdf = isom_gdfs.get("502D")
        mask_road_double = (highway.isin(["motorway", "trunk"])) & \
                           (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) & \
                           (bridge != "yes") & (access != "private")
        for sym, z in [("sym502Da", 45), ("sym502Db", 47), ("sym502Dc", 48)]:
            if cgdf is not None:
                plot_masked(sym_key=sym, zorder=z, mask=None, gdf=cgdf, ax=ax, to_mask=False)
            elif "SilniceDalnice" in zabaged_gdfs:
                mask = get_col(zabaged_gdfs["SilniceDalnice"], "typsil_k").isin(["D1", "D2", "M"])
                plot_masked(sym_key=sym, zorder=z, mask=mask, gdf=zabaged_gdfs["SilniceDalnice"], ax=ax)
            else:
                plot_masked(sym_key=sym, zorder=z, mask=mask_road_double, gdf=gdf_lines, ax=ax)

        # 502 - Široká silnice
        cgdf = isom_gdfs.get("502")
        mask_road_major = (highway.isin(["highway_link", "trunk_link", "primary", "primary_link",
                                          "secondary", "secondary_link", "residential", "tertiary",
                                          "living_street"])) & \
                          (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) & \
                          (bridge != "yes") & (access != "private")
        for sym, z in [("sym502a", 45), ("sym502b", 47)]:
            if cgdf is not None:
                plot_masked(sym_key=sym, zorder=z, mask=None, gdf=cgdf, ax=ax, to_mask=False)
            elif any(k in zabaged_gdfs for k in ["SilniceDalnice", "Ulice", "SilniceVeVastavbe"]):
                if "SilniceDalnice" in zabaged_gdfs:
                    mask = ~get_col(zabaged_gdfs["SilniceDalnice"], "typsil_k").isin(["D1", "D2", "M"])
                    plot_masked(sym_key=sym, zorder=z, mask=mask, gdf=zabaged_gdfs["SilniceDalnice"], ax=ax)
                if "Ulice" in zabaged_gdfs:
                    mask = get_col(zabaged_gdfs["Ulice"], "typulice_k").isin(["026", "926"])
                    plot_masked(sym_key=sym, zorder=z, mask=mask, gdf=zabaged_gdfs["Ulice"], ax=ax)
                if "SilniceVeVastavbe" in zabaged_gdfs:
                    plot_masked(sym_key=sym, zorder=z, mask=None, gdf=zabaged_gdfs["SilniceVeVastavbe"], ax=ax, to_mask=False)
            else:
                plot_masked(sym_key=sym, zorder=z, mask=mask_road_major, gdf=gdf_lines, ax=ax)

        # 503 - Silnice
        cgdf = isom_gdfs.get("503")
        mask_road_minor = ((highway.isin(["tertiary_link", "service"])) |
                           ((highway.isin(["track", "road", "cycleway", "unclassified"])) &
                            ((surface.isin(["concrete", "asphalt"])) | (tracktype == "grade1")))) & \
                          (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) & \
                          (bridge != "yes") & (access != "private")
        if cgdf is not None:
            plot_masked(sym_key="sym503", zorder=45, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif any(k in zabaged_gdfs for k in ["SilniceNeevidovana", "Cesta", "LyzarskyMustek"]):
            if "SilniceNeevidovana" in zabaged_gdfs:
                plot_masked(sym_key="sym503", zorder=45, mask=None, gdf=zabaged_gdfs["SilniceNeevidovana"], ax=ax, to_mask=False)
            if "Cesta" in zabaged_gdfs:
                mask = (get_col(zabaged_gdfs["Cesta"], "povrch_p").isin(
                    ["zpevněný (panel, dlažba)", "zpevněný (asfalt, beton)"])) & \
                       (get_col(zabaged_gdfs["Cesta"], "typcesty_p").isin(["cesta udržovaná"]))
                plot_masked(sym_key="sym503", zorder=45, mask=mask, gdf=zabaged_gdfs["Cesta"], ax=ax)
            if "LyzarskyMustek" in zabaged_gdfs:
                plot_masked(sym_key="sym503", zorder=46, mask=None, gdf=zabaged_gdfs["LyzarskyMustek"], ax=ax, to_mask=False)
        else:
            plot_masked(sym_key="sym503", zorder=45, mask=mask_road_minor, gdf=gdf_lines, ax=ax)

        # 504 - Vozová cesta
        cgdf = isom_gdfs.get("504")
        mask_track_major = ((highway.isin(["cycleway", "unclassified"])) &
                            (~surface.isin(["concrete", "asphalt"])) & (tracktype != "grade1")) & \
                           (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) & \
                           (bridge != "yes") & (access != "private")
        if cgdf is not None:
            plot_masked(sym_key="sym504", zorder=45, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "Cesta" in zabaged_gdfs:
            mask = (get_col(zabaged_gdfs["Cesta"], "povrch_p").isin([
                "zpevněný (nosný terén, štěrk, kalený povrch)",
                "nedostatečně zpevněný (tráva, hlína, písek, kamení)", "neurčeno", "NULL"])) & \
                   (get_col(zabaged_gdfs["Cesta"], "typcesty_p").isin(["cesta udržovaná"]))
            plot_masked(sym_key="sym504", zorder=45, mask=mask, gdf=zabaged_gdfs["Cesta"], ax=ax)
        else:
            plot_masked(sym_key="sym504", zorder=45, mask=mask_track_major, gdf=gdf_lines, ax=ax)

        # 505 - Pěší cesta
        cgdf = isom_gdfs.get("505")
        mask_track_minor = ((highway.isin(["pedestrian", "road", "footway", "track", "bridleway"])) |
                            ((highway == "cycleway") & (~surface.isin(["concrete", "asphalt"])) &
                             (tracktype != "grade1"))) & \
                           (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) & \
                           (bridge != "yes") & (access != "private")
        if cgdf is not None:
            plot_masked(sym_key="sym505", zorder=45, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif any(k in zabaged_gdfs for k in ["Ulice", "Cesta"]):
            if "Ulice" in zabaged_gdfs:
                mask = ~get_col(zabaged_gdfs["Ulice"], "typulice_k").isin(["925", "025"])
                plot_masked(sym_key="sym505", zorder=45, mask=mask, gdf=zabaged_gdfs["Ulice"], ax=ax)
            if "Cesta" in zabaged_gdfs:
                mask = get_col(zabaged_gdfs["Cesta"], "typcesty_p").isin(["cesta neudržovaná"])
                plot_masked(sym_key="sym505", zorder=45, mask=mask, gdf=zabaged_gdfs["Cesta"], ax=ax)
        else:
            plot_masked(sym_key="sym505", zorder=45, mask=mask_track_minor, gdf=gdf_lines, ax=ax)

        # 506 - Pěšina
        cgdf = isom_gdfs.get("506")
        mask_path_major = (highway == "path") & \
                          (~trail_visibility.isin(["low", "poor", "bad", "very_bad", "horrible", "no"])) & \
                          (bridge != "yes") & (access != "private")
        if cgdf is not None:
            plot_masked(sym_key="sym506", zorder=45, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "Pesina" in zabaged_gdfs:
            plot_masked(sym_key="sym506", zorder=45, mask=None, gdf=zabaged_gdfs["Pesina"], ax=ax, to_mask=False)
        else:
            plot_masked(sym_key="sym506", zorder=45, mask=mask_path_major, gdf=gdf_lines, ax=ax)

        # 507 - Nezřetelná pěšina
        cgdf = isom_gdfs.get("507")
        if cgdf is not None:
            plot_masked(sym_key="sym507", zorder=45, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            mask_path_minor = (highway == "path") & \
                              (trail_visibility.isin(["low", "poor", "bad", "very_bad", "horrible"])) & \
                              (bridge != "yes") & (access != "private")
            plot_masked(sym_key="sym507", zorder=45, mask=mask_path_minor, gdf=gdf_lines, ax=ax)

        # 508 - Průsek
        cgdf = isom_gdfs.get("508")
        if cgdf is not None:
            plot_masked(sym_key="sym508", zorder=38, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "LesniPrusek" in zabaged_gdfs:
            plot_masked(sym_key="sym508", zorder=38, mask=None, gdf=zabaged_gdfs["LesniPrusek"], ax=ax, to_mask=False)
        else:
            plot_masked(sym_key="sym508", zorder=38, mask=(man_made == "cutline"), gdf=gdf_lines, ax=ax)

        # Mosty - dálnice
        mask_bridge_double = (highway.isin(["motorway", "trunk"])) & \
                             (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) & \
                             (bridge == "yes") & (access != "private")
        for sym, z in zip(["sym502DBa", "sym502DBb", "sym502Da", "sym502Db", "sym502Dc"],
                          [65, 66, 67, 68, 69]):
            plot_masked(sym_key=sym, zorder=z, mask=mask_bridge_double, gdf=gdf_lines, ax=ax)

        # Mosty - hlavní silnice
        mask_bridge_major = (highway.isin(["highway_link", "trunk_link", "primary", "primary_link",
                                            "secondary", "secondary_link", "residential", "tertiary",
                                            "living_street"])) & \
                            (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) & \
                            (bridge == "yes") & (access != "private")
        for sym, z in zip(["sym502Ba", "sym502Bb", "sym502a", "sym502b"], [65, 66, 67, 68]):
            plot_masked(sym_key=sym, zorder=z, mask=mask_bridge_major, gdf=gdf_lines, ax=ax)

        # Mosty - vedlejší silnice
        mask_bridge_minor = (highway.isin(["tertiary_link", "service", "track", "road", "unclassified"])) & \
                            (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) & \
                            (bridge == "yes") & (access != "private")
        for sym, z in zip(["sym503Ba", "sym503Bb", "sym503"], [65, 66, 67]):
            plot_masked(sym_key=sym, zorder=z, mask=mask_bridge_minor, gdf=gdf_lines, ax=ax)

        # Lávka
        mask_bridge_path = (highway.isin(["path", "cycleway", "footway", "bridleway"])) & \
                           (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) & \
                           (bridge == "yes") & (access != "private")
        if "Lavka" in zabaged_gdfs:
            plot_masked(sym_key="sym503", zorder=40, mask=None, gdf=zabaged_gdfs["Lavka"], ax=ax, to_mask=False)
        else:
            plot_masked(sym_key="sym503", zorder=67, mask=mask_bridge_path, gdf=gdf_lines, ax=ax)

        # Železnice
        cgdf = isom_gdfs.get("509")
        mask_railway = ((railway.isin(["rail", "disused", "funicular", "light-rail", "narrow_gauge"])) &
                        (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) &
                        (~bridge.isin(["yes"])))
        for sym, z in [("sym509a", 40), ("sym509b", 41)]:
            if cgdf is not None:
                plot_masked(sym_key=sym, zorder=z, mask=None, gdf=cgdf, ax=ax, to_mask=False)
            elif any(k in zabaged_gdfs for k in ["ZeleznicniTrat", "ZeleznicniVlecka"]):
                for zk in ["ZeleznicniTrat", "ZeleznicniVlecka"]:
                    if zk in zabaged_gdfs:
                        plot_masked(sym_key=sym, zorder=z, mask=None, gdf=zabaged_gdfs[zk], ax=ax, to_mask=False)
            else:
                plot_masked(sym_key=sym, zorder=z, mask=mask_railway, gdf=gdf_lines, ax=ax)

        # Železniční most
        mask_bridge_railway = (railway.isin(["rail", "disused", "funicular", "light-rail", "narrow_gauge"])) & \
                              (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) & \
                              (bridge == "yes")
        for sym, z in zip(["sym509Ba", "sym509Bb", "sym509a", "sym509b"], [60, 61, 62, 63]):
            plot_masked(sym_key=sym, zorder=z, mask=mask_bridge_railway, gdf=gdf_lines, ax=ax)

    # MAN MADE FEATURES
    if visibility.get("man_made", True):

        # 510 - El. vedení / lanovky
        cgdf = isom_gdfs.get("510")
        mask_cable_low = (power == "low") | (aerialway.isin([
            "line", "cable_car", "gondola", "mixed_lift", "chair_lift",
            "drag_lift", "t-bar", "j-bar", "platter", "rope_tow", "zip_line", "goods"
        ]))
        if cgdf is not None:
            plot_masked(sym_key="sym510", zorder=70, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "ElektrickeVedeni" in zabaged_gdfs:
            mask = ~get_col(zabaged_gdfs["ElektrickeVedeni"], "napeti").isin(["400", "110"])
            plot_masked(sym_key="sym510", zorder=70, mask=mask, gdf=zabaged_gdfs["ElektrickeVedeni"], ax=ax)
        elif "LanovaDrahaLyzarskyVlek" in zabaged_gdfs:
            plot_masked(sym_key="sym510", zorder=70, mask=None, gdf=zabaged_gdfs["LanovaDrahaLyzarskyVlek"], ax=ax, to_mask=False)
        else:
            plot_masked(sym_key="sym510", zorder=70, mask=mask_cable_low, gdf=gdf_lines, ax=ax)

        # 511 - VVN
        if "ElektrickeVedeni" in zabaged_gdfs:
            mask = get_col(zabaged_gdfs["ElektrickeVedeni"], "napeti").isin(["400", "110"])
            plot_masked(sym_key="sym510", zorder=70, mask=mask, gdf=zabaged_gdfs["ElektrickeVedeni"], ax=ax)
        else:
            plot_masked(sym_key="sym510", zorder=70, mask=(power.isin(["line", "minor_line"])), gdf=gdf_lines, ax=ax)

        # 513.1 - Zeď
        cgdf = isom_gdfs.get("513.1")
        if cgdf is not None:
            plot_masked(sym_key="sym513-1a", zorder=30, mask=None, gdf=cgdf, ax=ax, to_mask=False)
            plot_masked(sym_key="sym513-1b", zorder=30, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif any(k in zabaged_gdfs for k in ["Zed", "PrehradniHrazJez", "Hrad"]):
            for zk in ["Zed", "Hrad", "PrehradniHrazJez"]:
                if zk in zabaged_gdfs:
                    plot_masked(sym_key="sym513-1a", zorder=30, mask=None, gdf=zabaged_gdfs[zk], ax=ax, to_mask=False)
                    plot_masked(sym_key="sym513-1b", zorder=30, mask=None, gdf=zabaged_gdfs[zk], ax=ax, to_mask=False)
        else:
            plot_masked(sym_key="sym513-1a", zorder=30, mask=(barrier == "wall"), gdf=gdf_lines, ax=ax)
            plot_masked(sym_key="sym513-1b", zorder=30, mask=(barrier == "wall"), gdf=gdf_lines, ax=ax)

        # 515 - Nepřekonatelná zeď
        cgdf = isom_gdfs.get("515")
        mask_wall_high = (barrier.isin(["city_wall"]))
        if cgdf is not None:
            plot_masked(sym_key="sym515a", zorder=30, mask=None, gdf=cgdf, ax=ax, to_mask=False)
            plot_masked(sym_key="sym515b", zorder=30, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "Zed" in zabaged_gdfs:
            mask = get_col(zabaged_gdfs["Zed"], "typzed_p").isin(
                ["protihluková stěna", "zeď vodního díla", "zeď ostatní"])
            plot_masked(sym_key="sym515a", zorder=30, mask=mask, gdf=zabaged_gdfs["Zed"], ax=ax)
        elif "HradbaValBastaOpevneni" in zabaged_gdfs:
            plot_masked(sym_key="sym515b", zorder=30, mask=None, gdf=zabaged_gdfs["HradbaValBastaOpevneni"], ax=ax, to_mask=False)
        else:
            plot_masked(sym_key="sym515a", zorder=30, mask=mask_wall_high, gdf=gdf_lines, ax=ax)
            plot_masked(sym_key="sym515b", zorder=30, mask=mask_wall_high, gdf=gdf_lines, ax=ax)

        # 520 - Privátní oblast
        if visibility.get("private", True):
            cgdf = isom_gdfs.get("520")
            if cgdf is not None:
                plot_masked(sym_key="sym520", zorder=1.5, mask=None, gdf=cgdf, ax=ax, to_mask=False)
            else:
                garden_layers = ["Hrbitov", "Kolejiste", "Letiste", "OstatniPlochaVSidlech",
                                 "OvocnySadZahrada", "PovrchovaTezbaLom", "Elektrarna",
                                 "ArealUceloveZastavby", "Skladka"]
                found = False
                for gl in garden_layers:
                    if gl in zabaged_gdfs:
                        plot_masked(sym_key="sym520", zorder=1.5, mask=None, gdf=zabaged_gdfs[gl], ax=ax, to_mask=False)
                        found = True
                if not found:
                    mask_garden = (landuse.isin([
                        "residential", "allotments", "brownfield", "military", "commercial",
                        "construction", "industrial", "retail", "education", "animal_keeping",
                        "cemetery", "landfill", "quarry", "depot", "religious", "farmyard"
                    ])) | (leisure.isin(["pitch", "sports_centre"]))
                    plot_masked(sym_key="sym520", zorder=1.5, mask=mask_garden, gdf=gdf_polygons, ax=ax)

        # 523 - Zřícenina / Bunkr
        cgdf = isom_gdfs.get("523")
        if cgdf is not None:
            plot_masked(sym_key="sym523", zorder=35, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "RozvalinaZricenina" in zabaged_gdfs:
            plot_masked(sym_key="sym523", zorder=35, mask=None, gdf=zabaged_gdfs["RozvalinaZricenina"], ax=ax, to_mask=False)
        elif "Bunkr" in zabaged_gdfs:
            plot_masked(sym_key="sym523", zorder=56, mask=None, gdf=zabaged_gdfs["Bunkr"], ax=ax, to_mask=False)
        else:
            mask_ruin = (building == "ruins") | (historic == "ruins") | (military == "bunker")
            plot_masked(sym_key="sym523", zorder=35, mask=mask_ruin, gdf=gdf_polygons, ax=ax)

        # 524 - Vysoká věž
        cgdf = isom_gdfs.get("524")
        if cgdf is not None:
            plot_masked(sym_key="sym524a", zorder=56, mask=None, gdf=cgdf, ax=ax, to_mask=False)
            plot_masked(sym_key="sym524b", zorder=56, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            tower_layers = ["Silo", "TezniVez", "TovarniKomin", "VetrnyMotor", "VetrnyMlyn",
                            "VodojemVezovy", "VezovitaStavba"]
            found = False
            for tl in tower_layers:
                if tl in zabaged_gdfs:
                    plot_masked(sym_key="sym524a", zorder=56, mask=None, gdf=zabaged_gdfs[tl], ax=ax, to_mask=False)
                    plot_masked(sym_key="sym524b", zorder=56, mask=None, gdf=zabaged_gdfs[tl], ax=ax, to_mask=False)
                    found = True
            if not found:
                mask_tower_high = (man_made.isin([
                    "tower", "transformer_tower", "water_tower", "communications_tower",
                    "mast", "chimney", "crane", "flagpole", "obelisk", "column", "beacon", "lighthouse"
                ])) | (historic == "round_tower") | (building == "clock_tower")
                plot_masked(sym_key="sym524a", zorder=56, mask=mask_tower_high, gdf=gdf_points, ax=ax)
                plot_masked(sym_key="sym524b", zorder=56, mask=mask_tower_high, gdf=gdf_points, ax=ax)

        # 525 - Malá věž / Posed
        cgdf = isom_gdfs.get("525")
        if cgdf is not None:
            plot_masked(sym_key="sym525", zorder=56, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            mask_tower_low = amenity == "hunting_stand" 
            plot_masked(sym_key="sym525", zorder=56, mask=mask_tower_low, gdf=gdf_points, ax=ax)

        # 526 - Pomník
        cgdf = isom_gdfs.get("526")
        if cgdf is not None:
            plot_masked(sym_key="sym526a", zorder=56, mask=None, gdf=cgdf, ax=ax, to_mask=False)
            plot_masked(sym_key="sym526b", zorder=56, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "MohylaPomnikNahrobek" in zabaged_gdfs:
            plot_masked(sym_key="sym526a", zorder=56, mask=None, gdf=zabaged_gdfs["MohylaPomnikNahrobek"], ax=ax, to_mask=False)
            plot_masked(sym_key="sym526b", zorder=56, mask=None, gdf=zabaged_gdfs["MohylaPomnikNahrobek"], ax=ax, to_mask=False)
        else:
            mask_memorial = (historic.isin(["boundary_stone", "memorial", "wayside_cross"])) | \
                            ((man_made.isin(["cross", "survey_point", "obelisk"])) &
                             (~building.isin(["plaque"])))
            plot_masked(sym_key="sym526a", zorder=56, mask=mask_memorial, gdf=gdf_centroids, ax=ax)
            plot_masked(sym_key="sym526b", zorder=56, mask=mask_memorial, gdf=gdf_centroids, ax=ax)

        # 531 - Výrazný umělý objekt
        cgdf = isom_gdfs.get("531")
        if cgdf is not None:
            plot_masked(sym_key="sym531", zorder=56, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "KrizSloupKulturnihoVyznamu" in zabaged_gdfs:
            plot_masked(sym_key="sym531", zorder=56, mask=None, gdf=zabaged_gdfs["KrizSloupKulturnihoVyznamu"], ax=ax, to_mask=False)
        else:
            plot_masked(sym_key="sym531", zorder=56, mask=(man_made.isin(["insect_hotel", "street_cabinet"])), gdf=gdf_centroids, ax=ax)

        # 532 - Schody
        mask_stairs = (highway == "steps")
        for sym, z in zip(["sym532a", "sym532b", "sym532c"], [49, 50, 51]):
            plot_masked(sym_key=sym, zorder=z, mask=mask_stairs, gdf=gdf_lines, ax=ax)

    # BUILDINGS
    if visibility.get("buildings", True):
        # 521 - Budova
        cgdf = isom_gdfs.get("521")
        if cgdf is not None:
            plot_masked(sym_key="sym521", zorder=50, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "BudovaJednotlivaNeboBlokBudov" in zabaged_gdfs:
            plot_masked(sym_key="sym521", zorder=50, mask=None, gdf=zabaged_gdfs["BudovaJednotlivaNeboBlokBudov"], ax=ax, to_mask=False)
        else:
            mask_building = (building.notna()) & (building != '') & (~building.isin(["roof", "ruins"]))
            plot_masked(sym_key="sym521", zorder=50, mask=mask_building, gdf=gdf_polygons, ax=ax)

        # 522 - Zastřešení
        cgdf = isom_gdfs.get("522")
        if cgdf is not None:
            plot_masked(sym_key="sym522", zorder=36, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            plot_masked(sym_key="sym522", zorder=36, mask=(building == "roof"), gdf=gdf_polygons, ax=ax)

    print(" Vše vykresleno")


def add_custom_isom_layers(ax, isom_gdfs):
    print("Drawing other imported files...")
    already_handled = ["105", "215", "412", "413", "414", "417", "418",
                       "502D", "502", "509", "513", "515", "524", "526"]

    for filename, gdf in isom_gdfs.items():
        base_name = filename.split('.')[0]
        if base_name in already_handled:
            continue

        if base_name in SYMBOL_LIBRARY:
            sym_key = base_name
        elif f"sym{base_name}" in SYMBOL_LIBRARY:
            sym_key = f"sym{base_name}"
        else:
            print(f"  -> Warning: Symbol for '{filename}' was not found.")
            gdf.plot(ax=ax, color='red', linewidth=1, zorder=100)
            continue

        sym_def = SYMBOL_LIBRARY.get(sym_key, {})
        default_zorder = sym_def.get('props', {}).get('zorder', 50)
        print(f"  -> Kreslím '{filename}' jako '{sym_key}' (zorder ~{default_zorder})")
        plot_masked(sym_key=sym_key, zorder=default_zorder, mask=None, gdf=gdf, ax=ax, to_mask=False)


def select_file(entry_widget, title, file_types):
    path = filedialog.askopenfilename(title=title, filetypes=file_types)
    if path:
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, path)
        print(f"Vybrán soubor: {path}")


def select_multiple_files_to_listbox(listbox_widget):
    paths = filedialog.askopenfilenames(title="Vyberte SHP soubory", filetypes=SHP_FILES)
    if paths:
        for path in paths:
            listbox_widget.insert(tk.END, path)
            print(f"Přidán soubor: {path}")


def remove_selected_from_listbox(listbox_widget):
    selected_indices = listbox_widget.curselection()
    for index in reversed(selected_indices):
        print(f"Odebírám: {listbox_widget.get(index)}")
        listbox_widget.delete(index)


def setup_map_figure(extent_original_data, paper_format="A4 (Landscape)"):
    PAPER_SIZES_IN = {
        "A4 (Landscape)": (11.693, 8.268),
        "A4 (Portrait)": (8.268, 11.693),
        "A3 (Landscape)": (16.535, 11.693),
        "A3 (Portrait)": (11.693, 16.535)
    }

    minx_orig, maxx_orig, miny_orig, maxy_orig = extent_original_data

    if paper_format == "Data Extent":
        meters_per_inch = 0.0254 * SCALE
        FIG_WIDTH_IN = (maxx_orig - minx_orig) / meters_per_inch
        FIG_HEIGHT_IN = (maxy_orig - miny_orig) / meters_per_inch
        print(f"Custom Size (Data Extent): {FIG_WIDTH_IN:.2f}×{FIG_HEIGHT_IN:.2f} in")
        minx, maxx, miny, maxy = minx_orig, maxx_orig, miny_orig, maxy_orig
    else:
        FIG_WIDTH_IN, FIG_HEIGHT_IN = PAPER_SIZES_IN.get(paper_format, PAPER_SIZES_IN["A4 (Landscape)"])
        map_width_meters = in2m(FIG_WIDTH_IN, SCALE)
        map_height_meters = in2m(FIG_HEIGHT_IN, SCALE)
        print(f"Paper format {paper_format} (1:{SCALE}): {map_width_meters:.2f}×{map_height_meters:.2f} m")
        center_x = (minx_orig + maxx_orig) / 2
        center_y = (miny_orig + maxy_orig) / 2
        minx = center_x - map_width_meters / 2
        maxx = center_x + map_width_meters / 2
        miny = center_y - map_height_meters / 2
        maxy = center_y + map_height_meters / 2

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect('equal')
    ax.axis('off')

    return fig, ax, (minx, maxx, miny, maxy)


# ATOM Downloading
CUZK_ATOM_DMR5G  = "https://atom.cuzk.gov.cz/DMR5G-SJTSK/DMR5G-SJTSK.xml"
CUZK_ATOM_DMP1G  = "https://atom.cuzk.gov.cz/DMP1G-SJTSK/DMP1G-SJTSK.xml"
CUZK_ATOM_DMPOK  = "https://atom.cuzk.gov.cz/DMPOK-SJTSK-LAZ/DMPOK-SJTSK-LAZ.xml"
# ZABAGED WFS – seznam vrstev dostupných přes ČÚZK ATOM/WFS
# Klíč = název vrstvy (použije se jako název souboru i klíč v zabaged_gdfs)


def _parse_atom_feed_tiles(atom_url):
    tiles = []
    try:
        req = urllib.request.Request(atom_url, headers={"User-Agent": "OMapMaker/6"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            raw = resp.read()
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "georss": "http://www.georss.org/georss",
        }
        root_el = ET.fromstring(raw)
        for entry in root_el.findall("atom:entry", ns):
            id_el = entry.find("atom:id", ns)
            tile_id = id_el.text.strip() if id_el is not None else ""
            link_el = entry.find("atom:link[@rel='alternate']", ns)
            if link_el is None:
                link_el = entry.find("atom:link", ns)
            sub_url = link_el.get("href", "") if link_el is not None else ""
            poly_el = entry.find("georss:polygon", ns)
            bbox = None
            if poly_el is not None and poly_el.text:
                coords = list(map(float, poly_el.text.strip().split()))
                lats = coords[0::2]
                lons = coords[1::2]
                bbox = (min(lats), min(lons), max(lats), max(lons))
            else:
                box_el = entry.find("georss:box", ns)
                if box_el is not None and box_el.text:
                    p = list(map(float, box_el.text.strip().split()))
                    bbox = (p[0], p[1], p[2], p[3])
            if sub_url and bbox:
                tiles.append((tile_id, sub_url, bbox[0], bbox[1], bbox[2], bbox[3]))
    except Exception as e:
        print(f"Error while handling ATOM feedu: {e}")
    return tiles


def _get_download_url_from_subfeed(sub_feed_url):
    try:
        req = urllib.request.Request(sub_feed_url, headers={"User-Agent": "OMapMaker/6"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read()
        root_el = ET.fromstring(raw)
        for link_el in root_el.iter("{http://www.w3.org/2005/Atom}link"):
            href = link_el.get("href", "")
            if href.lower().endswith(".zip"):
                return href
    except Exception as e:
        print(f"Error sub-feedu ({sub_feed_url}): {e}")
    return None


def open_cuzk_downloader(parent, on_complete_callback):
    import tkintermapview

    dlg = tk.Toplevel(parent)
    dlg.title("ATOM download")
    dlg.geometry("960x720")
    dlg.resizable(True, True)
    dlg.grab_set()

    S = {
        "mode": "pan",          # "pan" nebo "select"
        "sel_wgs84": None,
        "map_polygon": None,
        "drag_x0": None, "drag_y0": None,
        "dragging": False,
        "atom_dmr": None, "atom_dmp": None, "atom_dmpok": None,
    }

    # Download Layout
    dlg.geometry("1000x600")
    body = ttk.Frame(dlg)
    body.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
    body.columnconfigure(0, weight=3)
    body.columnconfigure(1, weight=1)
    body.rowconfigure(0, weight=1)

    # Map
    left = ttk.Frame(body)
    left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
    left.rowconfigure(1, weight=1)
    left.columnconfigure(0, weight=1)

    # Toolbar
    top = ttk.Frame(left)
    top.grid(row=0, column=0, sticky="ew", pady=(0, 4))
    ttk.Label(top, text="Režim:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)

    mode_var = tk.StringVar(value="pan")

    def _set_mode(m):
        S["mode"] = m
        overlay.config(cursor="fleur" if m == "pan" else "crosshair")

    ttk.Radiobutton(top, text="✋ Pan",  variable=mode_var, value="pan",
                    command=lambda: _set_mode("pan")).pack(side=tk.LEFT, padx=(6, 2))
    ttk.Radiobutton(top, text="⬜ Select", variable=mode_var, value="select",
                    command=lambda: _set_mode("select")).pack(side=tk.LEFT, padx=(0, 8))
    ttk.Button(top, text="Cancel selection", command=lambda: _clear()).pack(side=tk.LEFT, padx=4)

    map_container = ttk.Frame(left)
    map_container.grid(row=1, column=0, sticky="nsew")

    map_widget = tkintermapview.TkinterMapView(map_container, width=800, height=560, corner_radius=0)
    map_widget.pack(fill=tk.BOTH, expand=True)
    map_widget.set_position(49.8, 15.5)
    map_widget.set_zoom(7)

    overlay = map_widget.canvas
    overlay.config(cursor="fleur")

    info_lbl = ttk.Label(left, text="Chose by dragging.",
                         foreground="gray", padding=(0, 3))
    info_lbl.grid(row=2, column=0, sticky="ew")

    # Right collumn
    right = ttk.Frame(body)
    right.grid(row=0, column=1, sticky="nsew")
    right.columnconfigure(0, weight=1)

    # Složka
    df = ttk.Labelframe(right, text="Output folder:", padding=6)
    df.grid(row=0, column=0, sticky="ew", pady=(0, 6))
    df.columnconfigure(0, weight=1)
    dir_entry = ttk.Entry(df)
    dir_entry.grid(row=0, column=0, sticky="ew")
    ttk.Button(df, text="...", width=3,
               command=lambda: _pick_dir()).grid(row=0, column=1, padx=(4, 0))

    # DSM 
    dmp_frame = ttk.Labelframe(right, text="DMP:", padding=6)
    dmp_frame.grid(row=1, column=0, sticky="ew", pady=(0, 6))
    dmp_var = tk.StringVar(value="DMP1G")
    ttk.Radiobutton(dmp_frame, text="DMP 1G",
                    variable=dmp_var, value="DMP1G").pack(anchor="w")
    ttk.Radiobutton(dmp_frame, text="DMP OK (recomended)",
                    variable=dmp_var, value="DMPOK").pack(anchor="w")
    ttk.Label(dmp_frame, text="DMP OK (Does not cover all Czehia)",
              foreground="gray", font=("Segoe UI", 8)).pack(anchor="w", pady=(2, 0))

    # Progress + Button
    af = ttk.Frame(right, padding=(0, 4))
    af.grid(row=2, column=0, sticky="ew")
    af.columnconfigure(0, weight=1)
    prog = ttk.Progressbar(af, orient="horizontal", mode="determinate")
    prog.grid(row=0, column=0, sticky="ew", pady=(0, 4))
    prog_lbl = ttk.Label(af, text="", font=("Segoe UI", 8))
    prog_lbl.grid(row=1, column=0, sticky="w")
    dl_btn = ttk.Button(af, text="Stahovat", command=lambda: _start_download(), state="disabled")
    dl_btn.grid(row=2, column=0, sticky="ew", pady=(8, 0), ipady=4)

    # Helpers
    def _pick_dir():
        d = filedialog.askdirectory(title="Output folder", parent=dlg)
        if d:
            dir_entry.delete(0, tk.END)
            dir_entry.insert(0, d)

    def _clear():
        overlay.delete("sel")
        if S["map_polygon"]:
            S["map_polygon"].delete()
        S.update(sel_wgs84=None, map_polygon=None,
                 dragging=False, drag_x0=None, drag_y0=None)
        dl_btn.config(state="disabled")
        info_lbl.config(text="Drag for chosing area.", foreground="gray")

    def _px_to_lonlat(cx, cy):
        """Převede pixel overlay canvasu na lon/lat (Mercator, přes střed a zoom mapy)."""
        w, h = overlay.winfo_width(), overlay.winfo_height()
        center_lat, center_lon = map_widget.get_position()
        deg_per_px  = 360.0 / (256 * (2 ** map_widget.zoom))
        lon = center_lon + (cx - w / 2) * deg_per_px
        lat = center_lat - (cy - h / 2) * deg_per_px * math.cos(math.radians(center_lat))
        return lon, lat

    # Drag events
    def _on_press(ev):
        if S["mode"] != "select":
            return         
        _clear()
        S["dragging"] = True
        S["drag_x0"] = ev.x
        S["drag_y0"] = ev.y
        return "break"

    def _on_motion(ev):
        if S["mode"] != "select" or not S["dragging"]:
            return
        x0, y0, x1, y1 = S["drag_x0"], S["drag_y0"], ev.x, ev.y
        overlay.delete("sel")
        overlay.create_rectangle(
            min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1),
            outline="#e63946", width=2, fill="", stipple="gray25", tags="sel")
        for rx, ry in [(min(x0,x1), min(y0,y1)), (max(x0,x1), min(y0,y1)),
                       (max(x0,x1), max(y0,y1)), (min(x0,x1), max(y0,y1))]:
            overlay.create_oval(rx-4, ry-4, rx+4, ry+4,
                                fill="#e63946", outline="white", width=2, tags="sel")
        return "break"      # blokuje pan handler tkintermapview

    def _on_release(ev):
        if not S["dragging"]:
            return
        S["dragging"] = False
        x0, y0, x1, y1 = S["drag_x0"], S["drag_y0"], ev.x, ev.y

        if abs(x1 - x0) < 8 or abs(y1 - y0) < 8:
            info_lbl.config(text="Výběr příliš malý – zkuste znovu.", foreground="orange")
            overlay.delete("sel")
            return

        lon0, lat0 = _px_to_lonlat(min(x0, x1), min(y0, y1))
        lon1, lat1 = _px_to_lonlat(max(x0, x1), max(y0, y1))
        mn_lat, mx_lat = min(lat0, lat1), max(lat0, lat1)
        mn_lon, mx_lon = min(lon0, lon1), max(lon0, lon1)
        S["sel_wgs84"] = (mn_lat, mn_lon, mx_lat, mx_lon)

        if S["map_polygon"]:
            S["map_polygon"].delete()
        S["map_polygon"] = map_widget.set_polygon(
            [(mx_lat, mn_lon), (mx_lat, mx_lon), (mn_lat, mx_lon), (mn_lat, mn_lon)],
            outline_color="red", fill_color=None, border_width=2)

        km_lat = (mx_lat - mn_lat) * 111
        km_lon = (mx_lon - mn_lon) * 111 * math.cos(math.radians((mn_lat + mx_lat) / 2))
        n_est  = max(1, round(km_lat / 2)) * max(1, round(km_lon / 2))
        info_lbl.config(
            text=(f"Oblast: ~{km_lat:.1f} × {km_lon:.1f} km  (~{n_est * 2} souborů LAZ)  |  "
                  f"{mn_lat:.4f}–{mx_lat:.4f} N,  {mn_lon:.4f}–{mx_lon:.4f} E"),
            foreground="#1a5276")
        dl_btn.config(state="normal")

    # --- Drag events ---
    # Strategie: uložíme původní tkintermapview binding stringy a přepínáme
    # mezi nimi a našimi podle aktivního režimu.

    def _on_press_select(ev):
        _clear()
        S["dragging"] = True
        S["drag_x0"] = ev.x
        S["drag_y0"] = ev.y
        return "break"

    def _on_motion_select(ev):
        if not S["dragging"]:
            return "break"
        x0, y0, x1, y1 = S["drag_x0"], S["drag_y0"], ev.x, ev.y
        overlay.delete("sel")
        overlay.create_rectangle(
            min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1),
            outline="#e63946", width=2, fill="", stipple="gray25", tags="sel")
        for rx, ry in [(min(x0,x1), min(y0,y1)), (max(x0,x1), min(y0,y1)),
                       (max(x0,x1), max(y0,y1)), (min(x0,x1), max(y0,y1))]:
            overlay.create_oval(rx-4, ry-4, rx+4, ry+4,
                                fill="#e63946", outline="white", width=2, tags="sel")
        return "break"

    def _on_release_select(ev):
        if not S["dragging"]:
            return "break"
        S["dragging"] = False
        x0, y0, x1, y1 = S["drag_x0"], S["drag_y0"], ev.x, ev.y
        if abs(x1 - x0) < 8 or abs(y1 - y0) < 8:
            info_lbl.config(text="Výběr příliš malý – zkuste znovu.", foreground="orange")
            overlay.delete("sel")
            return "break"
        lon0, lat0 = _px_to_lonlat(min(x0, x1), min(y0, y1))
        lon1, lat1 = _px_to_lonlat(max(x0, x1), max(y0, y1))
        mn_lat, mx_lat = min(lat0, lat1), max(lat0, lat1)
        mn_lon, mx_lon = min(lon0, lon1), max(lon0, lon1)
        S["sel_wgs84"] = (mn_lat, mn_lon, mx_lat, mx_lon)
        if S["map_polygon"]:
            S["map_polygon"].delete()
        S["map_polygon"] = map_widget.set_polygon(
            [(mx_lat, mn_lon), (mx_lat, mx_lon), (mn_lat, mx_lon), (mn_lat, mn_lon)],
            outline_color="red", fill_color=None, border_width=2)
        km_lat = (mx_lat - mn_lat) * 111
        km_lon = (mx_lon - mn_lon) * 111 * math.cos(math.radians((mn_lat + mx_lat) / 2))
        n_est  = max(1, round(km_lat / 2)) * max(1, round(km_lon / 2))
        info_lbl.config(
            text=(f"Oblast: ~{km_lat:.1f} × {km_lon:.1f} km  (~{n_est * 2} souborů LAZ)  |  "
                  f"{mn_lat:.4f}–{mx_lat:.4f} N,  {mn_lon:.4f}–{mx_lon:.4f} E"),
            foreground="#1a5276")
        dl_btn.config(state="normal")
        return "break"

    # Uložíme původní tkintermapview handlery (jsou to tcl skripty jako stringy)
    _orig = {
        "<ButtonPress-1>":   overlay.bind("<ButtonPress-1>"),
        "<B1-Motion>":       overlay.bind("<B1-Motion>"),
        "<ButtonRelease-1>": overlay.bind("<ButtonRelease-1>"),
    }

    def _set_mode(m):
        S["mode"] = m
        if m == "select":
            overlay.bind("<ButtonPress-1>",   _on_press_select)
            overlay.bind("<B1-Motion>",       _on_motion_select)
            overlay.bind("<ButtonRelease-1>", _on_release_select)
            overlay.config(cursor="crosshair")
        else:
            # Obnovíme původní tkintermapview handlery
            for seq, script in _orig.items():
                if script:
                    overlay.bind(seq, script)
                else:
                    overlay.unbind(seq)
            overlay.config(cursor="fleur")

    mode_var.trace_add("write", lambda *_: _set_mode(mode_var.get()))

    # Scroll → zoom mapy
    def _on_scroll(ev):
        delta = 1 if (ev.delta > 0 or ev.num == 4) else -1
        map_widget.set_zoom(max(1, min(19, map_widget.zoom + delta)))
    overlay.bind("<MouseWheel>", _on_scroll)
    overlay.bind("<Button-4>",   _on_scroll)
    overlay.bind("<Button-5>",   _on_scroll)

    # --- Download helpers ---
    def _set_prog(v, txt):
        dlg.after(0, lambda: prog.config(value=v))
        dlg.after(0, lambda: prog_lbl.config(text=txt))

    def _start_download():
        if not S["sel_wgs84"]:
            return
        out = dir_entry.get().strip()
        if not out:
            messagebox.showwarning("Folder", "Select folder.", parent=dlg)
            return
        os.makedirs(out, exist_ok=True)
        dl_btn.config(state="disabled")
        threading.Thread(target=_do_download,
                         args=(S["sel_wgs84"], out, dmp_var.get() == "DMPOK"),
                         daemon=True).start()

    def _merge_laz_files(input_paths, output_path, label, clip_bbox_wgs84=None):
        clip_bounds_native = None
        if clip_bbox_wgs84 is not None:
            try:
                mn_lat, mn_lon, mx_lat, mx_lon = clip_bbox_wgs84
                with laspy.open(input_paths[0]) as fh_tmp:
                    try:
                        src_crs = fh_tmp.header.parse_crs()
                        if src_crs is None:
                            raise ValueError("No CRS")
                    except Exception:
                        src_crs = CRS.from_epsg(5514)
                wgs84 = CRS.from_epsg(4326)
                if src_crs != wgs84:
                    t = Transformer.from_crs(wgs84, src_crs, always_xy=True)
                    cx, cy = t.transform([mn_lon, mx_lon, mn_lon, mx_lon],
                                         [mn_lat, mn_lat, mx_lat, mx_lat])
                    clip_bounds_native = (min(cx), max(cx), min(cy), max(cy))
                else:
                    clip_bounds_native = (mn_lon, mx_lon, mn_lat, mx_lat)
                print(f"  Clip bbox: X[{clip_bounds_native[0]:.1f}, {clip_bounds_native[1]:.1f}]"
                      f"  Y[{clip_bounds_native[2]:.1f}, {clip_bounds_native[3]:.1f}]")
            except Exception as e:
                print(f"  Warning: clip bbox failed ({e}), merging without clip.")

        if len(input_paths) == 1 and clip_bounds_native is None:
            import shutil
            shutil.copy2(input_paths[0], output_path)
            return True

        try:
            print(f"  Merging {len(input_paths)} files -> {os.path.basename(output_path)}")
            all_x, all_y, all_z, all_cls = [], [], [], []
            header_ref = None
            for path in input_paths:
                with laspy.open(path) as fh:
                    if header_ref is None:
                        header_ref = fh.header
                    for chunk in fh.chunk_iterator(500_000):
                        cx = np.array(chunk.x); cy = np.array(chunk.y)
                        cz = np.array(chunk.z); cc = np.array(chunk.classification)
                        if clip_bounds_native is not None:
                            bx0, bx1, by0, by1 = clip_bounds_native
                            m = (cx >= bx0) & (cx <= bx1) & (cy >= by0) & (cy <= by1)
                            cx, cy, cz, cc = cx[m], cy[m], cz[m], cc[m]
                        if len(cx):
                            all_x.append(cx); all_y.append(cy)
                            all_z.append(cz); all_cls.append(cc)
            if not all_x:
                print(f"  WARNING: no points remain after clipping for {label}!")
                return False
            x = np.concatenate(all_x); y = np.concatenate(all_y)
            z = np.concatenate(all_z); cls = np.concatenate(all_cls)
            las_out = laspy.LasData(header=laspy.LasHeader(
                point_format=header_ref.point_format, version=header_ref.version))
            las_out.header.offsets = np.array([x.min(), y.min(), z.min()])
            las_out.header.scales  = np.array([0.01, 0.01, 0.01])
            try:
                if header_ref.parse_crs() is not None:
                    las_out.header.set_crs(header_ref.parse_crs())
            except Exception:
                pass
            las_out.x = x; las_out.y = y; las_out.z = z
            las_out.classification = cls
            las_out.write(output_path)
            print(f"  Merged {len(x):,} points -> {os.path.basename(output_path)}")
            return True
        except Exception as e:
            print(f"  ERROR merging {label}: {e}")
            return False

    def _do_download(bbox, out_dir, use_dmpok=False):
        mn_lat, mn_lon, mx_lat, mx_lon = bbox
        dmp_label       = "DMP OK" if use_dmpok else "DMP 1G"
        dmp_atom_url    = CUZK_ATOM_DMPOK if use_dmpok else CUZK_ATOM_DMP1G
        dmp_atom_key    = "atom_dmpok" if use_dmpok else "atom_dmp"
        dmp_merged_name = "DMPOK_merged.laz" if use_dmpok else "DMP1G_merged.laz"

        _set_prog(2, "Loading ATOM DMR 5G...")
        if S["atom_dmr"] is None:
            S["atom_dmr"] = _parse_atom_feed_tiles(CUZK_ATOM_DMR5G)
        _set_prog(5, f"Loading ATOM {dmp_label}...")
        if S.get(dmp_atom_key) is None:
            S[dmp_atom_key] = _parse_atom_feed_tiles(dmp_atom_url)

        def _overlap(tiles):
            return [(tid, su) for (tid, su, tla, tlo, txa, txo) in tiles
                    if txa >= mn_lat and tla <= mx_lat and txo >= mn_lon and tlo <= mx_lon]

        dmr_t = _overlap(S["atom_dmr"])
        dmp_t = _overlap(S[dmp_atom_key])

        if not dmr_t:
            dlg.after(0, lambda: messagebox.showwarning(
                "Žádná data", "No DTM tiles found.\nIs this area really in Czehia?.",
                parent=dlg))
            dlg.after(0, lambda: dl_btn.config(state="normal"))
            return

        if use_dmpok and not dmp_t:
            answer = [None]
            ev = threading.Event()
            def _ask():
                answer[0] = messagebox.askyesno(
                    "No DMP OK tiles found",
                    "DMP OK does not cover all Czechia.\n\nDo you want to download DMP 1G instead?",
                    parent=dlg)
                ev.set()
            dlg.after(0, _ask)
            ev.wait()
            if answer[0]:
                dmp_label       = "DMP 1G (fallback)"
                dmp_merged_name = "DMP1G_merged.laz"
                _set_prog(6, "Loading ATOM DMP 1G (fallback)...")
                if S.get("atom_dmp") is None:
                    S["atom_dmp"] = _parse_atom_feed_tiles(CUZK_ATOM_DMP1G)
                dmp_t = _overlap(S["atom_dmp"])
            else:
                dlg.after(0, lambda: dl_btn.config(state="normal"))
                return

        total = len(dmr_t) + len(dmp_t)
        done  = [0]
        dmr_files, dmp_files = [], []
        dmr_raw_dir = os.path.join(out_dir, "dmr_tiles")
        dmp_raw_dir = os.path.join(out_dir, "dmp_tiles")
        os.makedirs(dmr_raw_dir, exist_ok=True)
        os.makedirs(dmp_raw_dir, exist_ok=True)

        def _dl_tile(tid, sub_url, dest, flist):
            zip_url = _get_download_url_from_subfeed(sub_url)
            if not zip_url:
                print(f"  Nelze najít ZIP pro {tid}")
                return
            name  = tid.split("_")[-1] if "_" in tid else tid.replace("/", "_")
            zpath = os.path.join(dest, f"{name}.zip")
            try:
                req = urllib.request.Request(zip_url, headers={"User-Agent": "OMapMaker/6"})
                with urllib.request.urlopen(req, timeout=120) as r, open(zpath, "wb") as f:
                    f.write(r.read())
                with zipfile.ZipFile(zpath) as zf:
                    for n2 in zf.namelist():
                        if n2.lower().endswith((".laz", ".las")):
                            zf.extract(n2, dest)
                            flist.append(os.path.join(dest, n2))
                os.remove(zpath)
                print(f"  OK: {name}")
            except Exception as e:
                print(f"  Error {name}: {e}")

        _set_prog(8, f"Downloading {len(dmr_t)} DMR 5G tiles...")
        for tid, su in dmr_t:
            _set_prog(8 + int(35 * done[0] / max(1, total)),
                      f"DMR {done[0]+1}/{len(dmr_t)}: downloading...")
            _dl_tile(tid, su, dmr_raw_dir, dmr_files)
            done[0] += 1

        _set_prog(45, f"Downloading {len(dmp_t)} tiles {dmp_label}...")
        for tid, su in dmp_t:
            pct = done[0] - len(dmr_t)
            _set_prog(45 + int(35 * pct / max(1, len(dmp_t))),
                      f"{dmp_label} {pct+1}/{len(dmp_t)}: stahuji...")
            _dl_tile(tid, su, dmp_raw_dir, dmp_files)
            done[0] += 1

        if not dmr_files:
            dlg.after(0, lambda: messagebox.showerror(
                "Error", "No DMR tiles downloaded succesfully.", parent=dlg))
            dlg.after(0, lambda: dl_btn.config(state="normal"))
            return

        _set_prog(82, f"Merging {len(dmr_files)} DMR tiles...")
        dmr_merged = os.path.join(out_dir, "DMR5G_merged.laz")
        ok_dmr = _merge_laz_files(dmr_files, dmr_merged, "DMR", clip_bbox_wgs84=bbox)

        dmp_merged = None
        ok_dmp = False
        if dmp_files:
            _set_prog(91, f"Merging {len(dmp_files)} tiles {dmp_label}...")
            dmp_merged = os.path.join(out_dir, dmp_merged_name)
            ok_dmp = _merge_laz_files(dmp_files, dmp_merged, dmp_label, clip_bbox_wgs84=bbox)


        _set_prog(100, "Hotovo!")

        def _finish():
            if not ok_dmr:
                messagebox.showerror("Error while merging tiles",
                                     " Merging tiles failed. Try smaller area.",
                                     parent=dlg)
                dl_btn.config(state="normal")
                return
            if not ok_dmp or dmp_merged is None:
                messagebox.showwarning(f"Error {dmp_label}",
                                       f"DMR downloaded: {dmr_merged}\n\n"
                                       f"{dmp_label} could not be downloaded or merged.\n"
                                       "Try insert DSM other way.", parent=dlg)
                on_complete_callback(dmr_merged, "")
            else:
                on_complete_callback(dmr_merged, dmp_merged)
            dlg.destroy()

        dlg.after(0, _finish)


def open_cuzk_downloader_from_gui():
    def _cb(dmr, dmp):
        dmr_entry.delete(0, tk.END)
        dmr_entry.insert(0, dmr)
        dmp_entry.delete(0, tk.END)
        dmp_entry.insert(0, dmp)
        status_label.config(text="Data from ČÚZK inserted.", foreground="darkgreen")
    open_cuzk_downloader(root, _cb)


# =============================================================================
# HLAVNÍ ANALÝZA
# =============================================================================

def run_main_analysis():
    global SCALE, SYMBOL_LIBRARY, CURRENT_CRS, OOM_EXPORT_LAYERS
    OOM_EXPORT_LAYERS = {}
    start_time = time.time()
    selected_crs_label = crs_var.get()
    CURRENT_CRS = CRS_MAP.get(selected_crs_label, selected_crs_label) or "EPSG:5514"
    print(f"--- STARTING ANALYSIS ---")
    print(f"CRS: {CURRENT_CRS}")

    selected_scale = scale_var.get()
    if selected_scale == "1:10 000":
        SCALE = 10000
        xml_file = "symbols10.xml"
    else:
        SCALE = 15000
        xml_file = "symbols15.xml"

    print(f"Scale 1:{SCALE}, symbols: {xml_file}")
    SYMBOL_LIBRARY = load_symbol_library(xml_file)
    if not SYMBOL_LIBRARY:
        messagebox.showerror("Error", f"file {xml_file} could not be loaded")

    dmr_path = dmr_entry.get()
    dmp_path = dmp_entry.get()
    should_save_png = save_var.get()
    selected_paper_format = paper_format_var.get()
    zabaged_paths = zabaged_listbox.get(0, tk.END)
    isom_paths = isom_listbox.get(0, tk.END)
    FIXED_PIXEL_SIZE = 0.5

    if not dmr_path or not dmp_path:
        messagebox.showerror("Input parameters missing:\n\n- DTM\n- DSM")
        status_label.config(text="Error: Input parameters missing.", foreground="red")
        return

    try:
        progress_bar["value"] = 0

        try:
            b1 = float(bin_entry_1.get())
            b2 = float(bin_entry_2.get())
            b3 = float(bin_entry_3.get())
            b4 = float(bin_entry_4.get())
            bins = [-1, 0, b1, b2, b3, b4]
            print(f"Vegetation classes: {bins}")
        except ValueError:
            messagebox.showwarning("Error on input", "Invalid classification data. Using default (0.5, 2, 5, 11).")
            b1, b2, b3, b4 = 0.5, 2.0, 5.0, 11.0
            bins = [-1, 0, b1, b2, b3, b4]

        try:
            dmr_smooth = float(contour_smooth.get().replace(",", "."))
        except ValueError:
            dmr_smooth = 6.5

        dmr_grid_cubic, grid_x, grid_y, extent, dmr_points, dmr_z = load_dmr_grid(
            dmr_path, target_crs_code=CURRENT_CRS, pixel_size=FIXED_PIXEL_SIZE, sigma_smooth=dmr_smooth)
        (minx, maxx, miny, maxy) = extent

        status_label.config(text="Creating clip mask...")
        root.update_idletasks()
        try:
            clip_polygon = MultiPoint(dmr_points).convex_hull
            if not clip_polygon.is_valid:
                clip_polygon = clip_polygon.buffer(0)
            print("  -> Clip mask created.")
        except Exception as e:
            print(f"  -> Clip mask error: {e}")
            clip_polygon = None

        print("Interpolating DTM...")
        status_label.config(text="Interpolating DTM...")
        root.update_idletasks()

        shift_x = np.mean(dmr_points[:, 0])
        shift_y = np.mean(dmr_points[:, 1])
        pts_shifted = dmr_points - np.array([shift_x, shift_y])
        gx_shifted = grid_x - shift_x
        gy_shifted = grid_y - shift_y

        dmr_grid_linear = griddata(pts_shifted, dmr_z, (gx_shifted, gy_shifted), method='linear')
        if np.isnan(dmr_grid_linear).all():
            print("Fallback DMR-Linear -> nearest.")
            dmr_grid_linear = griddata(pts_shifted, dmr_z, (gx_shifted, gy_shifted), method='nearest')

        progress_bar["value"] = 10

        dmp_grid = load_dmp_grid(dmp_path, grid_x, grid_y, extent, target_crs_code=CURRENT_CRS)
        progress_bar["value"] = 20

        status_label.config(text="Calculating vegetation height...")
        root.update_idletasks()
        vegetation_height = np.clip(dmp_grid - dmr_grid_linear, 0, None)
        progress_bar["value"] = 25

        # Other data inputs
        isom_gdfs = {}
        if isom_paths:
            print(f"Loading {len(isom_paths)} other layers...")
            status_label.config(text="Loading other layers...")
            root.update_idletasks()
            for path in isom_paths:
                filename = os.path.basename(path)
                try:
                    isom_gdf = gpd.read_file(path)
                    if not isom_gdf.empty:
                        if isom_gdf.crs is None:
                            print(f"'{filename}' no CRS defined. Expecting {CURRENT_CRS}.")
                            isom_gdf.set_crs(CURRENT_CRS, allow_override=True, inplace=True)
                        elif isom_gdf.crs != CURRENT_CRS:
                            print(f"  -> Transforming '{filename}' to {CURRENT_CRS}...")
                            try:
                                isom_gdf = isom_gdf.to_crs(CURRENT_CRS)
                            except Exception as e:
                                print(f" Error while transforming CRS: {filename}: {e}")
                                continue
                        if clip_polygon is not None:
                            try:
                                isom_gdf['geometry'] = isom_gdf.geometry.buffer(0)
                                isom_gdf = gpd.clip(isom_gdf, clip_polygon)
                            except Exception as e:
                                print(f"  -> Clipping {filename} failed ({e}), using whole extent.")
                        if not isom_gdf.empty:
                            key_name = filename.rsplit('.', 1)[0]
                            isom_gdfs[key_name] = isom_gdf
                            isom_gdfs[filename] = isom_gdf
                            print(f"  -> Loaded: {filename} ({len(isom_gdf)} items)")
                        else:
                            print(f"  -> '{filename}' is empty.")
                except Exception as e:
                    print(f" Error whiile loading {filename}: {e}")

        # OSM data
        print("Downloading OSM data...")
        status_label.config(text="Downloading OSM data...")
        try:
            ox.settings.log_console = True
            ox.settings.use_cache = True
            ox.settings.user_agent = "OMapMaker-App-v7"
            ox.settings.timeout = 300
        except Exception as e:
            print(f"Varování nastavení osmnx: {e}")

        root.update_idletasks()
        gdf_osm = None
        try:
            download_buffer = 300
            to_wgs = Transformer.from_crs(CURRENT_CRS, "EPSG:4326", always_xy=True)
            minlon, minlat = to_wgs.transform(minx - download_buffer, miny - download_buffer)
            maxlon, maxlat = to_wgs.transform(maxx + download_buffer, maxy + download_buffer)
            tags = {
                "highway": True, "building": True, "waterway": True, "access": True,
                "aerialway": True, "amenity": True, "barrier": True, "bridge": True,
                "covered": True, "emergency": True, "geological": True, "historic": True,
                "intermittent": True, "landuse": True, "leisure": True, "man_made": True,
                "military": True, "natural": True, "parking": True, "place": True,
                "power": True, "railway": True, "tunnel": True, "tracktype": True,
                "trail_visibility": True, "surface": True, "water": True, "wetland": True
            }
            gdf_osm = ox.features_from_bbox((minlon, minlat, maxlon, maxlat), tags=tags)
            gdf_osm = gdf_osm.to_crs(CURRENT_CRS)
            print(f"OSM data downloaded")
        except Exception as e:
            print(f"Error OSM: {e}")
            status_label.config(text="Download error.", foreground="red")

        if gdf_osm is not None and not gdf_osm.empty and clip_polygon is not None:
            try:
                gdf_osm = gpd.clip(gdf_osm, clip_polygon)
            except Exception as e:
                print(f"  -> Warning: Clip failed gdf_osm: {e}")

        # ZABAGED
        zabaged_gdfs = {}
        if zabaged_paths:
            print(f"Loading {len(zabaged_paths)} ZABAGED files...")
            status_label.config(text="Loading ZABAGED data...")
            root.update_idletasks()
            target_bbox = box(minx, miny, maxx, maxy)

            for path in zabaged_paths:
                filename = os.path.basename(path)
                status_label.config(text=f"Loading {filename}...")
                root.update_idletasks()
                try:
                    with fiona.open(path) as src:
                        file_crs_wkt = src.crs_wkt
                        file_crs = file_crs_wkt if file_crs_wkt else "EPSG:5514"

                    file_bbox_tuple = None
                    try:
                        crs_src = CRS.from_user_input(file_crs)
                        crs_dst = CRS.from_user_input(CURRENT_CRS)
                        if crs_src != crs_dst:
                            transformer_bbox = Transformer.from_crs(crs_dst, crs_src, always_xy=True)
                            b_minx, b_miny, b_maxx, b_maxy = target_bbox.bounds
                            tx, ty = transformer_bbox.transform([b_minx, b_maxx], [b_miny, b_maxy])
                            file_bbox_tuple = (min(tx), min(ty), max(tx), max(ty))
                        else:
                            file_bbox_tuple = target_bbox.bounds
                    except Exception as e:
                        print(f"  -> Error while transforming bbox: {e}, loading all.")

                    zabaged_gdf = gpd.read_file(path, bbox=file_bbox_tuple) if file_bbox_tuple \
                        else gpd.read_file(path)

                    if not zabaged_gdf.empty:
                        if zabaged_gdf.crs != CURRENT_CRS:
                            zabaged_gdf = zabaged_gdf.to_crs(CURRENT_CRS)
                        if clip_polygon:
                            zabaged_gdf = gpd.clip(zabaged_gdf, clip_polygon)

                    zabaged_gdfs[filename.rsplit(".", 1)[0]] = zabaged_gdf
                except Exception as e:
                    print(f" Error while loading {filename}: {e}")
                root.update_idletasks()

        shape = grid_x.shape
        transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy,
                                                   width=shape[0], height=shape[1])

        # Lesní maska
        forest_mask = np.zeros(shape, dtype=np.uint8)
        if gdf_osm is not None and not gdf_osm.empty:
            forest_polys = gdf_osm[
                (get_col(gdf_osm, 'natural') == 'wood') |
                (get_col(gdf_osm, 'landuse') == 'forest')
            ].geometry
            if not forest_polys.empty:
                print(f"Found {len(forest_polys)} forest polygons ...")
                forest_mask_transposed = rasterize(
                    forest_polys, out_shape=(shape[1], shape[0]),
                    transform=transform, fill=0, default_value=1, dtype=np.uint8
                )
                forest_mask = np.flipud(forest_mask_transposed).T
                print("Forest mask created.")

        status_label.config(text="Finding clearings...")
        root.update_idletasks()
        is_clearing = ((vegetation_height < b1) & (vegetation_height >= 0)) & (forest_mask == 1)
        print(f" {np.sum(is_clearing)} pixels of clearings found.")
        vegetation_height[is_clearing] = -1

        progress_bar["value"] = 45

        class_names = {
            2: 'Louka', 1: 'Paseka', 6: 'Les', 5: 'Vysoky_porost',
            4: 'Stredni_porost', 3: 'Nizky_porost', 0: 'Mimo_data'
        }
        color_map = {
            'Paseka': '#ffdd9a', 'Louka': '#ffba35',
            'Vysoky_porost': '#c3ed9a', 'Stredni_porost': '#4cc74c',
            'Nizky_porost': '#0e990e', 'Les': '#ffffff'
        }

        classified_raster = np.digitize(
            np.nan_to_num(vegetation_height, nan=-9999), bins
        ).astype(np.int32)

        # Ořezová maska pro gridy
        if clip_polygon:
            root.update_idletasks()
            try:
                clip_mask_transposed = rasterize(
                    [(clip_polygon, 1)], out_shape=(shape[1], shape[0]),
                    transform=transform, fill=0, default_value=1, dtype=np.uint8
                )
                clip_mask_grid = np.flipud(clip_mask_transposed).T.astype(bool)
                classified_raster[~clip_mask_grid] = 0
                dmr_grid_linear_viz_features = np.nan_to_num(dmr_grid_linear, nan=0)
                dmr_grid_linear_viz_features[~clip_mask_grid] = 0
                dmr_grid_cubic_viz = np.nan_to_num(dmr_grid_cubic, nan=0)
                dmr_grid_cubic_viz[~clip_mask_grid] = np.nan
                dmr_grid_for_contours = np.nan_to_num(dmr_grid_cubic, nan=0)
            except Exception as e:
                print(f"  -> Error maskování gridů: {e}")
                dmr_grid_linear_viz_features = np.nan_to_num(dmr_grid_linear, nan=0)
                dmr_grid_cubic_viz = np.nan_to_num(dmr_grid_cubic, nan=0)
        else:
            dmr_grid_linear_viz_features = np.nan_to_num(dmr_grid_linear, nan=0)
            dmr_grid_cubic_viz = np.nan_to_num(dmr_grid_cubic, nan=0)

        gdf_vegetation = vectorize_vegetation(classified_raster, class_names, transform, dmr_path)
        progress_bar["value"] = 60

        try:
            rock_slope_deg = float(slope_threshold_entry.get().replace(",", "."))
        except ValueError:
            rock_slope_deg = 54

        gdf_rocks = vectorize_rocks(grid_x, grid_y, dmr_grid_linear_viz_features,
                                    transform, slope_threshold_deg=rock_slope_deg)
        progress_bar["value"] = 70

        if not should_save_png:
            print("Creating PNG skipped.")
            status_label.config(text=" Vectorizatione done.", foreground="darkgreen")
            progress_bar["value"] = 100
            root.after(2000, lambda: status_label.config(text="Done.", foreground="darkgreen"))
            return

        # Generování mapy
        status_label.config(text="Generating map composition...")
        root.update_idletasks()

        fig, ax, map_extent = setup_map_figure(extent, selected_paper_format)
        (map_minx, map_maxx, map_miny, map_maxy) = map_extent
        clip_box_map = box(map_minx, map_miny, map_maxx, map_maxy)

        if LAYER_VISIBILITY["magnetic_lines"].get():
            try:
                north_rotation = float(north_rotation_entry.get().replace(",", "."))
            except ValueError:
                north_rotation = 0.0
            add_magnetic_north_lines(ax, map_extent, SCALE, rotation=north_rotation,
                                     spacing_mm=30, zorder=50)

        if clip_polygon:
            try:
                clip_patch = MplPolygon(np.array(clip_polygon.exterior.coords),
                                        transform=ax.transData)
                ax.set_clip_path(clip_patch)
            except Exception as e:
                print(f"  -> Error while aplicating clip mask: {e}")

        # Vegetace
        if LAYER_VISIBILITY["vegetation"].get():
            status_label.config(text="Drawing vegetation...")
            root.update_idletasks()
            
            if not gdf_vegetation.empty:
                try:
                    # 1. Ořez vegetace na rozsah mapy
                    veg_clipped = gpd.clip(gdf_vegetation, clip_box_map)
                    
                    if not veg_clipped.empty:
                        # 2. Definice mapování tříd z rastru na ISOM symboly v XML
                        VEG_ISOM_MAP = {
                            'Paseka': 'sym403',         
                            'Louka': 'sym401',         
                            'Les': 'sym405',   
                            'Vysoky_porost': 'sym406',
                            'Stredni_porost': 'sym408',
                            'Nizky_porost': 'sym410'
                        }

                        # 3. Pořadí vykreslování (zorder) - od podkladu po detaily
                        plot_order = [
                            ('Paseka', 1.0),
                            ('Louka', 1.1),
                            ('Les', 1.2),
                            ('Vysoky_porost', 1.3),
                            ('Stredni_porost', 1.4),
                            ('Nizky_porost', 1.5)
                        ]

                        for class_name, base_zorder in plot_order:
                            if class_name in veg_clipped['class_name'].values:
                                mask = (veg_clipped['class_name'] == class_name)
                                sym_key = VEG_ISOM_MAP.get(class_name)
                                
                                if sym_key:
                                    plot_masked(
                                        sym_key=sym_key,
                                        zorder=base_zorder,
                                        mask=mask,
                                        gdf=veg_clipped,
                                        ax=ax,
                                        to_mask=True,
                                        dmr_grid=None,
                                        grid_x=grid_x,
                                        grid_y=grid_y
                                    )
                                    
                        print(" Vegetation done")
                        
                except Exception as e:
                    print(f"  -> Error while drawing vwgwtation: {e}")

        # Skály + vrstevnice
        if LAYER_VISIBILITY["rocks"].get():
            if not gdf_rocks.empty:
                try:
                    rocks_clipped = gpd.clip(gdf_rocks, clip_box_map)
                    if not rocks_clipped.empty:
                        rocks_clipped.plot(ax=ax, color='black', zorder=26)
                        oom_collect('sym201', rocks_clipped)
                except Exception as e:
                    print(f"  -> Error while drawing cliffs: {e}")

            status_label.config(text="Drawing contours...")
            root.update_idletasks()
            add_contour_lines(ax, grid_x, grid_y, dmr_grid_cubic_viz, clip_polygon=clip_polygon)

        # Terénní detaily
        if LAYER_VISIBILITY["contours"].get():
            try:
                knoll_h = float(knoll_height_entry.get().replace(",", "."))
            except ValueError:
                knoll_h = 0.8
            try:
                dep_depth = float(dep_depth_entry.get().replace(",", "."))
            except ValueError:
                dep_depth = 0.3

            status_label.config(text="Drawing knolls and depressions...")
            root.update_idletasks()
            add_depressions(ax, grid_x, grid_y, dmr_grid_linear_viz_features,
                            pixel_size=FIXED_PIXEL_SIZE, min_diameter=1,
                            max_diameter=5, min_depth=dep_depth)
            add_knoll_symbols(ax, grid_x, grid_y, dmr_grid_linear_viz_features,
                              pixel_size=FIXED_PIXEL_SIZE, min_height=knoll_h)

        # Vektory
        status_label.config(text="Drawing OSM and ZABAGED data...")
        root.update_idletasks()
        visibility_settings = {k: v.get() for k, v in LAYER_VISIBILITY.items()}
        if gdf_osm is not None and not gdf_osm.empty:
            add_vector_layers(ax, gdf_osm.copy(), map_extent, zabaged_gdfs,
                              dmr_grid_linear_viz_features, grid_x, grid_y,
                              visibility=visibility_settings, isom_gdfs=isom_gdfs)

        if isom_gdfs:
            add_custom_isom_layers(ax, isom_gdfs)

        # Uložení PNG
        output_path = os.path.splitext(dmr_path)[0] + "_OMap.png"
        status_label.config(text="Saving PNG file...")
        root.update_idletasks()
        print(f"Ukládám: {output_path} (DPI=1000)")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(output_path, dpi=1000, bbox_inches='tight', pad_inches=0,
                    transparent=(selected_paper_format == "Data Extent"))
        plt.close(fig)

        progress_bar["value"] = 95

        # World file
        print("Generating World File...")
        try:
            with Image.open(output_path) as img:
                img_width_px, img_height_px = img.width, img.height
            pixel_size_x = (map_maxx - map_minx) / img_width_px
            pixel_size_y = (map_maxy - map_miny) / img_height_px
            world_file_content = (
                f"{pixel_size_x}\n0.0\n0.0\n{-pixel_size_y}\n"
                f"{map_minx + pixel_size_x / 2.0}\n{map_maxy - pixel_size_y / 2.0}\n"
            )
            world_file_path = os.path.splitext(output_path)[0] + ".pgw"
            with open(world_file_path, "w") as f:
                f.write(world_file_content)
            print(f" World File uložen: {world_file_path}")
        except Exception as e:
            print(f"⚠️ Error while creating World File: {e}")

        status_label.config(text=" Done.", foreground="darkgreen")
        progress_bar["value"] = 100

        # OOM export
        if save_oom_var.get():
            try:
                status_label.config(text="Exporting layers to GPKG...")
                root.update_idletasks()
                oom_path = os.path.splitext(dmr_path)[0] + "_OOM.gpkg"
                if os.path.exists(oom_path):
                    os.remove(oom_path)
                export_oom_gpkg(oom_path)
                status_label.config(text=" Done", foreground="darkgreen")
            except Exception as e:
                print(f"Error while exporting GPKG: {e}")

        end_time = time.time()
        elapsed = end_time - start_time
        mins, secs = divmod(elapsed, 60)
        time_str = f"{int(mins)} min {int(secs)} s"
        
        print(f"--- ANALYSIS FINISHED ---")
        print(f"Finished in: {time_str}")
        
        status_label.config(text=f"Hotovo (Čas: {time_str})", foreground="darkgreen")
        progress_bar["value"] = 100

        root.after(3000, lambda: status_label.config(text="Ready to start."))

    except Exception as e:
        end_time = time.time()
        elapsed = end_time - start_time
        mins, secs = divmod(elapsed, 60)
        print(f"--- ANALYSIS FAILED ---")
        print(f"Failed in: {int(mins)} min {int(secs)} s")
        messagebox.showerror("Error", str(e))
        status_label.config(text=f" Error: {str(e)}", foreground="red")
        progress_bar["value"] = 0


# GUI

print("--- OMapMaker v5: Starting GUI ---")
SCALE = 15000
SYMBOL_LIBRARY = {}

LAS_FILES = [("Lidar data", "*.las *.laz"), ("All files", "*.*")]
SHP_FILES = [("Shapefile", "*.shp"), ("All files", "*.*")]
LAS_TIF_FILES = [
    ("Supported data", "*.las *.laz *.tif *.tiff"),
    ("Lidar data", "*.las *.laz"),
    ("GeoTIFF", "*.tif *.tiff"),
    ("All files", "*.*")
]

root = tk.Tk()
root.title("OMapMaker")
root.state('zoomed')

paper_format_var = tk.StringVar()
paper_format_options = ["Data Extent", "A3 (Landscape)", "A3 (Portrait)", "A4 (Landscape)", "A4 (Portrait)"]

main_frame = ttk.Frame(root, padding="15")
main_frame.pack(fill=tk.BOTH, expand=True)

notebook = ttk.Notebook(main_frame)
notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

tab_data = ttk.Frame(notebook, padding="10")
tab_settings = ttk.Frame(notebook, padding="10")
tab_layers = ttk.Frame(notebook, padding="10")

notebook.add(tab_data, text="Data")
notebook.add(tab_settings, text="Map and Terrain Settings")
notebook.add(tab_layers, text="Layers to Draw")

# --- Tab 1: Data ---
required_frame = ttk.Labelframe(tab_data, text="Required LiDAR data (.las/.laz)", padding="10")
required_frame.pack(fill=tk.X, expand=False, pady=(0, 10))

cuzk_btn_frame = ttk.Frame(required_frame)
cuzk_btn_frame.pack(fill=tk.X, padx=5, pady=(0, 8))
ttk.Button(cuzk_btn_frame, text="Download data from ČÚZK",
           command=open_cuzk_downloader_from_gui).pack(side=tk.LEFT)
ttk.Label(cuzk_btn_frame, text="(DMR 5G, DMP 1G, DMP OK)",
          foreground="gray", font=("Segoe UI", 8)).pack(side=tk.LEFT)
ttk.Separator(required_frame, orient="horizontal").pack(fill=tk.X, padx=5, pady=(0, 8))

ttk.Label(required_frame, text="Select Digital Terrain Model (DTM):").pack(anchor="w", padx=5)
dmr_entry_frame = ttk.Frame(required_frame)
dmr_entry_frame.pack(fill=tk.X, expand=True, padx=5, pady=(0, 10))
dmr_entry = ttk.Entry(dmr_entry_frame, width=70)
dmr_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
ttk.Button(dmr_entry_frame, text="...", width=4,
           command=lambda: select_file(dmr_entry, "Select DTM (.las)", LAS_FILES)).pack(side=tk.RIGHT, padx=(5, 0))

ttk.Label(required_frame, text="Select Digital Surface Model (DSM):").pack(anchor="w", padx=5)
dmp_entry_frame = ttk.Frame(required_frame)
dmp_entry_frame.pack(fill=tk.X, expand=True, padx=5, pady=(0, 10))
dmp_entry = ttk.Entry(dmp_entry_frame, width=70)
dmp_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
ttk.Button(dmp_entry_frame, text="...", width=4,
           command=lambda: select_file(dmp_entry, "Select DSM (.las or .tif)", LAS_TIF_FILES)).pack(side=tk.RIGHT, padx=(5, 0))

optional_frame = ttk.Labelframe(tab_data, text="Optional vector data (.shp)", padding="10")
optional_frame.pack(fill=tk.BOTH, expand=True)

zabaged_container = ttk.Frame(optional_frame)
zabaged_container.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
ttk.Label(zabaged_container, text="Data from ZABAGED© (.shp):").pack(anchor="w")
zabaged_list_frame = ttk.Frame(zabaged_container)
zabaged_list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
zabaged_scrollbar = ttk.Scrollbar(zabaged_list_frame, orient=tk.VERTICAL)
zabaged_listbox = tk.Listbox(zabaged_list_frame, height=4, selectmode=tk.EXTENDED,
                              yscrollcommand=zabaged_scrollbar.set)
zabaged_scrollbar.config(command=zabaged_listbox.yview)
zabaged_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
zabaged_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
zabaged_buttons_frame = ttk.Frame(zabaged_container)
zabaged_buttons_frame.pack(fill=tk.X)
ttk.Button(zabaged_buttons_frame, text="Add files...",
           command=lambda: select_multiple_files_to_listbox(zabaged_listbox)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
ttk.Button(zabaged_buttons_frame, text="Remove selected",
           command=lambda: remove_selected_from_listbox(zabaged_listbox)).pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(5, 0))

isom_container = ttk.Frame(optional_frame)
isom_container.pack(fill=tk.BOTH, expand=True)
ttk.Label(isom_container, text="Custom ISOM layers (Filename = ISOM Code, e.g. 501.shp):").pack(anchor="w")
isom_list_frame = ttk.Frame(isom_container)
isom_list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
isom_scrollbar = ttk.Scrollbar(isom_list_frame, orient=tk.VERTICAL)
isom_listbox = tk.Listbox(isom_list_frame, height=4, selectmode=tk.EXTENDED,
                           yscrollcommand=isom_scrollbar.set)
isom_scrollbar.config(command=isom_listbox.yview)
isom_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
isom_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
isom_buttons_frame = ttk.Frame(isom_container)
isom_buttons_frame.pack(fill=tk.X)
ttk.Button(isom_buttons_frame, text="Add ISOM layers...",
           command=lambda: select_multiple_files_to_listbox(isom_listbox)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
ttk.Button(isom_buttons_frame, text="Remove selected",
           command=lambda: remove_selected_from_listbox(isom_listbox)).pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(5, 0))

# --- Tab 2: Settings ---
settings_col1 = ttk.Frame(tab_settings)
settings_col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
settings_col2 = ttk.Frame(tab_settings)
settings_col2.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

map_frame = ttk.Labelframe(settings_col1, text="Map parameters", padding="10")
map_frame.pack(fill=tk.X, pady=(0, 10))

CRS_MAP = {
    "S-JTSK (ČR) [EPSG:5514]": "EPSG:5514",
    "UTM Zone 29N (Ireland, Portugal) [EPSG:32629]": "EPSG:32629",
    "UTM Zone 30N (GB, Western France, Spain) [EPSG:32630]": "EPSG:32630",
    "UTM Zone 31N (Central France, Eastern Spain) [EPSG:32631]": "EPSG:32631",
    "UTM Zone 32N (Norway, Germany, Switzerland, North-west of Italy) [EPSG:32632]": "EPSG:32632",
    "UTM Zone 33N (Czechia, Sweden, Slovenia, South-east of Italy) [EPSG:32633]": "EPSG:32633",
    "UTM Zone 34N (North-east of Sweden, Slovakia, Hungary) [EPSG:32634]": "EPSG:32634",
    "UTM Zone 35N (Finland, Estonia, Latvia, Lithuania, Romania) [EPSG:32635]": "EPSG:32635",
    "S-JTSK (Slovensko - JTSK03) [EPSG:2065]": "EPSG:2065",
}

ttk.Label(map_frame, text="Coordinate system:").grid(row=0, column=0, sticky="w", pady=5)
crs_var = tk.StringVar(value="UTM Zone 33N (Czechia, Sweden, Slovenia, South-east of Italy) [EPSG:32633]")
crs_combo = ttk.Combobox(map_frame, textvariable=crs_var, values=list(CRS_MAP.keys()), state="readonly")
crs_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

ttk.Label(map_frame, text="Paper format:").grid(row=1, column=0, sticky="w", pady=5)
paper_format_combo = ttk.Combobox(map_frame, textvariable=paper_format_var,
                                   values=paper_format_options, state="readonly")
paper_format_combo.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
paper_format_combo.set(paper_format_options[3])

ttk.Label(map_frame, text="Scale:").grid(row=2, column=0, sticky="w", pady=5)
scale_var = tk.StringVar(value="1:10 000")
scale_combo = ttk.Combobox(map_frame, textvariable=scale_var,
                            values=["1:10 000", "1:15 000"], state="readonly")
scale_combo.grid(row=2, column=1, sticky="ew", padx=5, pady=5)

ttk.Label(map_frame, text="Magnetic declination (°):").grid(row=3, column=0, sticky="w", pady=5)
north_rotation_entry = ttk.Entry(map_frame)
north_rotation_entry.insert(0, "5")
north_rotation_entry.grid(row=3, column=1, sticky="ew", padx=5, pady=5)
map_frame.columnconfigure(1, weight=1)

veg_frame = ttk.Labelframe(settings_col1, text="Vegetation heights (m)", padding="10")
veg_frame.pack(fill=tk.X)
for row, (label_text, default) in enumerate([
    ("Open Land / Rough Open Land (up to):", "1"),
    ("Vegetation: fight (up to):", "2"),
    ("Vegetation: walk (up to):", "6"),
    ("Vegetation: slow running (up to):", "12"),
]):
    ttk.Label(veg_frame, text=label_text).grid(row=row, column=0, sticky="w", pady=5)
veg_frame.columnconfigure(1, weight=1)

bin_entry_1 = ttk.Entry(veg_frame); bin_entry_1.insert(0, "1");   bin_entry_1.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
bin_entry_2 = ttk.Entry(veg_frame); bin_entry_2.insert(0, "2"); bin_entry_2.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
bin_entry_3 = ttk.Entry(veg_frame); bin_entry_3.insert(0, "6");   bin_entry_3.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
bin_entry_4 = ttk.Entry(veg_frame); bin_entry_4.insert(0, "12");  bin_entry_4.grid(row=3, column=1, sticky="ew", padx=5, pady=5)

ter_frame = ttk.Labelframe(settings_col2, text="Terrain feature detection", padding="10")
ter_frame.pack(fill=tk.X)
ttk.Label(ter_frame, text="Rock slope threshold (°):").grid(row=0, column=0, sticky="w", pady=5)
slope_threshold_entry = ttk.Entry(ter_frame); slope_threshold_entry.insert(0, "45")
slope_threshold_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

ttk.Label(ter_frame, text="Minimal knoll height (m):").grid(row=1, column=0, sticky="w", pady=5)
knoll_height_entry = ttk.Entry(ter_frame); knoll_height_entry.insert(0, "0.2")
knoll_height_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5) 

ttk.Label(ter_frame, text="Minimal depression depth (m):").grid(row=2, column=0, sticky="w", pady=5)
dep_depth_entry = ttk.Entry(ter_frame); dep_depth_entry.insert(0, "0.5")
dep_depth_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=5)

ttk.Label(ter_frame, text="Contour smoothing (Sigma):").grid(row=3, column=0, sticky="w", pady=5)
contour_smooth = ttk.Entry(ter_frame); contour_smooth.insert(0, "6.5")
contour_smooth.grid(row=3, column=1, sticky="ew", padx=5, pady=5)
ter_frame.columnconfigure(1, weight=1)

# --- Tab 3: Layers ---
LAYER_VISIBILITY = {
    "contours":       tk.BooleanVar(value=True),
    "private":        tk.BooleanVar(value=True),
    "rocks":          tk.BooleanVar(value=True),
    "vegetation":     tk.BooleanVar(value=True),
    "water":          tk.BooleanVar(value=True),
    "roads":          tk.BooleanVar(value=True),
    "buildings":      tk.BooleanVar(value=True),
    "fences":         tk.BooleanVar(value=True),
    "man_made":       tk.BooleanVar(value=True),
    "magnetic_lines": tk.BooleanVar(value=False),
}

layer_grid_frame = ttk.Frame(tab_layers)
layer_grid_frame.pack(fill=tk.BOTH, expand=True, pady=10)

layer_checks = [
    ("Landforms and contours (101–115)", "contours", 0, 0),
    ("Rocks and boulders (201–215)",     "rocks",    1, 0),
    ("Water and marsh (301–313)",        "water",    2, 0),
    ("Vegetation (401–419)",             "vegetation",3, 0),
    ("Roads, tracks, paths (501–509)",   "roads",    4, 0),
    ("Fences and walls (510–519)",       "fences",   0, 1),
    ("Areas that should not be entered (520)", "private", 1, 1),
    ("Buildings (521, 522)",             "buildings",2, 1),
    ("Man-made features (523–532)",      "man_made", 3, 1),
    ("Magnetic north lines (601)",       "magnetic_lines", 4, 1),
]
for text, key, row, col in layer_checks:
    ttk.Checkbutton(layer_grid_frame, text=text, variable=LAYER_VISIBILITY[key]).grid(
        row=row, column=col, sticky="w", pady=5, padx=10)

btn_frame = ttk.Frame(tab_layers)
btn_frame.pack(fill=tk.X, pady=10)
ttk.Button(btn_frame, text="Check all",
           command=lambda: [v.set(True) for v in LAYER_VISIBILITY.values()],
           width=15).pack(side=tk.LEFT, padx=(10, 5))
ttk.Button(btn_frame, text="Clear all",
           command=lambda: [v.set(False) for v in LAYER_VISIBILITY.values()],
           width=15).pack(side=tk.LEFT, padx=5)

# --- Controls ---
controls_frame = ttk.Frame(main_frame, padding="10")
controls_frame.pack(fill=tk.X, expand=False)

save_options_frame = ttk.Frame(controls_frame)
save_options_frame.pack(fill=tk.X, pady=(0, 10))
save_var = tk.BooleanVar(value=True)
ttk.Checkbutton(save_options_frame, text="Save map to PNG", variable=save_var).pack(side=tk.LEFT, padx=(0, 20))
save_oom_var = tk.BooleanVar(value=True)
ttk.Checkbutton(save_options_frame, text="Export layers for OOM (GPKG)", variable=save_oom_var).pack(side=tk.LEFT)

run_button = ttk.Button(controls_frame, text="Generate map", command=run_main_analysis)
run_button.pack(fill=tk.X, ipady=8)

progress_bar = ttk.Progressbar(controls_frame, orient="horizontal", mode="determinate")
progress_bar.pack(pady=10, fill=tk.X)

status_label = tk.Label(controls_frame, text="Ready.", anchor="center",
                         foreground="darkgreen", font=("Segoe UI", 10, "bold"))
status_label.pack(fill=tk.X)

root.mainloop()
print("--- OMapMaker: GUI was closed ---")