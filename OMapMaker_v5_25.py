import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import xml.etree.ElementTree as ET
import time
from svgpath2mpl import parse_path
from ast import literal_eval
import laspy
import rasterio 
from rasterio.features import rasterize
from rasterio.enums import Resampling
import numpy as np
from scipy.interpolate import griddata, splprep, splev
import matplotlib.pyplot as plt 
#from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import (binary_dilation, binary_erosion, gaussian_filter, label, laplace, find_objects, minimum_filter, maximum_filter, binary_opening, binary_closing)
from matplotlib.collections import LineCollection
from matplotlib.transforms import Affine2D
from matplotlib.patches import PathPatch, Polygon as MplPolygon
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon, box, shape, Point, MultiPoint
from shapely.ops import unary_union, shape
from shapely import affinity
import fiona
import osmnx as ox
from pyproj import Transformer, CRS
import pandas as pd
#import math
from PIL import Image
from skimage import measure
import os
import re

CURRENT_CRS = "EPSG:5514"
def load_symbol_library(xml_file):
    """ Načte symboly a inteligentně opraví formáty (čárky v číslech vs. n-tice v závorkách) """
    print(f"--- Načítám symboly z: {xml_file} ---")
    library = {}
    
    if not os.path.exists(xml_file):
        print("❌ CHYBA: Soubor symbols.xml nenalezen.")
        return {}

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        for symbol in root.findall('symbol'):
            sid = symbol.get('id')
            stype = symbol.get('type')
            
            if not sid: continue

            # 1. Načtení stylů (Style props)
            style_elem = symbol.find('style')
            props = style_elem.attrib.copy() if style_elem is not None else {}
            ticks_elem = symbol.find('style_ticks')
            if ticks_elem is not None:
                for k, v in ticks_elem.attrib.items():
                    props[f"tick_{k}"] = v
            clean_props = {}
            for k, v in props.items():
                val_str = str(v).strip()
                
                # n-tice/tuple? (např. linestyle="(0, (1, 5))") musíme zachovat a NEKONVERTOVAT čárky, jinak to matplotlib nevezme
                if val_str.startswith('('):
                    try:
                        clean_props[k] = literal_eval(val_str)
                        continue # Povedlo se, jdeme dál
                    except:
                        pass # Nejde to, zkusíme další metody
                
                # číslo s čárkou? (např. "0,35")
                val_fixed = val_str.replace(',', '.')
                try:
                    clean_props[k] = float(val_fixed)
                    continue
                except ValueError:
                    pass
                
                # text (barva, string)
                clean_props[k] = val_str

            # 2. Načtení cesty (Path) pro bodové značky
            path_obj = None
            path_d = None
            path_elem = symbol.find('path')
            if path_elem is not None:
                path_d = path_elem.get('d')
                if path_d:
                    try:
                        path_obj = parse_path(path_d)
                        # Vycentrování značky na střed
                        ext = path_obj.get_extents()
                        center = (ext.xmin + ext.xmax) / 2, (ext.ymin + ext.ymax) / 2
                        path_obj.vertices -= center
                        # Otočení osy Y (SVG má Y dolů, matplotlib nahoru)
                        path_obj.vertices[:, 1] *= -1
                    except Exception as e:
                        print(f"Chyba parsování path u {sid}: {e}")

            library[sid] = {
                "type": stype,
                "props": clean_props,
                "path": path_obj,
                "path_d": path_d
            }

        print(f"✅ Načteno {len(library)} symbolů.")
        return library

    except Exception as e:
        print(f"❌ KRITICKÁ CHYBA XML: {e}")
        return {}
    
def load_dmr_grid(dmr_path, target_crs_code, pixel_size=0.5, sigma_smooth=6.5):
    """ 
    Načte DMR, rozšíří rastr o 10 px a okraje vyplní hodnotou nejbližšího bodu (extrapolace).
    """
    print(f"Načítám DMR s pixelem {pixel_size} m: {dmr_path}")
    print(f"  -> Cílový systém: {target_crs_code}")
    status_label.config(text=f"Načítám DMR ({pixel_size}m pixel)...")
    root.update_idletasks()
    
    xs, ys, zs = [], [], []
    transformer = None
    
    with laspy.open(dmr_path) as fh:
        try:
            source_crs = fh.header.parse_crs()
            if source_crs is None: raise ValueError("No CRS in header")
        except:
            print("  -> CRS v hlavičce LAS nenalezeno, předpokládám EPSG:5514 (S-JTSK).")
            source_crs = CRS.from_epsg(5514)
            
        try:
            target_crs_obj = CRS.from_string(target_crs_code)
            if source_crs != target_crs_obj:
                print(f"  -> Provádím transformaci bodů: {source_crs.to_string()} -> {target_crs_code}")
                transformer = Transformer.from_crs(source_crs, target_crs_obj, always_xy=True)
        except Exception as e:
            print(f"Chyba při přípravě transformace: {e}")

        for chunk in fh.chunk_iterator(1_000_000):
            clas = np.array(chunk.classification)
            mask = (clas == 2) | (clas == 8) # Ground points
            
            if np.any(mask):
                chunk_x = np.array(chunk.x[mask])
                chunk_y = np.array(chunk.y[mask])
                chunk_z = np.array(chunk.z[mask])
                
                if transformer:
                    chunk_x, chunk_y = transformer.transform(chunk_x, chunk_y)
                
                xs.append(chunk_x)
                ys.append(chunk_y)
                zs.append(chunk_z)

    if not xs:
        raise ValueError("V souboru DMR nebyly nalezeny žádné body země (class 2 nebo 8).")
        
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    z = np.concatenate(zs)
    
    print(f"  -> Načteno {len(x)} bodů DMR.")
    
    # --- 1. ROZŠÍŘENÍ EXTENTU (PADDING) ---
    buffer_pixels = 1
    buffer_dist = buffer_pixels * pixel_size
    
    min_x, max_x = x.min() - buffer_dist, x.max() + buffer_dist
    min_y, max_y = y.min() - buffer_dist, y.max() + buffer_dist
    
    extent = (min_x, max_x, min_y, max_y)
    print(f"  -> Rozsah rozšířen o {buffer_dist}m (protažení okrajů).")
    
    grid_x, grid_y = np.mgrid[extent[0]:extent[1]:pixel_size, 
                              extent[2]:extent[3]:pixel_size]
    
    print(f"Interpoluji DMR mřížku (rozměr: {grid_x.shape})...")
    status_label.config(text="Interpoluji DMR mřížku...")
    root.update_idletasks()
    
    points = np.vstack((x, y)).T
    
    # --- 2. REDUKCE BODŮ ---
    MAX_POINTS_DMR = 1_500_000
    if len(points) > MAX_POINTS_DMR:
        indices = np.random.choice(len(points), MAX_POINTS_DMR, replace=False)        
        points = points[indices] 
        z = z[indices]   

    # --- 3. NORMALIZACE SOUŘADNIC ---
    shift_x = np.mean(points[:, 0])
    shift_y = np.mean(points[:, 1])
    points_shifted = points - np.array([shift_x, shift_y])
    grid_x_shifted = grid_x - shift_x
    grid_y_shifted = grid_y - shift_y

    # --- 4. INTERPOLACE (Cubic + Nearest Fill) ---
    # a) Cubic - kvalitní uvnitř, ale NaN na okrajích (bufferu)
    print("  -> Metoda Cubic (vnitřek)...")
    dmr_grid = griddata(points_shifted, z, (grid_x_shifted, grid_y_shifted), method='cubic')
    
    # b) Zkontrolujeme, kde jsou NaN (okraje bufferu + díry)
    nan_mask = np.isnan(dmr_grid)
    
    if np.any(nan_mask):
        print("  -> Metoda Nearest (vyplnění okrajů/děr)...")
        # c) Nearest - "protáhne" hodnoty nejbližších bodů do stran
        dmr_nearest = griddata(points_shifted, z, (grid_x_shifted, grid_y_shifted), method='nearest')
        
        # d) Sloučení: Kde Cubic selhal (NaN), tam dáme Nearest
        dmr_grid[nan_mask] = dmr_nearest[nan_mask]
    
    # --- 5. VYHLAZENÍ ---
    # Lehké vyhlazení, aby přechod mezi Cubic a Nearest nebyl ostrý
    sigma_pixels = sigma_smooth # Nebo méně, pokud chceš ostřejší terén
    dmr_grid = gaussian_filter(dmr_grid, sigma=sigma_pixels)
            
    return dmr_grid, grid_x, grid_y, extent, points, z


def load_dmp_grid(dmp_path, grid_x, grid_y, extent, target_crs_code):
    """ 
    Načítá DMP s NÁHODNOU redukcí bodů a transformací souřadnic.
    Obsahuje fix pro interpolaci ve velkých souřadnicích (UTM).
    """
    print(f"Načítám DMP: {dmp_path}")
    status_label.config(text="Načítám DMP (Random sample)...")
    root.update_idletasks()
    MAX_POINTS_DMP = 1_500_000 
    file_ext = os.path.splitext(dmp_path)[1].lower()
    
    if file_ext in ['.las', '.laz']:        
        xs, ys, zs = [], [], []
        transformer = None 
        
        with laspy.open(dmp_path) as fh:
            try:
                source_crs = fh.header.parse_crs()
                if source_crs is None: source_crs = CRS.from_epsg(5514)
            except:
                source_crs = CRS.from_epsg(5514)
            
            try:
                target_crs_obj = CRS.from_string(target_crs_code)
                if source_crs != target_crs_obj:
                    transformer = Transformer.from_crs(source_crs, target_crs_obj, always_xy=True)
            except Exception as e:
                print(f"Chyba transformace DMP: {e}")

            total_points = fh.header.point_count
            if total_points > 0:
                fraction = min(1.0, MAX_POINTS_DMP / total_points)
            else:
                fraction = 1.0
            
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
        
        print(f"Interpoluji DMP...")
        status_label.config(text="Interpoluji DMP mřížku...")
        root.update_idletasks()

        # --- NORMALIZACE SOUŘADNIC (FIX PRO UTM/VELKÁ ČÍSLA) ---
        shift_x = np.mean(points[:, 0])
        shift_y = np.mean(points[:, 1])
        
        points_shifted = points - np.array([shift_x, shift_y])
        grid_x_shifted = grid_x - shift_x
        grid_y_shifted = grid_y - shift_y
        # -------------------------------------------------------
        
        dmp_grid = griddata(points_shifted, z, (grid_x_shifted, grid_y_shifted), method='linear')
        
        if np.isnan(dmp_grid).all():
             print("  -> Linear selhal, používám nearest.")
             dmp_grid = griddata(points_shifted, z, (grid_x_shifted, grid_y_shifted), method='nearest')
            
    elif file_ext in ['.tif', '.tiff']:
        print("⚠️ Upozornění: GeoTIFF DMP se automaticky netransformuje. Musí být ve zvoleném CRS!")
        with rasterio.open(dmp_path) as src:
            total_pixels = src.width * src.height
            if total_pixels > MAX_POINTS_DMP:
                scale = (MAX_POINTS_DMP / total_pixels) ** 0.5
                nw, nh = int(src.width * scale), int(src.height * scale)
                print(f"  -> Zmenšuji TIF pro interpolaci na {nw}x{nh} (Bilinear)...")
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

            # --- NORMALIZACE SOUŘADNIC (FIX PRO UTM/VELKÁ ČÍSLA) ---
            shift_x = np.mean(points[:, 0])
            shift_y = np.mean(points[:, 1])
            
            points_shifted = points - np.array([shift_x, shift_y])
            grid_x_shifted = grid_x - shift_x
            grid_y_shifted = grid_y - shift_y
            # -------------------------------------------------------

            dmp_grid = griddata(points_shifted, z, (grid_x_shifted, grid_y_shifted), method='linear')
            if np.isnan(dmp_grid).all():
                dmp_grid = griddata(points_shifted, z, (grid_x_shifted, grid_y_shifted), method='nearest')

    else:
        raise ValueError(f"Nepodporovaný formát DMP: {file_ext}")
        
    return dmp_grid

def generate_and_plot_contours(ax, padded_grid, levels, style_id, zorder, transform_info, clip_box):
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
    
    if clip_box is not None:
        gdf = gpd.clip(gdf, clip_box)
        
    if not gdf.empty:
        plot_masked(sym_key=style_id, zorder=zorder, mask=None, gdf=gdf, ax=ax, to_mask=False)

def add_contour_lines(ax, grid_x, grid_y, dmr_grid_unclipped, smoothing_s=2, clip_mask=None):
    print("Kreslím vrstevnice...")
    status_label.config(text="Kreslím vrstevnice...")
    root.update_idletasks()

    # 1. TVORBA MASKY PLATNÝCH DAT
    valid_data_mask = (dmr_grid_unclipped > 0) & (~np.isnan(dmr_grid_unclipped))
    
    # EROZE OKRAJŮ
    safe_mask = binary_erosion(valid_data_mask, iterations=3)
    
    # Aplikujeme masku na výškový model
    dmr_grid_plot = np.where(safe_mask, dmr_grid_unclipped, np.nan)

    # 2. NASTAVENÍ GEOMETRIE
    # Padding nastavíme na NaN, aby algoritmus neuzavíral čáry do nulových bodů
    pad_width = 10 
    dmr_padded = np.pad(dmr_grid_plot, pad_width=pad_width, mode='constant', constant_values=np.nan)

    min_x, max_x = grid_x.min(), grid_x.max()
    min_y, max_y = grid_y.min(), grid_y.max()
    px_step_x = (max_x - min_x) / (grid_x.shape[0] - 1)
    px_step_y = (max_y - min_y) / (grid_y.shape[1] - 1)
    
    transform_info = (min_x, min_y, px_step_x, px_step_y, pad_width)
    original_extent_box = box(min_x, min_y, max_x, max_y)

    # 3. VÝPOČET LEVELŮ (Vrstvy z dmr_grid_plot)
    min_z = np.nanmin(dmr_grid_plot)
    max_z = np.nanmax(dmr_grid_plot)
    
    # Hlavní vrstevnice (25m) a základní (5m) dle symboliky ISOM
    major_levels = np.arange(np.floor(min_z / 25) * 25, np.ceil(max_z / 25) * 25 + 1, 25)
    base_levels = np.arange(np.floor(min_z / 5) * 5, np.ceil(max_z / 5) * 5 + 1, 5)
    base_levels = np.setdiff1d(base_levels, major_levels)

    # 4. VYKRESLENÍ
    generate_and_plot_contours(ax, dmr_padded, major_levels, 'sym102', 25, transform_info, original_extent_box)
    generate_and_plot_contours(ax, dmr_padded, base_levels, 'sym101', 25, transform_info, original_extent_box)
    
    # 4. MASKA PRO DOPLŇKOVÉ VRSTEVNICE (Ztracená část)
    filled_mean = np.nanmean(dmr_grid_unclipped)
    dmr_grid_calc = np.nan_to_num(dmr_grid_unclipped, nan=filled_mean)

    # Výpočet křivosti (curvature) pro detekci terénních tvarů
    gy, gx = np.gradient(dmr_grid_calc) 
    gxx, _ = np.gradient(gx)
    _, gyy = np.gradient(gy)
    curvature = np.abs(gxx + gyy)
    
    # Detekce mírných sklonů (gentle slope)
    slope = np.hypot(gx, gy)
    
    curvature_threshold = np.percentile(curvature[safe_mask], 88)
    curvature_mask = (curvature > curvature_threshold) & safe_mask
    gentle_slope_mask = (slope < np.percentile(slope[safe_mask], 15)) & safe_mask
    combined_mask = curvature_mask | gentle_slope_mask
    dilated_mask = binary_dilation(combined_mask, iterations=10)
    dmr_grid_minor = np.where(dilated_mask, dmr_grid_plot, np.nan)
    dmr_padded_minor = np.pad(dmr_grid_minor, pad_width=pad_width, mode='constant', constant_values=np.nan)
    
    minor_levels = np.arange(np.floor(min_z / 2.5) * 2.5, np.ceil(max_z / 2.5) * 2.5 + 1, 2.5)
    minor_levels = np.setdiff1d(minor_levels, np.union1d(major_levels, base_levels))
    
    generate_and_plot_contours(ax, dmr_padded_minor, minor_levels, 'sym103', 25, transform_info, original_extent_box)
    
    print("✅ Vrstevnice vykresleny")

def vectorize_rocks(grid_x, grid_y, dmr_grid, transform, slope_threshold_deg=26):
    print("Vektorizuji skály...")
    status_label.config(text="Vektorizuji skály...")
    root.update_idletasks()
    
    # Výpočet sklonu
    dy, dx = np.gradient(dmr_grid)
    slope = np.rad2deg(np.arctan(np.hypot(dx, dy)))

    valid_data_mask = (dmr_grid > 0) & (~np.isnan(dmr_grid))
    
    safe_mask = binary_erosion(valid_data_mask, iterations=7)
    rock_mask_raw = (slope > slope_threshold_deg) & safe_mask
    
    rock_area = rock_mask_raw.astype(np.int32).T
    rock_area = np.flipud(rock_area)

    pixel_area = abs(transform.a * transform.e)
    min_area = 5 * pixel_area
    
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
        print("✅ Skály vykresleny")

        # Čištění geometrií
        gdf.geometry = gdf.geometry.buffer(1.0).buffer(-0.1).simplify(0.5)
        dissolved_gdf = gdf.dissolve(by='class_name').reset_index()

        return dissolved_gdf

    except Exception as e:
        print(f"Chyba skal: {e}")
        return gpd.GeoDataFrame(columns=['class_name', 'geometry'], crs=CURRENT_CRS)

def add_depressions(ax, grid_x, grid_y, dmr_grid, pixel_size=0.5, min_diameter=2, max_diameter=5, min_depth=0.7):
    print(f"Detekuji prohlubně (pixel: {pixel_size}m, min_hloubka: {min_depth}m)...")
    status_label.config(text="Detekuji prohlubně...")
    root.update_idletasks()
    
    sym_key = 'sym111'
    symbol_info = SYMBOL_LIBRARY.get(sym_key)
    if not symbol_info or symbol_info['type'] != 'point':
        print(f"Chyba: Symbol '{sym_key}' nebyl nalezen v knihovně.")
        return
    
    # Načtení vlastností symbolu
    symbol_props = symbol_info['props'].copy()
    symbol_path = symbol_info['path'] # Výchozí cesta z XML loaderu
    
    # Zkusíme načíst/opravit SVG cestu přímo z props, pokud existuje
    svg_path_string = symbol_props.get('path_d')
    if svg_path_string:
        try:
            marker_path = parse_path(svg_path_string)
            marker_path.vertices -= marker_path.vertices.mean(axis=0)
            # DŮLEŽITÉ: Použijeme nově naparsovanou cestu
            symbol_path = marker_path 
        except Exception as e:
            print(f"⚠️ Chyba SVG u {sym_key}: {e}")

    # Nastavení Z-order a scale
    symbol_props.setdefault('zorder', 20)
    scale_factor = 1.0
    
    # Vyčištění props pro matplotlib
    clean_keys = ['dotsize', 'dotdistance', 'dotcolor', 'marker_shape', 
                  'facecolor_alt', 'hatchdistance', 'hatchcolor', 
                  'hatchwidth', 'hatchstyle', 'd', 'path_d']
    for k in clean_keys:
        symbol_props.pop(k, None)

    # --- 1. DYNAMICKÉ FILTRY (V metrech, ne pixelech) ---
    # Sigma pro vyhlazení terénu
    sigma_smooth_meters = 0.5
    sigma_smooth = sigma_smooth_meters / pixel_size
    
    # Sigma pro referenční rovinu (musí "překlenout" díru) - cca 5 metrů
    sigma_ref = 8
    
    # Okno pro hledání minima (aby to nebyl jen dolík mezi kameny) - cca 2 metry
    window_size_meters = 30.0
    window_size = int(window_size_meters / pixel_size)
    if window_size < 3: window_size = 3 # Minimum 3x3

    print(f"  -> Nastavení filtrů: Smooth Sigma={sigma_smooth:.1f}px, Ref Sigma={sigma_ref:.1f}px, Window={window_size}px")

    # Aplikace filtrů
    smoothed = gaussian_filter(dmr_grid, sigma=sigma_smooth)
    
    # Hledáme lokální minima v okně
    local_min = (smoothed == minimum_filter(smoothed, size=window_size))
    
    # Referenční "víko" přes terén
    depth_reference = gaussian_filter(smoothed, sigma=sigma_ref)
    
    # Hloubka je rozdíl mezi "víkem" a dnem
    depth = depth_reference - smoothed
    
    # Maska kandidátů
    depression_mask = (local_min & (depth > min_depth))
    
    count_candidates = np.sum(depression_mask)
    print(f"  -> Nalezeno {count_candidates} kandidátů na prohlubeň (před filtrováním velikosti).")

    labeled, num_features = label(depression_mask)
    slices = find_objects(labeled)
    
    count_final = 0
    for slc in slices:
        region = labeled[slc]
        region_mask = (region > 0)
        
        # Rozměry v pixelech -> převod na metry
        ny, nx = region_mask.shape
        diameter_meters = max(ny, nx) * pixel_size
        
        # Kontrola průměru (příliš malé nebo příliš velké zahodíme)
        if not (min_diameter <= diameter_meters <= max_diameter):
            continue
            
        # Výpočet středu
        cy, cx = np.argwhere(region_mask).mean(axis=0)
        i0 = int(slc[0].start + cy)
        j0 = int(slc[1].start + cx)
        
        if i0 >= dmr_grid.shape[0] or j0 >= dmr_grid.shape[1]:
            continue
            
        x0 = grid_x[i0, j0]
        y0 = grid_y[i0, j0]

        # Vykreslení
        transform = Affine2D().rotate_deg(180).scale(scale_factor).translate(x0, y0) + ax.transData
        patch = PathPatch(
            symbol_path, # Zde už je správná (případně opravená) cesta
            transform=transform,
            **symbol_props
        )
        ax.add_patch(patch)
        count_final += 1
        
    print("✅ Prohlubně vykresleny")


def add_knoll_symbols(ax, grid_x, grid_y, dmr_grid, pixel_size=0.5, min_height=0.8, max_diameter=6.0):
    print("Kreslím kupky...")
    status_label.config(text="Kreslím kupky...")
    root.update_idletasks()
    
    sym_key = 'sym109'
    symbol_info = SYMBOL_LIBRARY.get(sym_key)
    if not symbol_info or symbol_info['type'] != 'point':
        return
    
    symbol_path = symbol_info['path']
    symbol_props = symbol_info['props'].copy()
    valid_data_mask = (dmr_grid > 0) & (~np.isnan(dmr_grid))
    safe_mask = binary_erosion(valid_data_mask, iterations=8)
    # 2. ANALÝZA TERÉNU
    # Vyhlazení pro odstranění drobného šumu
    smoothed = gaussian_filter(dmr_grid, sigma=0.5)
    
    # Hledání lokálních maxim v okně 5x5 pixelů
    local_max = (smoothed == maximum_filter(smoothed, size=20))
    height_reference = gaussian_filter(smoothed, sigma=8)
    height = smoothed - height_reference
    
    # Kupka musí být lokální maximum, mít minimální výšku a ležet v bezpečné zóně
    knoll_mask = (local_max & (height > min_height) & safe_mask)
    
    # 3. VYKRESLENÍ SYMBOLŮ
    labeled, _ = label(knoll_mask)
    slices = find_objects(labeled)  

    count = 0
    for slc in slices:
        region = labeled[slc]
        region_mask = (region > 0)
        ny, nx = region_mask.shape
        diameter = max(ny, nx) * pixel_size
        
        # Filtrování příliš rozsáhlých útvarů, které nejsou bodovými kupkami
        if diameter > max_diameter:
            continue

        cy, cx = np.argwhere(region_mask).mean(axis=0)
        i0 = int(slc[0].start + cy)
        j0 = int(slc[1].start + cx)
        
        if i0 >= dmr_grid.shape[0] or j0 >= dmr_grid.shape[1]:
            continue

        x0 = float(grid_x[i0, j0])
        y0 = float(grid_y[i0, j0])

        # Aplikace transformace pro správné umístění symbolu
        transform = Affine2D().scale(1.0).translate(x0, y0) + ax.transData
        
        patch = PathPatch(
            symbol_path,
            transform=transform,
            zorder=10,
            facecolor=symbol_props.get('facecolor', '#d15c00'),
            edgecolor=symbol_props.get('edgecolor', 'none'),
            linewidth=0
        )
        ax.add_patch(patch)
        count += 1
        
    print(f"✅ Kupky vykresleny")


def pt2m(pt, SCALE=10_000):
    return pt * 0.0003527 * SCALE


def in2m(inch, SCALE=10_000):
    return inch * 0.0254 * SCALE
    

def plot_dashed_hatch(ax, gdf, style_props, zorder): 
    """ Plots polygon fill covered with hatched pattern """   
    # Příprava stylů z XML
    hatch_distance = style_props.pop('hatchdistance')
    hatch_color = style_props.pop('hatchcolor')
    hatch_width = style_props.pop('hatchwidth')
    hatch_style = style_props.pop('hatchstyle')
    
    # Vykreslení pozadí
    gdf.plot(ax=ax, zorder=zorder, **style_props)

    # Příprava a vykreslení čárkované výplně
    if gdf.empty: return
    
    try:
        all_geoms = unary_union(gdf.geometry)
    except Exception:
        all_geoms = gdf.geometry.buffer(0).unary_union
    
    if all_geoms.is_empty: return
        
    minx, miny, maxx, maxy = all_geoms.bounds
    
    if pt2m(hatch_distance, SCALE):
        y_coords = np.arange(np.floor(miny / pt2m(hatch_distance, SCALE)) * pt2m(hatch_distance, SCALE), maxy, pt2m(hatch_distance, SCALE))
        h_lines = [LineString([(minx, y), (maxx, y)]) for y in y_coords]
    
        if not h_lines: return
            
        multi_lines = MultiLineString(h_lines)
        clipped_lines = multi_lines.intersection(all_geoms)
        
        if not clipped_lines.is_empty:
            plot_series = gpd.GeoSeries([clipped_lines])
            
            plot_series.plot(ax=ax, color=hatch_color, linewidth=hatch_width, linestyle=hatch_style, zorder=zorder + 0.1)

            if ax.collections:
                ax.collections[-1].set_linestyle(hatch_style)


def plot_dotted_hatch(ax, gdf, style_props, zorder):
    """ Plots polygon fill covered with dotted pattern """
    # Příprava stylů z XML
    dot_distance = style_props.pop('dotdistance')
    dot_size = style_props.pop('dotsize')
    dot_color = style_props.pop('dotcolor')
 
    # Vykreslení pozadí
    gdf.plot(ax=ax, zorder=zorder, **style_props)

    # Příprava a vykreslení tečkované výplně
    if gdf.empty: return

    try:
        all_geoms = gdf.geometry.union_all()
    except Exception:
        all_geoms = gdf.geometry.buffer(0).union_all()
    
    if all_geoms.is_empty: return   
    
    minx, miny, maxx, maxy = all_geoms.bounds
    
    if pt2m(dot_distance, SCALE):
        x_coords = np.arange(np.floor(minx / pt2m(dot_distance, SCALE)) * pt2m(dot_distance, SCALE), maxx, pt2m(dot_distance, SCALE))
        y_coords = np.arange(np.floor(miny / pt2m(dot_distance, SCALE)) * pt2m(dot_distance, SCALE), maxy, pt2m(dot_distance, SCALE))
        
        if len(x_coords) == 0 or len(y_coords) == 0: return

        xx, yy = np.meshgrid(x_coords, y_coords)
        points = [Point(x, y) for x, y in zip(xx.flatten(), yy.flatten())]

        if not points: return
        
        all_dots = MultiPoint(points)
        clipped_dots = all_dots.intersection(all_geoms)
        
        if not clipped_dots.is_empty:
            if isinstance(clipped_dots, Point):
                x_final = [clipped_dots.x]
                y_final = [clipped_dots.y]
            elif isinstance(clipped_dots, MultiPoint):
                x_final = [p.x for p in clipped_dots.geoms]
                y_final = [p.y for p in clipped_dots.geoms]
            else: # je to GeometryCollection, musíme být opatrní
                x_final = []
                y_final = []
                if hasattr(clipped_dots, 'geoms'):
                    for geom in clipped_dots.geoms:
                        if isinstance(geom, Point):
                            x_final.append(geom.x)
                            y_final.append(geom.y)
                        elif isinstance(geom, MultiPoint):
                            x_final.extend([p.x for p in geom.geoms])
                            y_final.extend([p.y for p in geom.geoms])
            
            ax.scatter(x_final, y_final, marker='.', color=dot_color, s=dot_size, zorder=zorder + 0.1, edgecolors='none')

def add_magnetic_north_lines(ax, extent, scale, rotation=0, spacing_mm=30, zorder=20):    
    # 1. Přepočet rozestupu z mm na metry
    spacing_meters = (spacing_mm / 1000.0) * scale
    
    minx, maxx, miny, maxy = extent
    
    # Vypočteme střed a diagonálu pro dostatečné pokrytí při rotaci
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2
    diagonal = np.hypot(maxx - minx, maxy - miny)
    
    # Generujeme čáry v širším rozsahu (kvůli rotaci)
    # Začneme od středu a jdeme na obě strany
    num_lines = int(diagonal / spacing_meters) + 2
    
    lines = []
    
    # Generování svislých čar
    for i in range(-num_lines, num_lines + 1):
        x = center_x + (i * spacing_meters)
        # Čára musí být dostatečně dlouhá, aby po otočení pokryla mapu
        line = LineString([(x, center_y - diagonal), (x, center_y + diagonal)])
        
        if rotation != 0:
            line = affinity.rotate(line, -rotation, origin=(center_x, center_y))
            
        lines.append(line)
        
    # Ořezání čar na viditelný rozsah mapy (Clip)
    map_box = box(minx, miny, maxx, maxy)
    visible_lines = []
    
    for line in lines:
        if line.intersects(map_box):
            visible_lines.append(line.intersection(map_box))
            
    if not visible_lines:
        return

    try:
        gdf_north = gpd.GeoDataFrame(geometry=visible_lines, crs=CURRENT_CRS)
        plot_masked(sym_key='sym601', zorder=zorder, mask=None, gdf=gdf_north, ax=ax, to_mask=False)
        if 'sym601' not in SYMBOL_LIBRARY:
             gdf_north.plot(ax=ax, color='#21d1ff', linewidth=0.35, zorder=zorder)        
    except Exception as e:
        print(f"⚠️ Chyba při kreslení poledníků: {e}")

def get_col(df, col_name):
    if col_name in df.columns:
        return df[col_name]
    else:
        return pd.Series([''] * len(df), index=df.index)
    
def smooth_line(line, s, k):
    """
    Pomocná funkce: Vyhladí jednu LineString pomocí B-spline.
    (Verze s opraveným uzavíráním smyček a kontrolou min. bodů)
    """
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
        
        num_points = len(x)
        u_new = np.linspace(u.min(), u.max(), num_points * 5)
        
        x_new, y_new = splev(u_new, tck)
        
        coords = np.vstack((x_new, y_new)).T
        if len(coords) < (k + 1):
             print(f"  -> Upozornění: B-spline vygeneroval příliš málo bodů ({len(coords)}). Vracím původní linii.")
             return line 
        if is_closed:
            coords[-1] = coords[0]
        return LineString(coords)
    except Exception as e:
        print(f"  -> Upozornění: B-spline selhalo: {e}. Vracím původní linii.")
        return line


def vectorize_vegetation(classified_raster_raw, class_names, transform, dmr_path, save_gpkg=False):
    """ Converts input raster in simplified vector layer using morphological cleaning first """
    print("Zahajuji vektorizaci rastru vegetace...")
    status_label.config(text="Vektorizuji vegetaci...")
    root.update_idletasks()
    
    # 1. Čištění rastru (Morfologické operace) - ZÁSADNÍ ZRYCHLENÍ
    # Odstraní "sůl a pepř" (šum) a vyhladí hrany pixelů před vektorizací
    print("  -> Předzpracování rastru...")
    
    # Zkopírujeme si raw data
    cleaned_raster = classified_raster_raw.copy()
    
    # Pro každou třídu provedeme lehké vyhlazení
    # Tím se zbavíme osamocených pixelů, které generují tisíce zbytečných polygonů
    unique_classes = np.unique(cleaned_raster)
    unique_classes = unique_classes[unique_classes != 0] # Ignorovat pozadí
    
    for c in unique_classes:
        mask = (cleaned_raster == c)
        # Opening odstraní malé tečky (šum)
        mask = binary_opening(mask, structure=np.ones((3,3)))
        # Closing zaplní malé díry uvnitř porostu
        mask = binary_closing(mask, structure=np.ones((3,3)))
        cleaned_raster[mask] = c
        # Kde maska zmizela (byl to jen šum), nastavíme 0
        cleaned_raster[(cleaned_raster == c) & (~mask)] = 0

    classified_raster_transposed = cleaned_raster.T
    classified_raster = np.flipud(classified_raster_transposed)

    pixel_area = abs(transform.a * transform.e)
    min_area = 50 * pixel_area 
    
    mask = (classified_raster != 0)
    
    try:
        results_generator = rasterio.features.shapes(
            classified_raster, 
            mask=mask, 
            transform=transform 
        )
    
        features = []
        for geom, value in results_generator:
            class_id = int(value)
            if class_id == 0: continue
            features.append({
                'geometry': shape(geom),
                'class_id': class_id,
                'class_name': class_names.get(class_id, 'Neznama')
            })
            
        if not features:
            print("Nebyly nalezeny žádné polygony k vektorizaci.")
            return gpd.GeoDataFrame(columns=['class_name', 'class_id', 'geometry'], crs=CURRENT_CRS) # ZMĚNA

        print(f"Nalezeno {len(features)} hrubých polygonů.")
        
        gdf = gpd.GeoDataFrame(features, crs=CURRENT_CRS)
        original_crs = gdf.crs

        # Filtrace malých ploch
        print(f"  -> Filtruji malé polygony...")
        # Zrychlení: Filtrace pomocí vektorizovaného numpy, ne iterace
        gdf = gdf[gdf.geometry.area >= min_area]

        if gdf.empty:
            return gpd.GeoDataFrame(columns=['class_name', 'class_id', 'geometry'], crs=original_crs)

        print("  -> Zjednodušuji geometrie...")
        status_label.config(text="Zjednodušuji polygony...")
        root.update_idletasks()
        
        # simplify bez topologie
        gdf.geometry = gdf.geometry.simplify(0.6, preserve_topology=False)
        
        # Buffer 0 je trik na opravu invalidních geometrií po simplify(False)
        gdf.geometry = gdf.geometry.buffer(1.7).buffer(0)
        
        print("  -> Spojuji polygony (Dissolve)...")
        dissolved_gdf = gdf.dissolve(by='class_name', aggfunc='first').reset_index()
        
        # Finální jemné zjednodušení už na spojených datech (s topologií, aby to vypadalo hezky)
        dissolved_gdf.geometry = dissolved_gdf.geometry.simplify(0.5, preserve_topology=True)

        dissolved_gdf = gpd.GeoDataFrame(
            dissolved_gdf, 
            geometry='geometry', 
            crs=original_crs
        )

        if save_gpkg:
            output_file = os.path.splitext(dmr_path)[0] + "_vegetace.gpkg"
            dissolved_gdf.to_file(output_file, driver="GPKG")
            print(f"✅ Vektorová vegetace uložena do: {output_file}")
        
        return dissolved_gdf

    except Exception as e:
        print(f"❌ Chyba při vektorizaci: {e}")
        status_label.config(text=f"Chyba při vektorizaci: {e}", fg="red")
        return gpd.GeoDataFrame(columns=['class_name', 'class_id', 'geometry'], crs="EPSG:5514")


    sym_data = SYMBOL_LIBRARY.get(sym_key, {})
    # DŮLEŽITÉ: .copy(), aby se props nemazaly z globální knihovny při volání pop()
    sym_props = sym_data.get('props', {}).copy() 
    if 'hatchdistance' in sym_props:
        plot_dashed_hatch(ax, subset, sym_props, zorder=zorder)
    elif 'dotdistance' in sym_props:
        plot_dotted_hatch(ax, subset, sym_props, zorder=zorder)
    else:
        # Odstranění vlastních klíčů, které by vadily standardnímu plot()
        custom_keys = ['dotsize', 'dotdistance', 'dotcolor', 'marker_shape', 'facecolor_alt', 
                       'hatchdistance', 'hatchcolor', 'hatchwidth', 'hatchstyle']
        for key in custom_keys:
            sym_props.pop(key, None)
        
        subset.plot(ax=ax, zorder=zorder, **sym_props)

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
        val = sym_props.pop('solid_capstyle')
        sym_props['capstyle'] = val

    if sym_key == 'sym510':
        subset.plot(ax=ax, zorder=zorder, **sym_props)
        
        tick_len = 10 
        tick_segments = []

        for geom in subset.geometry:
            if geom is None or geom.is_empty: continue
            parts = [geom] if geom.geom_type == 'LineString' else geom.geoms

            for line in parts:
                coords = np.array(line.coords)
                if len(coords) < 2: continue

                vectors = np.diff(coords, axis=0)
                norms = np.hypot(vectors[:, 0], vectors[:, 1])
                valid = norms > 0
                vectors = vectors[valid]
                norms = norms[valid]
                coords_clean = np.vstack([coords[0], coords[1:][valid]])
                
                if len(vectors) == 0: continue

                tangents = vectors / norms[:, None]

                for i in range(len(coords_clean)):
                    x, y = coords_clean[i]
                    
                    if i == 0:
                        t = tangents[0]
                    elif i == len(coords_clean) - 1:
                        t = tangents[-1]
                    else:
                        t = (tangents[i-1] + tangents[i])
                        nm = np.hypot(t[0], t[1])
                        if nm == 0: t = tangents[i-1] 
                        else: t = t / nm
                    
                    nx, ny = -t[1], t[0]
                    
                    p1 = (x - nx * tick_len / 2, y - ny * tick_len / 2)
                    p2 = (x + nx * tick_len / 2, y + ny * tick_len / 2)
                    tick_segments.append([p1, p2])

        if tick_segments:
            lc = LineCollection(tick_segments, colors=sym_props.get('color', 'black'), 
                              linewidths=sym_props.get('linewidth', 1.0), zorder=zorder)
            ax.add_collection(lc)
        return

    cliff_ids = ['sym104', 'sym201', 'sym202']
    is_cliff = sym_key in cliff_ids or 'tick_length' in sym_props

    if is_cliff and dmr_grid is not None and grid_x is not None and grid_y is not None:
        # Načtení parametrů (s fixem pro tick_linewidth z předchozího dotazu)
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
        epsilon = 0.1 # Malý posun v metrech pro výpočet tečny

        for geom in subset.geometry:
            if geom is None or geom.is_empty: continue
            
            parts = [geom] if geom.geom_type == 'LineString' else geom.geoms
            
            for line in parts:
                line_len = line.length
                if line_len < tick_space: continue
                
                # Generujeme body podél čáry
                distances = np.arange(tick_space / 2, line_len, tick_space)
                
                for dist in distances:
                    # Bod, kde má být čárka
                    pt = line.interpolate(dist)
                    
                    pt_ahead = line.interpolate(min(dist + epsilon, line_len))
                    
                    dx = pt_ahead.x - pt.x
                    dy = pt_ahead.y - pt.y
                    tan_len = np.hypot(dx, dy)
                    if tan_len == 0: continue
                    
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
                        else:
                            dot_product = (n1x * gx) + (n1y * gy)
                            
                            if dot_product < 0:
                                # n1 míří "proti" stoupání = dolů
                                final_nx, final_ny = n1x, n1y
                            else:
                                # n1 míří "se" stoupáním = nahoru, takže bereme n2
                                final_nx, final_ny = n2x, n2y

                        # 5. Vytvoření úsečky
                        end_x = pt.x + (final_nx * tick_len)
                        end_y = pt.y + (final_ny * tick_len)
                        
                        ticks_segments.append([(pt.x, pt.y), (end_x, end_y)])

        if ticks_segments:
            lc = LineCollection(ticks_segments, 
                                colors=tick_color, 
                                linewidths=tick_width, 
                                zorder=zorder)
            ax.add_collection(lc)
        return

    if 'hatchdistance' in sym_props:
        plot_dashed_hatch(ax, subset.dissolve(), sym_props, zorder=zorder)
        return
    elif 'dotdistance' in sym_props:
        plot_dotted_hatch(ax, subset.dissolve(), sym_props, zorder=zorder)
        return

    if sym_type == 'point' and sym_path is not None:
        clean_keys = ['dotsize', 'dotdistance', 'dotcolor', 'marker_shape', 
                      'facecolor_alt', 'hatchdistance', 'hatchcolor', 
                      'hatchwidth', 'hatchstyle', 'd', 'path_d']
        for key in clean_keys:
            sym_props.pop(key, None)

        for geom in subset.geometry:
            points_to_plot = []
            if geom.geom_type == 'Point':
                points_to_plot.append((geom.x, geom.y))
            elif geom.geom_type == 'MultiPoint':
                points_to_plot.extend([(p.x, p.y) for p in geom.geoms])
            
            for x, y in points_to_plot:
                transform = Affine2D().translate(x, y) + ax.transData
                
                patch = PathPatch(
                    sym_path,
                    transform=transform,
                    zorder=zorder,
                    **sym_props
                )
                ax.add_patch(patch)
        return 

    clean_keys = ['dotsize', 'dotdistance', 'dotcolor', 'marker_shape', 'facecolor_alt', 
                  'hatchdistance', 'hatchcolor', 'hatchwidth', 'hatchstyle', 'd', 'path_d']
    for key in clean_keys:
        sym_props.pop(key, None)

    try:
        subset.plot(ax=ax, zorder=zorder, **sym_props)
    except Exception as e:
        print(f"⚠️ Chyba plot u {sym_key}: {e}")
        
def clip_vecor_layers(gdf, extent):
    """ Bezpečné oříznutí vektorové vrstvy """
    if gdf is None or gdf.empty:
        return gdf
    
    try:
        # Vytvoření ořezového obdélníku z extentu (minx, maxx, miny, maxy)
        # Toto převede souřadnice na geometrický objekt, který GeoPandas vyžaduje
        clip_box = box(extent[0], extent[2], extent[1], extent[3])
        
        # Samotný clip – druhý parametr MUSÍ být objekt 'clip_box', nikoliv jen text nebo pole
        return gpd.clip(gdf, clip_box)
    except Exception as e:
        print(f"Chyba při ořezu: {e}")
        return gdf



def add_vector_layers(ax, gdf, extent, zabaged_gdfs, dmr_grid_linear_viz_features, grid_x, grid_y, visibility, isom_gdfs):
    """ 
    Kompletní vykreslování vektorových vrstev.
    Obsahuje přímou podmínku IF pro každý symbol pro upřednostnění vlastních dat.
    """  
    if visibility is None:
        visibility = {k: True for k in ["water", "roads", "buildings", "fences", "man_made", "vegetation"]}      
    
    # Ořezání vstupních dat na rozsah mapy
    gdf = clip_vecor_layers(gdf, extent)
    for zabaged_key in zabaged_gdfs:
        zabaged_gdfs[zabaged_key] = clip_vecor_layers(zabaged_gdfs[zabaged_key], extent)

    if (gdf is None or gdf.empty) and not zabaged_gdfs and not isom_gdfs:
        return

    # --- 1. Načtení atributů z OSM dat ---
    access = get_col(gdf, "access").fillna('')
    amenity = get_col(gdf, "amenity").fillna('')
    barrier = get_col(gdf, "barrier").fillna('')
    bridge = get_col(gdf, "bridge").fillna('')
    building = get_col(gdf, "building").fillna('')
    covered = get_col(gdf, "covered").fillna('')
    emergency = get_col(gdf, "emergency").fillna('')
    geological = get_col(gdf, "geological").fillna('')
    highway = get_col(gdf, "highway").fillna('')
    historic = get_col(gdf, "historic").fillna('')
    intermittent = get_col(gdf, "intermittent").fillna('')
    landuse = get_col(gdf, "landuse").fillna('')
    leisure = get_col(gdf, "leisure").fillna('')
    man_made = get_col(gdf, "man_made").fillna('')
    military = get_col(gdf, "military").fillna('')
    natural = get_col(gdf, "natural").fillna('')
    parking = get_col(gdf, "parking").fillna('')
    place = get_col(gdf, "place").fillna('')
    power = get_col(gdf, "power").fillna('')
    railway = get_col(gdf, "railway").fillna('')
    surface = get_col(gdf, "surface").fillna('')
    trail_visibility = get_col(gdf, "trail_visibility")
    tracktype = get_col(gdf, "tracktype").fillna('')
    tunnel = get_col(gdf, "tunnel").fillna('')
    water = get_col(gdf, "water").fillna('')
    waterway = get_col(gdf, "waterway").fillna('')
    wetland = get_col(gdf, "wetland").fillna('')
    aerialway = get_col(gdf, "aerialway").fillna('')

    # --- 2. Rozdělení na geometrické typy ---
    gdf_centroids = gdf.copy()
    gdf_centroids['geometry'] = gdf_centroids.geometry.centroid
    gdf_points = gdf[gdf.geometry.geom_type.isin(["Point"])].copy()
    gdf_lines = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    gdf_polygons = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    # ======================================================================
    # TERRAIN SHAPES (Terénní tvary)
    # ======================================================================
    
    # 104 - Zemní Sráz
    sym = "sym104"
    cgdf = isom_gdfs.get("104")
    if cgdf is not None:
        plot_masked(sym_key=sym, zorder=21, mask=None, gdf=cgdf, ax=ax, to_mask=False)
    elif "StupenSraz" in zabaged_gdfs:
        plot_masked(sym_key=sym, zorder=21, mask=None, gdf=zabaged_gdfs["StupenSraz"], ax=ax, to_mask=False, dmr_grid=dmr_grid_linear_viz_features, grid_x=grid_x, grid_y=grid_y)
    else:
        mask_embankment = (man_made == "embankment")
        plot_masked(sym_key=sym, zorder=21, mask=mask_embankment, gdf=gdf_lines, ax=ax, dmr_grid=dmr_grid_linear_viz_features, grid_x=grid_x, grid_y=grid_y)

    # 105 - Zemní val    
    cgdf = isom_gdfs.get("105")
    if cgdf is not None:
        plot_masked(sym_key="sym105-1a", zorder=21, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        plot_masked(sym_key="sym105-1b", zorder=21, mask=None, gdf=cgdf, ax=ax, to_mask=False)
    elif "HradbaValBastaOpevneni" in zabaged_gdfs:
        plot_masked(sym_key="sym105-1a", zorder=30, mask=None, gdf=zabaged_gdfs["HradbaValBastaOpevneni"], ax=ax, to_mask=False)
        plot_masked(sym_key="sym105-1b", zorder=30, mask=None, gdf=zabaged_gdfs["HradbaValBastaOpevneni"], ax=ax, to_mask=False)

    # 107 - Rýha / Výmol
    sym = "sym107"
    cgdf = isom_gdfs.get("107")
    if cgdf is not None:
        plot_masked(sym_key=sym, zorder=20, mask=None, gdf=cgdf, ax=ax, to_mask=False)
    elif "RokleVymol" in zabaged_gdfs:
        plot_masked(sym_key=sym, zorder=20, mask=None, gdf=zabaged_gdfs["RokleVymol"], ax=ax, to_mask=False)

    # 108 - Malá erozní rýha
    sym = "sym108"
    cgdf = isom_gdfs.get("108")
    if cgdf is not None:
        plot_masked(sym_key="sym", zorder=21, mask=None, gdf=cgdf, ax=ax, to_mask=False)
    else:
        pass

    # 109 - Malá erozní rýha
    sym = "sym109"
    cgdf = isom_gdfs.get("109")
    if cgdf is not None:
        plot_masked(sym_key="sym", zorder=21, mask=None, gdf=cgdf, ax=ax, to_mask=False)
    else:
        pass

    # 111 - Malá erozní rýha
    sym = "sym111"
    cgdf = isom_gdfs.get("111")
    if cgdf is not None:
        plot_masked(sym_key="sym", zorder=21, mask=None, gdf=cgdf, ax=ax, to_mask=False)
    else:
        pass

    # 112 - Malá erozní rýha
    sym = "sym112"
    cgdf = isom_gdfs.get("112")
    if cgdf is not None:
        plot_masked(sym_key="sym", zorder=21, mask=None, gdf=cgdf, ax=ax, to_mask=False)
    else:
        pass
    
    if visibility.get("rocks", True):
        '''
        # 201 - Skalní sráz
        sym = "sym201"
        cgdf = isom_gdfs.get("201")
        if cgdf is not None:
            plot_masked(sym_key="sym202", zorder=56, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            mask_cliff_high = (natural == "cliff")
            plot_masked(sym_key=sym, zorder=56, mask=mask_cliff_high, gdf=gdf_lines, ax=ax, dmr_grid=dmr_grid_linear_viz_features, grid_x=grid_x, grid_y=grid_y)  
        '''
        # 202 - Skalní sráz
        sym = "sym202"
        cgdf = isom_gdfs.get("202")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=56, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            pass
        
        # 203-1 - Jeskyně
        sym = "sym203-1"
        cgdf = isom_gdfs.get("203.1")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=56, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        if "VstupDoJeskyne" in zabaged_gdfs:
            plot_masked(sym_key=sym, zorder=56, mask=None, gdf=zabaged_gdfs["VstupDoJeskyne"], ax=ax, to_mask=False)
        else:
            mask_cave = (natural == "cave_entrance") | (man_made == "adit")
            plot_masked(sym_key=sym, zorder=56, mask=mask_cave, gdf=gdf_centroids, ax=ax)
        
        # 203-2 - Nebezpečná jáma
        sym = "sym203-2"
        cgdf = isom_gdfs.get("203.1")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=56, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            pass

        # 204 - Malý Balvan
        sym = "sym204"
        cgdf = isom_gdfs.get("204")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=56, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            pass

        # 205 - Balvan
        sym = "sym205"
        cgdf = isom_gdfs.get("205")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=56, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "OsamelyBalvanSkalaSkalniSuk" in zabaged_gdfs:
            plot_masked(sym_key=sym, zorder=56, mask=None, gdf=zabaged_gdfs["OsamelyBalvanSkalaSkalniSuk"], ax=ax, to_mask=False)
        else:
            mask_boulder = (natural.isin(["stone", "rock"])) | (geological == "glacial_erratic")
            plot_masked(sym_key=sym, zorder=56, mask=mask_boulder, gdf=gdf_centroids, ax=ax)
            
        # 206 - Skalní útvar 
        '''
        sym = "sym206"
        cgdf = isom_gdfs.get("206")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=56, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "SkalniUtvary" in zabaged_gdfs:
            plot_masked(sym_key=sym, zorder=56, mask=None, gdf=zabaged_gdfs["SkalniUtvary"], ax=ax, to_mask=False)
        else:
            mask_rock = (geological.isin(["tor", "hoodoo", "dyke"]))
            plot_masked(sym_key=sym, zorder=56, mask=mask_rock, gdf=gdf_centroids, ax=ax)
        '''
                
        # 207 - Skupina balvanů
        sym = "sym207"
        cgdf = isom_gdfs.get("207")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=56, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "SkupinaBalvanu_b" in zabaged_gdfs:
            plot_masked(sym_key=sym, zorder=56, mask=None, gdf=zabaged_gdfs["SkupinaBalvanu_b"], ax=ax, to_mask=False)
        
        # 208 - Kamenité pole (Blockfield)
        sym = "sym208"
        cgdf = isom_gdfs.get("208")
        if cgdf is not None:
            plot_masked(sym_key="sym208", zorder=18, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            pass

        # 209 - Kamenité pole (Blockfield)
        sym = "sym209"
        cgdf = isom_gdfs.get("209")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=18, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            mask_gravel1 = (natural == "blockfield")
            plot_masked(sym_key=sym, zorder=18, mask=mask_gravel1, gdf=gdf_polygons, ax=ax)
        
        # 210 - Suťoviště (Scree)
        sym = "sym210"
        cgdf = isom_gdfs.get("210")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=18, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            mask_gravel2 = (natural == "scree")
            plot_masked(sym_key=sym, zorder=18, mask=mask_gravel2, gdf=gdf_polygons, ax=ax)
        
        # 211 - Suťoviště (Scree)
        sym = "sym211"
        cgdf = isom_gdfs.get("211")
        if cgdf is not None:
            plot_masked(sym_key="sym210", zorder=18, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            pass

        # 212 - Suťoviště (Scree)
        sym = "sym212"
        cgdf = isom_gdfs.get("212")
        if cgdf is not None:
            plot_masked(sym_key="sym210", zorder=18, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            pass

        # 213 - Písek
        sym = "sym213"
        cgdf = isom_gdfs.get("213")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=15, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            mask_sand = (natural.isin(["sand", "dune"]))
            plot_masked(sym_key=sym, zorder=15, mask=mask_sand, gdf=gdf_polygons, ax=ax)
        
        # 214 - Skalní podloží
        sym = "sym214"
        cgdf = isom_gdfs.get("214")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=17, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            mask_bedrock = (natural == "bare_rock")
            plot_masked(sym_key=sym, zorder=17, mask=mask_bedrock, gdf=gdf_polygons, ax=ax)
        
        # 215 - Příkop (zákop) - SLOŽENÝ SYMBOL (Kontrola 2x, pro okraj a pro výplň)
        mask_ditch = (barrier == "ditch") | (military == "trench")
        
        sym = "sym215a"
        cgdf = isom_gdfs.get("215")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=21, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            plot_masked(sym_key=sym, zorder=21, mask=mask_ditch, gdf=gdf_lines, ax=ax)

        sym = "sym215b"
        cgdf = isom_gdfs.get("215")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=21, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            plot_masked(sym_key=sym, zorder=21, mask=mask_ditch, gdf=gdf_lines, ax=ax)

    # ======================================================================
    # WATER (Voda)
    # ======================================================================
    if visibility.get("water", True):

        # 301 - Vodní plocha
        sym = "sym301"
        zabaged_has = any(k in zabaged_gdfs for k in ["VodniPlocha", "PozemniNadrz"])
        cgdf = isom_gdfs.get("301")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=27, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif zabaged_has:
            if "VodniPlocha" in zabaged_gdfs:
                plot_masked(sym_key=sym, zorder=27, mask=None, gdf=zabaged_gdfs["VodniPlocha"], ax=ax, to_mask=False)
            if "PozemniNadrz" in zabaged_gdfs:
                plot_masked(sym_key=sym, zorder=27, mask=None, gdf=zabaged_gdfs["PozemniNadrz"], ax=ax, to_mask=False)
        else:
            mask_water_deep = (natural.isin(["lake", "water", "canal"])) | (water.isin(["lake", "river", "basin", "bay", "reservoir"])) | (landuse == "basin") | (leisure == "swimming_pool")
            plot_masked(sym_key=sym, zorder=27, mask=mask_water_deep, gdf=gdf_polygons, ax=ax)
        
        # 302 -  Mělké vodní těleso
        sym = "sym302"
        cgdf = isom_gdfs.get("302")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=27, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            mask_water_shallow = (water == "stream") 
            plot_masked(sym_key=sym, zorder=27, mask=mask_water_shallow, gdf=gdf_polygons, ax=ax)

        # 303 - Jáma s vodou
        sym = "sym303"
        cgdf = isom_gdfs.get("303")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=27, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            pass

        # 304 - Řeka (splavná)
        sym = "sym304"
        cgdf = isom_gdfs.get("304")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=26, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "VodniTok" in zabaged_gdfs:
                mask = (get_col(zabaged_gdfs["VodniTok"], "typtoku_p").isin(["povrchový splavný"])) & \
                    (get_col(zabaged_gdfs["VodniTok"], "vydattok_p").isin(["stálý"]))
                plot_masked(sym_key=sym, zorder=26, mask=mask, gdf=zabaged_gdfs["VodniTok"], ax=ax)
        else:
            mask_river = ((waterway == "river") | (waterway == "canal")) & \
                        (~tunnel.isin(["culvert", "yes", "pipe", "covered", "cave"])) & \
                        (~intermittent.isin(["yes", "dry"]))
            plot_masked(sym_key=sym, zorder=26, mask=mask_river, gdf=gdf_lines, ax=ax)

        # 305 - Potok (nesplavný, stálý)
        sym = "sym305"
        cgdf = isom_gdfs.get("305")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=26, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "VodniTok" in zabaged_gdfs:
            mask = (get_col(zabaged_gdfs["VodniTok"], "typtoku_p").isin(["povrchový nesplavný"])) & \
                   (get_col(zabaged_gdfs["VodniTok"], "vydattok_p").isin(["stálý"]))
            plot_masked(sym_key=sym, zorder=26, mask=mask, gdf=zabaged_gdfs["VodniTok"], ax=ax)
        else:
            mask_stream = ((waterway == "stream") | (waterway == "ditch")) & \
                          (~tunnel.isin(["culvert", "yes", "pipe", "covered", "cave"])) & \
                          (~intermittent.isin(["yes", "dry"]))
            plot_masked(sym_key=sym, zorder=26, mask=mask_stream, gdf=gdf_lines, ax=ax)
        
        # 306 - Drobný vodní tok
        sym = "sym306"
        cgdf = isom_gdfs.get("306")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=26, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "VodniTok" in zabaged_gdfs:
            mask = (get_col(zabaged_gdfs["VodniTok"], "typtoku_p").isin(["povrchový splavný", "povrchový nesplavný"])) & \
                   (get_col(zabaged_gdfs["VodniTok"], "vydattok_p").isin(["občasný"]))
            plot_masked(sym_key=sym, zorder=26, mask=mask, gdf=zabaged_gdfs["VodniTok"], ax=ax)
        else:
            mask_drain = ((waterway == "drain") | ((waterway.isin(["stream", "ditch"])) & (~intermittent.isin(["yes", "dry"])))) & \
                         (~tunnel.isin(["culvert", "yes", "pipe", "covered", "cave"])) 
            plot_masked(sym_key=sym, zorder=26, mask=mask_drain, gdf=gdf_lines, ax=ax)
        
        # 307 - Neprůchodná bažina
        sym = "sym307"
        cgdf = isom_gdfs.get("307")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=25, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "Raseliniste" in zabaged_gdfs:
            plot_masked(sym_key=sym, zorder=25, mask=None, gdf=zabaged_gdfs["Raseliniste"], ax=ax, to_mask=False)
        else:
            mask_wetland1 = (wetland == "reedbed")
            plot_masked(sym_key=sym, zorder=25, mask=mask_wetland1, gdf=gdf_polygons, ax=ax)
        
        # 308 - Bažina
        sym = "sym308"
        cgdf = isom_gdfs.get("308")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=25, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "BazinaMocal" in zabaged_gdfs:
            plot_masked(sym_key=sym, zorder=25, mask=None, gdf=zabaged_gdfs["BazinaMocal"], ax=ax, to_mask=False)
        else:
            mask_wetland2 = (natural == "wetland") & (~wetland.isin(["marsh", "wet_meadow", "reedbed"]))
            plot_masked(sym_key=sym, zorder=25, mask=mask_wetland2, gdf=gdf_polygons, ax=ax)
        
        # 309 - Úzká bažina
        sym = "sym309"
        cgdf = isom_gdfs.get("309")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=25, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            pass
        
        # 310 - Nezřetelná bažina
        sym = "sym310"
        cgdf = isom_gdfs.get("310")
        if cgdf is not None:
            plot_masked(sym_key="sym308", zorder=25, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            mask_wetland3 = (wetland == "marsh") | (water == "wet_meadow")
            plot_masked(sym_key="sym308", zorder=25, mask=mask_wetland3, gdf=gdf_polygons, ax=ax)
        
        # 311 - Studna / Nádrž
        sym = "sym311"
        cgdf = isom_gdfs.get("311")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=52, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            mask_well = (man_made == "water_well") | (amenity == "fountain") | (natural == "geysyer") | \
                        (((emergency == "water_tank") | (man_made == "storage_tank")) & (~covered.isin(["yes", "roof", "shelter"])))
            plot_masked(sym_key=sym, zorder=52, mask=mask_well, gdf=gdf_centroids, ax=ax)
        
        # 312 - Pramen
        sym = "sym312"
        cgdf = isom_gdfs.get("312")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=52, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "ZdrojPodzemnichVod" in zabaged_gdfs:
            plot_masked(sym_key=sym, zorder=52, mask=None, gdf=zabaged_gdfs["ZdrojPodzemnichVod"], ax=ax, to_mask=False)
        else:
            mask_spring = (natural == "spring") & (covered != "yes")
            plot_masked(sym_key=sym, zorder=52, mask=mask_spring, gdf=gdf_centroids, ax=ax)
    
        # 313 - Výrazný vodní objekt
        sym = "sym313"
        cgdf = isom_gdfs.get("313")
        if cgdf is not None:
            plot_masked(sym_key="sym312", zorder=52, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            pass 

    # ======================================================================
    # VEGETATION / LANDUSE (Vegetace)
    # ======================================================================
    if visibility.get("vegetation", True):
        # 401 - Otevřený prostor (Louka)
        sym = "sym401"
        cgdf = isom_gdfs.get("401")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=1.0, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "TrvalyTravniPorost" in zabaged_gdfs:
            plot_masked(sym_key=sym, zorder=1.0, mask=None, gdf=zabaged_gdfs["TrvalyTravniPorost"], ax=ax, to_mask=False)
        else:
            mask_grass_low = (landuse.isin(["grassland", "grass"])) | (natural == "grassland")
            plot_masked(sym_key=sym, zorder=1.0, mask=mask_grass_low, gdf=gdf_polygons, ax=ax)
            
            mask_grass_high = (landuse == "meadow") | (natural.isin(["fell", "heath"]))
            plot_masked(sym_key=sym, zorder=1.0, mask=mask_grass_high, gdf=gdf_polygons, ax=ax)

        # 402 - Otevřený prostor s roztroušenými stromy (Park)
        sym = "sym402"
        cgdf = isom_gdfs.get("402")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=1.0, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "OkrasnaZahradaPark" in zabaged_gdfs:
            plot_masked(sym_key=sym, zorder=1.0, mask=None, gdf=zabaged_gdfs["OkrasnaZahradaPark"], ax=ax, to_mask=False)
        else:
            mask_park = (leisure == "park")
            plot_masked(sym_key=sym, zorder=1.0, mask=mask_park, gdf=gdf_polygons, ax=ax)
        
        # 403 - Divoký otevřený prostor (Louka)
        sym = "sym403"
        cgdf = isom_gdfs.get("403")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=1.0, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            pass

        # 404 - Divoký otevřený prostor s roztroušenými stromy (Park)
        sym = "sym404"
        cgdf = isom_gdfs.get("404")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=1.0, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            pass

        # - 405–410 se vykreslují z Lidaru
        
        # 408 - Alej / Živý plot
        sym = "sym408l"
        cgdf = isom_gdfs.get("sym408") or isom_gdfs.get("sym408".replace("sym", "")) or isom_gdfs.get(re.sub(r'[a-zA-Z]+$', '', "sym408".replace("sym", "")))
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=19, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "LiniovaVegetace" in zabaged_gdfs:
            mask = (get_col(zabaged_gdfs["LiniovaVegetace"], "typveg_p").isin(["živý plot"]))
            plot_masked(sym_key=sym, zorder=19, mask=mask, gdf=zabaged_gdfs["LiniovaVegetace"], ax=ax)
        else:
            mask_alley = (natural == "tree_row")
            plot_masked(sym_key=sym, zorder=99, mask=mask_alley, gdf=gdf_polygons, ax=ax)
        
        # 412 - Orná půda 
        cgdf = isom_gdfs.get("412")

        if cgdf is not None:
            plot_masked(sym_key="sym412a", zorder=1.9, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "OrnaPudaAOstatniDaleNespecifikovanePlochy" in zabaged_gdfs:
            mask = (get_col(zabaged_gdfs["OrnaPudaAOstatniDaleNespecifikovanePlochy"], "typ_pudy_p").isin(["orná půda"]))
            plot_masked(sym_key="sym412a", zorder=1.9, mask=mask, gdf=zabaged_gdfs["OrnaPudaAOstatniDaleNespecifikovanePlochy"], ax=ax)
        else:
            mask_field = (landuse == "farmland")
            plot_masked(sym_key="sym412a", zorder=1.9, mask=mask_field, gdf=gdf_polygons, ax=ax)

        if cgdf is not None:
            plot_masked(sym_key="sym412b", zorder=15, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "OrnaPudaAOstatniDaleNespecifikovanePlochy" in zabaged_gdfs:
            mask = (get_col(zabaged_gdfs["OrnaPudaAOstatniDaleNespecifikovanePlochy"], "typ_pudy_p").isin(["orná půda"]))
            plot_masked(sym_key="sym412b", zorder=15, mask=mask, gdf=zabaged_gdfs["OrnaPudaAOstatniDaleNespecifikovanePlochy"], ax=ax)
        else:
            mask_field = (landuse == "farmland")
            plot_masked(sym_key="sym412b", zorder=15, mask=mask_field, gdf=gdf_polygons, ax=ax)

        # 413 - Sad
        sym = "sym413"
        cgdf = isom_gdfs.get("413")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=1.9, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            mask_orchad = (landuse == "orchard") 
            plot_masked(sym_key=sym, zorder=1.9, mask=mask_orchad, gdf=gdf_polygons, ax=ax)
        
        # 414 - Vinice / Chmelnice
        sym = "sym414"
        cgdf = isom_gdfs.get("414")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=1.9, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "Vinice" in zabaged_gdfs:
            plot_masked(sym_key=sym, zorder=1.9, mask=None, gdf=zabaged_gdfs["Vinice"], ax=ax, to_mask=False)
        elif "Chmelnice" in zabaged_gdfs:
            plot_masked(sym_key=sym, zorder=1.9, mask=None, gdf=zabaged_gdfs["Chmelnice"], ax=ax, to_mask=False)
        else:
            mask_vineyard = (landuse == "vineyard")
            plot_masked(sym_key=sym, zorder=1.9, mask=mask_vineyard, gdf=gdf_polygons, ax=ax)
        
        # - Zřetelná hranice obdělávané půdy (linie)
        sym = "sym216l"
        cgdf = isom_gdfs.get("415")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=15, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            pass
        
        # 416 - Zřetelná hranice vegetace
        sym = "sym416p"
        cgdf = isom_gdfs.get("416")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=1.8, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "LesniPudaSeStromyKategorizovana" in zabaged_gdfs:
            mask = (get_col(zabaged_gdfs["LesniPudaSeStromyKategorizovana"], "druh_k").isin(["J"])) & \
                   (get_col(zabaged_gdfs["LesniPudaSeStromyKategorizovana"], "vyska_k").isin(["3"]))
            plot_masked(sym_key=sym, zorder=1.8, mask=mask, gdf=zabaged_gdfs["LesniPudaSeStromyKategorizovana"], ax=ax)
        
        # 417 - Výrazný strom
        sym = "sym417a"
        cgdf = isom_gdfs.get("417")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=54, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "VyznamnyNeboOsamelyStromLesik" in zabaged_gdfs:
            plot_masked(sym_key=sym, zorder=54, mask=None, gdf=zabaged_gdfs["VyznamnyNeboOsamelyStromLesik"], ax=ax, to_mask=False)
        else:
            mask_tree = (natural == "tree")
            if not gdf_centroids[mask_tree].empty:
                plot_masked(sym_key=sym, zorder=54, mask=mask_tree, gdf=gdf_centroids, ax=ax)
        sym = "sym417b"     
        cgdf = isom_gdfs.get("417")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=54, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "VyznamnyNeboOsamelyStromLesik" in zabaged_gdfs:
            plot_masked(sym_key=sym, zorder=54, mask=None, gdf=zabaged_gdfs["VyznamnyNeboOsamelyStromLesik"], ax=ax, to_mask=False)
        else:
            mask_tree = (natural == "tree")
            if not gdf_centroids[mask_tree].empty:
                plot_masked(sym_key=sym, zorder=55, mask=mask_tree, gdf=gdf_centroids, ax=ax)

        # 418 - Výrazný keř
        sym = "sym418a"
        cgdf = isom_gdfs.get("418")
        if cgdf is not None:
             plot_masked(sym_key=sym, zorder=54, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            mask_shrub = (natural == "shrub")
            plot_masked(sym_key=sym, zorder=54, mask=mask_shrub, gdf=gdf_centroids, ax=ax)
        sym = "sym418b"
        if cgdf is not None:
             plot_masked(sym_key=sym, zorder=54, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            mask_shrub = (natural == "shrub")
            plot_masked(sym_key=sym, zorder=55, mask=mask_shrub, gdf=gdf_centroids, ax=ax)

        # 419 - Výrazný vegetační objek
        mask_stump = (natural == "tree_stump") 
        sym = "sym419"
        cgdf = isom_gdfs.get("419")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=54, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            plot_masked(sym_key=sym, zorder=54, mask=mask_stump, gdf=gdf_centroids, ax=ax)


    # ======================================================================
    # ROADS & SQUARES (Komunikace)
    # ======================================================================
    if visibility.get("roads", True):
        # 501 - Parkoviště
        sym = "sym501"
        zabaged_has = any(k in zabaged_gdfs for k in ["ParkovisteOdpocivka", "ArealUceloveZastavby"])
        cgdf = isom_gdfs.get("501")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=49, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif zabaged_has:
            if "ParkovisteOdpocivka" in zabaged_gdfs:
                plot_masked(sym_key=sym, zorder=49, mask=None, gdf=zabaged_gdfs["ParkovisteOdpocivka"], ax=ax, to_mask=False)
            if "ArealUceloveZastavby" in zabaged_gdfs:
                mask = (get_col(zabaged_gdfs["ArealUceloveZastavby"], "typzast_k").isin(["408"]))
                plot_masked(sym_key=sym, zorder=49, mask=mask, gdf=zabaged_gdfs["ArealUceloveZastavby"], ax=ax)
        else:
            mask_parking = ((amenity == "parking") & (~parking.isin(["garage", "underground"]))) | \
                        (place == "square") | (highway.isin(["service", "pedestrian", "footway"])) | \
                        (man_made == "bunker_silo")
            plot_masked(sym_key=sym, zorder=49, mask=mask_parking, gdf=gdf_polygons, ax=ax)

        # 502D - Dálnice (Dva symboly 502 vedle sebe)
        sym = "sym502Da"
        cgdf = isom_gdfs.get("502D")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=45, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "SilniceDalnice" in zabaged_gdfs:
            mask = (get_col(zabaged_gdfs["SilniceDalnice"], "typsil_k").isin(["D1", "D2", "M"]))
            plot_masked(sym_key=sym, zorder=45, mask=mask, gdf=zabaged_gdfs["SilniceDalnice"], ax=ax)
        else:
            mask_road_double = (highway.isin(["motorway", "trunk"])) & \
                               (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) & \
                               (bridge != "yes") & (access != "private")
            plot_masked(sym_key=sym, zorder=45, mask=mask_road_double, gdf=gdf_lines, ax=ax)

        sym = "sym502Db"
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=47, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "SilniceDalnice" in zabaged_gdfs:
            mask = (get_col(zabaged_gdfs["SilniceDalnice"], "typsil_k").isin(["D1", "D2", "M"]))
            plot_masked(sym_key=sym, zorder=47, mask=mask, gdf=zabaged_gdfs["SilniceDalnice"], ax=ax)
        else:
            mask_road_double = (highway.isin(["motorway", "trunk"])) & \
                               (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) & \
                               (bridge != "yes") & (access != "private")
            plot_masked(sym_key=sym, zorder=47, mask=mask_road_double, gdf=gdf_lines, ax=ax)

        sym = "sym502Dc"
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=48, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "SilniceDalnice" in zabaged_gdfs:
            mask = (get_col(zabaged_gdfs["SilniceDalnice"], "typsil_k").isin(["D1", "D2", "M"]))
            plot_masked(sym_key=sym, zorder=48, mask=mask, gdf=zabaged_gdfs["SilniceDalnice"], ax=ax)
        else:
            mask_road_double = (highway.isin(["motorway", "trunk"])) & \
                               (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) & \
                               (bridge != "yes") & (access != "private")
            plot_masked(sym_key=sym, zorder=48, mask=mask_road_double, gdf=gdf_lines, ax=ax)

        # 502 - Široká silnice
        sym = "sym502a"
        zabaged_has = any(k in zabaged_gdfs for k in ["SilniceDalnice", "Ulice", "SilniceVeVastavbe"])
        cgdf = isom_gdfs.get("502")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=45, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif zabaged_has:
            if "SilniceDalnice" in zabaged_gdfs:
                mask = ((~get_col(zabaged_gdfs["SilniceDalnice"], "typsil_k").isin(["D1", "D2", "M"])))
                plot_masked(sym_key=sym, zorder=45, mask=mask, gdf=zabaged_gdfs["SilniceDalnice"], ax=ax)
            if "Ulice" in zabaged_gdfs:
                mask = (get_col(zabaged_gdfs["Ulice"], "typulice_k").isin(["026", "926"]))
                plot_masked(sym_key=sym, zorder=45, mask=mask, gdf=zabaged_gdfs["Ulice"], ax=ax)
            if "SilniceVeVastavbe" in zabaged_gdfs:
                plot_masked(sym_key=sym, zorder=45, mask=None, gdf=zabaged_gdfs["SilniceVeVastavbe"], ax=ax, to_mask=False)
        else:
            mask_road_major = (highway.isin(["highway_link", "trunk_link", "primary", "primary_link", "secondary", "secondary_link", "residential", "tertiary", "living_street"])) & \
                            (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) & \
                            (bridge != "yes") & (access != "private")
            plot_masked(sym_key=sym, zorder=45, mask=mask_road_major, gdf=gdf_lines, ax=ax)

        sym = "sym502b"
        zabaged_has = any(k in zabaged_gdfs for k in ["SilniceDalnice", "Ulice", "SilniceVeVastavbe"])
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=47, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif zabaged_has:
            if "SilniceDalnice" in zabaged_gdfs:
                mask = ((~get_col(zabaged_gdfs["SilniceDalnice"], "typsil_k").isin(["D1", "D2", "M"])))
                plot_masked(sym_key=sym, zorder=47, mask=mask, gdf=zabaged_gdfs["SilniceDalnice"], ax=ax)
            if "Ulice" in zabaged_gdfs:
                mask = (get_col(zabaged_gdfs["Ulice"], "typulice_k").isin(["026", "926"]))
                plot_masked(sym_key=sym, zorder=47, mask=mask, gdf=zabaged_gdfs["Ulice"], ax=ax)
            if "SilniceVeVastavbe" in zabaged_gdfs:
                plot_masked(sym_key=sym, zorder=47, mask=None, gdf=zabaged_gdfs["SilniceVeVastavbe"], ax=ax, to_mask=False)
        else:
            mask_road_major = (highway.isin(["highway_link", "trunk_link", "primary", "primary_link", "secondary", "secondary_link", "residential", "tertiary", "living_street"])) & \
                            (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) & \
                            (bridge != "yes") & (access != "private")
            plot_masked(sym_key=sym, zorder=47, mask=mask_road_major, gdf=gdf_lines, ax=ax)

        # 503 - Silnice
        sym = "sym503"
        zabaged_has = any(k in zabaged_gdfs for k in ["SilniceNeevidovana", "Cesta", "LyzarskyMustek"])
        cgdf = isom_gdfs.get("503")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=45, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif zabaged_has:
            if "SilniceNeevidovana" in zabaged_gdfs:
                plot_masked(sym_key=sym, zorder=45, mask=None, gdf=zabaged_gdfs["SilniceNeevidovana"], ax=ax, to_mask=False)
            if "Cesta" in zabaged_gdfs:
                mask = (get_col(zabaged_gdfs["Cesta"], "povrch_p").isin(["zpevněný (panel, dlažba)", "zpevněný (asfalt, beton)"])) & \
                    (get_col(zabaged_gdfs["Cesta"], "typcesty_p").isin(["cesta udržovaná"]))
                plot_masked(sym_key=sym, zorder=45, mask=mask, gdf=zabaged_gdfs["Cesta"], ax=ax)
            if "LyzarskyMustek" in zabaged_gdfs:
                plot_masked(sym_key=sym, zorder=46, mask=None, gdf=zabaged_gdfs["LyzarskyMustek"], ax=ax, to_mask=False)
        else:
            mask_road_minor = ((highway.isin(["tertiary_link", "service"])) | \
                        ((highway.isin(["track", "road", "cycleway", "track", "unclassified"])) & \
                        ((surface.isin(["concrete", "asphalt"])) | (tracktype == "grade1")))) & \
                        (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) & \
                        (bridge != "yes") & (access != "private")
            plot_masked(sym_key=sym, zorder=45, mask=mask_road_minor, gdf=gdf_lines, ax=ax)

        # 504 - Vozová cesta
        sym = "sym504"
        cgdf = isom_gdfs.get("504")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=45, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "Cesta" in zabaged_gdfs:
            mask = (get_col(zabaged_gdfs["Cesta"], "povrch_p").isin(["zpevněný (nosný terén, štěrk, kalený povrch)", "nedostatečně zpevněný (tráva, hlína, písek, kamení)", "neurčeno", "NULL"])) & \
                (get_col(zabaged_gdfs["Cesta"], "typcesty_p").isin(["cesta udržovaná"]))
            plot_masked(sym_key=sym, zorder=45, mask=mask, gdf=zabaged_gdfs["Cesta"], ax=ax)
        else:
            mask_track_major = ((highway.isin(["cycleway", "unclassified"])) & (~surface.isin(["concrete", "asphalt"])) & \
                            (tracktype != "grade1")) & \
                            (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) & \
                            (bridge != "yes") & (access != "private")
            plot_masked(sym_key=sym, zorder=45, mask=mask_track_major, gdf=gdf_lines, ax=ax)

        # 505 - Pěší cesta
        sym = "sym505"
        zabaged_has = any(k in zabaged_gdfs for k in ["Ulice", "Cesta"])
        cgdf = isom_gdfs.get("505")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=45, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif zabaged_has:
            if "Ulice" in zabaged_gdfs:
                mask = ((~get_col(zabaged_gdfs["Ulice"], "typulice_k").isin(["925", "025"])))
                plot_masked(sym_key=sym, zorder=45, mask=mask, gdf=zabaged_gdfs["Ulice"], ax=ax)
            if "Cesta" in zabaged_gdfs:
                mask = (get_col(zabaged_gdfs["Cesta"], "typcesty_p").isin(["cesta neudržovaná"]))
                plot_masked(sym_key=sym, zorder=45, mask=mask, gdf=zabaged_gdfs["Cesta"], ax=ax)
        else:
            mask_track_minor = ((highway.isin(["pedestrian", "road", "footway", "track", "bridleway"])) | \
                            ((highway == "cycleway") & (~surface.isin(["concrete", "asphalt"])) & (tracktype != "grade1"))) & \
                            (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) & \
                            (bridge != "yes") & (access != "private")
            plot_masked(sym_key=sym, zorder=45, mask=mask_track_minor, gdf=gdf_lines, ax=ax)

        # 506 - Pěšina
        sym = "sym506"
        cgdf = isom_gdfs.get("506")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=45, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "Pesina" in zabaged_gdfs:
            plot_masked(sym_key=sym, zorder=45, mask=None, gdf=zabaged_gdfs["Pesina"], ax=ax, to_mask=False)
        else:
            mask_path_major = (highway == "path") & \
                              (~trail_visibility.isin(["low", "poor", "bad", "very_bad", "horrible", "no"])) & \
                              (bridge != "yes") & (access != "private")
            plot_masked(sym_key=sym, zorder=45, mask=mask_path_major, gdf=gdf_lines, ax=ax)
            
        # 507 - Nezřetelné pěšina
        sym = "sym507"
        cgdf = isom_gdfs.get("507")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=45, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            mask_path_minor = (highway == "path") & \
                              (trail_visibility.isin(["low", "poor", "bad", "very_bad", "horrible"])) & \
                              (bridge != "yes") & (access != "private")
            plot_masked(sym_key=sym, zorder=45, mask=mask_path_minor, gdf=gdf_lines, ax=ax)

        # 508 -  Průsek nebo liniová trasa terénem
        sym = "sym508"
        cgdf = isom_gdfs.get("508")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=38, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "LesniPrusek" in zabaged_gdfs:
            plot_masked(sym_key=sym, zorder=38, mask=None, gdf=zabaged_gdfs["LesniPrusek"], ax=ax, to_mask=False)
        else:
            mask_cutline = (man_made == "cutline")
            plot_masked(sym_key=sym, zorder=38, mask=mask_cutline, gdf=gdf_lines, ax=ax)

        # Mosty - jen z OSM (Dobře odpovídají jen u hlavních silnic)
        mask_bridge_double = (highway.isin(["motorway", "trunk"])) & \
                             (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) & \
                             (bridge == "yes") & (access != "private")
        
        sym_list = ["sym502DBa", "sym502DBb", "sym502Da", "sym502Db", "sym502Dc"]
        zorders = [65, 66, 67, 68, 69]
        for sym, z in zip(sym_list, zorders):
            cgdf = isom_gdfs.get(sym)
            if cgdf is not None:
                plot_masked(sym_key=sym, zorder=z, mask=None, gdf=cgdf, ax=ax, to_mask=False)
            else:
                plot_masked(sym_key=sym, zorder=z, mask=mask_bridge_double, gdf=gdf_lines, ax=ax)
        
        mask_bridge_major = (highway.isin(["highway_link", "trunk_link", "primary", "primary_link", "secondary", "secondary_link", "residential", "tertiary", "living_street"])) & \
                            (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) & \
                            (bridge == "yes") & (access != "private")
        sym_list = ["sym502Ba", "sym502Bb", "sym502a", "sym502b"]
        zorders = [65, 66, 67, 68]
        for sym, z in zip(sym_list, zorders):
            cgdf = isom_gdfs.get(sym)
            if cgdf is not None:
                plot_masked(sym_key=sym, zorder=z, mask=None, gdf=cgdf, ax=ax, to_mask=False)
            else:
                plot_masked(sym_key=sym, zorder=z, mask=mask_bridge_major, gdf=gdf_lines, ax=ax)
        mask_bridge_minor = (highway.isin(["tertiary_link", "service", "track", "road", "unclassified"])) & \
                            (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) & \
                            (bridge == "yes") & (access != "private")
        sym_list = ["sym503Ba", "sym503Bb", "sym503"]
        zorders = [65, 66, 67]
        for sym, z in zip(sym_list, zorders):
            cgdf = isom_gdfs.get(sym)
            if cgdf is not None:
                plot_masked(sym_key=sym, zorder=z, mask=None, gdf=cgdf, ax=ax, to_mask=False)
            else:
                plot_masked(sym_key=sym, zorder=z, mask=mask_bridge_minor, gdf=gdf_lines, ax=ax)

        # Lávka
        sym = "sym503"
        if "Lavka" in zabaged_gdfs:
            plot_masked(sym_key=sym, zorder=40, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            mask_bridge_path = (highway.isin(["path", "cycleway", "footway", "bridleway"])) & \
                                (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) & \
                                (bridge == "yes") & (access != "private")
            plot_masked(sym_key=sym, zorder=67, mask=mask_bridge_path, gdf=gdf_lines, ax=ax)

        # Železnice 
        sym = "sym509a"
        zabaged_has = any(k in zabaged_gdfs for k in ["ZeleznicniTrat", "ZeleznicniVlecka"])
        cgdf = isom_gdfs.get("509")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=40, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif zabaged_has:
            if "ZeleznicniTrat" in zabaged_gdfs:
                plot_masked(sym_key=sym, zorder=40, mask=None, gdf=zabaged_gdfs["ZeleznicniTrat"], ax=ax, to_mask=False)
            if "ZeleznicniVlecka" in zabaged_gdfs:
                plot_masked(sym_key=sym, zorder=40, mask=None, gdf=zabaged_gdfs["ZeleznicniVlecka"], ax=ax, to_mask=False)
        elif "ZeleznicniTrat" not in zabaged_gdfs and "ZeleznicniVlecka" not in zabaged_gdfs:
            mask_railway = ((railway == "rail") | (railway == "disused") | (railway == "funicular") | (railway == "light-rail")  | (railway == "narrow_gauge")) & \
                           (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) & \
                           (~bridge.isin(["yes"])) 
            plot_masked(sym_key=sym, zorder=40, mask=mask_railway, gdf=gdf_lines, ax=ax)

        sym = "sym509b"
        zabaged_has = any(k in zabaged_gdfs for k in ["ZeleznicniTrat", "ZeleznicniVlecka"])
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=41, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif zabaged_has:
            if"ZeleznicniTrat" in zabaged_gdfs:
                plot_masked(sym_key=sym, zorder=41, mask=None, gdf=zabaged_gdfs["ZeleznicniTrat"], ax=ax, to_mask=False)
            if "ZeleznicniVlecka" in zabaged_gdfs:
                plot_masked(sym_key=sym, zorder=41, mask=None, gdf=zabaged_gdfs["ZeleznicniVlecka"], ax=ax, to_mask=False)
        elif "ZeleznicniTrat" not in zabaged_gdfs and "ZeleznicniVlecka" not in zabaged_gdfs:
            mask_railway = ((railway == "rail") | (railway == "disused") | (railway == "funicular") | (railway == "light-rail")  | (railway == "narrow_gauge")) & \
                           (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) & \
                           (~bridge.isin(["yes"])) 
            plot_masked(sym_key=sym, zorder=41, mask=mask_railway, gdf=gdf_lines, ax=ax)

        # Železniční most
        mask_bridge_railway = ((railway == "rail") | (railway == "disused") | (railway == "funicular") | (railway == "light-rail")  | (railway == "narrow_gauge")) & \
                              (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) & \
                              ((bridge == "yes")) 
        
        sym_list = ["sym509Ba", "sym509Bb", "sym509a", "sym509b"]
        zorders = [60, 61, 62, 63]
        for sym, z in zip(sym_list, zorders):
            cgdf = isom_gdfs.get(sym)
            if cgdf is not None:
                plot_masked(sym_key=sym, zorder=z, mask=None, gdf=cgdf, ax=ax, to_mask=False)
            else:
                plot_masked(sym_key=sym, zorder=z, mask=mask_bridge_railway, gdf=gdf_lines, ax=ax)

    # ======================================================================
    # POWER LINES & MAN MADE (Vedení a umělé objekty)
    # ======================================================================
    if visibility.get("man_made", True):
        # 510 - El. vedení (nízké napětí) / Lanovky
        sym = "sym510"
        zabaged_has = any(k in zabaged_gdfs for k in ["LanovaDrahaLyzarskyVlek", "ElektrickeVedeni"])
        cgdf = isom_gdfs.get("510")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=70, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif zabaged_has:
            if"ElektrickeVedeni" in zabaged_gdfs:
                mask = (~get_col(zabaged_gdfs["ElektrickeVedeni"], "napeti").isin(["400", "110"]))
                plot_masked(sym_key=sym, zorder=70, mask=mask, gdf=zabaged_gdfs["ElektrickeVedeni"], ax=ax)
            elif "LanovaDrahaLyzarskyVlek" in zabaged_gdfs:
                plot_masked(sym_key=sym, zorder=70, mask=None, gdf=zabaged_gdfs["LanovaDrahaLyzarskyVlek"], ax=ax, to_mask=False)
        elif "ElektrickeVedeni" not in zabaged_gdfs and "LanovaDrahaLyzarskyVlek" not in zabaged_gdfs:
            mask_cable_low = (power == "low") | (aerialway.isin(["line", "cable_car", "gondola", "mixed_lift", "chair_lift", "chair_lift", "drag_lift", "t-bar", "j-bar", "platter", "rope_tow", "zip_line", "goods"]))
            plot_masked(sym_key=sym, zorder=70, mask=mask_cable_low, gdf=gdf_lines, ax=ax)

        '''
        # 510P - Stožár
        sym = "sym511P"
        cgdf = isom_gdfs.get("511")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=70, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "StozarElektrickehoVedeni" in zabaged_gdfs:
            plot_masked(sym_key=sym, zorder=70, mask=None, gdf=zabaged_gdfs["StozarElektrickehoVedeni"], ax=ax, to_mask=False)
        elif "StozarLanoveDrahy" in zabaged_gdfs:
            plot_masked(sym_key=sym, zorder=70, mask=None, gdf=zabaged_gdfs["StozarLanoveDrahy"], ax=ax, to_mask=False)
        elif "StozarElektrickehoVedeni" not in zabaged_gdfs and "StozarLanoveDrahy" not in zabaged_gdfs:
            mask_cable_tower = (aerialway == "pylon") | (man_made == "utility_pole") | (power.isin(["tower", "pole"]))
            plot_masked(sym_key=sym, zorder=70, mask=mask_cable_tower, gdf=gdf_centroids, ax=ax)
        '''
        # 511 - VVN 
        sym = "sym510"
        cgdf = isom_gdfs.get("511")
        if cgdf is not None:
             pass 
        elif "ElektrickeVedeni" in zabaged_gdfs:
            mask = (get_col(zabaged_gdfs["ElektrickeVedeni"], "napeti").isin(["400", "110"]))
            plot_masked(sym_key=sym, zorder=70, mask=mask, gdf=zabaged_gdfs["ElektrickeVedeni"], ax=ax)
        else:
            mask_cable_high = (power.isin(["line", "minor_line"]))
            plot_masked(sym_key=sym, zorder=70, mask=mask_cable_high, gdf=gdf_lines, ax=ax)

        # 513.1 - Zeď (Složený symbol a+b)
        sym = "sym513-1a"
        zabaged_has = any(k in zabaged_gdfs for k in ["Zed", "PrehradniHrazJez", "Hrad"])
        cgdf = isom_gdfs.get("513.1")
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=30, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif zabaged_has:
            if "Zed" in zabaged_gdfs:
                plot_masked(sym_key=sym, zorder=30, mask=None, gdf=zabaged_gdfs["Zed"], ax=ax, to_mask=False)
            if "Hrad" in zabaged_gdfs:
                plot_masked(sym_key=sym, zorder=30, mask=None, gdf=zabaged_gdfs["Hrad"], ax=ax, to_mask=False)
            if "PrehradniHrazJez" in zabaged_gdfs:
                plot_masked(sym_key=sym, zorder=30, mask=None, gdf=zabaged_gdfs["PrehradniHrazJez"], ax=ax, to_mask=False)
        else:
            mask_wall_low = (barrier == "wall")
            plot_masked(sym_key=sym, zorder=30, mask=mask_wall_low, gdf=gdf_lines, ax=ax)

        sym = "sym513-1b"
        zabaged_has = any(k in zabaged_gdfs for k in ["Zed", "PrehradniHrazJez", "Hrad"])
        cgdf = isom_gdfs.get(513.2)
        if cgdf is not None:
            plot_masked(sym_key=sym, zorder=30, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif zabaged_has:
            if "Zed" in zabaged_gdfs:
                mask = (get_col(zabaged_gdfs["Zed"], "typzed_p").isin(["zeď opěrná"]))
                plot_masked(sym_key=sym, zorder=30, mask=None, gdf=zabaged_gdfs["Zed"], ax=ax, to_mask=False)
            if "Hrad" in zabaged_gdfs:
                plot_masked(sym_key=sym, zorder=30, mask=None, gdf=zabaged_gdfs["Hrad"], ax=ax, to_mask=False)
            if "PrehradniHrazJez" in zabaged_gdfs:
                plot_masked(sym_key=sym, zorder=30, mask=None, gdf=zabaged_gdfs["PrehradniHrazJez"], ax=ax, to_mask=False)                              
        else:
            mask_wall_low = (barrier == "wall")
            plot_masked(sym_key=sym, zorder=30, mask=mask_wall_low, gdf=gdf_lines, ax=ax)

        # 515 - Nepřekonatelná zeď (Složený symbol a+b)
        sym = "sym515a"
        cgdf = isom_gdfs.get(515)
        if cgdf is not None:
             plot_masked(sym_key=sym, zorder=30, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "Zed" in zabaged_gdfs:
            mask = (get_col(zabaged_gdfs["Zed"], "typzed_p").isin(["protihluková stěna", "zeď vodního díla", "zeď ostatní"]))
            plot_masked(sym_key=sym, zorder=30, mask=mask, gdf=zabaged_gdfs["Zed"], ax=ax)
        else:
            mask_wall_high = (barrier.isin(["city_wall"]))
            plot_masked(sym_key=sym, zorder=30, mask=mask_wall_high, gdf=gdf_lines, ax=ax)

        sym = "sym515b"
        if cgdf is not None:
             plot_masked(sym_key=sym, zorder=30, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "HradbaValBastaOpevneni" in zabaged_gdfs:
            plot_masked(sym_key=sym, zorder=30, mask=None, gdf=zabaged_gdfs["HradbaValBastaOpevneni"], ax=ax, to_mask=False)
        else:
            mask_wall_high = (barrier.isin(["city_wall"]))
            plot_masked(sym_key=sym, zorder=30, mask=mask_wall_high, gdf=gdf_lines, ax=ax)

        # 516–519 zatím neřešeno

        # ======================================================================
        # PRIVATE AREAS (Privát)
        # ======================================================================
        if visibility.get("private", True):  
            sym = "sym520"
            cgdf = isom_gdfs.get("520")
            
            if cgdf is not None:
                plot_masked(sym_key=sym, zorder=1.5, mask=None, gdf=cgdf, ax=ax, to_mask=False)
            else:
                garden_layers = ["Hrbitov", "Kolejiste", "Letiste", "OstatniPlochaVSidlech", 
                                "OvocnySadZahrada", "PovrchovaTezbaLom", "Elektrarna", 
                                "ArealUceloveZastavby", "Skladka"]
                gardens_not_from_zabaged = True
                
                for gl in garden_layers:
                    if gl in zabaged_gdfs:
                        plot_masked(sym_key=sym, zorder=1.5, mask=None, gdf=zabaged_gdfs[gl], ax=ax, to_mask=False)
                        gardens_not_from_zabaged = False
                        
                if gardens_not_from_zabaged:
                    mask_garden = (landuse.isin(["residential", "allotments", "brownfield", "military", "commercial", "construction", "industrial", "retail", "education", "animal_keeping", "cemetery", "landfill", "quarry", "depot", "religious", "farmyard"])) | \
                                (leisure == "pitch")
                    plot_masked(sym_key=sym, zorder=1.5, mask=mask_garden, gdf=gdf_polygons, ax=ax)
        # 523 - Zřícenina
        sym = "sym523"
        cgdf = isom_gdfs.get("523")
        if cgdf is not None:
             plot_masked(sym_key=sym, zorder=35, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "RozvalinaZricenina" in zabaged_gdfs:
            plot_masked(sym_key=sym, zorder=35, mask=None, gdf=zabaged_gdfs["RozvalinaZricenina"], ax=ax, to_mask=False)
        else:
            mask_ruin = (building == "ruins") | (historic == "ruins")
            plot_masked(sym_key=sym, zorder=35, mask=mask_ruin, gdf=gdf_polygons, ax=ax)

        # 524 - Vysoká věž 
        sym_a = "sym524a"
        sym_b = "sym524b"
        cgdf = isom_gdfs.get("524")
        if cgdf is not None:
            plot_masked(sym_key=sym_a, zorder=56, mask=None, gdf=cgdf, ax=ax, to_mask=False)
            plot_masked(sym_key=sym_b, zorder=56, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            zabaged_layers = ["Silo", "TezniVez", "TovarniKomin", "VetrnyMotor", "VetrnyMlyn", "VodojemVezovy", "VezovitaStavba"]
            found_tower_zabaged = False
            for tl in zabaged_layers:
                if tl in zabaged_gdfs:
                    plot_masked(sym_key=sym_a, zorder=56, mask=None, gdf=zabaged_gdfs[tl], ax=ax, to_mask=False)
                    plot_masked(sym_key=sym_b, zorder=56, mask=None, gdf=zabaged_gdfs[tl], ax=ax, to_mask=False)
                    found_tower_zabaged = True
            if not found_tower_zabaged:
                mask_tower_high = (man_made.isin(["tower", "transformer_tower", "water_tower", "communications_tower", "mast", "chimney", "crane", "flagpole", "obelisk"])) | \
                                  (historic == "round_tower")
                plot_masked(sym_key=sym_a, zorder=56, mask=mask_tower_high, gdf=gdf_points, ax=ax)
                plot_masked(sym_key=sym_b, zorder=56, mask=mask_tower_high, gdf=gdf_points, ax=ax)

        # 525 - Malá věž / Posed
        sym = "sym525"
        cgdf = isom_gdfs.get("525")
        if cgdf is not None:
             plot_masked(sym_key=sym, zorder=56, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            mask_tower_low = (man_made.isin(["column", "beacon", "lighthouse"])) | \
                             (amenity == "hunting_stand") | (building == "clock_tower")
            plot_masked(sym_key=sym, zorder=56, mask=mask_tower_low, gdf=gdf_points, ax=ax)

        # 526 - Pomník (Složený symbol a+b)
        sym_a = "sym526a"
        sym_b = "sym526b"
        cgdf = isom_gdfs.get("526") 
        if cgdf is not None:
             plot_masked(sym_key=sym_a, zorder=56, mask=None, gdf=cgdf, ax=ax, to_mask=False)
             plot_masked(sym_key=sym_b, zorder=56, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "MohylaPomnikNahrobek" in zabaged_gdfs:
            plot_masked(sym_key=sym_a, zorder=56, mask=None, gdf=zabaged_gdfs["MohylaPomnikNahrobek"], ax=ax, to_mask=False)
            plot_masked(sym_key=sym_b, zorder=56, mask=None, gdf=zabaged_gdfs["MohylaPomnikNahrobek"], ax=ax, to_mask=False)
        else:
            mask_memorial = (historic.isin(["boundary_stone", "memorial", "wayside_cross"])) | \
                            ((man_made.isin(["cross", "survey_point", "obelisk"])) & (~building.isin(["plaque"])))
            plot_masked(sym_key=sym_a, zorder=56, mask=mask_memorial, gdf=gdf_centroids, ax=ax)
            plot_masked(sym_key=sym_b, zorder=56, mask=mask_memorial, gdf=gdf_centroids, ax=ax)

        # 523 (Varianta) - Bunkr
        sym = "sym523"
        cgdf = isom_gdfs.get("523")
        if cgdf is not None:
             pass
        elif "Bunkr" in zabaged_gdfs:
            plot_masked(sym_key=sym, zorder=56, mask=None, gdf=zabaged_gdfs["Bunkr"], ax=ax, to_mask=False)
        else:
            mask_circle = (military == "bunker")
            plot_masked(sym_key=sym, zorder=56, mask=mask_circle, gdf=gdf_points, ax=ax)
            
        # 531 - Výrazný umělý objekt
        sym = "sym531"
        cgdf = isom_gdfs.get("531")
        if cgdf is not None:
             plot_masked(sym_key=sym, zorder=56, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "KrizSloupKulturnihoVyznamu" in zabaged_gdfs:
            plot_masked(sym_key=sym, zorder=56, mask=None, gdf=zabaged_gdfs["KrizSloupKulturnihoVyznamu"], ax=ax, to_mask=False)
        else:
            mask_man_made = (man_made.isin(["insect_hotel", "street_cabinet"]))
            plot_masked(sym_key=sym, zorder=56, mask=mask_man_made, gdf=gdf_centroids, ax=ax)

        # 532 - Schody (Složený symbol a,b,c)
        mask_stairs = (highway == "steps")
        sym_list = ["sym532a", "sym532b", "sym532c"]
        zorders = [49, 50, 51]
        for sym, z in zip(sym_list, zorders):
             cgdf = isom_gdfs.get("523")
             if cgdf is not None:
                 plot_masked(sym_key=sym, zorder=z, mask=None, gdf=cgdf, ax=ax, to_mask=False)
             else:
                 plot_masked(sym_key=sym, zorder=z, mask=mask_stairs, gdf=gdf_lines, ax=ax)
  
    
        
    # ======================================================================
    # BUILDINGS (Budovy)
    # ======================================================================
    if visibility.get("buildings", True):
        # 521 - Budova
        sym = "sym521"
        cgdf = isom_gdfs.get("521")
        if cgdf is not None:
             plot_masked(sym_key=sym, zorder=50, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        elif "BudovaJednotlivaNeboBlokBudov" in zabaged_gdfs:
            plot_masked(sym_key=sym, zorder=50, mask=None, gdf=zabaged_gdfs["BudovaJednotlivaNeboBlokBudov"], ax=ax, to_mask=False)
        else:
            mask_building = (building.notna()) & (building != '') & (~building.isin(["roof", "ruins"]))
            plot_masked(sym_key=sym, zorder=50, mask=mask_building, gdf=gdf_polygons, ax=ax)
            
        # 522 - Zastřešení
        sym = "sym522"
        cgdf = isom_gdfs.get("522")
        if cgdf is not None:
             plot_masked(sym_key=sym, zorder=36, mask=None, gdf=cgdf, ax=ax, to_mask=False)
        else:
            mask_roof = (building == "roof")
            plot_masked(sym_key=sym, zorder=36, mask=mask_roof, gdf=gdf_polygons, ax=ax)
    
    print("✅ Vše vykresleno")

def add_custom_isom_layers(ax, isom_gdfs):
    """
    Vykreslí vlastní uživatelské vrstvy pojmenované podle kódů ISOM.
    Tyto vrstvy se kreslí "navrch" (s vysokou prioritou) respektive s prioritou definovanou v XML.
    """
    print("Kreslím vlastní ISOM vrstvy...")
    already_handled = ["105", "215", "412", "413", "414", "417", "418", "502D", "502", "509", "513", "515", "524", "526"]

    for filename, gdf in isom_gdfs.items():
        base_name = filename.split('.')[0]
        if base_name in already_handled:
            continue
        # Zkoušíme najít klíč v SYMBOL_LIBRARY
        # 1. Přesná shoda (např. 'sym501' pokud se soubor jmenuje sym501.shp)
        if base_name in SYMBOL_LIBRARY:
            sym_key = base_name
        # 2. Přidání prefixu 'sym' (např. 'sym501' pokud se soubor jmenuje 501.shp)
        elif f"sym{base_name}" in SYMBOL_LIBRARY:
            sym_key = f"sym{base_name}"
        else:
            print(f"  -> Varování: Pro soubor '{filename}' nebyl nalezen odpovídající symbol v knihovně.")
            # Fallback vykreslení, aby uživatel viděl data aspoň nějak
            gdf.plot(ax=ax, color='red', linewidth=1, zorder=100)
            continue
            
        # Získání zorderu z definice symbolu, nebo defaultně vysoké číslo
        sym_def = SYMBOL_LIBRARY.get(sym_key, {})
        props = sym_def.get('props', {})
        # Pokud chceme, aby tyto vrstvy měly vyšší prioritu než standardní, 
        # můžeme k zorderu přičíst malou hodnotu nebo prostě použít definovaný zorder
        # a spoléhat na to, že se kreslí jako poslední.
        default_zorder = props.get('zorder', 50)
        
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
    
    # --- DATA EXTENT ---
    if paper_format == "Data Extent":
        # Spočítáme reálné rozměry dat v metrech
        data_width_m = maxx_orig - minx_orig
        data_height_m = maxy_orig - miny_orig
        
        # 1 palec = 0.0254 metru * Měřítko (10 000)
        # Tedy 1 palec na papíře = 254 metrů v terénu (při 1:10 000)
        meters_per_inch = 0.0254 * SCALE
        
        FIG_WIDTH_IN = data_width_m / meters_per_inch
        FIG_HEIGHT_IN = data_height_m / meters_per_inch
        
        print(f"Setting Custom Size (Data Extent): {data_width_m:.2f}×{data_height_m:.2f} m -> {FIG_WIDTH_IN:.2f}×{FIG_HEIGHT_IN:.2f} in")
        
        minx, maxx, miny, maxy = minx_orig, maxx_orig, miny_orig, maxy_orig
        
    else:
        # --- A4/A3 ---
        FIG_WIDTH_IN, FIG_HEIGHT_IN = PAPER_SIZES_IN.get(paper_format, PAPER_SIZES_IN["A4 (Landscape)"])
        
        map_width_meters = in2m(FIG_WIDTH_IN, SCALE)
        map_height_meters = in2m(FIG_HEIGHT_IN, SCALE)
        
        print(f"Setting paper format {paper_format} (1:{SCALE}): {map_width_meters:.2f}×{map_height_meters:.2f} m")

        # Najdeme střed původní datové oblasti
        center_x = (minx_orig + maxx_orig) / 2
        center_y = (miny_orig + maxy_orig) / 2
        
        # Vypočítáme nový rozsah (extent) pro plátno, vycentrovaný
        minx = center_x - (map_width_meters / 2)
        maxx = center_x + (map_width_meters / 2)
        miny = center_y - (map_height_meters / 2)
        maxy = center_y + (map_height_meters / 2)

    # Vytvoření plátna
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    
    ax.set_aspect('equal') 
    ax.axis('off') 
    
    extent_final = (minx, maxx, miny, maxy)
    return fig, ax, extent_final


def run_main_analysis():
    global SCALE, SYMBOL_LIBRARY, CURRENT_CRS  
    selected_crs_label = crs_var.get()
    if selected_crs_label in CRS_MAP:
        CURRENT_CRS = CRS_MAP[selected_crs_label]
    else:
        CURRENT_CRS = selected_crs_label
    if not CURRENT_CRS:
        CURRENT_CRS = "EPSG:5514"
    print(f"--- START ANALÝZY ---")
    print(f"Vybraný systém: {selected_crs_label}")
    print(f"Kód CRS pro výpočet: {CURRENT_CRS}")
    selected_scale = scale_var.get()
    if selected_scale == "1:10 000":
        SCALE = 10000
        xml_file = "symbols10.xml"
    else: # Default 1:15 000
        SCALE = 15000
        xml_file = "symbols15.xml"
        
    print(f"Nastaveno CRS: {CURRENT_CRS}")    
    print(f"Nastaveno měřítko 1:{SCALE}. Načítám knihovnu: {xml_file}")
    
    # Znovu načtení knihovny symbolů podle vybraného souboru
    SYMBOL_LIBRARY = load_symbol_library(xml_file)
    
    if not SYMBOL_LIBRARY:
        messagebox.showerror("Chyba", f"Nepodařilo se načíst soubor symbolů: {xml_file}\nZkontrolujte, zda soubor existuje ve složce.")
    dmr_path = dmr_entry.get()
    dmp_path = dmp_entry.get()
    should_save_png = save_var.get()
    should_save_vector = save_vector_var.get()
    selected_paper_format = paper_format_var.get()
    zabaged_paths = zabaged_listbox.get(0, tk.END) 
    isom_paths = isom_listbox.get(0, tk.END)
    FIXED_PIXEL_SIZE = 0.5
    if not dmr_path or not dmp_path:
        messagebox.showerror("Chyba vstupu", "Musíte vyplnit povinné parametry:\n\n- DMR\n- DMP")
        status_label.config(text="Chyba: Chybí povinné vstupy.", foreground="red")
        return
        
    try:
        progress_bar["value"] = 0
        
        try:
            b1 = float(bin_entry_1.get())
            b2 = float(bin_entry_2.get())
            b3 = float(bin_entry_3.get())
            b4 = float(bin_entry_4.get())
            bins = [-1, 0, b1, b2, b3, b4]
            print(f"Používám vlastní hranice klasifikace: {bins}")
        except ValueError:
            messagebox.showwarning("Chyba vstupu", "Neplatné hodnoty v 'Nastavení klasifikace'. Používám výchozí hodnoty (0.5, 2, 5, 11).")
            b1, b2, b3, b4 = 0.5, 2.0, 5.0, 11.0
            bins = [-1, 0, b1, b2, b3, b4]

        try:
            dmr_smooth = float(contour_smooth.get().replace(",", "."))
        except ValueError: dmr_smooth = 6.5
        
        dmr_grid_cubic, grid_x, grid_y, extent, dmr_points, dmr_z = load_dmr_grid(dmr_path, target_crs_code=CURRENT_CRS, pixel_size=FIXED_PIXEL_SIZE, sigma_smooth=dmr_smooth)
        (minx, maxx, miny, maxy) = extent
        status_label.config(text="Vytvářím ořezovou masku...")
        root.update_idletasks()
        try:
            # Vytvoříme polygon z vnějšího obrysu všech DMR bodů
            clip_polygon = MultiPoint(dmr_points).convex_hull
            if not clip_polygon.is_valid:
                print("  -> Varování: Convex hull není validní, zkouším buffer(0)")
                clip_polygon = clip_polygon.buffer(0)
            print("  -> Ořezová maska (Convex Hull) vytvořena.")
        except Exception as e:
            print(f"  -> CHYBA při tvorbě ořezové masky: {e}. Maskování bude přeskočeno.")
            clip_polygon = None
        
        print("Interpoluji mřížku  pro skály...")
        status_label.config(text="Interpoluji DMR mřížku (Linear)...")
        root.update_idletasks()
        
        dmr_grid_linear = griddata(dmr_points, dmr_z, (grid_x, grid_y), method='linear')
        
        if np.isnan(dmr_grid_linear).all():
            print("Fallback (DMR-Linear): 'Linear' selhala, používám 'nearest'.")
            dmr_grid_linear = griddata(dmr_points, dmr_z, (grid_x, grid_y), method='nearest')
        
        print("DMR mřížky (Cubic a Linear) připraveny.")
        
        progress_bar["value"] = 10
        
        dmp_grid = load_dmp_grid(dmp_path, grid_x, grid_y, extent, target_crs_code=CURRENT_CRS)
        progress_bar["value"] = 20
        
        status_label.config(text="Počítám výšku vegetace...")
        root.update_idletasks()
        vegetation_height = np.clip(dmp_grid - dmr_grid_linear, 0, None)
        progress_bar["value"] = 25
        
        # ---------------------------------------------------------
        # 3. Načtení Vlastních ISOM vrstev s ošetřením CRS
        # ---------------------------------------------------------
        isom_gdfs = {}
        if isom_paths:
            print(f"Nalezeno {len(isom_paths)} vlastních ISOM souborů...")
            status_label.config(text="Načítám a transformuji ISOM vrstvy...")
            root.update_idletasks()
            
            for path in isom_paths:
                filename = os.path.basename(path)
                try:
                    isom_gdf = gpd.read_file(path)
                    
                    if not isom_gdf.empty:
                        if isom_gdf.crs is None:
                            print(f"⚠️ Varování: Vrstva '{filename}' nemá definovaný CRS. Předpokládám {CURRENT_CRS}.")
                            isom_gdf.set_crs(CURRENT_CRS, allow_override=True, inplace=True)
                        
                        elif isom_gdf.crs != CURRENT_CRS:
                            print(f"  -> Transformuji '{filename}' z {isom_gdf.crs.to_string()} do {CURRENT_CRS}...")
                            try:
                                isom_gdf = isom_gdf.to_crs(CURRENT_CRS)
                            except Exception as e:
                                print(f"❌ Chyba transformace CRS u {filename}: {e}")
                                continue

                        # --- OŘEZ NA ROZSAH MAPY (CLIP) ---
                        if clip_polygon is not None:
                            try:
                                isom_gdf['geometry'] = isom_gdf.geometry.buffer(0)
                                isom_gdf = gpd.clip(isom_gdf, clip_polygon)
                            except Exception as e:
                                print(f"  -> Varování: Ořez vrstvy {filename} selhal ({e}), používám celou vrstvu.")

                        if not isom_gdf.empty:
                            # Příprava klíčů: "502a.shp" -> "502a"
                            key_name = filename.rsplit('.', 1)[0]
                            
                            # Uložíme pod čistým názvem (bez přípony)
                            isom_gdfs[key_name] = isom_gdf
                            # Pro jistotu uložíme i pod celým názvem
                            isom_gdfs[filename] = isom_gdf
                            
                            print(f"  -> Načteno: {filename} ({len(isom_gdf)} prvků)")
                        else:
                            print(f"  -> Varování: Vrstva {filename} je po ořezu prázdná.")

                except Exception as e:
                    print(f"❌ Chyba při načítání souboru {filename}: {e}")
        
        # Vytvoříme polygon z vnějšího obrysu všech DMR bodů
        print("Vytvářím ořezovou masku...")
        status_label.config(text="Vytvářím ořezovou masku...")
        root.update_idletasks()
        
        clip_polygon = MultiPoint(dmr_points).convex_hull
        if not clip_polygon.is_valid:
            print("  -> Varování: Convex hull není validní, zkouším buffer(0)")
            clip_polygon = clip_polygon.buffer(0)

        print("  -> Ořezová maska (Convex Hull) vytvořena.")

        # Stahování OSM dat
        print("Stahuji OSM data (základ)...")
        status_label.config(text="Stahuji OSM data...")
        try:
            print("Konfiguruji OSMnx (user_agent, cache)...")
            ox.settings.log_console = True
            ox.settings.use_cache = True
            ox.settings.user_agent = "OMapMaker-App-v4.5" 
        except Exception as e:
            print(f"Varování: Nelze nastavit osmnx: {e}")

        root.update_idletasks()
        gdf_osm = None
        try:
            download_buffer = 300
            download_minx = minx - download_buffer 
            download_maxx = maxx + download_buffer 
            download_miny = miny - download_buffer 
            download_maxy = maxy + download_buffer
            
            # Transformace z vybraného CURRENT_CRS do WGS84 (EPSG:4326) pro OSM
            to_wgs = Transformer.from_crs(CURRENT_CRS, "EPSG:4326", always_xy=True)
            minlon, minlat = to_wgs.transform(download_minx, download_miny)
            maxlon, maxlat = to_wgs.transform(download_maxx, download_maxy)
            
            tags = {
                "highway": True, "building": True, "waterway": True, "access": True, 
                "aerialway": True, "amenity": True, "barrier": True, "bridge": True, 
                "covered": True, "emergency": True, "geological": True, "historic": True, 
                "intermittent": True, "landuse": True, "leisure": True, "man_made": True, 
                "military": True, "natural": True, "parking": True, "place": True, 
                "power": True, "railway": True, "tunnel": True, "tracktype": True, 
                "trail_visibility": True, "surface": True, "water": True, "wetland": True
            } #TODO: UPŘESNIT VÝBĚR POUŽITÝCH TAGŮ
            bbox = (minlon, minlat, maxlon, maxlat) 
            gdf_osm = ox.features_from_bbox(bbox, tags=tags)
        
            gdf_osm = gdf_osm.to_crs(CURRENT_CRS)
            print(f"OSM data stažena a převedena do {CURRENT_CRS}.")
        except Exception as e:
            print(f"Chyba při stahování OSM: {e}")
            status_label.config(text="Chyba stahování OSM.", fg="red")
        except Exception as e:
            print(f"Chyba při stahování OSM: {e}")
            status_label.config(text="Chyba stahování OSM.", fg="red")
        
        try:
            print("Ořezávám hlavní GDF (OSM/ZABAGED) na ořezovou masku...")
            status_label.config(text="Ořezávám vektory (Data)...")
            root.update_idletasks()
            if gdf_osm is not None and not gdf_osm.empty:
                try:
                    # Ořez GDF na convex hull
                    gdf_osm = gpd.clip(gdf_osm, clip_polygon)
                except Exception as e:
                    print(f"  -> Varování: Selhal ořez gdf_osm: {e}")

        except Exception as e:
            print(f"  -> CHYBA při tvorbě/ořezu ořezové masky: {e}. Maskování vektorů bude přeskočeno.")
            clip_polygon = None

        # 2. Načtení ZABAGED (pokud je)
        zabaged_gdfs = {}
        if zabaged_paths: 
            print(f"Nalezeno {len(zabaged_paths)} souborů ZABAGED...")
            status_label.config(text="Načítám ZABAGED (Smart Load)...")
            root.update_idletasks()
            
            # Vytvoříme BBOX vašeho území v cílovém CRS
            # (minx, miny, maxx, maxy) už máte z load_dmr_grid
            target_bbox = box(minx, miny, maxx, maxy)
            
            for path in zabaged_paths:
                filename = os.path.basename(path)
                status_label.config(text=f"Načítám {filename}...")
                root.update_idletasks()

                try:
                    # 1. Zjistíme CRS souboru, aniž bychom ho celý četli
                    with fiona.open(path) as src:
                        file_crs_wkt = src.crs_wkt
                        if not file_crs_wkt: 
                            file_crs = "EPSG:5514"
                        else:
                            file_crs = file_crs_wkt
                    
                    # 2. Přepočítáme NÁŠ ořezový obdélník do CRS toho souboru
                    bbox_for_loading = target_bbox
                    
                    try:
                        crs_src = CRS.from_user_input(file_crs)
                        crs_dst = CRS.from_user_input(CURRENT_CRS)
                        
                        if crs_src != crs_dst:
                            transformer_bbox = Transformer.from_crs(crs_dst, crs_src, always_xy=True)
                            b_minx, b_miny, b_maxx, b_maxy = target_bbox.bounds
                            
                            xs = [b_minx, b_maxx]
                            ys = [b_miny, b_maxy]
                            tx, ty = transformer_bbox.transform(xs, ys)
                            
                            file_bbox_tuple = (min(tx), min(ty), max(tx), max(ty))
                        else:
                            file_bbox_tuple = target_bbox.bounds
                            
                    except Exception as e:
                        print(f"  -> Chyba transformace bboxu: {e}, načtu vše.")
                        file_bbox_tuple = None

                    # 3. Načteme JEN data v bboxu (obrovské zrychlení)
                    if file_bbox_tuple:
                        zabaged_gdf = gpd.read_file(path, bbox=file_bbox_tuple)
                    else:
                        zabaged_gdf = gpd.read_file(path) # Fallback

                    # 4. Teď teprve transformujeme načtená data do našeho CRS
                    if not zabaged_gdf.empty:
                        if zabaged_gdf.crs != CURRENT_CRS:
                            zabaged_gdf = zabaged_gdf.to_crs(CURRENT_CRS)
                        
                        if clip_polygon:
                            zabaged_gdf = gpd.clip(zabaged_gdf, clip_polygon)

                    zabaged_gdfs[filename.rsplit(".", 1)[0]] = zabaged_gdf
                    
                except Exception as e:
                    print(f"❌ Chyba při načítání {filename}: {e}")

                root.update_idletasks()
        
        shape = grid_x.shape 
        transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width=shape[0], height=shape[1])

        forest_mask = np.zeros(shape, dtype=np.uint8)
        if gdf_osm is not None and not gdf_osm.empty:
            forest_polys = gdf_osm[
                (get_col(gdf_osm, 'natural') == 'wood') | 
                (get_col(gdf_osm, 'landuse') == 'forest')
            ].geometry
            if not forest_polys.empty:
                print(f"Nalezeno {len(forest_polys)} lesních polygonů, rasterizuji masku...")
                forest_mask_transposed = rasterize(
                    forest_polys, 
                    out_shape=(shape[1], shape[0]), 
                    transform=transform,
                    fill=0, default_value=1, dtype=np.uint8
                )
                forest_mask = np.flipud(forest_mask_transposed).T
                print("Maska lesů vytvořena.")
        
        status_label.config(text="Detekuji paseky...")
        root.update_idletasks()
        
        is_meadow = (vegetation_height < b1) & (vegetation_height >= 0)
        is_forest = (forest_mask == 1) 
        is_clearing = is_meadow & is_forest
        print(f"Nalezeno {np.sum(is_clearing)} pixelů pasek (hranice < {b1}m).")
        vegetation_height[is_clearing] = -1
        
        progress_bar["value"] = 45

        class_names = {
            2: 'Louka',  1: 'Paseka', 6: 'Les', 5: 'Vysoky_porost', 4: 'Stredni_porost',  3: 'Nizky_porost', 0 : 'Mimo_data'
        }
        color_map = {
            'Paseka': '#ffdd9a',
            'Louka': '#ffba35',
            'Vysoky_porost': '#c3ed9a',
            'Stredni_porost': '#4cc74c',
            'Nizky_porost': '#0e990e',
            'Les': '#ffffff' 
        }
        
        vec_raster_input = np.nan_to_num(vegetation_height, nan=-9999)
        classified_raster = np.digitize(vec_raster_input, bins).astype(np.int32)
        # --- PŘIDANÝ KÓD: MASKOVÁNÍ RASTROVÝCH DAT ---
        if clip_polygon:
            print("Rasterizuji ořezovou masku (Convex Hull) pro gridy...")
            status_label.config(text="Rasterizuji ořezovou masku...")
            root.update_idletasks()
            try:
                clip_geoms = [(clip_polygon, 1)]
                clip_mask_transposed = rasterize(
                    clip_geoms,
                    out_shape=(shape[1], shape[0]), # (height, width)
                    transform=transform,
                    fill=0,
                    default_value=1,
                    dtype=np.uint8
                )
                clip_mask_grid = np.flipud(clip_mask_transposed).T.astype(bool)
                print("Aplikuji ořezovou masku na gridy...")
                classified_raster[~clip_mask_grid] = 0
                dmr_grid_linear_viz_mask_input = np.nan_to_num(dmr_grid_linear, nan=0)
                dmr_grid_linear_viz_mask_input[~clip_mask_grid] = 0
                dmr_grid_cubic_viz_mask_input = np.nan_to_num(dmr_grid_cubic, nan=0)
                dmr_grid_cubic_viz_mask_input[~clip_mask_grid] = np.nan
                dmr_grid_linear_viz_features = np.nan_to_num(dmr_grid_linear, nan=0)
                dmr_grid_linear_viz_features[~clip_mask_grid] = 0
            except Exception as e:
                print(f"  -> CHYBA: Selhalo maskování gridů: {e}")
                # Pokud selže, použijeme nemaskovaná data
                dmr_grid_linear_viz_mask_input = np.nan_to_num(dmr_grid_linear, nan=0)
                dmr_grid_cubic_viz_mask_input = np.nan_to_num(dmr_grid_cubic, nan=0)
                dmr_grid_linear_viz_features = np.nan_to_num(dmr_grid_linear, nan=0)
        else:
            # Pokud nebyl polygon (chyba v kroku 1), použijeme nemaskovaná data
            dmr_grid_linear_viz_mask_input = np.nan_to_num(dmr_grid_linear, nan=0)
            dmr_grid_cubic_viz_mask_input = np.nan_to_num(dmr_grid_cubic, nan=0)
            dmr_grid_linear_viz_features = np.nan_to_num(dmr_grid_linear, nan=0)
        gdf_vegetation = vectorize_vegetation(
            classified_raster, class_names, transform, dmr_path,
            save_gpkg=should_save_vector
        )
        progress_bar["value"] = 60
        
        dmr_grid_cubic_viz = dmr_grid_cubic_viz_mask_input
        dmr_grid_linear_viz = dmr_grid_linear_viz_mask_input
        
        try:
            rock_slope_deg = float(slope_threshold_entry.get().replace(",", "."))
        except ValueError:
            rock_slope_deg = 27.0
        
        gdf_rocks = vectorize_rocks(
            grid_x, grid_y, dmr_grid_linear_viz, transform, slope_threshold_deg=rock_slope_deg
        )
        progress_bar["value"] = 70
        
        # ... (zde končí předchozí část: vektorizace skal, progress_bar["value"] = 70) ...

        if not should_save_png:
            print("Vektorizace dokončena. Generování PNG přeskočeno.")
            status_label.config(text="✅ Vektorizace hotova.", fg="darkgreen")
            progress_bar["value"] = 100
            root.after(2000, lambda: status_label.config(text="Done.", foreground="darkgreen"))
            return 

        # --- GENEROvÁNÍ MAPY ---
        status_label.config(text="Generuji kompozici mapy...")
        root.update_idletasks()
        
        # 1. Nastavení plátna (jen jednou)
        fig, ax, map_extent = setup_map_figure(extent, selected_paper_format)
        (map_minx, map_maxx, map_miny, map_maxy) = map_extent
        
        # Ořezový box pro data
        clip_box_map = box(map_minx, map_miny, map_maxx, map_maxy)

        # --- SEVEROJIŽNÍ ČÁRY ---
        if LAYER_VISIBILITY["magnetic_lines"].get():
            # Načtení hodnoty z GUI
            try:
                rot_str = north_rotation_entry.get().replace(",", ".")
                north_rotation = float(rot_str)
            except ValueError:
                print("Neplatná hodnota rotace, používám 0°.")
                north_rotation = 0.0
            add_magnetic_north_lines(ax, map_extent, SCALE, rotation=north_rotation, spacing_mm=30, zorder=50)
        # Aplikace ořezové masky (Convex Hull) na celé plátno
        if clip_polygon:
            try:
                hull_coords = np.array(clip_polygon.exterior.coords)
                clip_patch = MplPolygon(hull_coords, transform=ax.transData)
                ax.set_clip_path(clip_patch)
            except Exception as e:
                print(f"  -> CHYBA: Nepodařilo se aplikovat ořezovou masku: {e}")

        # 2. Kreslení VEGETACE
        if LAYER_VISIBILITY["vegetation"].get():
            status_label.config(text="Kreslím vegetaci...")
            root.update_idletasks()
            
            if not gdf_vegetation.empty:
                try:
                    veg_data_only = gdf_vegetation[gdf_vegetation['class_name'] != 'Mimo_data']
                    if not veg_data_only.empty:
                        veg_clipped = gpd.clip(veg_data_only, clip_box_map)
                        if not veg_clipped.empty:
                            plot_order = [
                                (['Paseka', 'Louka'], 1.0),
                                (['Les'], 1.1),
                                (['Vysoky_porost'], 1.2),
                                (['Stredni_porost'], 1.3),
                                (['Nizky_porost'], 1.4)
                            ]
                            for class_list, z_level in plot_order:
                                subset_gdf = veg_clipped[veg_clipped['class_name'].isin(class_list)]
                                if not subset_gdf.empty:
                                    subset_colors = subset_gdf['class_name'].map(color_map).fillna('#FF00FF')
                                    subset_gdf.plot(ax=ax, color=subset_colors, zorder=z_level)
                except Exception as e:
                    print(f"  -> Chyba při kreslení vegetace: {e}")

        # 3. Kreslení SKAL A VRSTEVNIC
        if LAYER_VISIBILITY["rocks"].get():
            if not gdf_rocks.empty:
                try:
                    rocks_clipped = gpd.clip(gdf_rocks, clip_box_map)
                    if not rocks_clipped.empty:
                        rocks_clipped.plot(ax=ax, color='black', zorder=26)
                except Exception as e:
                    print(f"  -> Chyba při kreslení skal: {e}")

            status_label.config(text="Kreslím vrstevnice...")
            root.update_idletasks()
            # Předáváme fixní pixel size
            add_contour_lines(ax, grid_x, grid_y, dmr_grid_cubic_viz) 
            
            # 5. Kreslení PRVKŮ (Kupky, prohlubně)
            try: knoll_h = float(knoll_height_entry.get().replace(",", "."))
            except ValueError: knoll_h = 0.8
            
            try: dep_depth = float(dep_depth_entry.get().replace(",", "."))
            except ValueError: dep_depth = 0.3
            
            status_label.config(text="Kreslím terénní detaily...")
            root.update_idletasks()
            add_depressions(ax, grid_x, grid_y, dmr_grid_linear_viz_features, 
                            pixel_size=FIXED_PIXEL_SIZE, 
                            min_diameter=0.5, max_diameter=3, min_depth=dep_depth)
                          
            add_knoll_symbols(ax, grid_x, grid_y, dmr_grid_linear_viz_features, 
                            pixel_size=FIXED_PIXEL_SIZE, min_height=knoll_h)
        
        # 6. Kreslení VEKTORŮ (OSM/Zabaged)
        status_label.config(text="Kreslím cesty a objekty...")
        root.update_idletasks()
        visibility_settings = {k: v.get() for k, v in LAYER_VISIBILITY.items()}
        if gdf_osm is not None and not gdf_osm.empty:
            add_vector_layers(ax, gdf_osm.copy(), map_extent, zabaged_gdfs, dmr_grid_linear_viz_features, grid_x, grid_y, visibility=visibility_settings, isom_gdfs=isom_gdfs)

        # 7. Kreslení Vlastních ISOM vrstev (Nejvyšší priorita)
        if isom_gdfs:
            add_custom_isom_layers(ax, isom_gdfs)

        # 8. Uložení FINÁLNÍHO SOUBORU
        output_path = os.path.splitext(dmr_path)[0] + "_OMap.png"
        status_label.config(text="Ukládám PNG soubor...")
        root.update_idletasks()
        print(f"Ukládám finální mapu: {output_path} (DPI=300)")
        paper_format = paper_format_var.get()
        if paper_format == "Data Extent":
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(output_path, dpi=1000, bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close(fig) 
        else:    
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(output_path, dpi=1000, bbox_inches='tight', pad_inches=0, transparent=False)
            plt.close(fig) 
        
        progress_bar["value"] = 95

        # 9. Generování WORLD FILE (.pgw)
        print(f"Generuji World File...")
        data_width_m = map_maxx - map_minx
        data_height_m = map_maxy - map_miny
        
        try:
            with Image.open(output_path) as img:
                img_width_px = img.width
                img_height_px = img.height

            pixel_size_x = data_width_m / img_width_px
            pixel_size_y = data_height_m / img_height_px
            
            # Střed levého horního pixelu
            x_center = map_minx + (pixel_size_x / 2.0)
            y_center = map_maxy - (pixel_size_y / 2.0) 
            
            world_file_content = (
                f"{pixel_size_x}\n"  
                f"0.0\n"              
                f"0.0\n"              
                f"{-pixel_size_y}\n" 
                f"{x_center}\n"       
                f"{y_center}\n"       
            )
            world_file_path = os.path.splitext(output_path)[0] + ".pgw"
            with open(world_file_path, "w") as f:
                f.write(world_file_content)
            print(f"✅ World File uložen: {world_file_path}")
        except Exception as e:
            print(f"⚠️ Chyba při tvorbě World File: {e}")

        status_label.config(text="✅ Hotovo.", fg="darkgreen")
        progress_bar["value"] = 100
        root.after(3000, lambda: status_label.config(text="Připraveno."))

    except Exception as e:
        messagebox.showerror("Chyba analýzy", str(e))
        status_label.config(text=f"❌ Chyba: {str(e)}", fg="red")
        progress_bar["value"] = 0


print("--- OMapMaker v5: Spouštím GUI ---")
SCALE = 15000 
MAX_GRIDDATA_POINTS = 2_500_000
SYMBOL_LIBRARY = {} 

LAS_FILES = [("Lidar data", "*.las *.laz"), ("Všechny soubory", "*.*")]
SHP_FILES = [("Shapefile", "*.shp"), ("Všechny soubory", "*.*")]
LAS_TIF_FILES = [
    ("Podporovaná data", "*.las *.laz *.tif *.tiff"),
    ("Lidar data", "*.las *.laz"),
    ("GeoTIFF", "*.tif *.tiff"),
    ("Všechny soubory", "*.*")
]
root = tk.Tk()
root.title("OMapMaker")
root.state('zoomed')
paper_format_var = tk.StringVar()
paper_format_options = ["Data Extent", "A3 (Landscape)", "A3 (Portrait)", "A4 (Landscape)", "A4 (Portrait)"]
main_frame = ttk.Frame(root, padding="15")
main_frame.pack(fill=tk.BOTH, expand=True)
required_frame = ttk.Labelframe(main_frame, text="LIDAR data (.las/.laz)", padding="10")
required_frame.pack(fill=tk.X, expand=False, pady=(0, 10))
ttk.Label(required_frame, text="Select Digital Terrain Model (DTM):").pack(anchor="w", padx=5)
dmr_entry_frame = ttk.Frame(required_frame)
dmr_entry_frame.pack(fill=tk.X, expand=True, padx=5, pady=(0, 10))
dmr_entry = ttk.Entry(dmr_entry_frame, width=70) 
dmr_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
ttk.Button(dmr_entry_frame, text="...", width=4, 
           command=lambda: select_file(dmr_entry, "Sellect DtM (.las)", LAS_FILES)).pack(side=tk.RIGHT, padx=(5, 0))
ttk.Label(required_frame, text="Select Digital Surface Model (DSM)):").pack(anchor="w", padx=5)
dmp_entry_frame = ttk.Frame(required_frame)
dmp_entry_frame.pack(fill=tk.X, expand=True, padx=5, pady=(0, 10))
dmp_entry = ttk.Entry(dmp_entry_frame, width=70)
dmp_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
ttk.Button(dmp_entry_frame, text="...", width=4, 
           command=lambda: select_file(dmp_entry, "Sellect DSM (.las or .tif)", LAS_TIF_FILES)).pack(side=tk.RIGHT, padx=(5, 0))
middle_container = ttk.Frame(main_frame)
middle_container.pack(fill=tk.BOTH, expand=True, pady=5)
middle_container.columnconfigure(0, weight=3) 
middle_container.columnconfigure(1, weight=1)
middle_container.rowconfigure(0, weight=1) 
optional_frame = ttk.Labelframe(middle_container, text="Add your files (.shp)", padding="10")
optional_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5)) 
ttk.Label(optional_frame, text="Data from ZABAGED© (.shp)").pack(anchor="w", padx=5)
zabaged_list_frame = ttk.Frame(optional_frame)
zabaged_list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))
zabaged_scrollbar = ttk.Scrollbar(zabaged_list_frame, orient=tk.VERTICAL)
zabaged_listbox = tk.Listbox(
    zabaged_list_frame, 
    height=4, 
    selectmode=tk.EXTENDED, 
    yscrollcommand=zabaged_scrollbar.set
)
zabaged_scrollbar.config(command=zabaged_listbox.yview)
zabaged_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
zabaged_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
zabaged_buttons_frame = ttk.Frame(optional_frame)
zabaged_buttons_frame.pack(fill=tk.X, padx=5, pady=(0, 10))
ttk.Button(zabaged_buttons_frame, text="Add files...", 
           command=lambda: select_multiple_files_to_listbox(zabaged_listbox)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
ttk.Button(zabaged_buttons_frame, text="Remove selected", 
           command=lambda: remove_selected_from_listbox(zabaged_listbox)).pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(5, 0))

# --- ISOM Custom Layers Listbox ---
ttk.Label(optional_frame, text="ISOM Custom Layers (Filename = ISOM Code, e.g. 501.shp)").pack(anchor="w", padx=5)
isom_list_frame = ttk.Frame(optional_frame)
isom_list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))
isom_scrollbar = ttk.Scrollbar(isom_list_frame, orient=tk.VERTICAL)
isom_listbox = tk.Listbox(
    isom_list_frame, 
    height=4, 
    selectmode=tk.EXTENDED, 
    yscrollcommand=isom_scrollbar.set
)
isom_scrollbar.config(command=isom_listbox.yview)
isom_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
isom_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
isom_buttons_frame = ttk.Frame(optional_frame)
isom_buttons_frame.pack(fill=tk.X, padx=5, pady=(0, 10))
ttk.Button(isom_buttons_frame, text="Add ISOM layers...", 
           command=lambda: select_multiple_files_to_listbox(isom_listbox)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
ttk.Button(isom_buttons_frame, text="Remove selected", 
           command=lambda: remove_selected_from_listbox(isom_listbox)).pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(5, 0))

settings_frame = ttk.Labelframe(middle_container, text="Settings:", padding="10")
settings_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
settings_frame.columnconfigure(1, weight=1)
ttk.Label(settings_frame, text="Vegetation height:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
ttk.Label(settings_frame, text="Open Land/Rough Open Land:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
bin_entry_1 = ttk.Entry(settings_frame)
bin_entry_1.insert(0, "1")
bin_entry_1.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
ttk.Label(settings_frame, text="Vegetation: fight").grid(row=2, column=0, sticky="w", padx=5, pady=5)
bin_entry_2 = ttk.Entry(settings_frame)
bin_entry_2.insert(0, "1.3")
bin_entry_2.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
ttk.Label(settings_frame, text="Vegetation: walk").grid(row=3, column=0, sticky="w", padx=5, pady=5)
bin_entry_3 = ttk.Entry(settings_frame)
bin_entry_3.insert(0, "6")
bin_entry_3.grid(row=3, column=1, sticky="ew", padx=5, pady=5)
ttk.Label(settings_frame, text="Vegetation: slow running").grid(row=4, column=0, sticky="w", padx=5, pady=5)
bin_entry_4 = ttk.Entry(settings_frame)
bin_entry_4.insert(0, "12")
bin_entry_4.grid(row=4, column=1, sticky="ew", padx=5, pady=5)
ttk.Label(settings_frame, text="Paper format:").grid(row=5, column=0, sticky="w", padx=5, pady=(25, 5))
paper_format_combo = ttk.Combobox(
    settings_frame,
    textvariable=paper_format_var,
    values=paper_format_options,
    state="readonly" 
)
paper_format_combo.grid(row=5, column=1, sticky="ew", padx=5, pady=(15, 5))
paper_format_combo.set(paper_format_options[3]) # Výchozí hodnota A4
ttk.Label(settings_frame, text="Scale:").grid(row=7, column=0, sticky="w", padx=5, pady=5)
scale_var = tk.StringVar(value="1:10 000") # Výchozí hodnota
scale_combo = ttk.Combobox(
    settings_frame,
    textvariable=scale_var,
    values=["1:10 000", "1:15 000"],
    state="readonly"
)
# --- PŘIDANÁ ČÁST PRO VÝBĚR CRS ---
ttk.Label(settings_frame, text="Cordinate system:").grid(row=6, column=0, sticky="w", padx=5, pady=5)
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

crs_var = tk.StringVar(value="UTM Zone 33N (Czechia, Sweden, Slovenia, South-east of Italy) [EPSG:32633]") 

crs_combo = ttk.Combobox(
    settings_frame,
    textvariable=crs_var,
    values=list(CRS_MAP.keys()),
    state="readonly"
)
crs_combo.grid(row=6, column=1, sticky="ew", padx=5, pady=5)
scale_combo.grid(row=7, column=1, sticky="ew", padx=5, pady=5)
controls_frame = ttk.Frame(main_frame, padding="10")
controls_frame.pack(fill=tk.X, expand=False, pady=(10, 0))
save_var = tk.BooleanVar(value=True)
ttk.Label(settings_frame, text="Magnetic declination (°):").grid(row=8, column=0, sticky="w", padx=5, pady=5)
north_rotation_entry = ttk.Entry(settings_frame)
north_rotation_entry.insert(0, "5")
north_rotation_entry.grid(row=8, column=1, sticky="ew", padx=5, pady=5)
ttk.Label(settings_frame, text="Other settings:").grid(row=9, column=0, sticky="w", padx=5, pady=5)
ttk.Label(settings_frame, text="Rock slope threshold (°):").grid(row=10, column=0, sticky="w", padx=5, pady=5)
slope_threshold_entry = ttk.Entry(settings_frame)
slope_threshold_entry.insert(0, "27")
slope_threshold_entry.grid(row=10, column=1, sticky="ew", padx=5, pady=5)

ttk.Label(settings_frame, text="Minimal knoll height (m):").grid(row=11, column=0, sticky="w", padx=5, pady=5)
knoll_height_entry = ttk.Entry(settings_frame)
knoll_height_entry.insert(0, "0.8")
knoll_height_entry.grid(row=11, column=1, sticky="ew", padx=5, pady=5)

ttk.Label(settings_frame, text="Minimal depression depth (m):").grid(row=12, column=0, sticky="w", padx=5, pady=5)
dep_depth_entry = ttk.Entry(settings_frame)
dep_depth_entry.insert(0, "0.3")
dep_depth_entry.grid(row=12, column=1, sticky="ew", padx=5, pady=5)

ttk.Label(settings_frame, text="Contour smoothing:").grid(row=13, column=0, sticky="w", padx=5, pady=5)
contour_smooth = ttk.Entry(settings_frame)
contour_smooth.insert(0, "6.5")
contour_smooth.grid(row=13, column=1, sticky="ew", padx=5, pady=5)

# VÝBĚR VRSTEV
layers_frame = ttk.Labelframe(middle_container, text="Symbols to draw", padding="10")
layers_frame.grid(row=0, column=2, sticky="nsew", padx=5)
LAYER_VISIBILITY = {
    "contours": tk.BooleanVar(value=True),
    "private": tk.BooleanVar(value=True),
    "rocks": tk.BooleanVar(value=True),
    "vegetation": tk.BooleanVar(value=True),
    "water": tk.BooleanVar(value=True),
    "roads": tk.BooleanVar(value=True),
    "buildings": tk.BooleanVar(value=True),
    "fences": tk.BooleanVar(value=True),
    "man_made": tk.BooleanVar(value=True),
    "magnetic_lines": tk.BooleanVar(value=False)
}
ttk.Checkbutton(layers_frame, text="Landforms (101–115)", variable=LAYER_VISIBILITY["contours"]).pack(anchor="w")
ttk.Checkbutton(layers_frame, text="Rocks and boulders(201–215) ", variable=LAYER_VISIBILITY["rocks"]).pack(anchor="w")
ttk.Checkbutton(layers_frame, text="Water and marsh(301–313)", variable=LAYER_VISIBILITY["water"]).pack(anchor="w")
ttk.Checkbutton(layers_frame, text="Vegetation (401–419)", variable=LAYER_VISIBILITY["vegetation"]).pack(anchor="w")
ttk.Checkbutton(layers_frame, text="Roads, tracks, paths (501–509)", variable=LAYER_VISIBILITY["roads"]).pack(anchor="w")
ttk.Checkbutton(layers_frame, text="Fences and walls (510–519)", variable=LAYER_VISIBILITY["fences"]).pack(anchor="w")
ttk.Checkbutton(layers_frame, text="Areas that should not be entered (520)", variable=LAYER_VISIBILITY["private"]).pack(anchor="w")
ttk.Checkbutton(layers_frame, text="Buildings (521,522)", variable=LAYER_VISIBILITY["buildings"]).pack(anchor="w")
ttk.Checkbutton(layers_frame, text="Man made features (523–532)", variable=LAYER_VISIBILITY["man_made"]).pack(anchor="w")
ttk.Checkbutton(layers_frame, text="Magnetic north lines (601)", variable=LAYER_VISIBILITY["magnetic_lines"]).pack(anchor="w")

btn_frame = ttk.Frame(layers_frame)
btn_frame.pack(fill=tk.X, pady=5)
def select_all_layers(state):
    for var in LAYER_VISIBILITY.values():
        var.set(state)

ttk.Button(btn_frame, text="Check all", command=lambda: select_all_layers(True), width=15).pack(side=tk.LEFT, padx=2)
ttk.Button(btn_frame, text="Clear all", command=lambda: select_all_layers(False), width=15).pack(side=tk.LEFT, padx=2)
ttk.Checkbutton(controls_frame, text="Save map to PNG", 
                variable=save_var).pack(pady=5)
save_vector_var = tk.BooleanVar(value=True) 
ttk.Checkbutton(controls_frame, text="Save Vegetation to file (GPKG)", 
                variable=save_vector_var).pack(pady=5)
run_button = ttk.Button(controls_frame, text="Generate map", 
                        command=run_main_analysis)
run_button.pack(pady=10, fill=tk.X, ipady=5)
progress_bar = ttk.Progressbar(controls_frame, orient="horizontal", length=400, mode="determinate")
progress_bar.pack(pady=10, fill=tk.X, expand=True)
status_label = tk.Label(controls_frame, text="Hotovo.", anchor="center", foreground="darkgreen")
status_label.pack(pady=5, fill=tk.X, expand=True)
root.mainloop()
print("--- OMapMaker: GUI has been shut down ---")