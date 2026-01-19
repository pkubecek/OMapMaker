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
#import fiona
import osmnx as ox
from pyproj import Transformer
import pandas as pd
#import math
from PIL import Image
import os

def load_symbol_library(xml_file_path):
    """ Načte symboly a inteligentně opraví formáty (čárky v číslech vs. n-tice v závorkách) """
    print(f"--- Načítám symboly z: {xml_file_path} ---")
    library = {}
    
    if not os.path.exists(xml_file_path):
        print("❌ CHYBA: Soubor symbols.xml nenalezen.")
        return {}

    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        for symbol in root.findall('symbol'):
            sid = symbol.get('id')
            stype = symbol.get('type')
            
            if not sid: continue

            # 1. Načtení stylů (Style props)
            style_elem = symbol.find('style')
            props = style_elem.attrib.copy() if style_elem is not None else {}
            
            clean_props = {}
            for k, v in props.items():
                val_str = str(v).strip()
                
                # A) Je to n-tice/tuple? (např. linestyle="(0, (1, 5))")
                # Tu musíme zachovat a NEKONVERTOVAT čárky, jinak to matplotlib nevezme
                if val_str.startswith('('):
                    try:
                        clean_props[k] = literal_eval(val_str)
                        continue # Povedlo se, jdeme dál
                    except:
                        pass # Nejde to, zkusíme další metody
                
                # B) Je to číslo s čárkou? (např. "0,35")
                val_fixed = val_str.replace(',', '.')
                try:
                    clean_props[k] = float(val_fixed)
                    continue
                except ValueError:
                    pass
                
                # C) Je to text (barva, string)
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
    
def load_dmr_grid(dmr_path, pixel_size=0.5):
    """ Loads digital terrain model from LiDAR data with fixed pixel size - Memory Optimized (Chunking) """
    print(f"Načítám DMR s pixelem {pixel_size} m: {dmr_path}")
    status_label.config(text=f"Načítám DMR ({pixel_size}m pixel)...")
    root.update_idletasks()
    
    xs, ys, zs = [], [], []
    
    # --- OPTIMALIZACE: Čtení po blocích (Chunks) ---
    # Tímto se vyhneme chybě 'LasReader object is not subscriptable' a šetříme RAM
    with laspy.open(dmr_path) as fh:
        print(f"  -> Skenuji soubor po blocích (celkem {fh.header.point_count} bodů)...")
        
        # Čteme po 1 000 000 bodech
        for chunk in fh.chunk_iterator(1_000_000):
            # Okamžitá filtrace klasifikace v paměti pro daný blok
            # (Země bývá class 2, někdy 8)
            clas = np.array(chunk.classification)
            mask = (clas == 2) | (clas == 8)
            
            if np.any(mask):
                xs.append(np.array(chunk.x[mask]))
                ys.append(np.array(chunk.y[mask]))
                zs.append(np.array(chunk.z[mask]))

    # Spojení vyfiltrovaných bloků do jednoho pole
    if not xs:
        raise ValueError("V souboru DMR nebyly nalezeny žádné body země (class 2 nebo 8).")
        
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    z = np.concatenate(zs)
    
    print(f"  -> Načteno {len(x)} bodů země.")
    
    # Výpočet rozsahu a mřížky
    extent = (x.min(), x.max(), y.min(), y.max())
    grid_x, grid_y = np.mgrid[extent[0]:extent[1]:pixel_size, 
                              extent[2]:extent[3]:pixel_size]
    
    print(f"Interpoluji DMR mřížku (rozměr: {grid_x.shape})...")
    status_label.config(text="Interpoluji DMR mřížku...")
    root.update_idletasks()
    
    points = np.vstack((x, y)).T
    
    # Finální pojistka pro Griddata (pokud by i samotných bodů země bylo moc)
    if len(points) > MAX_GRIDDATA_POINTS:
        print(f"  -> Dodatečná redukce bodů země pro interpolaci na {MAX_GRIDDATA_POINTS}.")
        indices = np.random.choice(len(points), MAX_GRIDDATA_POINTS, replace=False)        
        points = points[indices] 
        z = z[indices]   
        
    dmr_grid_interpolated = griddata(points, z, (grid_x, grid_y), method='cubic')
    
    if np.isnan(dmr_grid_interpolated).all():
        print("Fallback: 'Cubic' selhala, zkouším 'linear'.")
        dmr_grid_interpolated = griddata(points, z, (grid_x, grid_y), method='linear')
        if np.isnan(dmr_grid_interpolated).all():
            dmr_grid_interpolated = griddata(points, z, (grid_x, grid_y), method='nearest')

    print("  -> Aplikuji základní vyhlazení...")
    
    nan_mask = np.isnan(dmr_grid_interpolated)
    valid_mean = np.nanmean(dmr_grid_interpolated)
    if np.isnan(valid_mean): valid_mean = 0 
        
    dmr_grid_filled = np.nan_to_num(dmr_grid_interpolated, nan=valid_mean)
    
    sigma_pixels = 5
    dmr_grid = gaussian_filter(dmr_grid_filled, sigma=sigma_pixels)
    dmr_grid[nan_mask] = np.nan
            
    return dmr_grid, grid_x, grid_y, extent, points, z


def load_dmp_grid(dmp_path, grid_x, grid_y, extent):
    """ 
    Načítá DMP s NÁHODNOU redukcí bodů (Random Sampling),
    aby se předešlo pruhům ve vegetaci. Zachovává lineární interpolaci.
    """
    print(f"Načítám DMP: {dmp_path}")
    status_label.config(text="Načítám DMP (Random sample)...")
    root.update_idletasks()
    
    # Limit bodů pro rychlou lineární interpolaci
    LOCAL_POINT_LIMIT = 1_500_000 
    
    file_ext = os.path.splitext(dmp_path)[1].lower()
    
    if file_ext in ['.las', '.laz']:        
        xs, ys, zs = [], [], []
        
        with laspy.open(dmp_path) as fh:
            total_points = fh.header.point_count
            
            # Vypočítáme pravděpodobnost, s jakou bod ponecháme (např. 0.1 pro 10%)
            # Pokud je bodů méně než limit, fraction bude 1.0 (bereme vše)
            if total_points > 0:
                fraction = min(1.0, LOCAL_POINT_LIMIT / total_points)
            else:
                fraction = 1.0
            
            if fraction < 1.0:
                print(f"  -> Redukce DMP: Náhodný výběr {fraction*100:.1f}% bodů (z {total_points}).")
            
            # Iterace po blocích (šetří RAM)
            for chunk in fh.chunk_iterator(1_000_000):
                # Načtení souřadnic a klasifikace z bloku
                # Zde musíme načíst celý blok, abychom z něj mohli náhodně vybírat
                cx = np.array(chunk.x)
                cy = np.array(chunk.y)
                cz = np.array(chunk.z)
                cc = np.array(chunk.classification)
                
                # 1. Maska platných bodů (odfiltrování šumu - class 7)
                valid_mask = (cc != 7)
                
                # 2. Náhodná maska pro redukci
                if fraction < 1.0:
                    # Vygenerujeme náhodná čísla 0-1 pro celý blok
                    # Pokud je číslo menší než fraction, bod bereme
                    random_mask = np.random.rand(len(cx)) < fraction
                    # Kombinujeme obě masky (musí být platný AND náhodně vybraný)
                    final_mask = valid_mask & random_mask
                else:
                    final_mask = valid_mask

                # Pokud po filtraci něco zbylo, uložíme to
                if np.any(final_mask):
                    xs.append(cx[final_mask])
                    ys.append(cy[final_mask])
                    zs.append(cz[final_mask])
        
        if not xs: 
            raise ValueError("V DMP nejsou platná data.")
            
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        z = np.concatenate(zs)
        
        points = np.vstack((x, y)).T
        
        print(f"Interpoluji DMP mřížku (Linear, {len(points)} bodů)...")
        status_label.config(text="Interpoluji DMP mřížku...")
        root.update_idletasks()
        
        # POUŽITÍ LINEÁRNÍ INTERPOLACE
        dmp_grid = griddata(points, z, (grid_x, grid_y), method='linear')
        
        # Fallback na nearest pro okraje
        if np.isnan(dmp_grid).all():
             print("  -> Linear selhal, používám nearest.")
             dmp_grid = griddata(points, z, (grid_x, grid_y), method='nearest')
            
    elif file_ext in ['.tif', '.tiff']:
        # U TIFu random sampling neděláme, tam je resampling (zmenšení rozlišení)
        # matematicky správnější než náhodné vyzobávání pixelů.
        with rasterio.open(dmp_path) as src:
            total_pixels = src.width * src.height
            if total_pixels > LOCAL_POINT_LIMIT:
                scale = (LOCAL_POINT_LIMIT / total_pixels) ** 0.5
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
            
            print(f"Interpoluji DMP mřížku (Linear, {len(points)} bodů)...")
            status_label.config(text="Interpoluji DMP mřížku...")
            root.update_idletasks()
            
            dmp_grid = griddata(points, z, (grid_x, grid_y), method='linear')
            if np.isnan(dmp_grid).all():
                dmp_grid = griddata(points, z, (grid_x, grid_y), method='nearest')

    else:
        raise ValueError(f"Nepodporovaný formát DMP: {file_ext}")
        
    return dmp_grid


def plot_lines(ax, grid_x, grid_y, data_grid, levels, style_id, zorder=25, smooth_factor=1):
    """ Plots generated contour lines with custom zorder """
    style = SYMBOL_LIBRARY.get(style_id)
    if not style or style["type"] != 'line':
        print(f"Styl '{style_id}' nenalezen, použije se nouzový.")
        style_props = {'color': 'red', 'linewidth': 0.5, 'linestyle': ':'}
    else:
        style_props = style["props"].copy()

    contour_props = {
        'colors': style_props.get('color'),
        'linewidths': style_props.get('linewidth'),
        'linestyles': style_props.get('linestyle')
    }
    contour_props = {k: v for k, v in contour_props.items() if v is not None}

    if 'linestyles' in contour_props and isinstance(contour_props['linestyles'], tuple):
        contour_props.pop('linestyles')

    try:
        if np.isnan(data_grid).all():
            return 

        # Vykreslíme pomocné vrstevnice (ty se pak smažou)
        cs = ax.contour(grid_x, grid_y, data_grid, levels=levels, **contour_props)
        
        if not hasattr(cs, 'collections') or cs.collections is None:
             return

        for collection in cs.collections:
            paths = collection.get_paths()
            processed_segments = [] 
            
            for path in paths:
                verts = path.vertices
                num_verts = len(verts)
                
                if num_verts > 1:
                    # Filtrujeme příliš krátké čáry
                    diffs = np.diff(verts, axis=0)
                    total_length = np.sum(np.hypot(diffs[:, 0], diffs[:, 1]))
                    
                    if total_length >= 10.0: # Sníženo z 30 na 10 pro jistotu
                        processed_segments.append(verts)
            
            # Odstraníme původní (hrubou) kolekci
            collection.remove()

            if not processed_segments:
                continue

            try:
                # Vytvoření vektorů a jejich vykreslení ve správné vrstvě
                lines_shapely = [LineString(segment) for segment in processed_segments]
                gdf = gpd.GeoDataFrame(geometry=lines_shapely, crs="EPSG:5514")
                
                # ZDE se aplikuje zorder předaný parametrem
                plot_masked(sym_key=style_id, zorder=zorder, mask=None, gdf=gdf, ax=ax, to_mask=False)
            except Exception as e:
                print(f"⚠️ Chyba při vektorizaci vrstevnice ({style_id}): {e}")
            
    except Exception as e:
        print(f"⚠️ Kritická chyba v plot_lines ({style_id}): {e}")


def add_contour_lines(ax, grid_x, grid_y, dmr_grid_unclipped, smoothing_s=2, clip_mask=None):
    """ Generates contour lines from input elevation data """
    print("Kreslím vrstevnice...")
    status_label.config(text="Kreslím vrstevnice...")
    root.update_idletasks()

    if clip_mask is not None:
        print("  -> Aplikuji ořezovou masku na vrstevnice.")
        dmr_grid_plot = np.where(clip_mask, dmr_grid_unclipped, np.nan)
    else:
        dmr_grid_plot = dmr_grid_unclipped # Kreslíme vše

    min_z = np.nanmin(dmr_grid_plot)
    max_z = np.nanmax(dmr_grid_plot)
    
    major_levels = np.arange(np.floor(min_z / 25) * 25, np.ceil(max_z / 25) * 25 + 1, 25)
    base_levels = np.arange(np.floor(min_z / 5) * 5, np.ceil(max_z / 5) * 5 + 1, 5)
    minor_levels = np.arange(np.floor(min_z / 2.5) * 2.5, np.ceil(max_z / 2.5) * 2.5 + 1, 2.5)
    minor_levels = [lvl for lvl in minor_levels if lvl not in base_levels]

    # Kreslíme hlavní a základní vrstevnice z oříznuté 'dmr_grid_plot'
    plot_lines(ax, grid_x, grid_y, dmr_grid_plot, major_levels, 'sym102', zorder=25, smooth_factor=smoothing_s)
    plot_lines(ax, grid_x, grid_y, dmr_grid_plot, base_levels, 'sym101', zorder=25, smooth_factor=smoothing_s)
    
    print("  -> Počítám masku pro doplňkové vrstevnice (z plných dat)...")
    
    valid_data_mask = ~np.isnan(dmr_grid_unclipped)
    filled_mean = np.nanmean(dmr_grid_unclipped)
    if np.isnan(filled_mean): filled_mean = 0 
        
    dmr_grid_calc = np.nan_to_num(dmr_grid_unclipped, nan=filled_mean)

    curvature_mask = np.zeros_like(dmr_grid_calc, dtype=bool)
    gentle_slope_mask = np.zeros_like(dmr_grid_calc, dtype=bool)

    gy, gx = np.gradient(dmr_grid_calc) 

    gxx, _ = np.gradient(gx)
    _, gyy = np.gradient(gy)
    curvature = np.abs(gxx + gyy)
    
    safe_calculation_mask = binary_erosion(valid_data_mask, iterations=2)
    
    valid_curvature = curvature[safe_calculation_mask] 
    if valid_curvature.size > 0:
        curvature_threshold = np.percentile(valid_curvature, 95) 
        curvature_mask = (curvature > curvature_threshold) & valid_data_mask

    slope = np.hypot(gx, gy)
    valid_slope = slope[safe_calculation_mask]

    if valid_slope.size > 0:
        slope_min_threshold = np.percentile(valid_slope, 0) 
        slope_max_threshold = np.percentile(valid_slope, 20) 
        gentle_slope_mask = (slope > slope_min_threshold) & (slope < slope_max_threshold) & valid_data_mask

    combined_mask = curvature_mask | gentle_slope_mask
    dilated_mask = binary_dilation(combined_mask, iterations=10)
    dmr_grid_combined = np.where(dilated_mask, dmr_grid_plot, np.nan)
    
    plot_lines(ax, grid_x, grid_y, dmr_grid_combined, minor_levels, 'sym103', zorder=25, smooth_factor=smoothing_s)


def vectorize_rocks(grid_x, grid_y, dmr_grid, transform, slope_threshold_deg=28): #sklon/2!!!
    """ Simplifies and vectorises polygon representation of rocks """ 
    print("Vektorizuji skály...")
    status_label.config(text="Vektorizuji skály...")
    root.update_idletasks()
    
    dy, dx = np.gradient(dmr_grid)
    slope = np.rad2deg(np.arctan(np.hypot(dx, dy)))
    rock_mask_raw = slope > slope_threshold_deg 
    
    print("  -> Odstraňuji okrajové artefakty...")
    valid_data_mask = (dmr_grid != 0)
    eroded_valid_mask = minimum_filter(valid_data_mask, size=2) 
    final_rock_mask = rock_mask_raw & eroded_valid_mask    
    rock_area_raw = np.where(final_rock_mask, 1, 0).astype(np.int32) 
    rock_area_transposed = rock_area_raw.T
    rock_area = np.flipud(rock_area_transposed)

    pixel_area = abs(transform.a * transform.e)
    min_area = 4 * pixel_area
    print(f"  -> Skenuji polygony. Plocha 1 pixelu: {pixel_area:.2f} m².")
    print(f"  -> Minimální plocha polygonu: {min_area:.2f} m² (10 pixelů).")
    
    mask = (rock_area != 0)
    
    if not np.any(rock_area):
        print("  -> Nenalezen žádný skalní polygon.")
        return gpd.GeoDataFrame([], crs="EPSG:5514")
        
    try:
        results_generator = rasterio.features.shapes(rock_area, mask=mask, transform=transform)
    
        features = []
        for geom, value in results_generator:
            class_id = int(value)
            if class_id == 0: continue

            features.append({'geometry': shape(geom), 'class_id': class_id, 'class_name': 'Skala'})
            
        if not features:
            print("Nebyly nalezeny žádné polygony k vektorizaci.")
            return gpd.GeoDataFrame(columns=['class_name', 'class_id', 'geometry'], crs="EPSG:5514")

        print(f"Nalezeno {len(features)} hrubých polygonů, vytvářím GeoDataFrame...")
        gdf = gpd.GeoDataFrame(features, crs="EPSG:5514")
        original_crs = gdf.crs

        print(f"  -> Filtruji malé polygony skal (menší než {min_area:.2f} m²)...")
        original_count = len(gdf)
        
        is_small = gdf.geometry.area < min_area
        gdf = gdf[~is_small]
        
        print(f"  -> Ponecháno {len(gdf)} z {original_count} polygonů.")

        if gdf.empty:
            print("  -> Po filtrování nezbyly žádné polygony.")
            return gpd.GeoDataFrame(columns=['class_name', 'class_id', 'geometry'], crs=original_crs)

        print("  -> Opravuji a zjednodušuji geometrie PŘED spojením (může chvíli trvat)...")
        status_label.config(text="Zjednodušuji polygony (Skály)...")
        root.update_idletasks()
        
        gdf.geometry = gdf.geometry.buffer(0.3)
        
        gdf.geometry = gdf.geometry.simplify(0.6, preserve_topology=True) 
            
        print("  -> Zjednodušení PŘED spojením hotovo.")
        print("Spojuji polygony podle třídy (Dissolve)...")
        dissolved_gdf = gdf.dissolve(by='class_name', aggfunc='first')
        
        dissolved_gdf = dissolved_gdf.reset_index()

        dissolved_gdf = gpd.GeoDataFrame(dissolved_gdf, geometry='geometry', crs=original_crs)
        
        print(f"  -> Vytvořeno {len(dissolved_gdf)} spojených skalních polygonů.")
        
        return dissolved_gdf

    except Exception as e:
        print(f"❌ Chyba při vektorizaci skal: {e}")
        status_label.config(text=f"Chyba při vektorizaci skal: {e}", fg="red")

        return gpd.GeoDataFrame(columns=['class_name', 'class_id', 'geometry'], crs="EPSG:5514")

def add_depressions(ax, grid_x, grid_y, dmr_grid, pixel_size=0.5, min_diameter=0.5, max_diameter=3, min_depth=0.3):
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
    # Sigma pro vyhlazení terénu (aby zmizel šum) - cca 0.5 metru
    sigma_smooth_meters = 0.5
    sigma_smooth = sigma_smooth_meters / pixel_size
    
    # Sigma pro referenční rovinu (musí "překlenout" díru) - cca 5 metrů
    sigma_ref = 5
    
    # Okno pro hledání minima (aby to nebyl jen dolík mezi kameny) - cca 2 metry
    window_size_meters = 5.0
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
        
    print(f"  -> Vykresleno {count_final} prohlubní.")


def add_knoll_symbols(ax, grid_x, grid_y, dmr_grid, pixel_size=0.5, min_height=0.25, max_diameter=3.0):
    print("Detekuji kupky...")
    status_label.config(text="Detekuji kupky...")
    root.update_idletasks()
    
    sym_key = 'sym109'
    symbol_info = SYMBOL_LIBRARY.get(sym_key)
    if not symbol_info or symbol_info['type'] != 'point':
        print(f"Chyba: Symbol '{sym_key}' nebyl nalezen v knihovně.")
        return
    
    symbol_path = symbol_info['path']
    symbol_props = symbol_info['props'].copy()
    
    if sym_key == 'sym109':
        if symbol_props.get('facecolor') == 'none':
            symbol_props['facecolor'] = '#d15c00'
        if symbol_props.get('linewidth') == 0:
             symbol_props['linewidth'] = 0
    
    symbol_props['zorder'] = 10
    
    scale_factor = 1.0
    if 'dotsize' in symbol_props:
        symbol_props.pop('dotsize', None) 
    
    clean_keys = ['d', 'path_d']
    for k in clean_keys:
        symbol_props.pop(k, None)

    smoothed = gaussian_filter(dmr_grid, sigma=0.1)
    local_max = (smoothed == maximum_filter(smoothed, size=5))
    height_reference = gaussian_filter(smoothed, sigma=2)
    height = smoothed - height_reference
    
    knoll_mask = (local_max & (height > min_height))
    
    labeled, _ = label(knoll_mask)
    slices = find_objects(labeled)  

    count = 0
    for slc in slices:
        region = labeled[slc]
        region_mask = (region > 0)
        ny, nx = region_mask.shape
        diameter = max(ny, nx) * pixel_size
        
        if diameter > max_diameter:
            continue

        cy, cx = np.argwhere(region_mask).mean(axis=0)
        i0 = int(slc[0].start + cy)
        j0 = int(slc[1].start + cx)
        
        if i0 >= dmr_grid.shape[0] or j0 >= dmr_grid.shape[1]:
            continue

        x0 = float(grid_x[i0, j0])
        y0 = float(grid_y[i0, j0])

        transform = Affine2D().scale(scale_factor).translate(x0, y0) + ax.transData
        
        patch = PathPatch(
            symbol_path,
            transform=transform,
            **symbol_props
        )
        ax.add_patch(patch)
        count += 1
        
    print(f"  -> Vykresleno {count} kupek.")


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

    # Otočení pro rasterio (stejné jako v původním kódu)
    classified_raster_transposed = cleaned_raster.T
    classified_raster = np.flipud(classified_raster_transposed)

    pixel_area = abs(transform.a * transform.e)
    min_area = 60 * pixel_area 
    
    mask = (classified_raster != 0)
    
    # Generování tvarů
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
            return gpd.GeoDataFrame(columns=['class_name', 'class_id', 'geometry'], crs="EPSG:5514")

        print(f"Nalezeno {len(features)} hrubých polygonů.")
        gdf = gpd.GeoDataFrame(features, crs="EPSG:5514")
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
        
        # 2. OPTIMALIZACE VEKTOROVÝCH OPERACÍ
        # simplify bez topologie (rychlé odstranění "schodů" z rastru)
        gdf.geometry = gdf.geometry.simplify(0.5, preserve_topology=False)
        
        # Až potom buffer (zakulatí rohy a případně spojí blízké segmenty)
        # Buffer 0 je trik na opravu invalidních geometrií po simplify(False)
        gdf.geometry = gdf.geometry.buffer(2).buffer(0)
        
        print("  -> Spojuji polygony (Dissolve)...")
        # Dissolve je teď mnohem rychlejší, protože vstupní geometrie jsou jednodušší
        dissolved_gdf = gdf.dissolve(by='class_name', aggfunc='first').reset_index()
        
        # Finální jemné zjednodušení už na spojených datech (s topologií, aby to vypadalo hezky)
        dissolved_gdf.geometry = dissolved_gdf.geometry.simplify(0.3, preserve_topology=True)

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

def plot_masked(sym_key, zorder, mask, gdf, ax, to_mask=True):
    """ Vykreslí GDF podle stylu z XML (Oprava pro SVG body) """
    if gdf is None or gdf.empty:
        return None

    # --- A. Filtrace dat ---
    if to_mask:
        if mask is None: 
            return None
        # Ujistíme se, že maska odpovídá délce GDF
        if isinstance(mask, (pd.Series, gpd.GeoSeries)):
            mask = mask.reindex(gdf.index).fillna(False)
        
        subset = gdf[mask].copy()
        if subset.empty: 
            return None
    else:
        subset = gdf.copy()

    # --- B. Načtení dat symbolu ---
    sym_data = SYMBOL_LIBRARY.get(sym_key, {})
    sym_type = sym_data.get('type')
    sym_path = sym_data.get('path') # Načtená MPL cesta
    sym_props = sym_data.get('props', {}).copy()

    # --- C. Speciální výplně (Hatch/Dot) ---
    if 'hatchdistance' in sym_props:
        plot_dashed_hatch(ax, subset.dissolve(), sym_props, zorder=zorder)
        return
    elif 'dotdistance' in sym_props:
        plot_dotted_hatch(ax, subset.dissolve(), sym_props, zorder=zorder)
        return

    # --- D. Vykreslení BODŮ pomocí SVG (PathPatch) ---
    # Toto je nová část, která opravuje prameny
    if sym_type == 'point' and sym_path is not None:
        # Vyčistíme props, které PatchPath nezná
        clean_keys = ['dotsize', 'dotdistance', 'dotcolor', 'marker_shape', 
                      'facecolor_alt', 'hatchdistance', 'hatchcolor', 
                      'hatchwidth', 'hatchstyle', 'd', 'path_d']
        for key in clean_keys:
            sym_props.pop(key, None)

        # Iterujeme přes geometrie a vkládáme SVG značky
        for geom in subset.geometry:
            # Získáme seznam souřadnic (x, y)
            points_to_plot = []
            if geom.geom_type == 'Point':
                points_to_plot.append((geom.x, geom.y))
            elif geom.geom_type == 'MultiPoint':
                points_to_plot.extend([(p.x, p.y) for p in geom.geoms])
            
            for x, y in points_to_plot:
                # Vytvoření transformace (posun na souřadnice)
                # Zde lze přidat i rotaci, pokud bychom ji znali (např. .rotate_deg(angle))
                transform = Affine2D().translate(x, y) + ax.transData
                
                patch = PathPatch(
                    sym_path,
                    transform=transform,
                    zorder=zorder,
                    **sym_props
                )
                ax.add_patch(patch)
        return # Hotovo, nepokračujeme na standardní plot

    # --- E. Standardní vykreslení (Linie, Polygony) ---
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


def add_vector_layers(ax, gdf, extent, zabaged_gdfs):
    """ 
    Kompletní vykreslování vektorových vrstev (ZABAGED + OSM).
    Obsahuje terénní tvary, vodu, vegetaci, sídla a komunikace.
    Verze: Opraveno vykreslování mostů a vedení (OSM tagy + copy-paste chyby).
    """        
    # Oříznutí hlavního GDF a všech ZABAGED vrstev na zadaný rozsah
    gdf = clip_vecor_layers(gdf, extent)
    
    for zabaged_key in zabaged_gdfs:
        zabaged_gdfs[zabaged_key] = clip_vecor_layers(zabaged_gdfs[zabaged_key], extent)

    if (gdf is None or gdf.empty) and not zabaged_gdfs:
        return

    # --- 1. Načtení atributů (sloupců) s ošetřením chybějících hodnot ---
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
    gdf_lines = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    gdf_polygons = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    # TERRAIN SHAPES ======================================================================    
    # cliff_low
    if "StupenSraz" in zabaged_gdfs:
        # TODO: plot_masked(sym_key='sym104', zorder=20, mask=None, gdf=zabaged_gdfs["StupenSraz"], ax=ax, to_mask=False)
        pass
    else:
        mask_embankment = (
        (man_made == "embankment")
    )
    #TODO: plot_masked(sym_key='sym104', zorder=20, mask=mask_embankment, gdf=gdf_lines, ax=ax)   
    # groove
    if "RokleVymol" in zabaged_gdfs:
        # TODO: plot_masked(sym_key='sym107', zorder=20, mask=None, gdf=zabaged_gdfs["RokleVymol"], ax=ax, to_mask=False)
        pass
    # cliff_large
    mask_cliff_high = (
        (natural == "cliff")
    )
    #TODO: plot_masked(sym_key='sym201', zorder=21, mask=mask_cliff_high, gdf=gdf, ax=ax)
    # cave
    if "VstupDoJeskyne" in zabaged_gdfs:
        # TODO: plot_masked(sym_key='sym203-1', zorder=56, mask=None, gdf=zabaged_gdfs["VstupDoJeskyne"], ax=ax, to_mask=False)
        pass
    else:
        mask_cave = (
            (natural == "cave_entrance") | (man_made == "adit")
        )
        #TODO: plot_masked(sym_key='sym203-1', zorder=56, mask=mask_cave, gdf=gdf_centroids, ax=ax)
    # hole    
    mask_hole = (
        (natural == "sinkhole") | (man_made == "mineshaft") | (geological == "giants_kettle")
    )
    #TODO: plot_masked(sym_key='sym203-2', zorder=56, mask=mask_hole, gdf=gdf, ax=ax)
    # boulder
    if "OsamelyBalvanSkalaSkalniSuk" in zabaged_gdfs:
        plot_masked(sym_key='sym205', zorder=56, mask=None, gdf=zabaged_gdfs["OsamelyBalvanSkalaSkalniSuk"], ax=ax, to_mask=False)
    else:
        mask_boulder = (
            (natural.isin(["stone", "rock"])) | (geological == "glacial_erratic")
        )
        plot_masked(sym_key='sym205', zorder=56, mask=mask_boulder, gdf=gdf_centroids, ax=ax)
    # rock
    if "SkalniUtvary" in zabaged_gdfs:
        plot_masked(sym_key='sym206', zorder=56, mask=None, gdf=zabaged_gdfs["SkalniUtvary"], ax=ax, to_mask=False)
        pass
    else:
        mask_rock = (
            (geological.isin(["tor", "hoodoo", "dyke"]))
        )
        plot_masked(sym_key='sym206', zorder=56, mask=mask_rock, gdf=gdf_centroids, ax=ax)
    # boulder_group
    if "SkupinaBalvanu_b" in zabaged_gdfs:
        plot_masked(sym_key='sym207', zorder=56, mask=None, gdf=zabaged_gdfs["SkupinaBalvanu_b"], ax=ax, to_mask=False)
    mask_gravel1 = (
        (natural == "blockfield")
    )   
    plot_masked(sym_key='sym209', zorder=18, mask=mask_gravel1, gdf=gdf_polygons, ax=ax)
    # gravel 2
    mask_gravel2 = (
        (natural == "scree")
    )
    plot_masked(sym_key='sym210', zorder=18, mask=mask_gravel2, gdf=gdf_polygons, ax=ax)
    # sand
    mask_sand = (
        (natural.isin(["sand", "dune"]))
    )
    plot_masked(sym_key='sym213', zorder=15, mask=mask_sand, gdf=gdf_polygons, ax=ax)
    # bedrock
    mask_bedrock = (
        (natural == "bare_rock")
    )
    plot_masked(sym_key='sym214', zorder=17, mask=mask_bedrock, gdf=gdf_polygons, ax=ax)
    # ditch
    mask_ditch = (
        (barrier == "ditch") | (military == "trench")
    )
    plot_masked(sym_key='sym215a', zorder=21, mask=mask_ditch, gdf=gdf_lines, ax=ax)
    plot_masked(sym_key='sym215b', zorder=21, mask=mask_ditch, gdf=gdf_lines, ax=ax)

    # WATER ======================================================================
    # river
    if "VodniTok" in zabaged_gdfs:
        mask = (
            (get_col(zabaged_gdfs["VodniTok"], "typtoku_p").isin(["povrchový splavný"])) &
            (get_col(zabaged_gdfs["VodniTok"], "vydattok_p").isin(["stálý"]))
        )
        plot_masked(sym_key='sym304', zorder=26, mask=mask, gdf=zabaged_gdfs["VodniTok"], ax=ax)
    else:
        mask_river = (
            ((waterway == "river") | (waterway == "canal")) &
            (~tunnel.isin(["culvert", "yes", "pipe", "covered", "cave"])) &
            (~intermittent.isin(["yes", "dry"])) 
        )
        plot_masked(sym_key='sym304', zorder=26, mask=mask_river, gdf=gdf_lines, ax=ax)
    # stream
    if "VodniTok" in zabaged_gdfs:
        mask = (
            (get_col(zabaged_gdfs["VodniTok"], "typtoku_p").isin(["povrchový nesplavný"])) &
            (get_col(zabaged_gdfs["VodniTok"], "vydattok_p").isin(["stálý"]))
        )
        plot_masked(sym_key='sym305', zorder=26, mask=mask, gdf=zabaged_gdfs["VodniTok"], ax=ax)
    else:
        mask_stream = (
            ((waterway == "stream") | (waterway == "ditch")) &
            (~tunnel.isin(["culvert", "yes", "pipe", "covered", "cave"])) &
            (~intermittent.isin(["yes", "dry"])) 
        )
        plot_masked(sym_key='sym305', zorder=26, mask=mask_stream, gdf=gdf_lines, ax=ax)
    # water_deep
    if "VodniPlocha" in zabaged_gdfs:
        plot_masked(sym_key='sym301', zorder=25, mask=None, gdf=zabaged_gdfs["VodniPlocha"], ax=ax, to_mask=False)
        pass
    else:
        mask_water_deep = (
            (natural.isin(["lake", "water", "canal"])) | (water.isin(["lake", "river", "basin", "bay", "reservoir"])) | (landuse == "basin") | (leisure == "swimming_pool")
        )
        plot_masked(sym_key='sym301', zorder=25, mask=mask_water_deep, gdf=gdf_polygons, ax=ax)
    # water_shallow
    mask_water_shallow = (
        (water == "stream") 
    )
    plot_masked(sym_key='sym302', zorder=27, mask=mask_water_shallow, gdf=gdf_polygons, ax=ax)
    # water_drain
    if "VodniTok" in zabaged_gdfs:
        mask = (
            (get_col(zabaged_gdfs["VodniTok"], "typtoku_p").isin(["povrchový splavný", "povrchový nesplavný"])) &
            (get_col(zabaged_gdfs["VodniTok"], "vydattok_p").isin(["občasný"]))
        )
        plot_masked(sym_key='sym306', zorder=26, mask=mask, gdf=zabaged_gdfs["VodniTok"], ax=ax)
    else:
        mask_drain = (
            ((waterway == "drain") | ((waterway.isin(["stream", "ditch"])) & (~intermittent.isin(["yes", "dry"])))) &
            (~tunnel.isin(["culvert", "yes", "pipe", "covered", "cave"])) 
        )
        plot_masked(sym_key='sym306', zorder=26, mask=mask_drain, gdf=gdf_lines, ax=ax)
    # wetland1
    if "Raseliniste" in zabaged_gdfs:
        plot_masked(sym_key='sym307', zorder=25, mask=None, gdf=zabaged_gdfs["Raseliniste"], ax=ax, to_mask=False)
    else:
        mask_wetland1 = (
            (wetland == "reedbed")
        )
        plot_masked(sym_key='sym307', zorder=25, mask=mask_wetland1, gdf=gdf_polygons, ax=ax)
    # wetland2
    if "BazinaMocal" in zabaged_gdfs:
        plot_masked(sym_key='sym308', zorder=25, mask=None, gdf=zabaged_gdfs["BazinaMocal"], ax=ax, to_mask=False)
        pass
    else:
        mask_wetland2 = (
            (natural == "wetland") & (~wetland.isin(["marsh", "wet_meadow", "reedbed"]))
        )
        plot_masked(sym_key='sym308', zorder=25, mask=mask_wetland2, gdf=gdf_polygons, ax=ax)
    # wetland3
    mask_wetland3 = (
        (wetland == "marsh") | (water == "wet_meadow")
    )
    plot_masked(sym_key='sym310', zorder=25, mask=mask_wetland3, gdf=gdf_polygons, ax=ax)
    # well
    mask_well = (
        (man_made == "water_well") | (amenity == "fountain") | (natural == "geysyer") |
        (((emergency == "water_tank") | (man_made == "storage_tank")) & (~covered.isin(["yes", "roof", "shelter"])))
    )
    plot_masked(sym_key='sym311', zorder=52, mask=mask_well, gdf=gdf_centroids, ax=ax)
    # spring
    mask_spring = (
        (natural == "spring") & (covered != "yes")
    )
    plot_masked(sym_key='sym312', zorder=52, mask=mask_spring, gdf=gdf_centroids, ax=ax)

    # LANDCOVER ======================================================================
    # grass_low
    if "TrvalyTravniPorost" in zabaged_gdfs:
        plot_masked(sym_key='sym401', zorder=1.0, mask=None, gdf=zabaged_gdfs["TrvalyTravniPorost"], ax=ax, to_mask=False)
        pass
    else:
        mask_grass_low = (landuse.isin(["grassland", "grass"])) | (natural == "grassland")
        plot_masked(sym_key='sym401', zorder=1.0, mask=mask_grass_low, gdf=gdf_polygons, ax=ax)
    # grass_park
    if "OkrasnaZahradaPark" in zabaged_gdfs: #(ale nelze z toho klasifikovat, jestli se stromy nebo bez)
        plot_masked(sym_key='sym402', zorder=1.0, mask=None, gdf=zabaged_gdfs["OkrasnaZahradaPark"], ax=ax, to_mask=False)
    else:
        mask_park = (
            (leisure == "park")
        ) 
        plot_masked(sym_key='sym402', zorder=1.0, mask=mask_park, gdf=gdf_polygons, ax=ax)
    # grass_high
    mask_grass_high = (
        (landuse == "meadow") | (natural.isin(["fell", "heath"])) # Tohle jsou taky louky značené jako klasický Open Land
    )
    plot_masked(sym_key='sym401', zorder=1.0, mask=mask_grass_high, gdf=gdf_polygons, ax=ax)

    # LESY JEN POR JISTOTU KONCEPČNĚ, ALE POČÍTÁME, ŽE SE BUDOU DĚLAT Z LIDARU
    """
    if "LesniPudaSeStromyKategorizovana" in zabaged_gdfs: #(kategorizovat podle atributů)
        plot_masked(sym_key='sym405', zorder=2, mask=None, gdf=zabaged_gdfs["LesniPudaSeStromyKategorizovana"], ax=ax, to_mask=False)
        pass
    if "LesniPudaSKosodrevinou" in zabaged_gdfs: #(kategorizovat podle atributů)
        plot_masked(sym_key='sym410', zorder=5, mask=None, gdf=zabaged_gdfs["LesniPudaSKosodrevinou"], ax=ax, to_mask=False)
        pass
    if "LesniPudaSKrovinatymPorostem" in zabaged_gdfs: #(kategorizovat podle atributů)
        plot_masked(sym_key='sym408', zorder=4, mask=None, gdf=zabaged_gdfs["LesniPudaSKrovinatymPorostem"], ax=ax, to_mask=False)
        pass
    if "LesniPudaSeStromyKategorizovana" not in zabaged_gdfs and "LesniPudaSKosodrevinou" not in zabaged_gdfs and "LesniPudaSKrovinatymPorostem" not in zabaged_gdfs:
        mask_forest = (
            (landuse == "forest") | (natural == "wood")
        )
        plot_masked(sym_key='sym405', zorder=2, mask=mask_forest, gdf=gdf_polygons, ax=ax)
    """
    # alley
    zabaged_layer = "LiniovaVegetace"
    if zabaged_layer in zabaged_gdfs:
        plot_masked(sym_key='sym408l', zorder=19, mask=None, gdf=zabaged_gdfs["LiniovaVegetace"], ax=ax, to_mask=False)
        pass
    else:
        mask_alley = (
            (natural == "tree_row")
        )
        plot_masked(sym_key='sym408l', zorder=99, mask=mask_alley, gdf=gdf_polygons, ax=ax)
    # field
    if "OrnaPuda" in zabaged_gdfs:
        plot_masked(sym_key='sym412', zorder=1.9, mask=None, gdf=zabaged_gdfs["OrnaPuda"], ax=ax, to_mask=False)
    else:
        mask_field = (
            (landuse == "farmland")
        )
        plot_masked(sym_key='sym412', zorder=1.9, mask=mask_field, gdf=gdf_polygons, ax=ax)
    # orchad
    mask_orchad = (
        (landuse == "orchad")
    )
    plot_masked(sym_key='sym413', zorder=1.9, mask=mask_orchad, gdf=gdf_polygons, ax=ax)
    # vineyard
    if "Vinice" in zabaged_gdfs:
        plot_masked(sym_key='sym414', zorder=1.9, mask=None, gdf=zabaged_gdfs["Vinice"], ax=ax, to_mask=False)
        pass
    if "Chmelnice" in zabaged_gdfs:
        plot_masked(sym_key='sym414', zorder=1.9, mask=None, gdf=zabaged_gdfs["Chmelnice"], ax=ax, to_mask=False)
        pass
    if "Vinice" not in zabaged_gdfs and "Chmelnice" not in zabaged_gdfs:
        mask_vineyard = (
            (landuse == "vineyard")
        )
        plot_masked(sym_key='sym414', zorder=1.9, mask=mask_vineyard, gdf=gdf_polygons, ax=ax)
    # vegetation_change
    zabaged_layer = "HraniceUzivaniPudy"
    if zabaged_layer in zabaged_gdfs:
        plot_masked(sym_key='sym416', zorder=22, mask=None, gdf=zabaged_gdfs["HraniceUzivaniPudy"], ax=ax, to_mask=False)
    # tree
    if "VyznacnyStrom" in zabaged_gdfs and zabaged_gdfs["VyznacnyStrom"] is not None:
        plot_masked(sym_key='sym417a', zorder=54, mask=None, gdf=zabaged_gdfs["VyznacnyStrom"], ax=ax, to_mask=False)
        plot_masked(sym_key='sym417b', zorder=54, mask=None, gdf=zabaged_gdfs["VyznacnyStrom"], ax=ax, to_mask=False)
    else:
        mask_tree = (natural == "tree")
        if not gdf_centroids[mask_tree].empty:
            plot_masked(sym_key='sym417a', zorder=54, mask=mask_tree, gdf=gdf_centroids, ax=ax)
            plot_masked(sym_key='sym417b', zorder=55, mask=mask_tree, gdf=gdf_centroids, ax=ax)

    # shrub
    mask_shrub = (natural == "shrub")
    if not gdf_centroids[mask_shrub].empty:
        plot_masked(sym_key='sym418a', zorder=54, mask=mask_shrub, gdf=gdf_centroids, ax=ax)
        plot_masked(sym_key='sym418b', zorder=54, mask=mask_shrub, gdf=gdf_centroids, ax=ax)

    # stump
    mask_stump = (natural == "tree_stump")
    if not gdf_centroids[mask_stump].empty:
        plot_masked(sym_key='sym418a', zorder=54, mask=mask_stump, gdf=gdf_centroids, ax=ax)
        plot_masked(sym_key='sym418b', zorder=54, mask=mask_stump, gdf=gdf_centroids, ax=ax)
    # PARKINGS & SQUARES ======================================================================
    # parking
    if "ParkovisteOdpocivka" in zabaged_gdfs:
        plot_masked(sym_key='sym501', zorder=46, mask=None, gdf=zabaged_gdfs["ParkovisteOdpocivka"], ax=ax, to_mask=False)
        pass
    else:
        mask_parking = (
            ((amenity == "parking") & (~parking.isin(["garage", "underground"]))) |
            (place == "square") | (highway.isin(["service", "pedestrian", "footway"])) |
            (man_made == "bunker_silo")                                                
        )
        plot_masked(sym_key='sym501', zorder=46, mask=mask_parking, gdf=gdf_polygons, ax=ax)

    # ROADS ======================================================================
    # road_double
    if "SilniceDalnice" in zabaged_gdfs:
        mask = (
            (get_col(zabaged_gdfs["SilniceDalnice"], "typsil_k").isin(["D1", "D2", "M"]))
        )
        plot_masked(sym_key='sym502Da', zorder=45, mask=mask, gdf=zabaged_gdfs["SilniceDalnice"], ax=ax)
        plot_masked(sym_key='sym502Db', zorder=47, mask=mask, gdf=zabaged_gdfs["SilniceDalnice"], ax=ax)
        plot_masked(sym_key='sym502Dc', zorder=48, mask=mask, gdf=zabaged_gdfs["SilniceDalnice"], ax=ax)
    else:
        mask_road_double = (
            (highway.isin(["motorway", "trunk"])) &
            (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) &
            (bridge != "yes") & (access != "private")                                               
        )
        plot_masked(sym_key='sym502Da', zorder=45, mask=mask_road_double, gdf=gdf_lines, ax=ax)
        plot_masked(sym_key='sym502Db', zorder=47, mask=mask_road_double, gdf=gdf_lines, ax=ax)
        plot_masked(sym_key='sym502Dc', zorder=48, mask=mask_road_double, gdf=gdf_lines, ax=ax)
    # road_major
    if "SilniceDalnice" in zabaged_gdfs:
        mask = (
            (~get_col(zabaged_gdfs["SilniceDalnice"], "typsil_k").isin(["D1", "D2", "M"]))
        )
        plot_masked(sym_key='sym502a', zorder=45, mask=mask, gdf=zabaged_gdfs["SilniceDalnice"], ax=ax)
        plot_masked(sym_key='sym502b', zorder=47, mask=mask, gdf=zabaged_gdfs["SilniceDalnice"], ax=ax)     
    if "Ulice" in zabaged_gdfs:
        mask = (
            (get_col(zabaged_gdfs["Ulice"], "tyulice_k").isin(["026"]))
        )
        plot_masked(sym_key='sym502a', zorder=45, mask=mask, gdf=zabaged_gdfs["Ulice"], ax=ax)
        plot_masked(sym_key='sym502b', zorder=47, mask=mask, gdf=zabaged_gdfs["Ulice"], ax=ax)   
    if "SilniceDalnice" not in zabaged_gdfs and "Ulice" not in zabaged_gdfs:
        mask_road_major = (
            (highway.isin(["highway_link", "trunk_link", "primary", "secondary", "secondary_link", "residential", "tertiary", "living_street"])) &
            (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) &
            (bridge != "yes") & (access != "private")                                               
        )
        plot_masked(sym_key='sym502a', zorder=45, mask=mask_road_major, gdf=gdf_lines, ax=ax)
        plot_masked(sym_key='sym502b', zorder=47, mask=mask_road_major, gdf=gdf_lines, ax=ax)
    # road_minor    
    if "SilniceNeevidovana" in zabaged_gdfs:
        plot_masked(sym_key='sym503', zorder=45, mask=None, gdf=zabaged_gdfs["SilniceNeevidovana"], ax=ax, to_mask=False)
    if "Cesta" in zabaged_gdfs:
        mask = (
            (get_col(zabaged_gdfs["Cesta"], "povrch_p").isin(["zpevněný (panel, dlažba)", "zpevněný (asfalt, beton)"]))
        )
        plot_masked(sym_key='sym503', zorder=45, mask=mask, gdf=zabaged_gdfs["Cesta"], ax=ax)
    if "SilniceNeevidovana" not in zabaged_gdfs and "Cesta" not in zabaged_gdfs:
        mask_road_minor = (
            ((highway.isin(["tertiary_link", "service"])) |
            ((highway.isin(["track", "road", "cycleway", "track", "unclassified"])) & 
            ((surface.isin(["concrete", "asphalt"])) | (tracktype == "grade1")))) &
            (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) &
            (bridge != "yes") & (access != "private")                                               
        )
        plot_masked(sym_key='sym503', zorder=45, mask=mask_road_minor, gdf=gdf_lines, ax=ax)
    # track_major
    if "Cesta" in zabaged_gdfs:
        mask = (
            (get_col(zabaged_gdfs["Cesta"], "povrch_p").isin(["zpevněný (nosný terén, štěrk, kalený povrch)"]))
        )
        plot_masked(sym_key='sym504', zorder=45, mask=mask, gdf=zabaged_gdfs["Cesta"], ax=ax)
    else:
        mask_track_major = (
            ((highway.isin(["cycleway", "unclassified"])) & (~surface.isin(["concrete", "asphalt"])) & (tracktype != "grade1")) &
            (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) &
            (bridge != "yes") & (access != "private")                                               
        )
        plot_masked(sym_key='sym504', zorder=45, mask=mask_track_major, gdf=gdf_lines, ax=ax)
    # track_minor
    if "Ulice" in zabaged_gdfs:
        mask = (
            (~get_col(zabaged_gdfs["Ulice"], "typulice_k").isin(["225", "025"]))
        )
        plot_masked(sym_key='sym505', zorder=45, mask=mask, gdf=zabaged_gdfs["Ulice"], ax=ax)
    if "Cesta" in zabaged_gdfs:
        mask = (
            (get_col(zabaged_gdfs["Cesta"], "povrch_p").isin(["nedostatečně zpevněný (tráva, hlína, písek, kamení)", "neurčeno", ""]))
        )
        plot_masked(sym_key='sym505', zorder=45, mask=mask, gdf=zabaged_gdfs["Cesta"], ax=ax)
    if "Ulice" not in zabaged_gdfs and "Cesta" not in zabaged_gdfs:
        mask_track_minor = (
            ((highway.isin(["pedestrian", "road", "footway", "track", "bridleway"])) | ((highway == "cycleway") & (~surface.isin(["concrete", "asphalt"])) & (tracktype != "grade1"))) &
            (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) &
            (bridge != "yes") & (access != "private")                                               
        )
        plot_masked(sym_key='sym505', zorder=45, mask=mask_track_minor, gdf=gdf_lines, ax=ax)
    # path_major
    if "Pesina" in zabaged_gdfs:
        plot_masked(sym_key='sym506', zorder=45, mask=None, gdf=zabaged_gdfs["Pesina"], ax=ax, to_mask=False)
    else:
        mask_path_major = (
            (highway == "path") &
            (~trail_visibility.isin(["low", "poor", "bad", "very_bad", "horrible", "no"])) &
            (bridge != "yes") & (access != "private")                                               
        )
        plot_masked(sym_key='sym506', zorder=45, mask=mask_path_major, gdf=gdf_lines, ax=ax)
    # path_minor
        mask_path_minor = (
            (highway == "path") &
            (trail_visibility.isin(["low", "poor", "bad", "very_bad", "horrible"])) &
            (bridge != "yes") & (access != "private")                                               
        )
        plot_masked(sym_key='sym507', zorder=45, mask=mask_path_minor, gdf=gdf_lines, ax=ax)
    # clearing
    if "LesniPrusek" in zabaged_gdfs:
        plot_masked(sym_key='sym508', zorder=38, mask=None, gdf=zabaged_gdfs["LesniPrusek"], ax=ax, to_mask=False)
    else:
        mask_cutline = (
            (man_made == "cutline")
        )
        plot_masked(sym_key='sym508', zorder=38, mask=mask_cutline, gdf=gdf_lines, ax=ax)
    # bridge_double
    mask_bridge_double = (
        (highway.isin(["motorway", "trunk"])) &
        (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) &
        (bridge == "yes") & (access != "private")                                               
    )
    plot_masked(sym_key='sym502DBa', zorder=65, mask=mask_bridge_double, gdf=gdf_lines, ax=ax)
    plot_masked(sym_key='sym502DBb', zorder=66, mask=mask_bridge_double, gdf=gdf_lines, ax=ax)
    plot_masked(sym_key='sym502Da', zorder=67, mask=mask_bridge_double, gdf=gdf_lines, ax=ax)
    plot_masked(sym_key='sym502Db', zorder=68, mask=mask_bridge_double, gdf=gdf_lines, ax=ax)
    plot_masked(sym_key='sym502Dc', zorder=69, mask=mask_bridge_double, gdf=gdf_lines, ax=ax)
    # bridge_major
    mask_bridge_major = (
        (highway.isin(["highway_link", "trunk_link", "primary", "secondary", "secondary_link", "tertiary", "living_street"])) &
        (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) &
        (bridge == "yes") & (access != "private")                                               
    )
    plot_masked(sym_key='sym502Ba', zorder=65, mask=mask_bridge_major, gdf=gdf_lines, ax=ax)
    plot_masked(sym_key='sym502Bb', zorder=66, mask=mask_bridge_major, gdf=gdf_lines, ax=ax)
    plot_masked(sym_key='sym502a', zorder=67, mask=mask_bridge_major, gdf=gdf_lines, ax=ax)
    plot_masked(sym_key='sym502b', zorder=68, mask=mask_bridge_major, gdf=gdf_lines, ax=ax)
    # bridge_minor+track
    mask_bridge_minor = (
        (highway.isin(["tertiary_link", "residential", "service", "track", "road", "unclassified"])) &
        (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) &
        (bridge == "yes") & (access != "private")                                               
    )
    plot_masked(sym_key='sym503Ba', zorder=65, mask=mask_bridge_minor, gdf=gdf_lines, ax=ax)
    plot_masked(sym_key='sym503Bb', zorder=66, mask=mask_bridge_minor, gdf=gdf_lines, ax=ax)
    plot_masked(sym_key='sym503', zorder=67, mask=mask_bridge_minor, gdf=gdf_lines, ax=ax)
    # bridge_path
    if "Lavka" in zabaged_gdfs:
        plot_masked(sym_key='sym503', zorder=67, mask=None, gdf=zabaged_gdfs["Lavka"], ax=ax, to_mask=False)
    else:
        mask_bridge_path = (
            (highway.isin(["path", "cycleway", "footway", "bridleway"])) &
            (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) &
            (bridge == "yes") & (access != "private")                                               
        )
        plot_masked(sym_key='sym503', zorder=67, mask=mask_bridge_path, gdf=gdf_lines, ax=ax)

    # RAILWAYS ======================================================================
    # railway
    if "ZeleznicniTrat" in zabaged_gdfs:
        plot_masked(sym_key='sym509a', zorder=40, mask=None, gdf=zabaged_gdfs["ZeleznicniTrat"], ax=ax, to_mask=False)
        plot_masked(sym_key='sym509b', zorder=41, mask=None, gdf=zabaged_gdfs["ZeleznicniTrat"], ax=ax, to_mask=False)
    if "ZeleznicniVlecka" in zabaged_gdfs:
        plot_masked(sym_key='sym509a', zorder=40, mask=None, gdf=zabaged_gdfs["ZeleznicniVlecka"], ax=ax, to_mask=False)
        plot_masked(sym_key='sym509b', zorder=41, mask=None, gdf=zabaged_gdfs["ZeleznicniVlecka"], ax=ax, to_mask=False)
    if "ZeleznicniTrat" not in zabaged_gdfs and "ZeleznicniVlecka" not in zabaged_gdfs:
        mask_railway = (
            ((railway == "rail") | (railway == "disused") | (railway == "funicular") | (railway == "light-rail")  | (railway == "narrow_gauge")) &
            (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) &
            (~bridge.isin(["yes"])) 
        )
        plot_masked(sym_key='sym509a', zorder=40, mask=mask_railway, gdf=gdf_lines, ax=ax)
        plot_masked(sym_key='sym509b', zorder=41, mask=mask_railway, gdf=gdf_lines, ax=ax)
    # bridge_railway
    mask_bridge_railway = (
        ((railway == "rail") | (railway == "disused") | (railway == "funicular") | (railway == "light-rail")  | (railway == "narrow_gauge")) &
        (~tunnel.isin(["yes", "avalanche_protector", "building_passage"])) &
        ((bridge == "yes")) 
    )
    plot_masked(sym_key='sym509Ba', zorder=60, mask=mask_bridge_railway, gdf=gdf_lines, ax=ax)
    plot_masked(sym_key='sym509Bb', zorder=61, mask=mask_bridge_railway, gdf=gdf_lines, ax=ax)
    plot_masked(sym_key='sym509a', zorder=62, mask=mask_bridge_railway, gdf=gdf_lines, ax=ax)
    plot_masked(sym_key='sym509b', zorder=63, mask=mask_bridge_railway, gdf=gdf_lines, ax=ax)

    #TODO: ZABAGED Most (ale neklasifikovaný na dálnice, silnice, železnice atd.) ... nevím, jestli je to k něčemu použitelné vůbec

    # TUNNELS ================================= VUBEC NERESIT !!! =====================================
    # tunnel
    if "Tunel" in zabaged_gdfs:
        #plot_masked(sym_key='sym512T', zorder=23, mask=None, gdf=zabaged_gdfs["Tunel"], ax=ax, to_mask=False)
        pass
    else:
        mask_tunnel = (
            ((highway.fillna("") != "") | (railway.fillna("") != "") | (waterway.fillna("") != ""))  &
            (tunnel.isin(["yes", "avalanche_protector", "building_passage", "covered", "cave"]))
        )
        #plot_masked(sym_key='sym512T', zorder=23, mask=mask_tunnel, gdf=gdf_lines, ax=ax)
        pass

    # POWER LINES ======================================================================
    # cable_low
    if "ElektrickeVedeni" in zabaged_gdfs:
        mask = (
            (~get_col(zabaged_gdfs["ElektrickeVedeni"], "napeti").isin(["400", "110"]))
        )
        plot_masked(sym_key='sym510', zorder=70, mask=mask, gdf=zabaged_gdfs["ElektrickeVedeni"], ax=ax)
        pass
    if "LanovaDrahaLyzarskyVlek" in zabaged_gdfs:
        plot_masked(sym_key='sym510', zorder=70, mask=None, gdf=zabaged_gdfs["LanovaDrahaLyzarskyVlek"], ax=ax, to_mask=False)
        pass
    if "ElektrickeVedeni" not in zabaged_gdfs and "LanovaDrahaLyzarskyVlek" not in zabaged_gdfs:
        mask_cable_low = (
            (power == "low") | (aerialway.isin(["line", "cable_car", "gondola", "mixed_lift", "chair_lift", "chair_lift", "drag_lift", "t-bar", "j-bar", "platter", "rope_tow", "zip_line", "goods"]))
        )
        plot_masked(sym_key='sym510', zorder=70, mask=mask_cable_low, gdf=gdf_lines, ax=ax)
    # cable_tower
    if "StozarElektrickehoVedeni" in zabaged_gdfs:
        pass
        plot_masked(sym_key='sym511P', zorder=70, mask=None, gdf=zabaged_gdfs["StozarElektrickehoVedeni"], ax=ax, to_mask=False)
    if "StozarLanoveDrahy" in zabaged_gdfs:
        pass
        plot_masked(sym_key='sym511P', zorder=70, mask=None, gdf=zabaged_gdfs["StozarLanoveDrahy"], ax=ax, to_mask=False)
    if "StozarElektrickehoVedeni" not in zabaged_gdfs and "StozarLanoveDrahy" not in zabaged_gdfs:
        mask_cable_tower = (
            (aerialway == "pylon") | (man_made == "utility_pole") | (power == "tower")
        )
        plot_masked(sym_key='sym511P', zorder=70, mask=mask_cable_tower, gdf=gdf_centroids, ax=ax)
    # cable_high
    if "ElektrickeVedeni" in zabaged_gdfs:
        mask = (
            (get_col(zabaged_gdfs["ElektrickeVedeni"], "napeti").isin(["400", "110"]))
        )
        plot_masked(sym_key='sym10', zorder=70, mask=mask, gdf=zabaged_gdfs["ElektrickeVedeni"], ax=ax)
        pass
    else:
        mask_cable_high = (
            (power.isin(["line", "minor_line"]))
        )
        plot_masked(sym_key='sym510', zorder=70, mask=mask_cable_high, gdf=gdf_lines, ax=ax) #tady asi nema cenu resit co je jaky vedeni a bude lepsi to vykreslovat vsechno jednim
    # wall_low
    if "Zed" in zabaged_gdfs:
        plot_masked(sym_key='sym513-1a', zorder=30, mask=None, gdf=zabaged_gdfs["Zed"], ax=ax, to_mask=False)
        plot_masked(sym_key='sym513-1b', zorder=30, mask=None, gdf=zabaged_gdfs["Zed"], ax=ax, to_mask=False)
        pass
    else:
        mask_wall_low = (
            (barrier == "wall")
        )
        plot_masked(sym_key='sym513-1a', zorder=30, mask=mask_wall_low, gdf=gdf_lines, ax=ax) 
        plot_masked(sym_key='sym513-1b', zorder=30, mask=mask_wall_low, gdf=gdf_lines, ax=ax) 

    # retaining_wall  
    mask_retwall = (
        (barrier == "retaining_wall")
    )
    #TODO: plot_masked(sym_key='sym513-2', zorder=30, mask=mask_retwall, gdf=gdf_lines, ax=ax)
    # wall_high
    if "HradbaValBastaOpevneni" in zabaged_gdfs:
        plot_masked(sym_key='sym515a', zorder=30, mask=None, gdf=zabaged_gdfs["HradbaValBastaOpevneni"], ax=ax, to_mask=False)
        plot_masked(sym_key='sym515b', zorder=30, mask=None, gdf=zabaged_gdfs["HradbaValBastaOpevneni"], ax=ax, to_mask=False)

        pass
    else:
        mask_wall_high = (
            (barrier.isin(["city_wall"]))
        )
        plot_masked(sym_key='sym515a', zorder=30, mask=mask_wall_high, gdf=gdf_lines, ax=ax)
        plot_masked(sym_key='sym515b', zorder=30, mask=mask_wall_high, gdf=gdf_lines, ax=ax)

    # fence_low
    mask_fence_low = (
        (barrier.isin(["cable_barrier", "guard_rail", "hand_rail", "chain", "rope", "jersey_barrier"]))
    )
    #TODO: plot_masked(sym_key='sym516', zorder=30, mask=mask_fence_low, gdf=gdf_lines, ax=ax)
    # fence_high
    mask_fence_high = (
        (barrier == "fence")
    )
    #TODO: plot_masked(sym_key='sym518', zorder=30, mask=mask_fence_high, gdf=gdf_lines, ax=ax)
    # gate
    mask_gate = (
        (barrier.isin(["entrance", "gate", "stile", "wicket_gate", "full-height_turnstile"]))
    ) # NOT SURE HOW TO EFFICIENTLY DESIGN A SYMBOL
    #TODO: plot_masked(sym_key='sym519', zorder=31, mask=mask_gate, gdf=gdf_centroids, ax=ax)
  
    # GARDENS ======================================================================    
    # garden_etc
    gardens_not_from_zabaged = True
    if "Hrbitov" in zabaged_gdfs:
        plot_masked(sym_key='sym520', zorder=19, mask=None, gdf=zabaged_gdfs["Hrbitov"], ax=ax, to_mask=False)
        gardens_not_from_zabaged = False
    if "Kolejiste" in zabaged_gdfs:
        plot_masked(sym_key='sym520', zorder=19, mask=None, gdf=zabaged_gdfs["Kolejiste"], ax=ax, to_mask=False)
        gardens_not_from_zabaged = False
    if "Letiste" in zabaged_gdfs:
        plot_masked(sym_key='sym520', zorder=19, mask=None, gdf=zabaged_gdfs["Letiste"], ax=ax, to_mask=False)
        gardens_not_from_zabaged = False
    if "OstatniPlochaVSidlech" in zabaged_gdfs:
        plot_masked(sym_key='sym520', zorder=19, mask=None, gdf=zabaged_gdfs["OstatniPlochaVSidlech"], ax=ax, to_mask=False)
        gardens_not_from_zabaged = False
    if "OvocnySadZahrada" in zabaged_gdfs:
        plot_masked(sym_key='sym520', zorder=19, mask=None, gdf=zabaged_gdfs["OvocnySadZahrada"], ax=ax, to_mask=False)
        gardens_not_from_zabaged = False
    if "PovrchovaTezbaLom" in zabaged_gdfs:
        plot_masked(sym_key='sym520', zorder=19, mask=None, gdf=zabaged_gdfs["PovrchovaTezbaLom"], ax=ax, to_mask=False)
        gardens_not_from_zabaged = False
    if "ArealUceloveZastavby" in zabaged_gdfs:
        plot_masked(sym_key='sym520', zorder=19, mask=None, gdf=zabaged_gdfs["ArealUceloveZastavby"], ax=ax, to_mask=False)
        gardens_not_from_zabaged = False    
    if "Skladka" in zabaged_gdfs:
        plot_masked(sym_key='sym520', zorder=19, mask=None, gdf=zabaged_gdfs["Skladka"], ax=ax, to_mask=False)
        gardens_not_from_zabaged = False
    if gardens_not_from_zabaged:
        mask_garden = (
            (landuse.isin(["residential", "allotments", "brownfield", "military", "commercial", "construction", "industrial", "retail", "education", "animal_keeping", "cemetery", "landfill", "quarry", "depot", "religious", "farmyard"])) |
            (leisure == "pitch")                                                
        )
        plot_masked(sym_key='sym520', zorder=1.5, mask=mask_garden, gdf=gdf_polygons, ax=ax)
        
    # BUILDINGS ======================================================================
    # building
    if "BudovaJednotlivaNeboBlokBudov" in zabaged_gdfs:
        plot_masked(sym_key='sym521', zorder=37, mask=None, gdf=zabaged_gdfs["BudovaJednotlivaNeboBlokBudov"], ax=ax, to_mask=False)
    else:
        mask_building = (
            (building.notna()) &
            (building != '') &
            (~building.isin(["roof", "ruins"])) 
        )
        plot_masked(sym_key='sym521', zorder=37, mask=mask_building, gdf=gdf_polygons, ax=ax)
    # roof
    mask_roof = (
        (building == "roof") 
    )
    plot_masked(sym_key='sym522', zorder=36, mask=mask_roof, gdf=gdf_polygons, ax=ax)
    
    if "RozvalinaZricenina" in zabaged_gdfs:
        plot_masked(sym_key='sym523', zorder=35, mask=None, gdf=zabaged_gdfs["RozvalinaZricenina"], ax=ax, to_mask=False)
    else:
        mask_ruin = (
            (building == "ruins") | (historic == "ruins") 
        )
        plot_masked(sym_key='sym523', zorder=35, mask=mask_ruin, gdf=gdf_polygons, ax=ax)
    # tower_high
    if "Silo" in zabaged_gdfs:
        plot_masked(sym_key='524a', zorder=56, mask=None, gdf=zabaged_gdfs["Silo"], ax=ax, to_mask=False)
        plot_masked(sym_key='524b', zorder=56, mask=None, gdf=zabaged_gdfs["Silo"], ax=ax, to_mask=False)
        pass
    if "TezniVez" in zabaged_gdfs:
        plot_masked(sym_key='524a', zorder=56, mask=None, gdf=zabaged_gdfs["TezniVez"], ax=ax, to_mask=False)
        plot_masked(sym_key='524b', zorder=56, mask=None, gdf=zabaged_gdfs["TezniVez"], ax=ax, to_mask=False)
        pass
    if "TovarniKomin" in zabaged_gdfs:
        plot_masked(sym_key='524a', zorder=56, mask=None, gdf=zabaged_gdfs["TovarniKomin"], ax=ax, to_mask=False)
        plot_masked(sym_key='524b', zorder=56, mask=None, gdf=zabaged_gdfs["TovarniKomin"], ax=ax, to_mask=False)
        pass
    if "VetrnyMotor" in zabaged_gdfs:
        plot_masked(sym_key='524a', zorder=56, mask=None, gdf=zabaged_gdfs["VetrnyMotor"], ax=ax, to_mask=False)
        plot_masked(sym_key='524b', zorder=56, mask=None, gdf=zabaged_gdfs["VetrnyMotor"], ax=ax, to_mask=False)
        pass
    if "VodojemVezovy" in zabaged_gdfs:
        plot_masked(sym_key='524a', zorder=56, mask=None, gdf=zabaged_gdfs["VodojemVezovy"], ax=ax, to_mask=False)
        plot_masked(sym_key='524b', zorder=56, mask=None, gdf=zabaged_gdfs["VodojemVezovy"], ax=ax, to_mask=False)

        pass
    if "Silo" not in zabaged_gdfs and "TezniVez" not in zabaged_gdfs and "TovarniKomin" not in zabaged_gdfs and "VetrnyMotor" not in zabaged_gdfs and "VodojemVezovy" not in zabaged_gdfs:
        mask_tower_high = (
            (man_made.isin(["tower", "transformer_tower", "water_tower", "communications_tower", "mast", "chimney", "crane", "flagpole", "obelisk"])) |
            (historic == "round_tower")                                                
        )
        plot_masked(sym_key='sym524a', zorder=56, mask=mask_tower_high, gdf=gdf_centroids, ax=ax)
        plot_masked(sym_key='sym524b', zorder=56, mask=mask_tower_high, gdf=gdf_centroids, ax=ax)
    # tower_low
    if "VezovitaStavba" in zabaged_gdfs:
        plot_masked(sym_key='sym525', zorder=56, mask=None, gdf=zabaged_gdfs["VezovitaStavba"].assign(geometry=lambda x: x.geometry.centroid), ax=ax, to_mask=False)
        pass
    else:
        mask_tower_low = (
            (man_made.isin(["column", "beacon", "lighthouse"])) |
            (amenity == "hunting_stand") | (building == "clock_tower")                                               
        )
        plot_masked(sym_key='sym525', zorder=56, mask=mask_tower_low, gdf=gdf_centroids, ax=ax)
    # memorial
    if "BodPolohovehoBodovehoPole" in zabaged_gdfs: #Tohle se pro OB nemapuje
        #plot_masked(sym_key='sym526a', zorder=56, mask=None, gdf=zabaged_gdfs["BodPolohovehoBodovehoPole"], ax=ax, to_mask=False)
        #plot_masked(sym_key='sym526b', zorder=56, mask=None, gdf=zabaged_gdfs["BodPolohovehoBodovehoPole"], ax=ax, to_mask=False)
        pass
    if "BodZakladnihoTihovehoBodovehoPole" in zabaged_gdfs: #Tohle se pro OB nemapuje
        #plot_masked(sym_key='sym526a', zorder=56, mask=None, gdf=zabaged_gdfs["BodZakladnihoTihovehoBodovehoPole"], ax=ax, to_mask=False)
        #plot_masked(sym_key='sym526b', zorder=56, mask=None, gdf=zabaged_gdfs["BodZakladnihoTihovehoBodovehoPole"], ax=ax, to_mask=False)
        pass
    if "MohylaPomnikNahrobek" in zabaged_gdfs:
        plot_masked(sym_key='sym526a', zorder=56, mask=None, gdf=zabaged_gdfs["MohylaPomnikNahrobek"], ax=ax, to_mask=False)
        plot_masked(sym_key='sym526b', zorder=56, mask=None, gdf=zabaged_gdfs["MohylaPomnikNahrobek"], ax=ax, to_mask=False)
        pass
    if "BodPolohovehoBodovehoPole" not in zabaged_gdfs and "BodZakladnihoTihovehoBodovehoPole" not in zabaged_gdfs and "MohylaPomnikNahrobek" not in zabaged_gdfs:
        mask_memorial = (
            (historic.isin(["boundary_stone", "memorial"])) |
            (man_made == "survey_point") &
            (~building.isin(["plaque"]))                                           
        )
        plot_masked(sym_key='sym526a', zorder=56, mask=mask_memorial, gdf=gdf_centroids, ax=ax)
        plot_masked(sym_key='sym526b', zorder=56, mask=mask_memorial, gdf=gdf_centroids, ax=ax)
    # pipe
    if "DalkovyProduktovodDalkovePotrubi" in zabaged_gdfs:
        # TODO: plot_masked(sym_key='sym528', zorder=56, mask=None, gdf=zabaged_gdfs["DalkovyProduktovodDalkovePotrubi"], ax=ax, to_mask=False)
        pass
    if "DopravnikovyPas" in zabaged_gdfs:
        # TODO: plot_masked(sym_key='sym528', zorder=56, mask=None, gdf=zabaged_gdfs["DopravnikovyPas"], ax=ax, to_mask=False)
        pass
    if "DalkovyProduktovodDalkovePotrubi" not in zabaged_gdfs and "DopravnikovyPas" not in zabaged_gdfs:
        mask_pipe = (
            (man_made.isin(["goods_conveyor", "pipeline"]))                                              
        )
        #plot_masked(sym_key='sym528', zorder=56, mask=mask_pipe, gdf=gdf_lines, ax=ax)
        pass
    # circle
    if "Bunkr" in zabaged_gdfs:
        plot_masked(sym_key='sym523', zorder=56, mask=None, gdf=zabaged_gdfs["Bunkr"], ax=ax, to_mask=False)
        pass
    else:
        mask_circle = (
            (military == "bunker")                                               
        )
        plot_masked(sym_key='sym523', zorder=56, mask=mask_circle, gdf=gdf_centroids, ax=ax)
    # cross
    if "KrizSloupKulturnihoVyznamu" in zabaged_gdfs:
        plot_masked(sym_key='sym531', zorder=56, mask=None, gdf=zabaged_gdfs["KrizSloupKulturnihoVyznamu"], ax=ax, to_mask=False)
        pass
    else:
        mask_cross = (
            (man_made == "cross")                                               
        )
        plot_masked(sym_key='sym531', zorder=56, mask=mask_cross, gdf=gdf_centroids, ax=ax)
    # stairs
    mask_stairs = (
        (highway == "steps")                                               
    )
    plot_masked(sym_key='sym532a', zorder=49, mask=mask_stairs, gdf=gdf_lines, ax=ax)
    plot_masked(sym_key='sym532b', zorder=50, mask=mask_stairs, gdf=gdf_lines, ax=ax)
    plot_masked(sym_key='sym532c', zorder=51, mask=mask_stairs, gdf=gdf_lines, ax=ax);

    # if not power_towers.empty:
    #     symbol_id = 'power_tower'
    #     symbol_info = SYMBOL_LIBRARY.get(symbol_id)
        
    #     if not symbol_info or symbol_info['type'] != 'point':
    #         print(f"Chyba: Symbol '{symbol_id}' nebyl nalezen v knihovně.")
    #     else:
    #         symbol_path = symbol_info['path']
    #         symbol_props = symbol_info['props'].copy()
    #         symbol_props.setdefault('zorder', 70)
            
    #         for _, row in power_towers.iterrows():
    #             geom = row.geometry
    #             if geom.geom_type == 'Point':
    #                 x0, y0 = geom.x, geom.y
    #                 transform = Affine2D().translate(x0, y0) + ax.transData
    #                 patch = PathPatch(
    #                     symbol_path,
    #                     transform=transform,
    #                     **symbol_props
    #                 )
    #                 ax.add_patch(patch)
    #             elif geom.geom_type == 'MultiPoint':
    #                  for point in geom.geoms:
    #                     x0, y0 = point.x, point.y
    #                     transform = Affine2D().translate(x0, y0) + ax.transData
    #                     patch = PathPatch(
    #                         symbol_path,
    #                         transform=transform,
    #                         **symbol_props
    #                     )
    #                     ax.add_patch(patch)
def select_file(entry_widget, title, file_types):
    path = filedialog.askopenfilename(title=title, filetypes=file_types)
    if path:
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, path)
        print(f"Vybrán soubor: {path}")


def select_multiple_files_to_listbox(listbox_widget):
    paths = filedialog.askopenfilenames(title="Vyberte ZABAGED SHP soubory", filetypes=SHP_FILES)
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
    global SCALE, SYMBOL_LIBRARY
    selected_scale = scale_var.get()
    if selected_scale == "1:10 000":
        SCALE = 10000
        xml_file = "symbols10.xml"
    else: # Default 1:15 000
        SCALE = 15000
        xml_file = "symbols15.xml"
        
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
    other_shp_path = other_shp_entry.get()
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

        dmr_grid_cubic, grid_x, grid_y, extent, dmr_points, dmr_z = load_dmr_grid(dmr_path, pixel_size=FIXED_PIXEL_SIZE)
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
        
        dmp_grid = load_dmp_grid(dmp_path, grid_x, grid_y, extent)
        progress_bar["value"] = 20
        
        status_label.config(text="Počítám výšku vegetace...")
        root.update_idletasks()
        vegetation_height = np.clip(dmp_grid - dmr_grid_linear, 0, None)
        progress_bar["value"] = 25
        
        gdf_other = None

        print("Kontoluji, jestli jsou přiloženy další vektorové vrstvy...")
        status_label.config(text="Kontoluji přiložené vrstvy...")
        if other_shp_path:
            try:
                print(f"Načítám 'Jiné SHP': {other_shp_path}")
                status_label.config(text="Načítám 'Jiné SHP'...")
                root.update_idletasks()
                gdf_other = gpd.read_file(other_shp_path)
                if gdf_other.crs != "EPSG:5514":
                    print(f"  -> Přepočítávám CRS z {gdf_other.crs} na EPSG:5514...")
                    gdf_other = gdf_other.to_crs("EPSG:5514")

                # print("  -> Mapuji atributy pro 'Jiné SHP'...")
                # gdf_other = map_zabaged_to_osm(gdf_other, os.path.basename(other_shp_path))
            except Exception as e:
                print(f"CHYBA: Nepodařilo se načíst 'Jiné SHP': {e}")
                messagebox.showerror("Chyba vstupu", f"Nepodařilo se načíst soubor 'Jiné SHP':\n{e}")
                gdf_other = None
        
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
            
            to_wgs = Transformer.from_crs("EPSG:5514", "EPSG:4326", always_xy=True)
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
            gdf_osm = gdf_osm.to_crs("EPSG:5514")
            print("OSM data stažena.")
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
            status_label.config(text="Načítám ZABAGED soubory...")
            root.update_idletasks()
            
            for path in zabaged_paths:
                status_label.config(text=f"Načítám {os.path.basename(path)}...")

                zabaged_gdf = gpd.read_file(path)
                if zabaged_gdf is not None and not zabaged_gdf.empty:
                    try:
                        # Ořez GDF na convex hull
                        zabaged_gdf = gpd.clip(zabaged_gdf, clip_polygon)
                    except Exception as e:
                        print(f"  -> Varování: Selhal ořez ZABAGED vrstvy {os.path.basename(path).rsplit('.', 1)[0]}: {e}")

                zabaged_gdfs[os.path.basename(path).rsplit(".", 1)[0]] = zabaged_gdf
                root.update_idletasks()
        
        progress_bar["value"] = 35

        status_label.config(text="Rasterizuji lesní plochy...")
        root.update_idletasks()
        
        shape = grid_x.shape 
        transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width=shape[0], height=shape[1])

        forest_mask_raw = np.zeros(shape, dtype=np.uint8)
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
        
        gdf_rocks = vectorize_rocks(
            grid_x, grid_y, dmr_grid_linear_viz, transform 
        )
        progress_bar["value"] = 70
        
        # ... (zde končí předchozí část: vektorizace skal, progress_bar["value"] = 70) ...

        if not should_save_png:
            print("Vektorizace dokončena. Generování PNG přeskočeno.")
            status_label.config(text="✅ Vektorizace hotova.", fg="darkgreen")
            progress_bar["value"] = 100
            root.after(2000, lambda: status_label.config(text="Done.", foreground="darkgreen"))
            return 

        # --- GENEROWÁNÍ JEDNÉ MAPY (BEZ DLAŽDIC) ---
        print("Zahajuji generování finální mapy...")
        status_label.config(text="Generuji kompozici mapy...")
        root.update_idletasks()
        
        # 1. Nastavení plátna (jen jednou)
        fig, ax, map_extent = setup_map_figure(extent, selected_paper_format)
        (map_minx, map_maxx, map_miny, map_maxy) = map_extent
        
        # Ořezový box pro data
        clip_box_map = box(map_minx, map_miny, map_maxx, map_maxy)

        # Aplikace ořezové masky (Convex Hull) na celé plátno
        if clip_polygon:
            try:
                hull_coords = np.array(clip_polygon.exterior.coords)
                clip_patch = MplPolygon(hull_coords, transform=ax.transData)
                ax.set_clip_path(clip_patch)
            except Exception as e:
                print(f"  -> CHYBA: Nepodařilo se aplikovat ořezovou masku: {e}")

        # 2. Kreslení VEGETACE
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

        # 3. Kreslení SKAL
        if not gdf_rocks.empty:
            try:
                rocks_clipped = gpd.clip(gdf_rocks, clip_box_map)
                if not rocks_clipped.empty:
                    rocks_clipped.plot(ax=ax, color='black', zorder=21)
            except Exception as e:
                print(f"  -> Chyba při kreslení skal: {e}")

        # 4. Kreslení VRSTEVNIC
        status_label.config(text="Kreslím vrstevnice...")
        root.update_idletasks()
        # Předáváme fixní pixel size
        add_contour_lines(ax, grid_x, grid_y, dmr_grid_cubic_viz) 

        # 5. Kreslení PRVKŮ (Kupky, prohlubně)
        status_label.config(text="Kreslím terénní detaily...")
        root.update_idletasks()
        add_depressions(ax, grid_x, grid_y, dmr_grid_linear_viz_features, 
                        pixel_size=FIXED_PIXEL_SIZE, 
                        min_diameter=0.5, max_diameter=3, min_depth=0.2)
                        
        add_knoll_symbols(ax, grid_x, grid_y, dmr_grid_linear_viz_features, 
                          pixel_size=FIXED_PIXEL_SIZE)

        # 6. Kreslení VEKTORŮ (OSM/Zabaged)
        status_label.config(text="Kreslím cesty a objekty...")
        root.update_idletasks()
        if gdf_osm is not None and not gdf_osm.empty:
            add_vector_layers(ax, gdf_osm.copy(), map_extent, zabaged_gdfs)

        if gdf_other is not None and not gdf_other.empty:
            add_vector_layers(ax, gdf_other.copy(), map_extent, zabaged_gdfs)

        # 7. Uložení FINÁLNÍHO SOUBORU
        output_path = os.path.splitext(dmr_path)[0] + "_OMap.png"
        status_label.config(text="Ukládám PNG soubor...")
        root.update_idletasks()
        print(f"Ukládám finální mapu: {output_path} (DPI=300)")

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # Uložení přímo do cílového souboru
        plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0)
        plt.close(fig) 
        
        progress_bar["value"] = 95

        # 8. Generování WORLD FILE (.pgw)
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
print("--- Vytvářím GUI ---")

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
optional_frame = ttk.Labelframe(middle_container, text="Add your fies (.shp)", padding="10")
optional_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5)) 
ttk.Label(optional_frame, text="Names similar to ISOM 2017-2. Use CamelCase (WaterCourse, NarrowRide, VehicleTrack, ...)").pack(anchor="w", padx=5)
zabaged_list_frame = ttk.Frame(optional_frame)
zabaged_list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))
zabaged_scrollbar = ttk.Scrollbar(zabaged_list_frame, orient=tk.VERTICAL)
zabaged_listbox = tk.Listbox(
    zabaged_list_frame, 
    height=6, 
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
ttk.Label(optional_frame, text="Other files:").pack(anchor="w", padx=5)
other_shp_entry_frame = ttk.Frame(optional_frame)
other_shp_entry_frame.pack(fill=tk.X, expand=False, padx=5, pady=(0, 10))
other_shp_entry = ttk.Entry(other_shp_entry_frame, width=70)
other_shp_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
ttk.Button(other_shp_entry_frame, text="...", width=4, 
           command=lambda: select_file(other_shp_entry, "Chose Shapefile (.shp)", SHP_FILES)).pack(side=tk.RIGHT, padx=(5, 0))
classify_frame = ttk.Labelframe(middle_container, text="Custom vegetation height (m)", padding="10")
classify_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
classify_frame.columnconfigure(1, weight=1)
ttk.Label(classify_frame, text="Open Land/Rough Open Land:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
bin_entry_1 = ttk.Entry(classify_frame)
bin_entry_1.insert(0, "1")
bin_entry_1.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
ttk.Label(classify_frame, text="Vegetation: fight").grid(row=1, column=0, sticky="w", padx=5, pady=5)
bin_entry_2 = ttk.Entry(classify_frame)
bin_entry_2.insert(0, "1.3")
bin_entry_2.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
ttk.Label(classify_frame, text="Vegetation: walk").grid(row=2, column=0, sticky="w", padx=5, pady=5)
bin_entry_3 = ttk.Entry(classify_frame)
bin_entry_3.insert(0, "6")
bin_entry_3.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
ttk.Label(classify_frame, text="Vegetation: slow running").grid(row=3, column=0, sticky="w", padx=5, pady=5)
bin_entry_4 = ttk.Entry(classify_frame)
bin_entry_4.insert(0, "12")
bin_entry_4.grid(row=3, column=1, sticky="ew", padx=5, pady=5)
ttk.Label(classify_frame, text="Formát výstupu:").grid(row=4, column=0, sticky="w", padx=5, pady=(15, 5))
paper_format_combo = ttk.Combobox(
    classify_frame,
    textvariable=paper_format_var,
    values=paper_format_options,
    state="readonly" 
)
paper_format_combo.grid(row=4, column=1, sticky="ew", padx=5, pady=(15, 5))
paper_format_combo.set(paper_format_options[3]) # Výchozí hodnota A4
ttk.Label(classify_frame, text="Měřítko mapy:").grid(row=6, column=0, sticky="w", padx=5, pady=5)
scale_var = tk.StringVar(value="1:10 000") # Výchozí hodnota
scale_combo = ttk.Combobox(
    classify_frame,
    textvariable=scale_var,
    values=["1:10 000", "1:15 000"],
    state="readonly"
)
scale_combo.grid(row=6, column=1, sticky="ew", padx=5, pady=5)
controls_frame = ttk.Frame(main_frame, padding="10")
controls_frame.pack(fill=tk.X, expand=False, pady=(10, 0))
save_var = tk.BooleanVar(value=True) 
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