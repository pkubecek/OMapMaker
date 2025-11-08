import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import xml.etree.ElementTree as ET
from svgpath2mpl import parse_path  # Potřebuje 'pip install svgpath2mpl'
from ast import literal_eval
import laspy
import rasterio 
from rasterio.features import rasterize
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import (binary_dilation, gaussian_filter, label, find_objects, minimum_filter, maximum_filter)
from matplotlib.collections import LineCollection
from matplotlib.transforms import Affine2D
from matplotlib.patches import PathPatch
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, box, shape
from shapely.ops import unary_union, shape
import fiona
import osmnx as ox
from pyproj import Transformer
import pandas as pd
import math
from PIL import Image
import os 

def load_symbol_library(xml_file_path):
    """
    Načte knihovnu symbolů z XML souboru.
    Očekává, že 'style' tagy obsahují přímo matplotlib argumenty.
    (Převzato z OMapMaker_v3.py)
    """
    print(f"Načítám knihovnu symbolů z {xml_file_path}...")
    symbol_library = {}
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        for symbol_element in root.findall('symbol'):
            symbol_id = symbol_element.get('id')
            symbol_type = symbol_element.get('type') 
            if not symbol_id or not symbol_type:
                continue       
            style_element = symbol_element.find('style')
            if style_element is None:
                style_props = {}
            else:
                style_props = style_element.attrib.copy()  
            for key, value in style_props.items():
                try:
                    style_props[key] = float(value)
                except (ValueError, TypeError):
                    if value.startswith('('):
                        try:
                            style_props[key] = literal_eval(value)
                        except:
                            style_props[key] = value 
                    else:
                        style_props[key] = value

            if symbol_type == 'point':
                path_element = symbol_element.find('path')
                if path_element is None or path_element.get('d') is None: 
                    continue
                mpl_path = parse_path(path_element.get('d'))
                symbol_library[symbol_id] = {
                    "type": "point", 
                    "path": mpl_path, 
                    "props": style_props
                }
                print(f"  -> Načten bodový symbol: '{symbol_id}'")
            elif symbol_type in ['line', 'area']:
                symbol_library[symbol_id] = {
                    "type": symbol_type, 
                    "props": style_props
                }
                print(f"  -> Načten styl ({symbol_type}): '{symbol_id}'")  
    except FileNotFoundError:
        print(f"CHYBA: Soubor '{xml_file_path}' nenalezen. Knihovna symbolů nebude použita.")
        print("!!! Ujistěte se, že soubor 'symbols.xml' je ve stejném adresáři jako skript. !!!")
    except Exception as e:
        print(f"CHYBA při načítání symbolů: {e}")
    return symbol_library

SYMBOL_LIBRARY = load_symbol_library("symbols.xml")
print("--- Knihovna symbolů načtena, vytvářím GUI ---")
MAX_GRIDDATA_POINTS = 5_000_000 #Omezení na 5 000 000 bod, aby to nezajebalo počítač

def load_dmr_grid(dmr_path, resolution_points=1500j):
    """
    Načte DMR .las soubor, odfiltruje zem (včetně class 8) a
    interpoluje data na mřížku o daném rozlišení.
    
    (Verze s podvzorkováním pro úsporu paměti)
    
    Vrací: (dmr_grid, grid_x, grid_y, extent)
    """
    print(f"Načítám DMR: {dmr_path}")
    status_label.config(text="Načítám DMR...")
    root.update_idletasks()
    
    las = laspy.read(dmr_path)
    
    classification = np.array(las.classification) #Tohle ošetřuje český DMR, kde dělaj keyopoints problémy
    classification[classification == 8] = 2 

    mask = (classification == 2) # pak si z toho vezme ground points
    
    if not np.any(mask):
        raise ValueError("V souboru DMR nebyly nalezeny žádné body země (klasifikace 2 nebo 8).")
        
    x = las.x[mask]
    y = las.y[mask]
    z = las.z[mask]
    
    # Uložení rozsahu dat
    extent = (x.min(), x.max(), y.min(), y.max())

    grid_x, grid_y = np.mgrid[extent[0]:extent[1]:resolution_points, 
    extent[2]:extent[3]:resolution_points]
    
    print("Interpoluji DMR mřížku...")
    status_label.config(text="Interpoluji DMR mřížku...")
    root.update_idletasks()
    
    points = np.vstack((x, y)).T
    
    # --- OPRAVA: Blok pro podvzorkování ---
    # Je umístěn AŽ PO 'points = np.vstack...'
    if len(points) > MAX_GRIDDATA_POINTS:
        print(f"  -> VAROVÁNÍ: Počet DMR bodů ({len(points)}) je příliš vysoký pro griddata.")
        print(f"  -> Redukuji počet bodů na {MAX_GRIDDATA_POINTS} (náhodný výběr).")
        
        # Provedeme náhodný výběr indexů
        indices = np.random.choice(len(points), MAX_GRIDDATA_POINTS, replace=False)
        
        points = points[indices] # Použijeme jen vybrané body
        z = z[indices]           # DŮLEŽITÉ: musíme redukovat i 'z' hodnoty!
    # --- KONEC OPRAVY ---
        
    dmr_grid = griddata(points, z, (grid_x, grid_y), method='linear')
    
    if np.isnan(dmr_grid).all():
        print("Fallback (DMR): Lineární interpolace selhala, používám 'nearest'.")
        dmr_grid = griddata(points, z, (grid_x, grid_y), method='nearest')
        
    return dmr_grid, grid_x, grid_y, extent

def load_dmp_grid(dmp_path, grid_x, grid_y, extent):
    """
    Načte DMP (.las nebo .tif) a interpoluje ho na existující mřížku
    definovanou z DMR (grid_x, grid_y).
    
    (Verze s podvzorkováním pro úsporu paměti)

    Vrací: (dmp_grid)
    """
    print(f"Načítám DMP: {dmp_path}")
    status_label.config(text="Načítám DMP...")
    root.update_idletasks()
    
    file_ext = os.path.splitext(dmp_path)[1].lower()
    
    # Když je DMP LAS
    if file_ext in ['.las', '.laz']:        
        las = laspy.read(dmp_path)
        classification = np.array(las.classification)
        mask = (classification != 7) #Maska, aby to vyhodilo body šumu
        
        if not np.any(mask):
            raise ValueError("V DMP souboru (.las) nebyly nalezeny žádné body (kromě šumu).")
            
        points = np.vstack((las.x[mask], las.y[mask])).T
        z = las.z[mask]
        
        print("Interpoluji DMP mřížku (z LAS)...")
        status_label.config(text="Interpoluji DMP mřížku...")
        root.update_idletasks()
        
        # --- OPRAVA: Blok pro podvzorkování ---
        # Je umístěn AŽ PO 'points = np.vstack...'
        if len(points) > MAX_GRIDDATA_POINTS:
            print(f"  -> VAROVÁNÍ: Počet DMP bodů ({len(points)}) je příliš vysoký pro griddata.")
            print(f"  -> Redukuji počet bodů na {MAX_GRIDDATA_POINTS} (náhodný výběr).")
            
            indices = np.random.choice(len(points), MAX_GRIDDATA_POINTS, replace=False)
            
            points = points[indices]
            z = z[indices] 
        # --- KONEC OPRAVY ---
            
        dmp_grid = griddata(points, z, (grid_x, grid_y), method='linear')
        
        if np.isnan(dmp_grid).all():
            print("Fallback (DMP-LAS): Lineární interpolace selhala, používám 'nearest'.")
            dmp_grid = griddata(points, z, (grid_x, grid_y), method='nearest')
            
    # Když je DMP TIFF
    elif file_ext in ['.tif', '.tiff']:
        print("Detekován GeoTIFF, čtu data...")
        with rasterio.open(dmp_path) as src:
            data = src.read(1)
            rows, cols = np.indices(data.shape)
            xs, ys = rasterio.transform.xy(src.transform, rows.flatten(), cols.flatten())
            z = data.flatten()
            nodata = src.nodata
            if nodata is not None:
                mask = (z != nodata)
                xs, ys, z = np.array(xs)[mask], np.array(ys)[mask], z[mask]
                
            points = np.vstack((xs, ys)).T
            
            print("Interpoluji DMP mřížku (z TIF)...")
            status_label.config(text="Interpoluji DMP mřížku...")
            root.update_idletasks()
            
            # --- OPRAVA: Blok pro podvzorkování ---
            # Je umístěn AŽ PO 'points = np.vstack...'
            if len(points) > MAX_GRIDDATA_POINTS:
                print(f"  -> VAROVÁNÍ: Počet DMP-TIF bodů ({len(points)}) je příliš vysoký pro griddata.")
                print(f"  -> Redukuji počet bodů na {MAX_GRIDDATA_POINTS} (náhodný výběr).")
                
                indices = np.random.choice(len(points), MAX_GRIDDATA_POINTS, replace=False)
                
                points = points[indices]
                z = z[indices]
            # --- KONEC OPRAVY ---
                
            dmp_grid = griddata(points, z, (grid_x, grid_y), method='linear')
            if np.isnan(dmp_grid).all():
                print("Fallback (DMP-TIF): Lineární interpolace selhala, používám 'nearest'.")
                dmp_grid = griddata(points, z, (grid_x, grid_y), method='nearest')
    else:
        raise ValueError(f"Nepodporovaný formát DMP: '{file_ext}'. Použijte .las, .tif nebo .tiff.")
        
    return dmp_grid

def plot_lines(ax, grid_x, grid_y, data_grid, levels, style_id, zorder=15):
    """
    Pomocná funkce, která vykreslí sadu vrstevnic (contours) 
    na dané ose (ax) podle specifického stylu ze SYMBOL_LIBRARY.
    
    (Původně vnořená funkce v add_contour_lines)
    """
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
            print(f"⚠️ Chyba při kreslení vrstevnic ({style_id}): Žádná data (vše NaN).")
            return 

        cs = ax.contour(grid_x, grid_y, data_grid, levels=levels, **contour_props)
        
        lc_props = style_props.copy()
        if 'color' in lc_props: lc_props['colors'] = lc_props.pop('color')
        if 'edgecolor' in lc_props: lc_props['colors'] = lc_props.pop('edgecolor')

        custom_linestyle = None
        if 'linestyle' in lc_props and isinstance(lc_props['linestyle'], tuple):
            custom_linestyle = lc_props.pop('linestyle')

        if not hasattr(cs, 'collections') or cs.collections is None:
             return # Žádné vrstevnice pro tuto úroveň

        for collection in cs.collections:
            paths = collection.get_paths()
            segments = [path.vertices for path in paths if len(path.vertices) > 1]
            if not segments: continue
            
            lc = LineCollection(segments, **lc_props) 
            
            if custom_linestyle:
                lc.set_linestyle(custom_linestyle)
            
            ax.add_collection(lc)
            collection.remove() 
    except Exception as e:
        print(f"⚠️ Chyba při kreslení vrstevnic ({style_id}): {e}")


# --- Kreslení vrstevnic ---
def add_contour_lines(ax, grid_x, grid_y, dmr_grid):
    """
    Vykreslí tři úrovně vrstevnic (hlavní, základní, doplňkové)
    s využitím pomocné funkce plot_lines.
    
    (Logika kompletně převzata z OMapMaker_v3.py)
    """
    print("Kreslím vrstevnice...")
    status_label.config(text="Kreslím vrstevnice...")
    root.update_idletasks()

    min_z = np.nanmin(dmr_grid)
    max_z = np.nanmax(dmr_grid)
    
    # Definice úrovní
    major_levels = np.arange(np.floor(min_z / 25) * 25, np.ceil(max_z / 25) * 25 + 1, 25) # 25m
    base_levels = np.arange(np.floor(min_z / 5) * 5, np.ceil(max_z / 5) * 5 + 1, 5)     # 5m
    minor_levels = np.arange(np.floor(min_z / 2.5) * 2.5, np.ceil(max_z / 2.5) * 2.5 + 1, 2.5) # 2.5m
    minor_levels = [lvl for lvl in minor_levels if lvl not in base_levels] # Jen ty, co nejsou 5m

    # --- Kreslení ---
    
    # 1. Hlavní a Základní vrstevnice (celá plocha)
    plot_lines(ax, grid_x, grid_y, dmr_grid, major_levels, 'vrstevnice_major', zorder=16)
    plot_lines(ax, grid_x, grid_y, dmr_grid, base_levels, 'vrstevnice_base', zorder=15)
    
    # 2. Doplňkové vrstevnice (jen kde je vysoká křivost)
    gy, gx = np.gradient(dmr_grid)
    gxx, _ = np.gradient(gx)
    _, gyy = np.gradient(gy)
    curvature = np.abs(gxx + gyy)
    
    # Použijeme jen tam, kde je křivost v horním 1% (inspirováno v3)
    threshold = np.percentile(curvature[~np.isnan(curvature)], 99.2) 
    curvature_mask = curvature > threshold
    dilated_mask = binary_dilation(curvature_mask, iterations=20) # Mírné rozšíření
    dmr_grid_curved = np.where(dilated_mask, dmr_grid, np.nan)
    
    # (UPRAVENO: Volání externí funkce)
    plot_lines(ax, grid_x, grid_y, dmr_grid_curved, minor_levels, 'vrstevnice_doplnk')

def vectorize_rocks(grid_x, grid_y, dmr_grid, transform, slope_threshold_deg=60):
    """
    Detekuje strmé svahy a převede je na vektorové polygony.
    (Verze 4.4.8 - Oprava orientace rastru)
    Vrací GeoDataFrame.
    """
    print("Vektorizuji skály...")
    status_label.config(text="Vektorizuji skály...")
    root.update_idletasks()
    
    dy, dx = np.gradient(dmr_grid)
    slope = np.rad2deg(np.arctan(np.hypot(dx, dy)))
    rock_mask = slope > slope_threshold_deg
    rock_area_raw = np.where(rock_mask, 1, 0).astype(np.int32) 

    # --- ZMĚNA ZDE: Oprava orientace rastru ---
    # 1. Transpozice (X, Y) -> (Y, X)
    rock_area_transposed = rock_area_raw.T
    # 2. Vertikální otočení (aby [0, 0] byl horní-levý roh)
    rock_area = np.flipud(rock_area_transposed)
    # --- KONEC ZMĚNY ---

    if not np.any(rock_area):
        print("  -> Nalezeno 0 skalních polygonů.")
        return gpd.GeoDataFrame([], crs="EPSG:5514")
        
    try:
        results_generator = rasterio.features.shapes(
            rock_area, 
            mask=(rock_area == 1), 
            transform=transform # Transformace je již opravena v run_main_analysis
        )
        
        features = [{'geometry': shape(geom)} for geom, value in results_generator if value == 1]
        
        if not features:
            print("  -> Nalezeno 0 skalních polygonů.")
            return gpd.GeoDataFrame([], crs="EPSG:5514")

        gdf = gpd.GeoDataFrame(features, crs="EPSG:5514")
        
        gdf_dissolved = gpd.GeoSeries(unary_union(gdf.geometry))
        gdf_rocks = gpd.GeoDataFrame(geometry=gdf_dissolved, crs="EPSG:5514")
        
        # Vracíme se k jednoduchému zhlazení (bez oblých rohů)
        print("  -> Zhlazuji skalní polygony (jednoduché)...")
        geom = gdf_rocks.geometry.buffer(0)
        gdf_rocks.geometry = geom.simplify(0.5, preserve_topology=True)

        print(f"  -> Vytvořeno {len(gdf_rocks)} spojených skalních polygonů.")
        return gdf_rocks
        
    except Exception as e:
        print(f"⚠️ Chyba při vektorizaci skal: {e}")
        return gpd.GeoDataFrame([], crs="EPSG:5514")

def add_depressions(ax, grid_x, grid_y, dmr_grid, pixel_size=1.0, min_diameter=2.0, max_diameter=6.0, min_depth=0.3):
    """
    Detekuje prohlubně a vykreslí je pomocí symbolu 'prohluben' z XML.
    (Převzato z OMapMaker_v3.py)
    """
    print("Detekuji prohlubně...")
    status_label.config(text="Detekuji prohlubně...")
    root.update_idletasks()
    
    # Zkontrolujeme, zda symbol existuje
    symbol_id = 'prohluben'
    symbol_info = SYMBOL_LIBRARY.get(symbol_id)
    if not symbol_info or symbol_info['type'] != 'point':
        print(f"Chyba: Symbol '{symbol_id}' nebyl nalezen v knihovně.")
        return
    
    symbol_path = symbol_info['path']
    symbol_props = symbol_info['props'].copy()
    symbol_props.setdefault('zorder', 4) # Nastavíme zorder

    # Detekce
    smoothed = gaussian_filter(dmr_grid, sigma=1)
    local_min = (smoothed == minimum_filter(smoothed, size=5))
    depth_reference = gaussian_filter(smoothed, sigma=3)
    depth = depth_reference - smoothed
    depression_mask = (local_min & (depth > min_depth))
    labeled, _ = label(depression_mask)
    slices = find_objects(labeled)
    
    for slc in slices:
        region = labeled[slc]
        region_mask = (region > 0)
        ny, nx = region_mask.shape
        diameter = max(ny, nx) * pixel_size
        if not (min_diameter <= diameter <= max_diameter):
            continue
            
        cy, cx = np.argwhere(region_mask).mean(axis=0)
        i0 = int(slc[0].start + cy)
        j0 = int(slc[1].start + cx)
        if i0 >= dmr_grid.shape[0] or j0 >= dmr_grid.shape[1]:
            continue
            
        x0 = grid_x[i0, j0]
        y0 = grid_y[i0, j0]

        # Kreslení pomocí PathPatch
        transform = Affine2D().scale(1.0).translate(x0, y0) + ax.transData
        patch = PathPatch(
            symbol_path,
            transform=transform,
            **symbol_props  # Aplikuje styl z XML (facecolor, edgecolor, zorder...)
        )
        ax.add_patch(patch)

def add_knoll_symbols(ax, grid_x, grid_y, dmr_grid, pixel_size=1.0, min_height=0.3, max_diameter=5.0):
    """
    Detekuje kupky a vykreslí je pomocí symbolu 'kupka' z XML.
    (Převzato z OMapMaker_v3.py)
    """
    print("Detekuji kupky...")
    status_label.config(text="Detekuji kupky...")
    root.update_idletasks()
    
    # Zkontrolujeme, zda symbol existuje
    symbol_id = 'kupka'
    symbol_info = SYMBOL_LIBRARY.get(symbol_id)
    if not symbol_info or symbol_info['type'] != 'point':
        print(f"Chyba: Symbol '{symbol_id}' nebyl nalezen v knihovně.")
        return
    
    symbol_path = symbol_info['path']
    symbol_props = symbol_info['props'].copy()
    symbol_props.setdefault('zorder', 6) # Nastavíme zorder

    # Detekce
    smoothed = gaussian_filter(dmr_grid, sigma=1)
    local_max = (smoothed == maximum_filter(smoothed, size=5))
    height_reference = gaussian_filter(smoothed, sigma=3)
    height = smoothed - height_reference
    knoll_mask = (local_max & (height > min_height))
    labeled, _ = label(knoll_mask)
    slices = find_objects(labeled)  
    
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
            
        x0 = grid_x[i0, j0]
        y0 = grid_y[i0, j0]
        
        # Kreslení pomocí PathPatch
        transform = Affine2D().translate(x0, y0) + ax.transData
        patch = PathPatch(
            symbol_path,
            transform=transform,
            **symbol_props  # Aplikuje styl z XML (facecolor, edgecolor, zorder...)
        )
        ax.add_patch(patch)

def plot_dashed_hatch(ax, gdf, style_props, zorder=4):    
    # 1. Příprava stylů z XML
    hatch_props = style_props.copy()
    hatch_color = hatch_props.pop('edgecolor', '#28defc') # Barva čar
    hatch_width = hatch_props.pop('linewidth', 0.15)   # Tloušťka čar
    hatch_style = (0, (6, 3)) # Pevně daný styl (6m on, 3m off)
    
    border_props = style_props.copy()
    border_props['facecolor'] = 'none' 
    border_props['edgecolor'] = hatch_color 
    border_props['linewidth'] = hatch_width
    border_props['linestyle'] = 'solid' 
    
    # 2. Vykreslení okrajů
    gdf.plot(ax=ax, zorder=zorder, **border_props)

    # 3. Příprava a vykreslení čárkované výplně
    if gdf.empty: return
    try:
        all_geoms = unary_union(gdf.geometry)
    except Exception:
        all_geoms = gdf.geometry.buffer(0).unary_union
    if all_geoms.is_empty: return
        
    minx, miny, maxx, maxy = all_geoms.bounds
    
    spacing = 5.0 # Rozestup čar 5m
    y_coords = np.arange(np.floor(miny / spacing) * spacing, maxy, spacing)
    h_lines = [LineString([(minx, y), (maxx, y)]) for y in y_coords]
    if not h_lines: return
        
    multi_lines = MultiLineString(h_lines)
    clipped_lines = multi_lines.intersection(all_geoms)
    
    if not clipped_lines.is_empty:
        gpd.GeoSeries([clipped_lines]).plot(
            ax=ax, 
            color=hatch_color, 
            linewidth=hatch_width, 
            linestyle=hatch_style, 
            zorder=zorder - 1 
        )
def get_col(df, col_name):
    """
    Pomocná funkce: Bezpečně vrátí sloupec z GeoDataFrame, 
    i když neexistuje (vrátí prázdnou sérii).
    """
    if col_name in df.columns:
        return df[col_name]
    else:
        # Vytvoříme prázdnou sérii, ale se shodným indexem!
        return gpd.GeoSeries([None] * len(df), index=df.index, crs=df.crs)
 
def vectorize_vegetation(classified_raster_raw, class_names, transform, dmr_path, save_gpkg=False):
    """
    Převede klasifikovaný rastr na vektorové polygony (GeoPackage).
    Filtruje malé plochy a zhlazuje geometrie.
    
    (Verze 4.4.9 - Agresivní zhlazení pro odstranění pixelace)
    
    VŽDY vrací GeoDataFrame.
    Pokud je save_gpkg=True, uloží jej také na disk.
    """
    print("Zahajuji vektorizaci rastru vegetace...")
    status_label.config(text="Vektorizuji vegetaci...")
    root.update_idletasks()
    
    # --- ZMĚNA ZDE: Oprava orientace rastru ---
    classified_raster_transposed = classified_raster_raw.T
    classified_raster = np.flipud(classified_raster_transposed)
    # --- KONEC ZMĚNY ---

    pixel_area = abs(transform.a * transform.e)
    min_area = 10 * pixel_area
    print(f"  -> Skenuji polygony. Plocha 1 pixelu: {pixel_area:.2f} m^2.")
    print(f"  -> Minimální plocha polygonu: {min_area:.2f} m^2 (10 pixelů).")
    
    mask = (classified_raster != 0) # Ignorujeme třídu 0 (Mimo_data)
    
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
            return gpd.GeoDataFrame([], crs="EPSG:5514")

        print(f"Nalezeno {len(features)} hrubých polygonů, vytvářím GeoDataFrame...")
        gdf = gpd.GeoDataFrame(features, crs="EPSG:5514")

        print(f"  -> Filtruji malé polygony porostu (menší než {min_area:.2f} m^2)...")
        print("     (Ponechávám všechny 'Paseky' a 'Louky' bez ohledu na velikost)")
        original_count = len(gdf)
        is_small = gdf.geometry.area < min_area
        is_filterable_class = gdf['class_name'].isin([
            'Nizky_porost', 'Stredni_porost', 
            'Vysoky_porost', 'Les', 'Mimo_data'
        ])
        rows_to_remove = is_small & is_filterable_class
        gdf = gdf[~rows_to_remove]
        print(f"  -> Ponecháno {len(gdf)} z {original_count} polygonů.")

        print("Spojuji polygony podle třídy (Dissolve)...")
        dissolved_gdf = gdf.dissolve(by='class_name', aggfunc='first')
        
        print("  -> Opravuji a zhlazuji geometrie (agresivní oblé rohy)...")
        geom = dissolved_gdf.geometry.buffer(0)
        smooth_geom = geom.buffer(3.0).buffer(-3.0)
        dissolved_gdf.geometry = smooth_geom.simplify(1.0, preserve_topology=True)
        
        dissolved_gdf = dissolved_gdf.reset_index()

        if save_gpkg:
            output_file = os.path.splitext(dmr_path)[0] + "_vegetace.gpkg"
            dissolved_gdf.to_file(output_file, driver="GPKG")
            print(f"✅ Vektorová vegetace uložena do: {output_file}")
        
        return dissolved_gdf

    except Exception as e:
        print(f"❌ Chyba při vektorizaci: {e}")
        status_label.config(text=f"Chyba při vektorizaci: {e}", fg="red")
        return gpd.GeoDataFrame([], crs="EPSG:5514")
    
def add_vector_layers(ax, gdf, extent):        
    # Oříznutí dat na přesný rozsah (extent)
    minx, maxx, miny, maxy = extent
    clip_box = box(minx, miny, maxx, maxy)
    try:
        spatial_index = gdf.sindex
        possible_matches_index = list(spatial_index.intersection(clip_box.bounds))
        gdf = gdf.iloc[possible_matches_index]
        gdf = gpd.clip(gdf, clip_box)
    except Exception as e:
        print(f"Nepodařilo se oříznout GDF, chyba: {e}. Pokračuji s neúplnými daty.")

    if gdf.empty:
        #print("Žádná vektorová data k vykreslení v daném rozsahu.")
        return

    # --- ZMĚNA 1: Explicitní .copy() pro potlačení SettingWithCopyWarning ---
    roads = gdf[get_col(gdf, "highway").notna()].copy()
    railways = gdf[get_col(gdf, "railway").notna()].copy()
    buildings = gdf[get_col(gdf, "building").notna()].copy() # I budovy, i když je nezaoblujeme
    water_lines = gdf[get_col(gdf, "waterway").notna()].copy()
    water_areas = gdf[get_col(gdf, "natural") == "water"].copy()
    wetlands = gdf[get_col(gdf, "natural") == "wetland"].copy()
    gardens = gdf[get_col(gdf, "landuse").isin(["residential", "allotments", "brownfield", "military"])].copy()

    # Styly
    style_major_road = SYMBOL_LIBRARY.get('osm_road_major', {}).get('props', {})
    style_secondary_road = SYMBOL_LIBRARY.get('osm_road_secondary', {}).get('props', {})
    style_track = SYMBOL_LIBRARY.get('osm_road_track', {}).get('props', {})
    style_path = SYMBOL_LIBRARY.get('osm_road_path', {}).get('props', {})
    style_rail_main = SYMBOL_LIBRARY.get('osm_railway_main', {}).get('props', {})
    style_rail_detail = SYMBOL_LIBRARY.get('osm_railway_detail', {}).get('props', {})
    style_building = SYMBOL_LIBRARY.get('osm_building', {}).get('props', {})
    style_water_line = SYMBOL_LIBRARY.get('osm_water_line', {}).get('props', {})
    style_water_area = SYMBOL_LIBRARY.get('osm_water_area', {}).get('props', {})
    style_wetland = SYMBOL_LIBRARY.get('osm_wetland', {}).get('props', {})
    style_garden = SYMBOL_LIBRARY.get('osm_garden', {}).get('props', {})

    # --- Vykreslení ---
    
    # Cesty
    if not roads.empty:
        for _, row in roads.iterrows():
            highway_type = row["highway"]
            geom = row.geometry # Tato geometrie je již zaoblená (pokud není prázdná)
            
            # (Kontrola prázdné geometrie je teoreticky zbytečná díky ZMĚNĚ 2, ale pro jistotu...)
            if geom.is_empty: continue 
                        
            if highway_type in ["motorway", "trunk", "primary", "osm_road_major"]:
                gpd.GeoSeries([geom]).buffer(3.5).plot(ax=ax, zorder=5, **style_major_road)
            
            elif highway_type in ["secondary", "tertiary", "residential", "cycleway", "pedestrian", "service", "unclassified", "osm_road_secondary"]:
                gpd.GeoSeries([geom]).plot(ax=ax, zorder=4, **style_secondary_road)
            
            elif highway_type in ["road", "cycleway","service", "track", "osm_road_track"]:
                gpd.GeoSeries([geom]).plot(ax=ax, zorder=4, **style_track)
            
            elif highway_type in [ "path", "footway", "bridleway", "pedestrian", "osm_road_path"]:
                gpd.GeoSeries([geom]).plot(ax=ax, zorder=4, **style_path)

    # Železnice
    if not railways.empty:
        # Iterace je zde nutná pro styl "pražců"
        for _, row in railways.iterrows():
            geom = row.geometry
            if geom.is_empty: continue
            
            lines = [geom] if geom.geom_type == "LineString" else (geom.geoms if hasattr(geom, 'geoms') else [])
            for line in lines:
                if line.is_empty or not isinstance(line, LineString): continue
                x, y = line.xy
                ax.plot(x, y, zorder=10, **style_rail_main)
                ax.plot(x, y, zorder=11, **style_rail_detail)
                
    # Budovy (Tato logika byla vždy správně, měla by fungovat)
    if not buildings.empty:
        buildings.plot(ax=ax, zorder=5, **style_building)
    # Vodní toky
    if not water_lines.empty:
        water_lines.plot(ax=ax, zorder=3, **style_water_line)
    # Vodní plochy (Tato logika byla vždy správně)
    if not water_areas.empty:
        water_areas.plot(ax=ax, zorder=5, **style_water_area)
    # Mokřady
    if not wetlands.empty:
        plot_dashed_hatch(ax, wetlands, style_wetland, zorder=4)
    # Zahrady / obytné zóny
    if not gardens.empty:
        gardens.plot(ax=ax, **style_garden)
        
def map_zabaged_to_osm(gdf, filename):    
    print(f"Mapuji atributy souboru: {filename}")
    
    # 1. VODSTVO
    if 'WaterCourse' in filename:
        print("  -> Detekován 'WaterCourse', mapuji na 'waterway' = 'stream'")
        gdf['waterway'] = 'stream' 
        
    elif 'Lake' in filename:
        print("  -> Detekován 'Lake', mapuji na 'natural' = 'water'")
        gdf['natural'] = 'water'

    elif 'Marsh' in filename:
        print("  -> Detekován 'Marsh', mapuji na 'natural' = 'wetland'")
        gdf['natural'] = 'wetland'
    
    elif 'MajorRoad' in filename:
        print("  -> Detekován 'MajorRoad', mapuji na 'highway' = 'osm_road_major'")
        gdf['highway'] = 'osm_road_major'
    
    elif  'VehicleTrack' in filename:
        print ("  -> Detekován 'VehicleTrack', mapuji na 'highway' = 'osm_road_track'")
        gdf['highway'] = 'osm_road_track'
    
    elif 'NarrowRide' in filename:
        print ("  -> Detekován 'NarrowRide', mapuji na 'highway' = 'osm_road_path'")
        gdf['highway'] = 'osm_road_path'

    elif  'SmallFootpath' in filename:
        print ("  -> Detekován 'SmallFootpath', mapuji na 'highway' = 'osm_road_path'")
        gdf['highway'] = 'osm_road_path'
        
    elif 'Railway' in filename:
        print ("  -> Detekován 'Railway', mapuji na 'railway' = 'rail'")
        gdf['railway'] = 'rail'
        
    elif 'Building' in filename or 'Budova' in filename:
        print (f"  -> Detekován '{filename}', mapuji na 'building' = 'yes'")
        gdf['building'] = 'yes'

    elif 'Forest' in filename or 'Les' in filename:
        print (f"  -> Detekován '{filename}', mapuji na 'natural' = 'wood'")
        gdf['natural'] = 'wood'
        
    elif 'Meadow' in filename or 'Louka' in filename or 'Grass' in filename:
        print (f"  -> Detekován '{filename}', mapuji na 'landuse' = 'meadow'")
        gdf['landuse'] = 'meadow'
    
    else:
        print(f"  -> VAROVÁNÍ: Pro soubor '{filename}' nebylo nalezeno žádné mapování. Pokusím se ho přesto vykreslit, ale nemusí se zobrazit.")

    return gdf

print("--- OMapMaker v4: Spouštím GUI ---")
def select_file(entry_widget, title, file_types):
    """
    Otevře dialog pro výběr souboru a vloží cestu do zadaného Entry widgetu.
    """
    path = filedialog.askopenfilename(title=title, filetypes=file_types)
    if path:
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, path)
        print(f"Vybrán soubor: {path}")

def select_multiple_files_to_listbox(listbox_widget): # --- ZABAGED Listbox ---
    paths = filedialog.askopenfilenames(title="Vyberte ZABAGED SHP soubory", filetypes=SHP_FILES)
    if paths:
        for path in paths:
            listbox_widget.insert(tk.END, path)
            print(f"Přidán soubor: {path}")

def remove_selected_from_listbox(listbox_widget): # --- Funkce pro ZABAGED Listbox ---
    """
    Odebere vybrané položky ze zadaného Listboxu.
    """
    selected_indices = listbox_widget.curselection()
    # Musíme mazat odzadu, abychom neporušili indexy
    for index in reversed(selected_indices):
        print(f"Odebírám: {listbox_widget.get(index)}")
        listbox_widget.delete(index)

def setup_map_figure(extent): 
    minx, maxx, miny, maxy = extent
    METERS_PER_INCH = 254.0
    data_width_m = maxx - minx
    data_height_m = maxy - miny
    fig_width_in = data_width_m / METERS_PER_INCH
    fig_height_in = data_height_m / METERS_PER_INCH
    print(f"Nastavuji plátno: {fig_width_in:.2f}\" x {fig_height_in:.2f}\" @ 1:10 000")
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect('equal') 
    ax.axis('off') 
    return fig, ax

# --- (UPRAVENÁ) Hlavní funkce analýzy ---
def run_main_analysis():
    """
    Hlavní funkce, která sbírá vstupy z GUI a spouští analýzu.
    (Verze 4.5.1 - Oprava logiky načítání OSM/ZABAGED/Jiné SHP)
    """
    dmr_path = dmr_entry.get()
    dmp_path = dmp_entry.get()
    should_save_png = save_var.get()
    should_save_vector = save_vector_var.get()
    
    zabaged_paths = zabaged_listbox.get(0, tk.END) 
    other_shp_path = other_shp_entry.get()
    
    if not dmr_path or not dmp_path:
        messagebox.showerror("Chyba vstupu", "Musíte vyplnit povinné parametry:\n\n- DMR\n- DMP")
        status_label.config(text="Chyba: Chybí povinné vstupy.", foreground="red")
        return
        
    try:
        progress_bar["value"] = 0
        
        # --- KROK 1: Načtení hranic z GUI (HNED NA ZAČÁTKU) ---
        try:
            b1 = float(bin_entry_1.get()) # Hranice Louka / Nízký
            b2 = float(bin_entry_2.get()) # Hranice Nízký / Střední
            b3 = float(bin_entry_3.get()) # Hranice Střední / Vysoký
            b4 = float(bin_entry_4.get()) # Hranice Vysoký / Les
            bins = [-1, 0, b1, b2, b3, b4]
            print(f"Používám vlastní hranice klasifikace: {bins}")
        except ValueError:
            messagebox.showwarning("Chyba vstupu", "Neplatné hodnoty v 'Nastavení klasifikace'. Používám výchozí hodnoty (0.5, 2, 5, 11).")
            b1, b2, b3, b4 = 0.5, 2.0, 5.0, 11.0
            bins = [-1, 0, b1, b2, b3, b4]
        # --- KONEC NAČÍTÁNÍ HRANIC ---

        # --- KROK 2: Načtení DMR a definice mřížky ---
        dmr_grid, grid_x, grid_y, extent = load_dmr_grid(dmr_path)
        (minx, maxx, miny, maxy) = extent
        progress_bar["value"] = 10
        
        # --- KROK 3: Načtení DMP (LAS/TIF) na mřížku DMR ---
        dmp_grid = load_dmp_grid(dmp_path, grid_x, grid_y, extent)
        progress_bar["value"] = 20
        
        # --- KROK 4: Výpočet výšky vegetace ---
        status_label.config(text="Počítám výšku vegetace...")
        root.update_idletasks()
        vegetation_height = np.clip(dmp_grid - dmr_grid, 0, None)
        progress_bar["value"] = 25
        
        # --- KROK 5: Načtení vektorových dat ---
        gdf_main = None  # Pro OSM + ZABAGED
        gdf_other = None # Pro "Jiné SHP"

        # --- KROK 5A: Načtení "Jiné SHP" (kreslí se PŘES) ---
        # (Toto je moje oprava z minula, nyní na správném místě)
        if other_shp_path:
            try:
                print(f"Načítám 'Jiné SHP': {other_shp_path}")
                status_label.config(text="Načítám 'Jiné SHP'...")
                root.update_idletasks()
                gdf_other = gpd.read_file(other_shp_path)
                if gdf_other.crs != "EPSG:5514":
                    print(f"  -> Přepočítávám CRS z {gdf_other.crs} na EPSG:5514...")
                    gdf_other = gdf_other.to_crs("EPSG:5514")
                print("  -> Mapuji atributy pro 'Jiné SHP'...")
                gdf_other = map_zabaged_to_osm(gdf_other, os.path.basename(other_shp_path))
            except Exception as e:
                print(f"CHYBA: Nepodařilo se načíst 'Jiné SHP': {e}")
                messagebox.showerror("Chyba vstupu", f"Nepodařilo se načíst soubor 'Jiné SHP':\n{e}")
                gdf_other = None
        
        # --- KROK 5B: Načtení OSM a ZABAGED (které nahrazuje OSM) ---
        # (Toto je původní, funkční logika z v4, kterou jsem omylem smazal)
        
        # 1. VŽDY stáhnout OSM jako základ
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
            to_wgs = Transformer.from_crs("EPSG:5514", "EPSG:4326", always_xy=True)
            minlon, minlat = to_wgs.transform(minx, miny)
            maxlon, maxlat = to_wgs.transform(maxx, maxy)
            tags = {
                "highway": True, "building": True, "waterway": True,
                "railway": ["rail", "lightrail"],
                "landuse": ["residential", "allotments", "brownfield", "military", "forest", "meadow"],
                "natural": ["water", "wetland", "wood", "grassland"],
            }
            bbox = (minlon, minlat, maxlon, maxlat) 
            gdf_osm = ox.features_from_bbox(bbox, tags=tags)
            gdf_osm = gdf_osm.to_crs("EPSG:5514")
            print("OSM data stažena.")
        except Exception as e:
            print(f"Chyba při stahování OSM: {e}")
            status_label.config(text="Chyba stahování OSM.", fg="red")
        
        gdf_main = gdf_osm # Začneme s plnými OSM daty
        
        # 2. Načtení ZABAGED (pokud je)
        if zabaged_paths: 
            print(f"Nalezeno {len(zabaged_paths)} souborů ZABAGED...")
            status_label.config(text="Načítám ZABAGED soubory...")
            root.update_idletasks()
            all_gdfs = [] 
            zabaged_replaces = {
                'highway': False, 'building': False, 'waterway': False,
                'natural_water': False, 'natural_wetland': False,
                'landuse_residential_etc': False, 'railway': False,
                'forest_wood': False, 'landuse_meadow': False
            }
            for path in zabaged_paths:
                try:
                    status_label.config(text=f"Načítám {os.path.basename(path)}...")
                    root.update_idletasks()
                    gdf = gpd.read_file(path)
                    gdf = map_zabaged_to_osm(gdf, os.path.basename(path))
                    # Kontrola, které kategorie ZABAGED obsahuje
                    if 'highway' in gdf.columns and gdf['highway'].notna().any(): zabaged_replaces['highway'] = True
                    if 'building' in gdf.columns and gdf['building'].notna().any(): zabaged_replaces['building'] = True
                    if 'waterway' in gdf.columns and gdf['waterway'].notna().any(): zabaged_replaces['waterway'] = True
                    if 'railway' in gdf.columns and gdf['railway'].notna().any(): zabaged_replaces['railway'] = True
                    if 'natural' in gdf.columns:
                        if (gdf['natural'] == 'water').any(): zabaged_replaces['natural_water'] = True
                        if (gdf['natural'] == 'wetland').any(): zabaged_replaces['natural_wetland'] = True
                        if (gdf['natural'] == 'wood').any(): zabaged_replaces['forest_wood'] = True
                    if 'landuse' in gdf.columns:
                         if gdf['landuse'].isin(["residential", "allotments", "brownfield", "military"]).any(): zabaged_replaces['landuse_residential_etc'] = True
                         if (gdf['landuse'] == 'meadow').any(): zabaged_replaces['landuse_meadow'] = True
                         if (gdf['landuse'] == 'forest').any(): zabaged_replaces['forest_wood'] = True 
                    all_gdfs.append(gdf)
                except Exception as e:
                    print(f"Chyba při načítání souboru {path}: {e}")
            
            # 3. Sloučení OSM a ZABAGED (TOTO JE KLÍČOVÁ LOGIKA)
            if all_gdfs:
                print("Sloučuji vrstvy ZABAGED...")
                gdf_zabaged = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs="EPSG:5514")
                
                if gdf_osm is not None and not gdf_osm.empty:
                    # Filtrování OSM dat
                    print(f"Filtruji OSM data na základě ZABAGED vstupů: {zabaged_replaces}")
                    osm_mask = pd.Series(True, index=gdf_osm.index)
                    if zabaged_replaces['highway'] and 'highway' in gdf_osm.columns: osm_mask &= gdf_osm['highway'].isna()
                    if zabaged_replaces['building'] and 'building' in gdf_osm.columns: osm_mask &= gdf_osm['building'].isna()
                    if zabaged_replaces['waterway'] and 'waterway' in gdf_osm.columns: osm_mask &= gdf_osm['waterway'].isna()
                    if zabaged_replaces['railway'] and 'railway' in gdf_osm.columns: osm_mask &= gdf_osm['railway'].isna()
                    if 'natural' in gdf_osm.columns:
                        if zabaged_replaces['natural_water']: osm_mask &= ~(gdf_osm['natural'] == 'water')
                        if zabaged_replaces['natural_wetland']: osm_mask &= ~(gdf_osm['natural'] == 'wetland')
                        if zabaged_replaces['forest_wood']: osm_mask &= ~(gdf_osm['natural'] == 'wood')
                    if 'landuse' in gdf_osm.columns:
                        if zabaged_replaces['landuse_residential_etc']: osm_mask &= ~gdf_osm['landuse'].isin(["residential", "allotments", "brownfield", "military"])
                        if zabaged_replaces['landuse_meadow']: osm_mask &= ~(gdf_osm['landuse'] == 'meadow')
                        if zabaged_replaces['forest_wood']: osm_mask &= ~(gdf_osm['landuse'] == 'forest')
                    
                    gdf_osm_filtered = gdf_osm[osm_mask]
                    print("Spojuji filtrovaná OSM data a ZABAGED data...")
                    # Finální GDF = filtrované OSM + plné ZABAGED
                    gdf_main = gpd.GeoDataFrame(pd.concat([gdf_osm_filtered, gdf_zabaged], ignore_index=True), crs="EPSG:5514")                
                else:
                    gdf_main = gdf_zabaged # Pokud není OSM, použij jen ZABAGED
        # --- KONEC OPRAVENÉHO BLOKU ---
        
        progress_bar["value"] = 35

        # --- KROK 6: Tvorba Masek a Klasifikace (Stále v rastru) ---
        status_label.config(text="Rasterizuji lesní plochy...")
        root.update_idletasks()
        
        shape = grid_x.shape 
        transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width=shape[0], height=shape[1])

        forest_mask_raw = np.zeros(shape, dtype=np.uint8)
        if gdf_main is not None and not gdf_main.empty:
            forest_polys = gdf_main[
                (get_col(gdf_main, 'natural') == 'wood') | 
                (get_col(gdf_main, 'landuse') == 'forest')
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

        # --- KROK 7: VEKTORIZACE VŠEHO (Před smyčkou) ---
        class_names = {
            0: 'Mimo_data', 1: 'Paseka', 2: 'Louka', 3: 'Nizky_porost',
            4: 'Stredni_porost', 5: 'Vysoky_porost', 6: 'Les'
        }
        color_map = {
            'Paseka': '#FFFFB3', 'Louka': '#f7b667', 'Vysoky_porost': '#c3ed9a',
            'Stredni_porost': '#4cc74c', 'Nizky_porost': '#0e990e', 'Les': '#ffffff', 
        }
        
        vec_raster_input = np.nan_to_num(vegetation_height, nan=-9999)
        classified_raster = np.digitize(vec_raster_input, bins).astype(np.int32)
        
        gdf_vegetation = vectorize_vegetation(
            classified_raster, class_names, transform, dmr_path,
            save_gpkg=should_save_vector
        )
        progress_bar["value"] = 60
        
        dmr_grid_viz = np.nan_to_num(dmr_grid, nan=0) 
        gdf_rocks = vectorize_rocks(
            grid_x, grid_y, dmr_grid_viz, transform 
        )
        progress_bar["value"] = 70
        
        # --- KROK 8: PŘÍPRAVA PRO PNG (Pokud je požadováno) ---
        if not should_save_png:
            print("Vektorizace dokončena. Generování PNG přeskočeno.")
            status_label.config(text="✅ Vektorizace hotova.", fg="darkgreen")
            progress_bar["value"] = 100
            root.after(2000, lambda: status_label.config(text="Připraven.", foreground="darkgreen"))
            return 

        # --- KROK 9: Nastavení DLAŽDICOVÁNÍ (TILING) ---
        tile_divisions = 2 
        print(f"Detekováno ukládání PNG, bude použito {tile_divisions}x{tile_divisions} dlaždic.")
        
        tile_width = (maxx - minx) / tile_divisions
        tile_height = (maxy - miny) / tile_divisions
        temp_files = []
        total_tiles = tile_divisions * tile_divisions
        
        # --- KROK 10: Smyčka pro generování DLAŽDIC ---
        for i in range(tile_divisions): # Řádky (y)
            for j in range(tile_divisions): # Sloupce (x)
                
                tile_num = i * tile_divisions + j + 1
                status_label.config(text=f"Generuji PNG dlaždici {tile_num}/{total_tiles}...")
                print(f"--- Generuji PNG dlaždici ({i}, {j}) ---")
                root.update_idletasks()

                tile_minx = minx + j * tile_width
                tile_maxx = minx + (j + 1) * tile_width
                tile_miny = maxy - (i + 1) * tile_height
                tile_maxy = maxy - i * tile_height
                tile_extent = (tile_minx, tile_maxx, tile_miny, tile_maxy)
                clip_box_tile = box(tile_minx, tile_miny, tile_maxx, tile_maxy)

                fig, ax = setup_map_figure(tile_extent)
                
                status_label.config(text=f"Kreslím vektorový podklad... ({tile_num}/{total_tiles})")
                root.update_idletasks()

                # 1. Vykreslení vegetace (zorder 1)
                if not gdf_vegetation.empty:
                    try:
                        veg_data_only = gdf_vegetation[gdf_vegetation['class_name'] != 'Mimo_data']
                        if not veg_data_only.empty:
                            veg_clipped = gpd.clip(veg_data_only, clip_box_tile)
                            if not veg_clipped.empty:
                                veg_colors = veg_clipped['class_name'].map(color_map).fillna('#FF00FF')
                                veg_clipped.plot(ax=ax, color=veg_colors, zorder=1)
                    except Exception as e:
                        print(f"  -> Chyba při kreslení vegetace: {e}")

                # 2. Vykreslení skal (zorder 10)
                if not gdf_rocks.empty:
                    try:
                        rocks_clipped = gpd.clip(gdf_rocks, clip_box_tile)
                        if not rocks_clipped.empty:
                            rocks_clipped.plot(ax=ax, color='black', zorder=10)
                    except Exception as e:
                        print(f"  -> Chyba při kreslení skal: {e}")
                
                # 3. Vrstevnice (zorder 14-16)
                status_label.config(text=f"Kreslím vrstevnice... ({tile_num}/{total_tiles})")
                root.update_idletasks()
                add_contour_lines(ax, grid_x, grid_y, dmr_grid_viz) 

                # 4. Kupky/Prohlubně (zorder 4, 6)
                add_depressions(ax, grid_x, grid_y, dmr_grid_viz, pixel_size=1.0, min_diameter=1, max_diameter=5, min_depth=0.2)
                add_knoll_symbols(ax, grid_x, grid_y, dmr_grid_viz, pixel_size=1.0)
                
                # 5. Cesty, budovy, voda... (zorder 3-11)
                status_label.config(text=f"Kreslím vektory... ({tile_num}/{total_tiles})")
                root.update_idletasks()
                
                # Kreslení hlavní vrstvy (OSM + ZABAGED)
                if gdf_main is not None and not gdf_main.empty:
                    add_vector_layers(ax, gdf_main.copy(), tile_extent) 
                
                # Kreslení "Jiné SHP" PŘES
                if gdf_other is not None and not gdf_other.empty:
                    print("Kreslím 'Jiné SHP' přes...")
                    add_vector_layers(ax, gdf_other.copy(), tile_extent)
                
                # --- UKLÁDÁNÍ ---
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

                temp_file_path = os.path.splitext(dmr_path)[0] + f"_temp_tile_{i}_{j}.png"
                print(f"Ukládám dlaždici: {temp_file_path} (DPI=2540)")
                plt.savefig(temp_file_path, dpi=2540, bbox_inches='tight', pad_inches=0)
                temp_files.append(temp_file_path)

                plt.close(fig) 
                progress_val = 80 + (tile_num / total_tiles) * 15 
                progress_bar["value"] = progress_val
        
        # --- KROK 11: Spojení dlaždic (pokud ukládáme PNG) ---
        if temp_files:
            status_label.config(text="Spojuji PNG dlaždice do finální mapy...")
            print("Spojuji PNG dlaždice...")
            root.update_idletasks()
            
            tile_images_refs = {}
            for i in range(tile_divisions):
                tile_images_refs[i] = {}
                for j in range(tile_divisions):
                    idx = i * tile_divisions + j
                    tile_images_refs[i][j] = Image.open(temp_files[idx])
            total_width = sum(tile_images_refs[0][j].width for j in range(tile_divisions))
            total_height = sum(tile_images_refs[i][0].height for i in range(tile_divisions))
            final_image = Image.new('RGBA', (total_width, total_height))
            current_y = 0
            for i in range(tile_divisions):
                current_x = 0
                max_row_height = 0
                for j in range(tile_divisions):
                    img = tile_images_refs[i][j]
                    final_image.paste(img, (current_x, current_y))
                    current_x += img.width
                    max_row_height = max(max_row_height, img.height)
                    img.close() 
                current_y += max_row_height
            
            output_path = os.path.splitext(dmr_path)[0] + "_OMap.png"
            print(f"Ukládám finální mapu: {output_path}")
            final_image.save(output_path, "PNG")
            
            print("Mažu dočasné soubory dlaždic...")
            for f in temp_files:
                try: os.remove(f)
                except Exception as e: print(f"Nelze smazat temp soubor {f}: {e}")
            
            status_label.config(text=f"✅ PNG uloženo; Vektorizace dokončena.", fg="darkgreen")

        elif not should_save_png and should_save_vector:
             status_label.config(text="✅ Vektorizace hotova.", fg="darkgreen")

        progress_bar["value"] = 100
        root.after(2000, lambda: status_label.config(text="Připraven.", foreground="darkgreen"))

    except Exception as e:
        messagebox.showerror("Chyba analýzy", str(e))
        status_label.config(text=f"❌ Chyba: {str(e)}", fg="red")
        progress_bar["value"] = 0

# --- Definice typů souborů pro dialogová okna ---
LAS_FILES = [("Lidar data", "*.las *.laz"), ("Všechny soubory", "*.*")]
SHP_FILES = [("Shapefile", "*.shp"), ("Všechny soubory", "*.*")]
LAS_TIF_FILES = [
    ("Podporovaná data", "*.las *.laz *.tif *.tiff"),
    ("Lidar data", "*.las *.laz"),
    ("GeoTIFF", "*.tif *.tiff"),
    ("Všechny soubory", "*.*")
]

# -------------------------------
# 5. Spuštění GUI
# -------------------------------
root = tk.Tk() # Hlavní okno
root.title("OMapMaker v4")
# Změníme geometrii na širší, ale (potenciálně) kratší
root.geometry("1000x800") 

main_frame = ttk.Frame(root, padding="15") # Hlavní kontejner s vnitřním odsazením
main_frame.pack(fill=tk.BOTH, expand=True)

# 1. Rámec pro povinné vstupy (Zůstává nahoře)
required_frame = ttk.Labelframe(main_frame, text="Povinné vstupy (.las)", padding="10")
required_frame.pack(fill=tk.X, expand=False, pady=(0, 10))

# --- DMR ---
ttk.Label(required_frame, text="Digitální model reliéfu (DMR):").pack(anchor="w", padx=5)
dmr_entry_frame = ttk.Frame(required_frame)
dmr_entry_frame.pack(fill=tk.X, expand=True, padx=5, pady=(0, 10))
dmr_entry = ttk.Entry(dmr_entry_frame, width=70) 
dmr_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
ttk.Button(dmr_entry_frame, text="...", width=4, 
           command=lambda: select_file(dmr_entry, "Vyberte DMR (.las)", LAS_FILES)).pack(side=tk.RIGHT, padx=(5, 0))

# --- DMP ---
ttk.Label(required_frame, text="Digitální model povrchu (DMP):").pack(anchor="w", padx=5)
dmp_entry_frame = ttk.Frame(required_frame)
dmp_entry_frame.pack(fill=tk.X, expand=True, padx=5, pady=(0, 10))
dmp_entry = ttk.Entry(dmp_entry_frame, width=70)
dmp_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
ttk.Button(dmp_entry_frame, text="...", width=4, 
           command=lambda: select_file(dmp_entry, "Vyberte DMP (.las nebo .tif)", LAS_TIF_FILES)).pack(side=tk.RIGHT, padx=(5, 0))


# --- NOVÝ KONTENJER PRO PROSTŘEDNÍ ČÁST (2 sloupcce) ---
middle_container = ttk.Frame(main_frame)
middle_container.pack(fill=tk.BOTH, expand=True, pady=5)
# Nastavíme váhu sloupců - levý (s listboxem) může být větší
middle_container.columnconfigure(0, weight=3) 
middle_container.columnconfigure(1, weight=1)
middle_container.rowconfigure(0, weight=1) 


# 2. Rámec pro volitelné vstupy (LEVÝ SLOUPEC)
optional_frame = ttk.Labelframe(middle_container, text="Volitelné datové vrstvy (.shp)", padding="10")
# Umístíme do gridu vlevo
optional_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5)) 

# --- ZABAGED (nyní jako Listbox) ---
ttk.Label(optional_frame, text="ZABAGED (nahradí OSM):").pack(anchor="w", padx=5)
zabaged_list_frame = ttk.Frame(optional_frame)
zabaged_list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5)) # Tento expand=True je klíčový
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
ttk.Button(zabaged_buttons_frame, text="Přidat soubory...", 
           command=lambda: select_multiple_files_to_listbox(zabaged_listbox)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
ttk.Button(zabaged_buttons_frame, text="Odebrat vybrané", 
           command=lambda: remove_selected_from_listbox(zabaged_listbox)).pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(5, 0))

# --- Jiné SHP ---
ttk.Label(optional_frame, text="Jiné SHP (kreslí se přes):").pack(anchor="w", padx=5)
other_shp_entry_frame = ttk.Frame(optional_frame)
other_shp_entry_frame.pack(fill=tk.X, expand=False, padx=5, pady=(0, 10)) # expand=False
other_shp_entry = ttk.Entry(other_shp_entry_frame, width=70)
other_shp_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
ttk.Button(other_shp_entry_frame, text="...", width=4, 
           command=lambda: select_file(other_shp_entry, "Vyberte jiný Shapefile (.shp)", SHP_FILES)).pack(side=tk.RIGHT, padx=(5, 0))


# 3. Rámec pro Nastavení klasifikace (PRAVÝ SLOUPEC)
classify_frame = ttk.Labelframe(middle_container, text="Nastavení klasifikace (metry)", padding="10")
# Umístíme do gridu vpravo
classify_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

# Použijeme grid pro snadné zarovnání
classify_frame.columnconfigure(1, weight=1) # Dáme Entry widgetům prostor

# -- Hranice 1: Louka -> Nízký porost --
ttk.Label(classify_frame, text="Louka / Nízký porost:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
bin_entry_1 = ttk.Entry(classify_frame)
bin_entry_1.insert(0, "1") # Výchozí hodnota
bin_entry_1.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

# -- Hranice 2: Nízký -> Střední --
ttk.Label(classify_frame, text="Nízký / Střední porost:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
bin_entry_2 = ttk.Entry(classify_frame)
bin_entry_2.insert(0, "2") # Výchozí hodnota
bin_entry_2.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

# -- Hranice 3: Střední -> Vysoký --
ttk.Label(classify_frame, text="Střední / Vysoký porost:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
bin_entry_3 = ttk.Entry(classify_frame)
bin_entry_3.insert(0, "5") # Výchozí hodnota
bin_entry_3.grid(row=2, column=1, sticky="ew", padx=5, pady=5)

# -- Hranice 4: Vysoký -> Les --
ttk.Label(classify_frame, text="Vysoký porost / Les:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
bin_entry_4 = ttk.Entry(classify_frame)
bin_entry_4.insert(0, "11") # Výchozí hodnota
bin_entry_4.grid(row=3, column=1, sticky="ew", padx=5, pady=5)


# 4. Ovládací prvky (Zůstávají dole)
controls_frame = ttk.Frame(main_frame, padding="10")
controls_frame.pack(fill=tk.X, expand=False, pady=(10, 0))

# Checkbox pro uložení PNG
save_var = tk.BooleanVar(value=True) 
ttk.Checkbutton(controls_frame, text="Uložit výslednou mapu jako PNG", 
                variable=save_var).pack(pady=5)

# Checkbox pro uložení Vektoru
save_vector_var = tk.BooleanVar(value=True) 
ttk.Checkbutton(controls_frame, text="Uložit vegetaci jako vektor (GPKG)", 
                variable=save_vector_var).pack(pady=5)

# Spouštěcí tlačítko
run_button = ttk.Button(controls_frame, text="Generovat mapu", 
                        command=run_main_analysis)
run_button.pack(pady=10, fill=tk.X, ipady=5) 

# Progress bar
progress_bar = ttk.Progressbar(controls_frame, orient="horizontal", length=400, mode="determinate")
progress_bar.pack(pady=10, fill=tk.X, expand=True)

# Stavová hláška
status_label = tk.Label(controls_frame, text="Hotovo.", anchor="center", foreground="darkgreen")
status_label.pack(pady=5, fill=tk.X, expand=True)


# Spuštění smyčky
root.mainloop()

print("--- OMapMaker v4: GUI ukončeno ---")