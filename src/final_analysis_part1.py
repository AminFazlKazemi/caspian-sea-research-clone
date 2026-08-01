#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
کد فوق‌بهینه برای سرور - با رفع گلوگاه‌های عملکرد
- بارگذاری یکباره داده‌ها با xr.open_mfdataset و کش کردن
- ساخت یکباره RegularGridInterpolator برای کل داده‌ها
- استفاده از ThreadPoolExecutor (برای I/O-bound)
- کاهش حجم داده‌های ارسالی به Workerها
- ذخیره کش با فاصله بیشتر (هر ۱۰۰ تسک)
- استفاده از Dask برای بارگذاری موازی
"""

import os
import sys
import logging
import pickle
import time
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator, splprep, splev, interp1d
from scipy.stats import ttest_ind
from shapely.geometry import Polygon
from shapely import minimum_rotated_rectangle
import pyproj
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# ۱. تنظیمات مسیرها
# ============================================================
EXCEL_PATH = r"I:\North_Center_South_Caspian Sea.xlsx"
IVT_ZARR_BASE = r"J:\ivt_data"
OUTPUT_BASE = r"I:\caspian_flux_output"
os.makedirs(OUTPUT_BASE, exist_ok=True)

PROGRESS_FILE = os.path.join(OUTPUT_BASE, 'progress.txt')
RESULTS_FILE = os.path.join(OUTPUT_BASE, 'all_monthly_results.pkl')
LOG_FILE = os.path.join(OUTPUT_BASE, 'analysis_optimized_v2.log')

YEARS = list(range(1940, 2026))
MONTHS = list(range(1, 13))
SPECIAL_PERIODS = {
    'Low_Water': list(range(1976, 1979)),
    'High_Water': list(range(1994, 1997))
}
SHEET_NAMES = {'S_CS': 'South', 'C_CS': 'Center', 'N_CS': 'North'}
INTERPOLATE_POINTS = 300
QC_THRESHOLD = 3000.0
CACHE_SAVE_INTERVAL = 100  # افزایش به ۱۰۰
N_THREADS = max(1, os.cpu_count() - 1)  # استفاده از ThreadPool

PROJ_STR = "+proj=laea +lat_0=42 +lon_0=51 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
transformer = pyproj.Transformer.from_crs("EPSG:4326", PROJ_STR, always_xy=True)
transformer_inv = pyproj.Transformer.from_crs(PROJ_STR, "EPSG:4326", always_xy=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# ۲. بارگذاری یکباره داده‌های Zarr (با Dask)
# ============================================================
def load_all_ivt_data():
    """بارگذاری همه داده‌های Zarr به‌صورت یکبار با Dask"""
    logger.info("🔄 بارگذاری همه داده‌های Zarr با Dask...")
    pattern = os.path.join(IVT_ZARR_BASE, "*.zarr")
    ds = xr.open_mfdataset(
        pattern,
        combine='nested',
        concat_dim='time',
        parallel=True,
        chunks={'time': 1, 'latitude': 90, 'longitude': 180},  # chunks مناسب
        engine='zarr'
    )
    ds = ds.sortby('time')
    logger.info(f"✅ داده بارگذاری شد: {ds.time.min().values} تا {ds.time.max().values}")
    return ds

# ============================================================
# ۳. ساخت یکباره Interpolator برای همه زمان‌ها
# ============================================================
def build_interpolators_for_all_times(ds):
    """ساخت RegularGridInterpolator برای همه زمان‌ها (یکبار)"""
    logger.info("🔄 ساخت Interpolator برای همه زمان‌ها...")
    lon_grid = ds.viwve.longitude.values
    lat_grid = ds.viwve.latitude.values
    if lon_grid[0] > lon_grid[-1]:
        lon_grid = lon_grid[::-1]
    if lat_grid[0] > lat_grid[-1]:
        lat_grid = lat_grid[::-1]
    
    # بارگذاری همه داده‌ها به‌صورت یکبار (با Dask)
    u_all = ds.viwve.compute()  # محاسبه همه داده‌ها
    v_all = ds.viwvn.compute()
    
    # اطمینان از ترتیب
    u_all = u_all.values
    v_all = v_all.values
    
    # ساخت یک لیست از interpolatorها برای هر زمان
    interpolators = []
    for t in tqdm(range(len(ds.time)), desc="ساخت Interpolator"):
        u_vals = u_all[t, :, :]
        v_vals = v_all[t, :, :]
        # اطمینان از ترتیب
        if lon_grid[0] > lon_grid[-1]:
            u_vals = u_vals[:, ::-1]
            v_vals = v_vals[:, ::-1]
        if lat_grid[0] > lat_grid[-1]:
            u_vals = u_vals[::-1, :]
            v_vals = v_vals[::-1, :]
        interp_u = RegularGridInterpolator(
            (lat_grid, lon_grid), u_vals,
            method='linear', bounds_error=False, fill_value=np.nan
        )
        interp_v = RegularGridInterpolator(
            (lat_grid, lon_grid), v_vals,
            method='linear', bounds_error=False, fill_value=np.nan
        )
        interpolators.append((interp_u, interp_v))
    
    logger.info(f"✅ {len(interpolators)} Interpolator ساخته شد.")
    return interpolators

# ============================================================
# ۴. توابع هندسی (همانند قبل)
# ============================================================
def lonlat_to_proj(lons, lats):
    return transformer.transform(lons, lats)

def proj_to_lonlat(xs, ys):
    return transformer_inv.transform(xs, ys)

def get_inside_point_proj(lons, lats):
    polygon = Polygon(zip(lons, lats))
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    centroid = polygon.centroid
    if polygon.contains(centroid):
        inside_lon, inside_lat = centroid.x, centroid.y
    else:
        rep = polygon.representative_point()
        inside_lon, inside_lat = rep.x, rep.y
    inside_x, inside_y = lonlat_to_proj(np.array([inside_lon]), np.array([inside_lat]))
    return inside_x[0], inside_y[0]

def interpolate_boundary_spline(lons, lats, num_points):
    n = len(lons)
    if num_points <= n or num_points == 0:
        return lons.copy(), lats.copy()
    mask = np.isfinite(lons) & np.isfinite(lats)
    if not np.all(mask):
        lons, lats = lons[mask], lats[mask]
        if len(lons) < 4:
            return interpolate_boundary_linear(lons, lats, num_points)
    if not np.allclose([lons[0], lats[0]], [lons[-1], lats[-1]]):
        lons_ext = np.append(lons, lons[0])
        lats_ext = np.append(lats, lats[0])
    else:
        lons_ext, lats_ext = lons.copy(), lats.copy()
    if len(lons_ext) < 4:
        return interpolate_boundary_linear(lons, lats, num_points)
    try:
        tck, u = splprep([lons_ext, lats_ext], s=0, per=True)
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)
        return np.array(x_new), np.array(y_new)
    except:
        return interpolate_boundary_linear(lons, lats, num_points)

def interpolate_boundary_linear(lons, lats, num_points):
    n = len(lons)
    if n < 2:
        return lons.copy(), lats.copy()
    dist = np.zeros(n)
    for i in range(1, n):
        dist[i] = dist[i-1] + np.hypot(lons[i]-lons[i-1], lats[i]-lats[i-1])
    total_dist = dist[-1] + np.hypot(lons[0]-lons[-1], lats[0]-lats[-1])
    dist_ext = np.append(dist, total_dist)
    lons_ext = np.append(lons, lons[0])
    lats_ext = np.append(lats, lats[0])
    new_dist = np.linspace(0, total_dist, num_points)
    f_lon = interp1d(dist_ext, lons_ext, kind='linear')
    f_lat = interp1d(dist_ext, lats_ext, kind='linear')
    return f_lon(new_dist), f_lat(new_dist)

def compute_segments_and_normals(xs, ys, inside_x, inside_y):
    n = len(xs)
    if n < 2:
        return {}
    n_seg = n - 1
    mid_x = np.zeros(n_seg)
    mid_y = np.zeros(n_seg)
    lengths = np.zeros(n_seg)
    normals = np.zeros((n_seg, 2))
    for i in range(n_seg):
        mid_x[i] = (xs[i] + xs[i+1]) / 2
        mid_y[i] = (ys[i] + ys[i+1]) / 2
        dx = xs[i+1] - xs[i]
        dy = ys[i+1] - ys[i]
        lengths[i] = np.hypot(dx, dy)
        if lengths[i] > 0:
            nx = -dy / lengths[i]
            ny = dx / lengths[i]
            to_inside = np.array([inside_x - mid_x[i], inside_y - mid_y[i]])
            if np.dot([nx, ny], to_inside) > 0:
                nx = -nx
                ny = -ny
            normals[i, 0] = nx
            normals[i, 1] = ny
    return {'mid_x': mid_x, 'mid_y': mid_y, 'lengths': lengths, 'normals': normals, 'xs': xs, 'ys': ys}

def split_into_sides_advanced(xs, ys):
    n = len(xs)
    if n < 10:
        return split_into_sides_simple(xs, ys)
    dist = np.zeros(n)
    for i in range(1, n):
        dist[i] = dist[i-1] + np.hypot(xs[i]-xs[i-1], ys[i]-ys[i-1])
    total_dist = dist[-1] + np.hypot(xs[0]-xs[-1], ys[0]-ys[-1])
    polygon = Polygon(zip(xs, ys))
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    try:
        mbr = minimum_rotated_rectangle(polygon)
        corners = np.array(list(mbr.exterior.coords)[:-1])
    except:
        return split_into_sides_curvature(xs, ys)
    break_points = []
    for cx, cy in corners:
        idx = np.argmin(np.hypot(xs-cx, ys-cy))
        if len(break_points) > 0:
            if min(abs(dist[idx]-dist[bp]) for bp in break_points) < 0.1*total_dist:
                continue
        break_points.append(idx)
    if len(break_points) < 4:
        return split_into_sides_curvature(xs, ys)
    break_points = sorted(break_points, key=lambda i: dist[i])
    sides = {}
    for i in range(4):
        start, end = break_points[i], break_points[(i+1)%4]
        if start < end:
            idx = np.arange(start, end+1)
        else:
            idx = np.concatenate([np.arange(start, n), np.arange(0, end+1)])
        mx, my = np.mean(xs[idx]), np.mean(ys[idx])
        lon, lat = proj_to_lonlat(np.array([mx]), np.array([my]))
        lons_all, lats_all = proj_to_lonlat(xs, ys)
        lat_mean, lon_mean = np.mean(lats_all), np.mean(lons_all)
        name = 'North' if lat > lat_mean else 'South' if lat < lat_mean else 'East' if lon > lon_mean else 'West'
        sides[name] = {'xs': xs[idx], 'ys': ys[idx], 'indices': idx}
    if len(sides) < 4 or not all(s in sides for s in ['North','South','East','West']):
        return split_into_sides_curvature(xs, ys)
    return sides

def split_into_sides_curvature(xs, ys):
    n = len(xs)
    if n < 10:
        return split_into_sides_simple(xs, ys)
    curvature = np.zeros(n)
    for i in range(1, n-1):
        dx1, dy1 = xs[i]-xs[i-1], ys[i]-ys[i-1]
        dx2, dy2 = xs[i+1]-xs[i], ys[i+1]-ys[i]
        len1, len2 = np.hypot(dx1, dy1), np.hypot(dx2, dy2)
        if len1 > 1e-10 and len2 > 1e-10:
            curvature[i] = (dx1*dy2 - dy1*dx2) / (len1*len2)
    dist = np.zeros(n)
    for i in range(1, n):
        dist[i] = dist[i-1] + np.hypot(xs[i]-xs[i-1], ys[i]-ys[i-1])
    total_dist = dist[-1] + np.hypot(xs[0]-xs[-1], ys[0]-ys[-1])
    sorted_idx = np.argsort(np.abs(curvature))[::-1]
    corners = []
    for idx in sorted_idx:
        if len(corners) >= 4:
            break
        if all(min(abs(dist[idx]-dist[c]), total_dist-abs(dist[idx]-dist[c])) >= 0.1*total_dist for c in corners):
            corners.append(idx)
    if len(corners) < 4:
        return split_into_sides_simple(xs, ys)
    corners = sorted(corners, key=lambda i: dist[i])
    sides = {}
    for i in range(4):
        start, end = corners[i], corners[(i+1)%4]
        if start < end:
            idx = np.arange(start, end+1)
        else:
            idx = np.concatenate([np.arange(start, n), np.arange(0, end+1)])
        mx, my = np.mean(xs[idx]), np.mean(ys[idx])
        lon, lat = proj_to_lonlat(np.array([mx]), np.array([my]))
        lons_all, lats_all = proj_to_lonlat(xs, ys)
        lat_mean, lon_mean = np.mean(lats_all), np.mean(lons_all)
        name = 'North' if lat > lat_mean else 'South' if lat < lat_mean else 'East' if lon > lon_mean else 'West'
        sides[name] = {'xs': xs[idx], 'ys': ys[idx], 'indices': idx}
    return sides

def split_into_sides_simple(xs, ys):
    n = len(xs)
    if n < 4:
        return {}
    dist = np.zeros(n)
    for i in range(1, n):
        dist[i] = dist[i-1] + np.hypot(xs[i]-xs[i-1], ys[i]-ys[i-1])
    sorted_idx = np.argsort(dist)
    n_per_side = n // 4
    sides = {}
    for i in range(4):
        start, end = i*n_per_side, (i+1)*n_per_side if i<3 else n
        idx = sorted_idx[start:end]
        if len(idx) == 0:
            continue
        mx, my = np.mean(xs[idx]), np.mean(ys[idx])
        lon, lat = proj_to_lonlat(np.array([mx]), np.array([my]))
        lons_all, lats_all = proj_to_lonlat(xs, ys)
        lat_mean, lon_mean = np.mean(lats_all), np.mean(lons_all)
        name = 'North' if lat > lat_mean else 'South' if lat < lat_mean else 'East' if lon > lon_mean else 'West'
        sides[name] = {'xs': xs[idx], 'ys': ys[idx], 'indices': idx}
    return sides

def compute_month_flux_for_segments(year, month, segments, interp_u, interp_v):
    try:
        mid_x, mid_y = segments['mid_x'], segments['mid_y']
        lons, lats = proj_to_lonlat(mid_x, mid_y)
        points = np.column_stack((lats, lons))
        u_mid = interp_u(points)
        v_mid = interp_v(points)
        bad_mask = (np.abs(u_mid) > QC_THRESHOLD) | (np.abs(v_mid) > QC_THRESHOLD)
        if np.any(bad_mask):
            valid_mask = ~bad_mask
            if np.sum(valid_mask) > 2:
                interp_u_bad = NearestNDInterpolator(points[valid_mask], u_mid[valid_mask])
                interp_v_bad = NearestNDInterpolator(points[valid_mask], v_mid[valid_mask])
                u_mid[bad_mask] = interp_u_bad(points[bad_mask])
                v_mid[bad_mask] = interp_v_bad(points[bad_mask])
            else:
                u_mid[bad_mask] = np.nanmean(u_mid[valid_mask])
                v_mid[bad_mask] = np.nanmean(v_mid[valid_mask])
        normals, lengths = segments['normals'], segments['lengths']
        ivt_dot_n = u_mid * normals[:,0] + v_mid * normals[:,1]
        return np.sum(-np.minimum(ivt_dot_n, 0) * lengths), np.sum(np.maximum(ivt_dot_n, 0) * lengths)
    except Exception as e:
        return None, None

# ============================================================
# ۵. بارگذاری داده‌های مرزی
# ============================================================
def load_boundary_data():
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"فایل اکسل {EXCEL_PATH} یافت نشد.")
    xl = pd.ExcelFile(EXCEL_PATH)
    boundary_data = {}
    for sheet_name, sector_name in SHEET_NAMES.items():
        df_coords = pd.read_excel(xl, sheet_name=sheet_name, header=None, names=['lon','lat'])
        lons, lats = df_coords['lon'].values, df_coords['lat'].values
        coords = pd.DataFrame({'lon': lons, 'lat': lats}).drop_duplicates().values
        if len(coords) < len(lons):
            logger.info(f"{sheet_name}: {len(lons)-len(coords)} نقطه تکراری حذف شد.")
        lons, lats = coords[:,0], coords[:,1]
        if not np.allclose([lons[0], lats[0]], [lons[-1], lats[-1]]):
            lons = np.append(lons, lons[0])
            lats = np.append(lats, lats[0])
        boundary_data[sector_name] = {'lons': lons, 'lats': lats}
    return boundary_data

def prepare_sectors(boundary_data):
    sectors = {}
    for sector_name, data in boundary_data.items():
        lons, lats = data['lons'], data['lats']
        xs, ys = lonlat_to_proj(lons, lats)
        xs, ys = interpolate_boundary_spline(xs, ys, INTERPOLATE_POINTS)
        inside_x, inside_y = get_inside_point_proj(lons, lats)
        sides = split_into_sides_advanced(xs, ys)
        sector_sides = {}
        for side_name, side_data in sides.items():
            if len(side_data['xs']) < 3:
                continue
            segments = compute_segments_and_normals(side_data['xs'], side_data['ys'], inside_x, inside_y)
            if len(segments) == 0:
                continue
            sector_sides[side_name] = {
                'segments': segments,
                'xs': side_data['xs'],
                'ys': side_data['ys']
            }
        sectors[sector_name] = sector_sides
    return sectors

# ============================================================
# ۶. پردازش موازی با ThreadPool (بهینه برای I/O)
# ============================================================
def process_one_month_worker(args):
    """Worker برای پردازش یک ماه با interpolator از پیش ساخته شده"""
    year, month, sector_name, side_name, segments_data, interp_u, interp_v = args
    try:
        segments = {
            'mid_x': np.array(segments_data['mid_x']),
            'mid_y': np.array(segments_data['mid_y']),
            'lengths': np.array(segments_data['lengths']),
            'normals': np.array(segments_data['normals']),
            'xs': np.array(segments_data['xs']),
            'ys': np.array(segments_data['ys'])
        }
        infl, outfl = compute_month_flux_for_segments(year, month, segments, interp_u, interp_v)
        if infl is not None:
            return {'year': year, 'month': month, 'sector': sector_name, 'side': side_name, 'inflow': infl, 'outflow': outfl}
        return None
    except Exception as e:
        return None

# ============================================================
# ۷. توابع کش
# ============================================================
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            lines = f.readlines()
            if lines:
                return lines[0].strip()
    return None

def save_progress(last_key):
    with open(PROGRESS_FILE, 'w') as f:
        f.write(last_key)

def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

def save_results(results):
    with open(RESULTS_FILE, 'wb') as f:
        pickle.dump(results, f)

# ============================================================
# ۸. پردازش اصلی با ThreadPool
# ============================================================
def process_file_by_file_optimized():
    logger.info("🚀 شروع پردازش فوق‌بهینه با ThreadPool...")
    start_time = time.time()
    
    # بارگذاری داده‌های مرزی
    boundary_data = load_boundary_data()
    sectors = prepare_sectors(boundary_data)
    
    # بارگذاری یکباره داده‌های Zarr
    ds = load_all_ivt_data()
    
    # ساخت یکباره Interpolatorها
    interpolators = build_interpolators_for_all_times(ds)
    
    # بارگذاری نتایج قبلی و وضعیت پیشرفت
    results = load_results()
    last_key = load_progress()
    
    # ساخت لیست تمام ماه‌ها
    all_months = [(y, m) for y in YEARS for m in MONTHS]
    total_months = len(all_months)
    
    logger.info(f"تعداد کل ماه‌ها: {total_months}")
    logger.info(f"تعداد نتایج موجود: {len(results)}")
    
    # تعیین ماه شروع
    start_index = 0
    if last_key and last_key != "DONE":
        parts = last_key.split('_')
        if len(parts) == 4:
            sector, side, year_str, month_str = parts
            last_year, last_month = int(year_str), int(month_str)
            for idx, (y, m) in enumerate(all_months):
                if y == last_year and m == last_month:
                    start_index = idx + 1
                    break
            logger.info(f"ادامه از ماه: {all_months[start_index] if start_index < total_months else 'پایان'}")
    
    # ساخت لیست کارهای باقی‌مانده
    tasks = []
    for idx in range(start_index, total_months):
        year, month = all_months[idx]
        # محاسبه ایندکس زمانی
        time_idx = (year - 1940) * 12 + (month - 1)
        if time_idx < 0 or time_idx >= len(interpolators):
            continue
        interp_u, interp_v = interpolators[time_idx]
        for sector_name, sector_sides in sectors.items():
            for side_name, side_data in sector_sides.items():
                key = f"{sector_name}_{side_name}_{year}_{month:02d}"
                if key in results:
                    continue
                seg_data = {
                    'mid_x': side_data['segments']['mid_x'].tolist(),
                    'mid_y': side_data['segments']['mid_y'].tolist(),
                    'lengths': side_data['segments']['lengths'].tolist(),
                    'normals': side_data['segments']['normals'].tolist(),
                    'xs': side_data['segments']['xs'].tolist(),
                    'ys': side_data['segments']['ys'].tolist()
                }
                tasks.append((year, month, sector_name, side_name, seg_data, interp_u, interp_v))
    
    if not tasks:
        logger.info("✅ همه ماه‌ها قبلاً پردازش شده‌اند.")
    else:
        logger.info(f"تعداد کارهای باقی‌مانده: {len(tasks)}")
        
        # پردازش با ThreadPool
        logger.info(f"شروع پردازش با {N_THREADS} ترد...")
        processed = 0
        with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
            futures = {executor.submit(process_one_month_worker, task): task for task in tasks}
            for future in tqdm(as_completed(futures), total=len(tasks), desc="پردازش", unit="task"):
                try:
                    res = future.result(timeout=120)
                    if res is not None:
                        key = f"{res['sector']}_{res['side']}_{res['year']}_{res['month']:02d}"
                        results[key] = res
                        processed += 1
                    if processed % CACHE_SAVE_INTERVAL == 0:
                        save_results(results)
                        last_task = futures[future]
                        save_progress(f"{last_task[2]}_{last_task[3]}_{last_task[0]}_{last_task[1]:02d}")
                        logger.info(f"پیشرفت: {processed}/{len(tasks)} کار جدید (ذخیره شد)")
                except Exception as e:
                    logger.error(f"خطا در پردازش یک کار: {str(e)}")
        
        # ذخیره نهایی
        save_results(results)
        save_progress("DONE")
        logger.info(f"✅ {processed} کار جدید پردازش شد.")
    
    # ============================================================
    # ۹. تولید فایل‌های نهایی
    # ============================================================
    logger.info("📊 تولید فایل‌های نهایی...")
    
    all_records = []
    for key, val in results.items():
        all_records.append(val)
    
    if not all_records:
        logger.error("هیچ نتیجه‌ای وجود ندارد!")
        return
    
    df_all = pd.DataFrame(all_records)
    
    for sector_name in SHEET_NAMES.values():
        df_sector = df_all[df_all['sector'] == sector_name]
        if df_sector.empty:
            continue
        
        # سری ماهانه
        df_monthly = df_sector.pivot_table(
            index=['year', 'month'],
            columns='side',
            values=['inflow', 'outflow'],
            aggfunc='first'
        ).reset_index()
        
        df_monthly.columns = ['year', 'month'] + [f'{col[1]}_{col[0]}' for col in df_monthly.columns[2:]]
        df_monthly.to_csv(os.path.join(OUTPUT_BASE, f'monthly_{sector_name}.csv'), index=False, encoding='utf-8-sig')
        logger.info(f"✅ monthly_{sector_name}.csv")
        
        # سری سالانه
        df_annual = df_monthly.groupby('year').mean(numeric_only=True).reset_index()
        df_annual.insert(0, 'sector', sector_name)
        df_annual.to_csv(os.path.join(OUTPUT_BASE, f'annual_{sector_name}.csv'), index=False, encoding='utf-8-sig')
        logger.info(f"✅ annual_{sector_name}.csv")
        
        # آمار دوره‌ها
        df_merged = df_monthly.copy()
        df_merged['sector'] = sector_name
        period_stats = []
        periods = list(SPECIAL_PERIODS.items())
        for i, (pname1, years1) in enumerate(periods):
            for pname2, years2 in periods[i+1:]:
                mask1 = df_merged['year'].isin(years1)
                mask2 = df_merged['year'].isin(years2)
                if mask1.sum()==0 or mask2.sum()==0:
                    continue
                num_cols = [c for c in df_merged.columns if 'inflow' in c or 'outflow' in c]
                row = {'period1': pname1, 'period2': pname2}
                for col in num_cols:
                    vals1 = df_merged.loc[mask1, col].dropna().values
                    vals2 = df_merged.loc[mask2, col].dropna().values
                    if len(vals1) > 1 and len(vals2) > 1:
                        monthly_means1 = df_merged.loc[mask1].groupby('month')[col].mean()
                        monthly_means2 = df_merged.loc[mask2].groupby('month')[col].mean()
                        anomalies1 = vals1 - np.array([monthly_means1.get(m,0) for m in df_merged.loc[mask1,'month']])
                        anomalies2 = vals2 - np.array([monthly_means2.get(m,0) for m in df_merged.loc[mask2,'month']])
                        t_stat, p_val = ttest_ind(anomalies1, anomalies2, equal_var=False)
                        row[f'{col}_mean1'] = np.mean(vals1)
                        row[f'{col}_std1'] = np.std(vals1)
                        row[f'{col}_mean2'] = np.mean(vals2)
                        row[f'{col}_std2'] = np.std(vals2)
                        row[f'{col}_pval'] = p_val
                period_stats.append(row)
        if period_stats:
            pd.DataFrame(period_stats).to_csv(os.path.join(OUTPUT_BASE, f'period_stats_{sector_name}.csv'), index=False, encoding='utf-8-sig')
            logger.info(f"✅ period_stats_{sector_name}.csv")
        
        # Conservation
        sides = [c.split('_')[0] for c in df_monthly.columns if '_inflow' in c]
        if len(sides) == 4:
            df_merged['total_inflow'] = df_merged[[f'{s}_inflow' for s in sides]].sum(axis=1)
            df_merged['total_outflow'] = df_merged[[f'{s}_outflow' for s in sides]].sum(axis=1)
            df_merged['net_flux'] = df_merged['total_inflow'] - df_merged['total_outflow']
            df_merged['conservation_error'] = np.abs(df_merged['net_flux']) / (df_merged['total_inflow'] + df_merged['total_outflow'] + 1e-10) * 100
            df_merged[['year','month','total_inflow','total_outflow','net_flux','conservation_error']].to_csv(
                os.path.join(OUTPUT_BASE, f'conservation_{sector_name}.csv'), index=False, encoding='utf-8-sig')
            logger.info(f"✅ conservation_{sector_name}.csv")
        
        # خلاصه
        df_merged.describe(percentiles=[.25,.5,.75]).to_csv(os.path.join(OUTPUT_BASE, f'summary_{sector_name}.csv'), encoding='utf-8-sig')
        logger.info(f"✅ summary_{sector_name}.csv")
    
    # فایل پایان کار
    done_file = os.path.join(OUTPUT_BASE, 'DONE.txt')
    with open(done_file, 'w', encoding='utf-8') as f:
        f.write(f"پردازش در {time.ctime()} کامل شد.\n")
        f.write(f"تعداد کل رکوردها: {len(all_records)}\n")
        f.write(f"تعداد فایل‌های خروجی: {len(os.listdir(OUTPUT_BASE))}\n")
        f.write("همه فایل‌های CSV مستقل از داده خام هستند.\n")
        f.write("برای انتقال به K: کافی است کل پوشه کپی شود.\n")
    
    elapsed = time.time() - start_time
    logger.info(f"✅ تمام محاسبات در {elapsed:.1f} ثانیه انجام شد.")
    logger.info(f"📂 خروجی‌ها در: {OUTPUT_BASE}")

if __name__ == "__main__":
    process_file_by_file_optimized()