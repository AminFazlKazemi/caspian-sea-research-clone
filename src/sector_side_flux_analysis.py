#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
محاسبه شار رطوبتی ورودی و خروجی برای هر ضلع (شمال، جنوب، شرق، غرب)
در سه بخش شمالی، میانی و جنوبی دریای خزر
با موازی‌سازی و نمایش پیشرفت
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata
import concurrent.futures
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ============================================
# ۱. تنظیمات مسیرها
# ============================================
EXCEL_PATH = r"K:\gozareshha\Dr Farjami\Dr Farjami\140503\North_Center_South_Caspian Sea.xlsx"
IVT_ZARR_BASE = r"J:\ivt_data"
OUTPUT_BASE = r"K:\gozareshha\Dr Farjami\Dr Farjami\140503\sector_side_flux"
os.makedirs(OUTPUT_BASE, exist_ok=True)

CACHE_FILE = os.path.join(OUTPUT_BASE, 'all_monthly_fluxes.pkl')
SHEET_NAMES = {'S_CS': 'South', 'C_CS': 'Center', 'N_CS': 'North'}
SPECIAL_PERIODS = {
    'Low_Water': list(range(1976, 1979)),   # 1976-1978
    'High_Water': list(range(1994, 1997))   # 1994-1996
}
YEARS = list(range(1940, 2026))

# ============================================
# ۲. توابع کمکی (بدون تغییر)
# ============================================
def haversine_distance(lon1, lat1, lon2, lat2):
    R = 6371
    dlon = np.radians(lon2 - lon1)
    dlat = np.radians(lat2 - lat1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def compute_normals(lons, lats):
    n = len(lons)
    normals = np.zeros((n, 2))
    seg_lengths = np.zeros(n)
    for i in range(n):
        j = (i + 1) % n
        dx = lons[j] - lons[i]
        dy = lats[j] - lats[i]
        length = np.sqrt(dx**2 + dy**2)
        seg_lengths[i] = length
        if length > 0:
            normals[i, 0] = -dy / length
            normals[i, 1] = dx / length
    return normals, seg_lengths

def split_into_sides(lons, lats):
    center_lon = np.mean(lons)
    center_lat = np.mean(lats)
    angles = np.degrees(np.arctan2(lats - center_lat, lons - center_lon))
    angles = (angles + 360) % 360
    side_labels = []
    for a in angles:
        if 45 <= a < 135:
            side_labels.append('North')
        elif 135 <= a < 225:
            side_labels.append('West')
        elif 225 <= a < 315:
            side_labels.append('South')
        else:
            side_labels.append('East')
    side_labels = np.array(side_labels)
    sides = {}
    for side in ['North', 'South', 'East', 'West']:
        mask = (side_labels == side)
        if np.sum(mask) > 0:
            sides[side] = {
                'lons': lons[mask],
                'lats': lats[mask],
                'indices': np.where(mask)[0]
            }
    return sides

# ============================================
# ۳. محاسبه شار برای یک ماه (برای یک مجموعه نقطه)
# ============================================
def compute_flux_for_points(year, month, lons, lats, normals, seg_lengths):
    ivt_path = os.path.join(IVT_ZARR_BASE, f"{year}{month:02d}.zarr")
    if not os.path.exists(ivt_path):
        return np.nan, np.nan
    try:
        ds = xr.open_zarr(ivt_path)
        lat_min, lat_max = lats.min()-0.5, lats.max()+0.5
        lon_min, lon_max = lons.min()-0.5, lons.max()+0.5
        u = ds.viwve.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
        v = ds.viwvn.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
        u_mean = u.mean(dim='valid_time').load()
        v_mean = v.mean(dim='valid_time').load()

        lon_crop = u_mean.longitude.values
        lat_crop = u_mean.latitude.values
        lon2d, lat2d = np.meshgrid(lon_crop, lat_crop, indexing='ij')
        points = np.column_stack((lon2d.ravel(), lat2d.ravel()))
        u_flat = u_mean.values.ravel()
        v_flat = v_mean.values.ravel()

        u_border = griddata(points, u_flat, (lons, lats), method='linear')
        v_border = griddata(points, v_flat, (lons, lats), method='linear')

        ivt_dot_n = u_border * normals[:,0] + v_border * normals[:,1]
        inflow = np.sum(-np.minimum(ivt_dot_n, 0) * seg_lengths) * 1000.0
        outflow = np.sum(np.maximum(ivt_dot_n, 0) * seg_lengths) * 1000.0
        ds.close()
        return inflow, outflow
    except:
        return np.nan, np.nan

# ============================================
# ۴. تابع پردازش یک ماه کامل (همه بخش‌ها و ضلع‌ها)
# ============================================
def process_all_sectors_for_month(year, month, sector_data):
    """
    sector_data: دیکشنری شامل اطلاعات هر بخش (نقاط، نرمال‌ها، ضلع‌ها)
    خروجی: دیکشنری با کلیدهای (sector, side)_inflow و _outflow
    """
    result = {'year': year, 'month': month}
    for sector_name, sides in sector_data.items():
        for side, data in sides.items():
            inflow, outflow = compute_flux_for_points(
                year, month,
                data['lons'], data['lats'],
                data['normals'], data['seg_lengths']
            )
            result[f'{sector_name}_{side}_inflow'] = inflow
            result[f'{sector_name}_{side}_outflow'] = outflow
    return result

# ============================================
# ۵. بارگذاری/ساخت کش
# ============================================
def build_cache():
    print("📂 خواندن فایل اکسل و آماده‌سازی داده‌های بخش‌ها...")
    xl = pd.ExcelFile(EXCEL_PATH)
    sector_data = {}

    for sheet_name, sector_name in SHEET_NAMES.items():
        df_coords = pd.read_excel(xl, sheet_name=sheet_name, header=None, names=['lon', 'lat'])
        lons = df_coords['lon'].values
        lats = df_coords['lat'].values

        # حذف تکراری‌ها
        coords = np.column_stack((lons, lats))
        unique_coords = np.unique(coords, axis=0)
        if len(unique_coords) < len(coords):
            print(f"   ⚠️ {sheet_name}: {len(coords)-len(unique_coords)} نقطه تکراری حذف شد.")
        lons, lats = unique_coords[:,0], unique_coords[:,1]

        # بسته کردن پلی‌لاین
        if not np.allclose([lons[0], lats[0]], [lons[-1], lats[-1]]):
            lons = np.append(lons, lons[0])
            lats = np.append(lats, lats[0])

        sides = split_into_sides(lons, lats)
        # محاسبه نرمال‌ها برای هر ضلع
        sides_with_normals = {}
        for side, data in sides.items():
            if len(data['lons']) < 3:
                continue
            normals, seg_lengths = compute_normals(data['lons'], data['lats'])
            sides_with_normals[side] = {
                'lons': data['lons'],
                'lats': data['lats'],
                'normals': normals,
                'seg_lengths': seg_lengths
            }
        sector_data[sector_name] = sides_with_normals

    print("✅ داده‌های بخش‌ها آماده شد.")
    return sector_data

def get_cached_or_compute():
    if os.path.exists(CACHE_FILE):
        print(f"📂 کش قبلی یافت شد: {CACHE_FILE}")
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)

    sector_data = build_cache()
    all_tasks = [(y, m) for y in YEARS for m in range(1, 13)]
    print(f"🔄 شروع محاسبه {len(all_tasks)} ماه با {os.cpu_count()} هسته...")

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()-1) as executor:
        futures = {executor.submit(process_all_sectors_for_month, y, m, sector_data): (y, m)
                   for y, m in all_tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="پیشرفت"):
            res = future.result()
            if res is not None:
                results.append(res)

    # ذخیره در کش
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(results, f)
    print(f"✅ کش در {CACHE_FILE} ذخیره شد.")
    return results

# ============================================
# ۶. تولید خروجی‌های نهایی از داده‌های کش
# ============================================
def generate_outputs(cached_data):
    df_all = pd.DataFrame(cached_data)
    sectors = list(SHEET_NAMES.values())
    sides = ['North', 'South', 'East', 'West']

    # ۱. سری زمانی ماهانه برای هر بخش و ضلع
    for sector in sectors:
        cols = ['year', 'month']
        for side in sides:
            cols.append(f'{sector}_{side}_inflow')
            cols.append(f'{sector}_{side}_outflow')
        df_sector = df_all[cols].copy()
        df_sector.to_csv(os.path.join(OUTPUT_BASE, f'monthly_{sector}.csv'), index=False)
        print(f"   ✅ monthly_{sector}.csv")

    # ۲. سری سالانه
    df_annual = df_all.groupby('year').mean(numeric_only=True).reset_index()
    df_annual.to_csv(os.path.join(OUTPUT_BASE, 'annual_all_sectors.csv'), index=False)
    print(f"   ✅ annual_all_sectors.csv")

    # ۳. میانگین دوره‌های خاص
    period_list = []
    for period_name, years_list in SPECIAL_PERIODS.items():
        mask = df_all['year'].isin(years_list)
        if mask.sum() == 0:
            continue
        period_mean = df_all[mask].mean(numeric_only=True).to_frame().T
        period_mean.insert(0, 'period', period_name)
        period_list.append(period_mean)
    if period_list:
        df_periods = pd.concat(period_list, ignore_index=True)
        df_periods.to_csv(os.path.join(OUTPUT_BASE, 'period_means_all_sectors.csv'), index=False)
        print(f"   ✅ period_means_all_sectors.csv")

    print("✅ تمام خروجی‌ها تولید شد.")

# ============================================
# ۷. اجرای اصلی
# ============================================
if __name__ == '__main__':
    print("🚀 شروع پردازش...")
    cached = get_cached_or_compute()
    print("📊 تولید خروجی‌های نهایی...")
    generate_outputs(cached)
    print(f"\n✅ تمام محاسبات با موفقیت انجام شد.")
    print(f"📂 خروجی‌ها در: {OUTPUT_BASE}")