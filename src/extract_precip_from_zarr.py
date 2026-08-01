#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
================================================================================
استخراج بارش روزانه از فایل‌های Zarr روزانه (۱۹۶۵–۲۰۲۵) – خروجی روزانه
================================================================================
- بارش روزانه بر حسب متر → تبدیل به میلی‌متر (×۱۰۰۰)
- میانگین وزنی مساحت با earth_radius_function
- ماسک‌های حوضه آبریز و دریای خزر از Shapefile
- پردازش موازی با ThreadPoolExecutor
- خروجی: Excel و CSV با سری‌های زمانی روزانه
================================================================================
"""

import os
import re
import glob
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# ۱. تنظیمات
# ============================================================
ZARR_DIR = r"I:\rrr24"
OUTPUT_DIR = r"I:\caspian_climate_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SHP_BASIN = r"K:\gozareshha\Dr Farjami\Dr Farjami\140503\LAND3.shp"
SHP_LAKE = r"K:\gozareshha\Dr Farjami\Dr Farjami\140503\caspian_lake_boundary\caspian_polygon_fixed.shp"

START_YEAR = 1965
END_YEAR = 2025
MAX_WORKERS = 8

# ============================================================
# ۲. توابع هندسی
# ============================================================
def earth_radius_function(lats):
    a = 6378137.0
    b = 6356752.3
    return np.sqrt(
        (np.power(a * a * np.cos(np.radians(lats)), 2) +
         np.power(b * b * np.sin(np.radians(lats)), 2)) /
        (np.power(a * np.cos(np.radians(lats)), 2) +
         np.power(b * np.sin(np.radians(lats)), 2))
    )

def get_pixel_area(lats, res_x, res_y):
    radius = earth_radius_function(lats)
    pixel_area = -np.radians(res_x) * np.radians(res_y) * (radius ** 2) * np.cos(np.radians(lats))
    return pixel_area[:, np.newaxis]

def get_polygon_from_shp(shp_path):
    gdf = gpd.read_file(shp_path)
    if len(gdf) > 1:
        polygon = gdf.union_all()
    else:
        polygon = gdf.geometry.iloc[0]
    if gdf.crs is not None and not gdf.crs.is_geographic:
        polygon = gdf.to_crs("EPSG:4326").geometry.iloc[0]
    return polygon

def create_mask_from_polygon(polygon, lons, lats):
    lon2d, lat2d = np.meshgrid(lons, lats, indexing='xy')
    points = np.column_stack([lon2d.ravel(), lat2d.ravel()])
    mask_flat = np.array([polygon.contains(Point(p)) for p in points])
    mask = mask_flat.reshape(lon2d.shape)
    return mask

# ============================================================
# ۳. شبکه مرجع
# ============================================================
def get_reference_grid():
    zarr_files = sorted(glob.glob(os.path.join(ZARR_DIR, "*.zarr")))
    if not zarr_files:
        raise FileNotFoundError("هیچ فایل Zarr یافت نشد.")
    ds = xr.open_zarr(zarr_files[0], consolidated=False)
    lons = ds.longitude.values
    lats = ds.latitude.values
    res_x = np.abs(np.diff(lons)[0]) if len(lons) > 1 else 1
    res_y = np.abs(np.diff(lats)[0]) if len(lats) > 1 else 1
    ds.close()
    return lons, lats, res_x, res_y

# ============================================================
# ۴. پردازش یک فایل Zarr (خروجی روزانه)
# ============================================================
def process_zarr_file_daily(zarr_path, mask, lons_ref, lats_ref, pixel_area):
    try:
        match = re.search(r'(\d{6})\.zarr', os.path.basename(zarr_path))
        if not match:
            return None
        yyyymm = match.group(1)
        month_start = pd.to_datetime(yyyymm, format='%Y%m')

        ds = xr.open_zarr(zarr_path, consolidated=False)
        tp = ds['tp']  # (valid_time, lat, lon)

        tp_mm = tp * 1000

        lats = tp.latitude.values
        lons = tp.longitude.values
        if lats[0] < lats[-1]:
            tp_mm = tp_mm.isel(latitude=slice(None, None, -1))
            lats = lats[::-1]

        if len(lons) != len(lons_ref) or len(lats) != len(lats_ref):
            tp_mm = tp_mm.interp(latitude=lats_ref, longitude=lons_ref, method='linear')

        tp_mm_values = tp_mm.values
        pixel_area_2d = np.ones((len(lats_ref), len(lons_ref))) * pixel_area

        mask_bool = mask.astype(bool)
        if not mask_bool.any():
            return None

        # محاسبه میانگین وزنی برای هر روز
        daily_results = []
        for t in range(tp_mm_values.shape[0]):
            day_data = tp_mm_values[t, :, :]
            masked_vals = day_data[mask_bool]
            masked_weights = pixel_area_2d[mask_bool]
            daily_mean = np.average(masked_vals, weights=masked_weights)
            day_date = month_start + pd.Timedelta(days=t)
            daily_results.append({
                'date': day_date,
                'precip_mm': daily_mean
            })

        ds.close()
        return daily_results

    except Exception as e:
        print(f"⚠️ خطا در {os.path.basename(zarr_path)}: {e}")
        return None

# ============================================================
# ۵. پردازش موازی
# ============================================================
def process_all_files_daily(file_list, mask, lons_ref, lats_ref, pixel_area, desc):
    all_results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_zarr_file_daily, f, mask, lons_ref, lats_ref, pixel_area): f
            for f in file_list
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            result = future.result()
            if result is not None:
                all_results.extend(result)
    return all_results

# ============================================================
# ۶. اجرای اصلی
# ============================================================
def main():
    print("=" * 80)
    print("🌧️ استخراج بارش روزانه از فایل‌های Zarr روزانه (۱۹۶۵–۲۰۲۵)")
    print("=" * 80)

    # خواندن مرزها
    print("\n📂 خواندن مرزها...")
    polygon_basin = get_polygon_from_shp(SHP_BASIN)
    polygon_lake = get_polygon_from_shp(SHP_LAKE)
    print(f"✅ حوضه آبریز: {polygon_basin.area:.2f} درجه مربع")
    print(f"✅ دریای خزر: {polygon_lake.area:.2f} درجه مربع")

    # شبکه مرجع
    print("\n📐 استخراج شبکه مرجع...")
    lons_ref, lats_ref, res_x, res_y = get_reference_grid()
    pixel_area = get_pixel_area(lats_ref, res_x, res_y)
    print(f"✅ ابعاد شبکه: lat {len(lats_ref)} × lon {len(lons_ref)}")

    # ماسک‌ها
    print("\n🔄 ایجاد ماسک‌ها...")
    mask_basin = create_mask_from_polygon(polygon_basin, lons_ref, lats_ref)
    mask_lake = create_mask_from_polygon(polygon_lake, lons_ref, lats_ref)
    print(f"✅ ماسک حوضه: {int(mask_basin.sum())} پیکسل")
    print(f"✅ ماسک دریاچه: {int(mask_lake.sum())} پیکسل")

    # لیست فایل‌ها
    zarr_files = sorted(glob.glob(os.path.join(ZARR_DIR, "*.zarr")))
    filtered_files = []
    for f in zarr_files:
        match = re.search(r'(\d{4})(\d{2})\.zarr', os.path.basename(f))
        if match:
            year = int(match.group(1))
            if START_YEAR <= year <= END_YEAR:
                filtered_files.append(f)
    zarr_files = filtered_files
    print(f"\n📂 تعداد فایل‌ها: {len(zarr_files)}")

    # پردازش بارش برای حوضه
    print("\n🔄 پردازش بارش روزانه (حوضه آبریز)...")
    precip_basin = process_all_files_daily(zarr_files, mask_basin, lons_ref, lats_ref, pixel_area, "بارش روزانه - حوضه")

    # پردازش بارش برای دریاچه
    print("\n🔄 پردازش بارش روزانه (دریای خزر)...")
    precip_lake = process_all_files_daily(zarr_files, mask_lake, lons_ref, lats_ref, pixel_area, "بارش روزانه - دریاچه")

    # تبدیل به DataFrame
    df_basin = pd.DataFrame(precip_basin) if precip_basin else pd.DataFrame(columns=['date', 'precip_basin'])
    df_lake = pd.DataFrame(precip_lake) if precip_lake else pd.DataFrame(columns=['date', 'precip_lake'])

    # ادغام
    df_merged = df_basin.merge(df_lake, on='date', how='outer')
    df_merged = df_merged.sort_values('date').reset_index(drop=True)

    # استخراج سال و ماه برای راحتی
    df_merged['year'] = df_merged['date'].dt.year
    df_merged['month'] = df_merged['date'].dt.month
    df_merged['day'] = df_merged['date'].dt.day

    # ذخیره
    excel_path = os.path.join(OUTPUT_DIR, 'caspian_precip_daily_1965_2025.xlsx')
    print(f"\n💾 ذخیره در Excel: {excel_path}")
    df_merged.to_excel(excel_path, sheet_name='Daily Precipitation', index=False)
    print(f"✅ Excel ذخیره شد.")

    csv_path = os.path.join(OUTPUT_DIR, 'caspian_precip_daily_1965_2025.csv')
    df_merged.to_csv(csv_path, index=False)
    print(f"✅ CSV ذخیره شد.")

    # گزارش
    print("\n📊 خلاصه:")
    for col in ['precip_basin', 'precip_lake']:
        if col in df_merged.columns and not df_merged[col].isna().all():
            valid = df_merged[col].dropna()
            print(f"   {col}: {len(valid)} مقدار معتبر (میانگین {valid.mean():.2f} mm)")

    print(f"\n📂 خروجی: {OUTPUT_DIR}")
    print("✅ تمام شد. فایل‌ها در پوشه خروجی ذخیره شده‌اند.")

if __name__ == "__main__":
    main()