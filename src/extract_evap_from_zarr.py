#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
================================================================================
استخراج تبخیر ماهانه از فایل‌های Zarr سالانه (۱۹۴۰–۲۰۲۵) – نسخه نهایی
================================================================================
- تبخیر روزانه بر حسب متر (با علامت منفی) → تبدیل به میلی‌متر مثبت (×-1000)
- میانگین روزانه هر ماه → ضربدر تعداد روزهای آن ماه
- میانگین وزنی مساحت با earth_radius_function
- ماسک‌های حوضه آبریز و دریای خزر از Shapefile
- پردازش موازی با ThreadPoolExecutor
- خروجی: Excel با سری‌های زمانی ماهانه
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
EVAP_DIR = r"I:\evaporation\evaporation_zarr_yearly"
OUTPUT_DIR = r"I:\caspian_climate_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SHP_BASIN = r"K:\gozareshha\Dr Farjami\Dr Farjami\140503\LAND3.shp"
SHP_LAKE = r"K:\gozareshha\Dr Farjami\Dr Farjami\140503\caspian_lake_boundary\caspian_polygon_fixed.shp"

START_YEAR = 1940
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
    zarr_files = sorted(glob.glob(os.path.join(EVAP_DIR, "*.zarr")))
    if not zarr_files:
        raise FileNotFoundError("هیچ فایل Zarr تبخیر یافت نشد.")
    ds = xr.open_zarr(zarr_files[0], consolidated=False)
    lons = ds.longitude.values
    lats = ds.latitude.values
    res_x = np.abs(np.diff(lons)[0]) if len(lons) > 1 else 1
    res_y = np.abs(np.diff(lats)[0]) if len(lats) > 1 else 1
    ds.close()
    return lons, lats, res_x, res_y

# ============================================================
# ۴. پردازش یک فایل Zarr سالانه (تبخیر)
# ============================================================
def process_evap_file(zarr_path, mask, lons_ref, lats_ref, pixel_area):
    try:
        match = re.search(r'(\d{4})\.zarr', os.path.basename(zarr_path))
        if not match:
            return None
        year = int(match.group(1))
        if year < START_YEAR or year > END_YEAR:
            return None

        ds = xr.open_zarr(zarr_path, consolidated=False)
        evap = ds['e']  # (valid_time, lat, lon) – متر در روز (علامت منفی)

        evap_mm = evap * -1000  # تبدیل به میلی‌متر مثبت

        lats = evap.latitude.values
        if lats[0] < lats[-1]:
            evap_mm = evap_mm.isel(latitude=slice(None, None, -1))
            lats = lats[::-1]

        lons = evap.longitude.values
        if len(lons) != len(lons_ref) or len(lats) != len(lats_ref):
            evap_mm = evap_mm.interp(latitude=lats_ref, longitude=lons_ref, method='linear')

        evap_values = evap_mm.values
        pixel_area_2d = np.ones((len(lats_ref), len(lons_ref))) * pixel_area

        mask_bool = mask.astype(bool)
        if not mask_bool.any():
            return None

        daily_means = []
        for t in range(evap_values.shape[0]):
            day_data = evap_values[t, :, :]
            masked_vals = day_data[mask_bool]
            masked_weights = pixel_area_2d[mask_bool]
            daily_mean = np.average(masked_vals, weights=masked_weights)
            daily_means.append(daily_mean)

        times = ds.valid_time.values
        df_temp = pd.DataFrame({
            'date': times,
            'evap_daily': daily_means
        })
        df_temp['year'] = pd.DatetimeIndex(df_temp['date']).year.astype(int)
        df_temp['month'] = pd.DatetimeIndex(df_temp['date']).month.astype(int)

        monthly_avg = df_temp.groupby(['year', 'month'])['evap_daily'].mean().reset_index()
        monthly_avg['year'] = monthly_avg['year'].astype(int)
        monthly_avg['month'] = monthly_avg['month'].astype(int)
        monthly_avg['days'] = monthly_avg.apply(
            lambda row: pd.Timestamp(year=int(row['year']), month=int(row['month']), day=1).days_in_month,
            axis=1
        )
        monthly_avg['evap_mm'] = monthly_avg['evap_daily'] * monthly_avg['days']

        results = []
        for _, row in monthly_avg.iterrows():
            results.append({
                'year': int(row['year']),
                'month': int(row['month']),
                'date': pd.Timestamp(year=int(row['year']), month=int(row['month']), day=1),
                'evap_mm': row['evap_mm']
            })

        ds.close()
        return results

    except Exception as e:
        print(f"⚠️ خطا در {os.path.basename(zarr_path)}: {e}")
        return None

# ============================================================
# ۵. پردازش موازی
# ============================================================
def process_all_files(file_list, mask, lons_ref, lats_ref, pixel_area, desc):
    all_results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_evap_file, f, mask, lons_ref, lats_ref, pixel_area): f
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
    print("🌊 استخراج تبخیر ماهانه از فایل‌های Zarr سالانه (۱۹۴۰–۲۰۲۵)")
    print("=" * 80)

    print("\n📂 خواندن مرزها...")
    polygon_basin = get_polygon_from_shp(SHP_BASIN)
    polygon_lake = get_polygon_from_shp(SHP_LAKE)
    print(f"✅ حوضه آبریز: {polygon_basin.area:.2f} درجه مربع")
    print(f"✅ دریای خزر: {polygon_lake.area:.2f} درجه مربع")

    print("\n📐 استخراج شبکه مرجع...")
    lons_ref, lats_ref, res_x, res_y = get_reference_grid()
    pixel_area = get_pixel_area(lats_ref, res_x, res_y)
    print(f"✅ ابعاد شبکه: lat {len(lats_ref)} × lon {len(lons_ref)}")

    print("\n🔄 ایجاد ماسک‌ها...")
    mask_basin = create_mask_from_polygon(polygon_basin, lons_ref, lats_ref)
    mask_lake = create_mask_from_polygon(polygon_lake, lons_ref, lats_ref)
    print(f"✅ ماسک حوضه: {int(mask_basin.sum())} پیکسل")
    print(f"✅ ماسک دریاچه: {int(mask_lake.sum())} پیکسل")

    zarr_files = sorted(glob.glob(os.path.join(EVAP_DIR, "*.zarr")))
    filtered_files = []
    for f in zarr_files:
        match = re.search(r'(\d{4})\.zarr', os.path.basename(f))
        if match:
            year = int(match.group(1))
            if START_YEAR <= year <= END_YEAR:
                filtered_files.append(f)
    zarr_files = filtered_files
    print(f"\n📂 تعداد فایل‌ها: {len(zarr_files)}")

    print("\n🔄 پردازش تبخیر (حوضه آبریز)...")
    evap_basin_raw = process_all_files(zarr_files, mask_basin, lons_ref, lats_ref, pixel_area, "تبخیر - حوضه")

    print("\n🔄 پردازش تبخیر (دریای خزر)...")
    evap_lake_raw = process_all_files(zarr_files, mask_lake, lons_ref, lats_ref, pixel_area, "تبخیر - دریاچه")

    # 🔧 اصلاح: تبدیل به DataFrame با تغییر نام ستون‌ها
    if evap_basin_raw:
        df_basin = pd.DataFrame(evap_basin_raw)
        df_basin.rename(columns={'evap_mm': 'evap_basin'}, inplace=True)
    else:
        df_basin = pd.DataFrame(columns=['year', 'month', 'evap_basin'])

    if evap_lake_raw:
        df_lake = pd.DataFrame(evap_lake_raw)
        df_lake.rename(columns={'evap_mm': 'evap_lake'}, inplace=True)
    else:
        df_lake = pd.DataFrame(columns=['year', 'month', 'evap_lake'])

    # ادغام
    df_merged = df_basin[['year', 'month', 'evap_basin']].merge(
        df_lake[['year', 'month', 'evap_lake']],
        on=['year', 'month'],
        how='outer'
    )
    df_merged = df_merged.sort_values(['year', 'month']).reset_index(drop=True)

    # ذخیره
    excel_path = os.path.join(OUTPUT_DIR, 'caspian_evap_1940_2025.xlsx')
    print(f"\n💾 ذخیره در Excel: {excel_path}")
    df_merged.to_excel(excel_path, sheet_name='Evaporation', index=False)
    print(f"✅ Excel ذخیره شد.")

    csv_path = os.path.join(OUTPUT_DIR, 'caspian_evap_1940_2025.csv')
    df_merged.to_csv(csv_path, index=False)
    print(f"✅ CSV ذخیره شد.")

    print("\n📊 خلاصه:")
    for col in ['evap_basin', 'evap_lake']:
        if col in df_merged.columns and not df_merged[col].isna().all():
            valid = df_merged[col].dropna()
            print(f"   {col}: {len(valid)} مقدار معتبر (میانگین {valid.mean():.2f} mm)")

    print(f"\n📂 خروجی: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()