#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
================================================================================
استخراج سری‌های زمانی ماهانه بارش و دما از فایل‌های TIF
با میانگین وزنی مساحت (روش earth_radius) – نسخه نهایی
================================================================================
- بارش: K:\WMO\prec\pYYYYMM.tif (میلی‌متر در ماه)
- دما: K:\WMO\tmean\tavgYYYYMM.tif (درجه سلسیوس)
- مرز حوضه: LAND3.shp
- مرز دریاچه: caspian_polygon_fixed.shp
- خروجی: Excel با سری‌های زمانی ماهانه (۱۹۴۰–۲۰۲۵)
================================================================================
"""

import os
import re
import numpy as np
import rioxarray
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
PREC_DIR = r"K:\WMO\prec"
TEMP_DIR = r"K:\WMO\tmean"
OUTPUT_DIR = r"I:\caspian_climate_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SHP_BASIN = r"K:\gozareshha\Dr Farjami\Dr Farjami\140503\LAND3.shp"
SHP_LAKE = r"K:\gozareshha\Dr Farjami\Dr Farjami\140503\caspian_lake_boundary\caspian_polygon_fixed.shp"

START_YEAR = 1940
END_YEAR = 2025
MAX_WORKERS = 8

# ============================================================
# ۲. توابع هندسی (روش شما)
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
    return pixel_area[:, np.newaxis]  # (lat, 1)

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
    prec_files = sorted([f for f in os.listdir(PREC_DIR) if f.endswith('.tif')])
    if not prec_files:
        raise FileNotFoundError("هیچ فایل بارشی یافت نشد.")
    first_file = os.path.join(PREC_DIR, prec_files[0])
    ds = rioxarray.open_rasterio(first_file)
    lons = ds.x.values
    lats = ds.y.values
    res_x = np.abs(np.diff(lons)[0]) if len(lons) > 1 else 1
    res_y = np.abs(np.diff(lats)[0]) if len(lats) > 1 else 1
    ds.close()
    return lons, lats, res_x, res_y

# ============================================================
# ۴. پردازش یک فایل (با گسترش pixel_area به ابعاد کامل)
# ============================================================
def process_file(file_path, mask, lons_ref, lats_ref, pixel_area, var_type):
    try:
        basename = os.path.basename(file_path)
        if var_type == 'precip':
            match = re.search(r'p(\d{6})\.tif', basename)
        else:
            match = re.search(r'tavg(\d{6})\.tif', basename)
        if not match:
            return None
        yyyymm = match.group(1)
        date = pd.to_datetime(yyyymm, format='%Y%m')
        year = date.year
        month = date.month

        ds = rioxarray.open_rasterio(file_path)
        data = ds.isel(band=0)

        # اطمینان از ترتیب (y, x)
        if data.dims == ('x', 'y'):
            data = data.transpose('y', 'x')
        elif data.dims != ('y', 'x'):
            try:
                data = data.rio.set_spatial_dims(x_dim="x", y_dim="y")
                data = data.transpose('y', 'x')
            except:
                pass

        # درون‌یابی به شبکه مرجع
        if len(data.x) != len(lons_ref) or len(data.y) != len(lats_ref):
            data = data.interp(x=lons_ref, y=lats_ref, method='linear')
            data = data.transpose('y', 'x')

        values = data.values  # (lat, lon)

        # گسترش pixel_area به ابعاد کامل (lat, lon)
        pixel_area_2d = np.ones_like(values) * pixel_area

        mask_bool = mask.astype(bool)
        if not mask_bool.any():
            return None

        masked_vals = values[mask_bool]
        masked_weights = pixel_area_2d[mask_bool]

        weighted_mean = np.average(masked_vals, weights=masked_weights)

        ds.close()
        return {
            'year': year,
            'month': month,
            'date': date,
            'value': weighted_mean
        }

    except Exception as e:
        print(f"⚠️ خطا در {os.path.basename(file_path)}: {e}")
        return None

# ============================================================
# ۵. پردازش موازی
# ============================================================
def process_all_files(file_list, mask, lons_ref, lats_ref, pixel_area, var_type, desc):
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_file, f, mask, lons_ref, lats_ref, pixel_area, var_type): f
            for f in file_list
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            result = future.result()
            if result is not None:
                results.append(result)
    return results

# ============================================================
# ۶. اجرای اصلی
# ============================================================
def main():
    print("=" * 80)
    print("🌡️🌧️ استخراج بارش و دما از TIF – نسخه نهایی")
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
    prec_files = sorted([os.path.join(PREC_DIR, f) for f in os.listdir(PREC_DIR) if f.endswith('.tif')])
    temp_files = sorted([os.path.join(TEMP_DIR, f) for f in os.listdir(TEMP_DIR) if f.endswith('.tif')])
    print(f"\n📂 بارش: {len(prec_files)} فایل")
    print(f"📂 دما: {len(temp_files)} فایل")

    # پردازش بارش
    print("\n🔄 پردازش بارش...")
    precip_basin = process_all_files(prec_files, mask_basin, lons_ref, lats_ref, pixel_area, 'precip', "بارش - حوضه")
    precip_lake = process_all_files(prec_files, mask_lake, lons_ref, lats_ref, pixel_area, 'precip', "بارش - دریاچه")

    # پردازش دما
    print("\n🔄 پردازش دما...")
    temp_basin = process_all_files(temp_files, mask_basin, lons_ref, lats_ref, pixel_area, 'temp', "دما - حوضه")
    temp_lake = process_all_files(temp_files, mask_lake, lons_ref, lats_ref, pixel_area, 'temp', "دما - دریاچه")

    # ترکیب
    print("\n🔗 ترکیب داده‌ها...")
    def to_df(results, name):
        if not results:
            return pd.DataFrame(columns=['year', 'month', name])
        df = pd.DataFrame(results)
        df = df[['year', 'month', 'value']].rename(columns={'value': name})
        return df

    df_pb = to_df(precip_basin, 'precip_basin')
    df_pl = to_df(precip_lake, 'precip_lake')
    df_tb = to_df(temp_basin, 'temp_basin')
    df_tl = to_df(temp_lake, 'temp_lake')

    df_merged = df_pb.merge(df_pl, on=['year', 'month'], how='outer')
    df_merged = df_merged.merge(df_tb, on=['year', 'month'], how='outer')
    df_merged = df_merged.merge(df_tl, on=['year', 'month'], how='outer')
    df_merged = df_merged.sort_values(['year', 'month']).reset_index(drop=True)

    # ذخیره
    excel_path = os.path.join(OUTPUT_DIR, 'caspian_climate_1940_2025.xlsx')
    print(f"\n💾 ذخیره در Excel: {excel_path}")
    df_merged.to_excel(excel_path, sheet_name='Climate', index=False)
    print(f"✅ Excel ذخیره شد.")

    csv_path = os.path.join(OUTPUT_DIR, 'caspian_climate_1940_2025.csv')
    df_merged.to_csv(csv_path, index=False)
    print(f"✅ CSV ذخیره شد.")

    # گزارش
    print("\n📊 خلاصه:")
    for col in ['precip_basin', 'precip_lake', 'temp_basin', 'temp_lake']:
        if col in df_merged.columns and not df_merged[col].isna().all():
            valid = df_merged[col].dropna()
            print(f"   {col}: {len(valid)} مقدار معتبر (میانگین {valid.mean():.2f})")

    print(f"\n📂 خروجی: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()