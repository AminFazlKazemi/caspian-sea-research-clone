#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
مرحله اول - آماده‌سازی مرز ایران از LAND3.shp
- هموارسازی قوی (sigma=5)
- تراکم نقاط: نصف حالت قبل (فاصله مرزی 10km، فاصله نقاط شروع 20km)
- نمایش تمام نقاط شروع در نقشه (بدون زیرنمونه)
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf

# ============================================================
# مسیرها
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SHAPEFILE_PATH = os.path.join(SCRIPT_DIR, "LAND3.shp")
OUTPUT_BASE = SCRIPT_DIR

print(f"پوشه کد: {SCRIPT_DIR}")

# ============================================================
# 1. خواندن و استخراج مرز اصلی
# ============================================================
print("در حال خواندن Shapefile...")
gdf = gpd.read_file(SHAPEFILE_PATH)
print(f"تعداد Features: {len(gdf)}")

if hasattr(gdf.geometry, 'union_all'):
    unified_geom = gdf.geometry.union_all()
else:
    unified_geom = gdf.geometry.unary_union

if isinstance(unified_geom, MultiPolygon):
    main_poly = max(unified_geom.geoms, key=lambda p: p.length)
else:
    main_poly = unified_geom

coarse_coords = np.array(main_poly.exterior.coords)
if np.allclose(coarse_coords[0], coarse_coords[-1]):
    coarse_coords = coarse_coords[:-1]

print(f"تعداد نقاط اولیه (خام): {len(coarse_coords)}")

# ============================================================
# 2. هموارسازی قوی‌تر با sigma=5.0 (قابل تنظیم)
# ============================================================
def smooth_coordinates(coords, sigma=5.0):
    lons = coords[:, 0]
    lats = coords[:, 1]
    lons_smooth = gaussian_filter1d(lons, sigma=sigma, mode='wrap')
    lats_smooth = gaussian_filter1d(lats, sigma=sigma, mode='wrap')
    return np.column_stack((lons_smooth, lats_smooth))

border_smoothed = smooth_coordinates(coarse_coords, sigma=5.0)
print(f"پس از هموارسازی (sigma=5.0): {len(border_smoothed)} نقطه")

# ============================================================
# 3. درون‌یابی با فاصله 10 کیلومتر (نصف تراکم حالت قبل)
# ============================================================
def haversine_distance(lon1, lat1, lon2, lat2):
    R = 6371
    dlon = np.radians(lon2 - lon1)
    dlat = np.radians(lat2 - lat1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def interpolate_border(border, spacing_km):
    dist = [0.0]
    for i in range(1, len(border)):
        d = haversine_distance(border[i-1,0], border[i-1,1], border[i,0], border[i,1])
        dist.append(dist[-1] + d)
    total_len = dist[-1]
    targets = np.arange(0, total_len, spacing_km)
    points = []
    j = 0
    for t in targets:
        while j < len(dist)-1 and dist[j+1] < t:
            j += 1
        if j >= len(border)-1:
            break
        frac = (t - dist[j]) / (dist[j+1] - dist[j])
        lon = border[j,0] + frac * (border[j+1,0] - border[j,0])
        lat = border[j,1] + frac * (border[j+1,1] - border[j,1])
        points.append([lon, lat])
    return np.array(points)

# فاصله 10 کیلومتر برای مرز چگال
border_dense = interpolate_border(border_smoothed, spacing_km=10)
print(f"تعداد نقاط مرزی (فاصله 10km): {len(border_dense)}")

# ============================================================
# 4. محاسبه نرمال درون‌سو با بررسی جهت
# ============================================================
def is_ccw(coords):
    area = 0
    for i in range(len(coords)):
        x1, y1 = coords[i]
        x2, y2 = coords[(i+1) % len(coords)]
        area += (x2 - x1) * (y2 + y1)
    return area > 0

if not is_ccw(border_dense):
    print("جهت مرز ساعت‌گرد است. معکوس می‌شود.")
    border_dense = border_dense[::-1]

def compute_normals(border):
    n = len(border)
    normals = []
    for i in range(n):
        i_prev = (i - 1 + n) % n
        i_next = (i + 1) % n
        dx = border[i_next,0] - border[i_prev,0]
        dy = border[i_next,1] - border[i_prev,1]
        nx = -dy
        ny = dx
        norm = np.hypot(nx, ny)
        if norm > 0:
            nx /= norm
            ny /= norm
        normals.append([nx, ny])
    return np.array(normals)

normals = compute_normals(border_dense)

# بررسی جهت نرمال نسبت به مرکز ایران
center_lat = np.mean(border_dense[:,1])
center_lon = np.mean(border_dense[:,0])
sample_idx = len(border_dense)//2
vec = normals[sample_idx]
point = border_dense[sample_idx]
to_center = np.array([center_lon - point[0], center_lat - point[1]])
to_center = to_center / (np.hypot(to_center[0], to_center[1]) + 1e-8)
if np.dot(vec, to_center) < 0:
    print("نرمال‌ها به سمت بیرون هستند. معکوس می‌شوند.")
    normals = -normals
else:
    print("نرمال‌ها به سمت داخل هستند.")

# ============================================================
# 5. تولید نقاط شروع با فاصله 20 کیلومتر (نصف تراکم)
# ============================================================
start_points = interpolate_border(border_dense, spacing_km=20)
print(f"تعداد نقاط شروع (برای مسیرها): {len(start_points)}")
pd.DataFrame(start_points, columns=['Longitude','Latitude']).to_csv(
    os.path.join(OUTPUT_BASE, "start_points.csv"), index=False)

# ذخیره مرز و نرمال‌ها
df_border = pd.DataFrame(border_dense, columns=['Longitude','Latitude'])
df_border['nx'] = normals[:,0]
df_border['ny'] = normals[:,1]
df_border.to_csv(os.path.join(OUTPUT_BASE, "border_with_normals.csv"), index=False)
print("فایل border_with_normals.csv ذخیره شد.")

# ============================================================
# 6. رسم نقشه (نمایش همه نقاط شروع - بدون step)
# ============================================================
def plot_all_start_points(border, start_pts, normals, arrow_scale=0.6, save_path=None):
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([border[:,0].min()-2, border[:,0].max()+2,
                   border[:,1].min()-2, border[:,1].max()+2], crs=ccrs.PlateCarree())
    ax.add_feature(cf.COASTLINE, linewidth=0.5)
    ax.add_feature(cf.BORDERS, linestyle=':', linewidth=0.5)
    ax.add_feature(cf.OCEAN, alpha=0.3)
    ax.add_feature(cf.LAND, alpha=0.1)
    
    # رسم مرز هموار
    ax.plot(border[:,0], border[:,1], 'b-', linewidth=1.5, transform=ccrs.PlateCarree(), label='مرز هموار')
    
    # پیدا کردن نزدیک‌ترین نرمال برای هر نقطه شروع
    selected_norm = []
    for (lon, lat) in start_pts:
        dists = np.hypot(border[:,0]-lon, border[:,1]-lat)
        idx = np.argmin(dists)
        selected_norm.append(normals[idx])
    selected_norm = np.array(selected_norm)
    
    # نمایش همه نقاط شروع
    ax.scatter(start_pts[:,0], start_pts[:,1], c='red', s=15, zorder=5,
               transform=ccrs.PlateCarree(), label='نقاط شروع')
    
    # رسم فلش برای همه نقاط (اگر تعداد زیاد است، ممکن است شلوغ شود؛ اما به خواسته کاربر)
    for (lon, lat), (nx, ny) in zip(start_pts, selected_norm):
        ax.arrow(lon, lat, nx*arrow_scale, ny*arrow_scale,
                 head_width=0.15, head_length=0.15, fc='green', ec='green', alpha=0.6,
                 transform=ccrs.PlateCarree())
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'مرز ایران با نرمال درون‌سو - {len(start_pts)} نقطه شروع (همه نمایش داده شده)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"نقشه ذخیره شد: {save_path}")
    else:
        plt.show()

plot_all_start_points(border_dense, start_points, normals, arrow_scale=0.6,
                      save_path=os.path.join(OUTPUT_BASE, "iran_border_normals_inward.png"))

print("\n✅ مرحله اول با موفقیت انجام شد.")
print(f"   - تعداد نقاط مرزی: {len(border_dense)} (فاصله 10km)")
print(f"   - تعداد نقاط شروع: {len(start_points)} (فاصله 20km)")
print("   - همه نقاط شروع در نقشه نمایش داده شده‌اند.")
print("   - هموارسازی با sigma=5 انجام شده.")