# 🌊 Caspian Sea Hydroclimate Research Framework

**A comprehensive, reproducible scientific framework for analyzing Caspian Sea Level variability, atmospheric moisture transport, hydroclimatic drivers, and future projections (1940–2025).**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-success)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-blue)](#system-requirements)
[![LaTeX](https://img.shields.io/badge/Documentation-LaTeX-darkgreen)](https://www.latex-project.org/)
[![Status](https://img.shields.io/badge/Status-Research-orange)](#)
[![DOI](https://img.shields.io/badge/DOI-10.xxxx%2Fxxxxx-blue)](https://doi.org/10.xxxx/xxxxx)

---

## 📖 Table of Contents

- [Overview](#overview)
- [Research Background](#research-background)
- [Key Scientific Contributions](#key-scientific-contributions)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Sources](#data-sources)
- [Methodology](#methodology)
- [Scientific Workflow](#scientific-workflow)
- [Running the Analyses](#running-the-analyses)
- [Compiling the Paper](#compiling-the-paper)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

## Overview

This repository contains the complete, reproducible research workflow used to investigate **long-term variations in the Caspian Sea Level (CSL)** and the **hydroclimatic mechanisms** controlling these fluctuations. The project integrates multiple independent observational and reanalysis datasets—including **ERA5 atmospheric reanalysis**, **hydrological observations**, **satellite altimetry**, **reservoir databases**, and **population data**—into a unified processing pipeline.

The framework is capable of reproducing **all analyses, figures, statistical results, and future projections** presented in the accompanying research paper. Rather than focusing on a single environmental factor, this work combines:

- Atmospheric moisture transport
- Hydrological balance
- Remote sensing products
- Human influences (reservoirs, population)
- Advanced statistical learning methods

to provide a **comprehensive understanding** of Caspian Sea variability and its drivers.

---

## Research Background

The **Caspian Sea** is the largest enclosed inland water body on Earth and has experienced substantial water‑level fluctuations throughout the last century. During recent decades, **accelerated declines** in water level have become a major environmental concern because they directly affect:

- Coastal ecosystems and biodiversity
- Fisheries and aquaculture
- Navigation and shipping routes
- Coastal infrastructure and urban areas
- Freshwater resources and agriculture
- Socioeconomic activities of surrounding countries
- International coastal management and geopolitics

Understanding these fluctuations requires integrating **atmospheric circulation**, **terrestrial hydrology**, and **human influences** within a common analytical framework. This project provides such a framework by combining modern climate reanalysis, satellite observations, hydrological records, and statistical modelling into one **fully reproducible research workflow**.

---

## Key Scientific Contributions

1. **Atmospheric Moisture Transport**  
   Quantification of Integrated Vapor Transport (IVT) into and out of the Caspian basin, revealing a persistent **North Atlantic–Volga transport corridor** and a strong **west‑to‑east asymmetry** in net moisture flux.

2. **Hydrological Controls**  
   Identification of **Volga River discharge** as the dominant hydroclimatic driver of Caspian Sea Level variability, with secondary contributions from basin precipitation and evaporation.

3. **Change‑Point Detection**  
   Detection of **abrupt hydroclimatic transitions** using three complementary algorithms (binary segmentation, window‑based analysis, and voting strategy), providing robust estimates of regime shifts.

4. **Causal Relationships**  
   Investigation of **directional causal links** among atmospheric, hydrological, and sea‑level variables using statistical causality methods.

5. **Future Projections**  
   Estimation of **Caspian Sea Level scenarios through 2030** using Ridge Regression and ensemble learning techniques, with uncertainty quantification.

6. **Reproducible Workflow**  
   A **complete, end‑to‑end** processing pipeline that regenerates all figures, tables, and statistical outputs from raw data, ensuring full reproducibility and transparency.

---

## Repository Structure

caspian-sea-research/
│
├── src/ # All Python analysis scripts
│ ├── stage1_caspianforecast.py
│ ├── stage2_caspianforecast.py
│ ├── ...
│ ├── final_analysis.py
│ ├── comprehensive_final_analysis.py
│ ├── SHAP_OSNAP_Validation.py
│ ├── extract_precip_from_zarr.py
│ ├── extract_evap_from_zarr.py
│ ├── causality.py
│ ├── change_point.py
│ ├── ultimate_test.py
│ ├── analysis_final_code.py
│ └── ...
│
├── docs/ # LaTeX source and PDF reports
│ ├── caspian_analysis_fa.tex
│ ├── caspian_analysis_en.tex
│ ├── paper_fa.tex
│ ├── paper_en.tex
│ ├── analysis_report_final.tex
│ ├── Caspian_Complete_Report.pdf
│ ├── caspian_analysis_fa.pdf
│ ├── caspian_analysis_en.pdf
│ └── ...
│
├── data/ # Key datasets (sample & processed)
│ ├── caspian_sea_level_raw.csv
│ ├── volga_discharge.csv
│ ├── AMOC_extended_1870_2023.csv
│ ├── teleconnection_indices.csv
│ ├── start_points.csv
│ ├── border_with_normals.csv
│ ├── granger_causality_full.csv
│ ├── predictions.csv
│ ├── indices_complete.xlsx
│ ├── Data_compilation.xlsx
│ └── ...
│
├── figures/ # Publication‑quality figures
│ ├── sea_level_forecast_final.png
│ ├── shap_summary_final.png
│ ├── residual_analysis.png
│ ├── final_forecast.png
│ ├── correlation_heatmap_kendall.png
│ ├── correlation_heatmap_spearman.png
│ ├── prediction_with_ci.png
│ ├── time_series_all_variables.png
│ ├── pairplot_analysis.png
│ ├── wavelet_evaporation.png
│ ├── wavelet_level_m.png
│ ├── LSTM_Prediction.png
│ ├── Extended_Prediction.png
│ └── ...
│
├── manual/ # User documentation
│ └── manual.tex
│
├── README.md # This file
├── COMPLETE_GUIDE.md # Comprehensive step‑by‑step guide (English)
├── requirements.txt # Python dependencies
├── LICENSE # MIT License
├── compile.py # One‑click LaTeX compilation
└── enrich_report.json # Report from file transfer utility
text


---

## Installation

### System Requirements

| Component | Recommended Specification |
|-----------|---------------------------|
| **Python** | 3.10 or newer |
| **Operating System** | Windows 10/11, Linux (Ubuntu 20.04+), or macOS |
| **LaTeX** | MiKTeX (Windows) or TeX Live (Linux/macOS) |
| **Disk Space** | ≥ 50 GB (recommended for ERA5 data storage) |
| **RAM** | ≥ 16 GB (for parallel processing of large datasets) |
| **Storage** | SSD recommended for faster I/O |

---

### Clone the Repository

```bash
git clone https://github.com/your-username/caspian-sea-research.git
cd caspian-sea-research

Install Python Dependencies

All required Python packages are listed in requirements.txt. Install them using pip:
bash

pip install -r requirements.txt

If you encounter installation errors, ensure your pip is up‑to‑date:
bash

pip install --upgrade pip

For GPU‑accelerated machine learning (optional), you may install tensorflow-gpu or torch with CUDA support separately.
Configure CDS API (for ERA5 download)

To download ERA5 data, you need a Copernicus Climate Data Store (CDS) account and API credentials.

    Register at https://cds.climate.copernicus.eu

    Create a .cdsapirc file in your home directory with the following content:

ini

url: https://cds.climate.copernicus.eu/api/v2
key: <UID>:<API-Key>

Replace <UID> and <API-Key> with your actual credentials (available in your CDS profile).
Additional GIS Datasets

The analysis requires the following GIS shapefiles (provided in the repository under data/):

    LAND3.shp – global land mask

    caspian_polygon_fixed.shp – Caspian Sea polygon (fixed)

    border_with_normals.csv – boundary segments with outward normals

Ensure these files are present in the data/ directory before running the extraction scripts.
Quick Start

After installation, you can run individual processing modules from the src/ directory. The workflow is modular; each script can be executed independently.
1. Extract Precipitation from ERA5 Zarr Archives
bash

python src/extract_precip_from_zarr.py

This script reads monthly ERA5 Zarr datasets, extracts precipitation fields, and prepares data for subsequent analyses. Outputs are saved in the data/ directory.
2. Extract Evaporation
bash

python src/extract_evap_from_zarr.py

Similar to the precipitation extraction, this script processes evaporation fields from ERA5.
3. Run the Complete Analysis Pipeline

To execute the entire workflow and regenerate all figures and statistical outputs:
bash

python src/analysis_final_code.py

This script will run all processing stages sequentially. Depending on the data volume, this may take several hours.
4. Compile the Scientific Paper
bash

python compile.py

This will generate both Persian and English PDF versions of the manuscript from the LaTeX sources.
5. Generate Future Projections
bash

python src/ultimate_test.py

This script estimates future Caspian Sea Level scenarios using the trained Ridge Regression model.
Data Sources

The project integrates multiple independent datasets to ensure robust and reliable scientific conclusions.
Atmospheric Reanalysis

ERA5 (Copernicus Climate Change Service, C3S) – the primary atmospheric dataset, covering 1940–2024 at monthly resolution.

Required variables:
Variable	Description
q	Specific Humidity
u	Zonal Wind Component
v	Meridional Wind Component
ps	Surface Pressure
tp	Total Precipitation
e	Evaporation
Hydrological Observations

    Volga River Discharge – observed daily/monthly discharge from multiple gauging stations, compiled from international databases.

    Caspian Sea Level – satellite altimetry (Hydroweb) and tide‑gauge records, covering 1940–2025.

Remote Sensing

    Satellite altimetry – TOPEX/Poseidon, Jason‑1/2/3, Sentinel‑6.

    Shoreline observations – from various international monitoring services.

Human Influence

    Reservoir Database – locations, surface area, and temporal evolution of major reservoirs in the Volga basin.

    Population Density – WorldPop gridded population products (2000–2020).

Geographic Data

    Watershed boundaries – Caspian basin mask.

    Shoreline polygons – Caspian Sea polygon (fixed and dynamic versions).

Methodology

The analysis follows a modular scientific workflow, as illustrated below.
Atmospheric Moisture Flux Computation

Vertically Integrated Moisture Flux (VIMF) is calculated using the equation:
VIMF=1g∫psfcptopV q dp
VIMF=g1​∫psfc​ptop​​Vqdp

where:

    V=(u,v)V=(u,v) is the horizontal wind vector,

    qq is specific humidity,

    psfcpsfc​ is surface pressure,

    ptopptop​ is the upper pressure level (typically 300 hPa),

    gg is the acceleration due to gravity.

Net Boundary Flux

The net moisture flux across the basin boundary is computed by integrating the inward component of VIMF along the polygon perimeter:
Fnet=∮VIMF⋅n dl
Fnet​=∮VIMF⋅ndl

where nn is the outward unit normal vector and dldl is the boundary segment length.
Change‑Point Detection

Three complementary algorithms are applied:

    Binary Segmentation – recursively detects breakpoints in time series.

    Window‑Based Analysis – uses sliding windows to identify localized changes.

    Voting Strategy – combines results from multiple algorithms to increase robustness.

Causal Analysis

Granger causality and Liang–Kleeman information flow are used to investigate directional influences between atmospheric, hydrological, and sea‑level variables.
Forecast Model

A Ridge Regression model is trained on historical data (1940–2024) to project future Caspian Sea Level (2025–2030). Predictors include:

    Volga River discharge (annual/seasonal)

    Basin precipitation

    Evaporation

    AMOC index

    Teleconnection indices (NAO, ENSO)

    Lagged sea‑level values

Model performance is evaluated using R2R2 and RMSE, with independent validation on withheld data.
Scientific Workflow
Running the Analyses

All analytical workflows are implemented as independent Python modules in the src/ directory. Below is a detailed description of each key script.
Precipitation Extraction

File: src/extract_precip_from_zarr.py

Purpose: Reads monthly ERA5 Zarr archives, extracts precipitation fields, and outputs processed data for further analysis.