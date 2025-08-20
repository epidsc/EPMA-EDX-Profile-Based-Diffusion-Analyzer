# EPMA/EDX Diffusion Profiler

A PyQt5-based tool for **EPMA/EDX diffusion profiling**, combining gradient computation with clustering and multi-model fitting.
It allows rapid and accurate interpretation of spatial concentration data with interactive heatmaps, vector fields, and 3D surface plots.

---

## 🎯 Purpose

In diffusion studies using **EPMA (Electron Probe Microanalysis)** or **EDS/WDS (Energy/Wavelength Dispersive Spectroscopy)**, researchers often collect spatially resolved concentration maps.
Manually identifying **dominant diffusion directions** and extracting **diffusion parameters (Diffussion Coefficient Diffusion profile geometries, fits)** from such datasets is tedious and error-prone.

This tool provides:

* Automated **gradient computation** from spatial data.
* **Clustering of gradients (∇C)** to identify dominant diffusion pathways.
* **Curve fitting** of profiles with multiple diffusion models (erfc, Gaussian, slab, cylindrical, spherical).
* RMSE-based ranking to **suggest the best fitting model**.
* Easy visualization: heatmaps, vector fields, 3D surface plots.

In short: it bridges raw spectroscopic maps → meaningful diffusion parameters.

---

## 📂 Input Format

Input should be a **CSV file** with three columns:

```csv
x,y,c
0,0,12.5
1,0,12.2
0,1,11.9
1,1,11.5
```

* **x, y** → spatial coordinates (μm or chosen unit).
* **c** → measured concentration (wt% or at%).

This is compatible with output from most EPMA/EDS/WDS mapping software after simple export.

---

## 🔬 Applications

* **Materials Science**: alloy interdiffusion, oxidation studies, thin-film diffusion.
* **Geoscience**: diffusion zoning in minerals (e.g., olivine, feldspar).
* **Semiconductors**: dopant diffusion in wafers.
* **General microanalysis**: whenever diffusion profiles need to be quantified from spatial concentration data.

---

## ✨ Features

* Load CSV data and visualize as **tables, heatmaps, vector fields, and 3D surfaces**.
* **Gradient computation (∇C)** and **KMeans clustering** to extract dominant diffusion directions.
* **Multi-model diffusion fitting**:

  * 1D erfc
  * Gaussian (2D)
  * Slab
  * Cylindrical
  * Spherical
  * Exponential Decay
* **Auto-suggest best fit model** using RMSE ranking.
* Export computed gradient data as CSV.

---

## ⚙️ Requirements

* PyQt5
* pandas
* numpy
* matplotlib
* scikit-learn
* scipy

---

## 🚀 Installation

```bash
git clone https://github.com/epidsc/EPMA-EDX-Diffusion-Profiler
cd EPMA-EDX-Profile-Based-Diffusion-Analyzer
pip install -r requirements.txt
```

---

## 📸 Demonstration Video
https://github.com/user-attachments/assets/c6129147-b94c-41ab-8327-88eb916cc9ce

---

## 📸 Screenshot

https://github.com/epidsc/EPMA-EDX-Profile-Based-Diffusion-Analyzer/blob/main/screen%20shots/App/1_Initiation.png
https://github.com/epidsc/EPMA-EDX-Profile-Based-Diffusion-Analyzer/blob/main/screen%20shots/App/2_Browse%20and%20Loading%20EPMA%20or%20EDX%20sample%20data%20csv%20file.png
https://github.com/epidsc/EPMA-EDX-Profile-Based-Diffusion-Analyzer/blob/main/screen%20shots/App/3_Visualizing%20as%20heat%20map.png
https://github.com/epidsc/EPMA-EDX-Profile-Based-Diffusion-Analyzer/blob/main/screen%20shots/App/4_Finding%20Diffusion%20directions.png
https://github.com/epidsc/EPMA-EDX-Profile-Based-Diffusion-Analyzer/blob/main/screen%20shots/App/5_Analyzing%20Diffusion%20Dominant%20Directions%20and%20Clustering.png
https://github.com/epidsc/EPMA-EDX-Profile-Based-Diffusion-Analyzer/blob/main/screen%20shots/App/6_%20C(x%2Cy)%20vs%20x%2Cy%20visualization.png
https://github.com/epidsc/EPMA-EDX-Profile-Based-Diffusion-Analyzer/blob/main/screen%20shots/App/7_%20Fitting%20data.png

---



## ▶️ Usage

```bash
python "EPMA EDX Diffusion Profile Analyzer.py"
```

* Load a CSV dataset.
* View and explore **data table, heatmap, vector field, 3D surface**.
* Compute gradients and apply clustering.
* Fit different diffusion models and compare RMSE.
* Export ∇C data as CSV.


