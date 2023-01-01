import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob as glb
import rioxarray
from geopandas import read_file as gpd_read_file
from datetime import date, datetime, timedelta

from masking import xr_shp_to_grid, xr_mask
from dates import get_first_date_of_current_month, get_last_date_of_month


shp_Peru = gpd_read_file('C:/Users/FELIPEOH/ClimatologiaTempPeru/data/shps/Departamentos.shp')

# PISCOt v1.1 (tx:temperatura máxima)
PISCOtx = xr.open_dataset("C:/Users/FELIPEOH/ClimatologiaTempPeru/data/PISCO_temperature/tx/PISCOdtx_v1.1.nc")

shp_exp_grid = xr_shp_to_grid(shp_i = shp_Peru,
                              netcdf_array = PISCOtx.tx)


PISCOtx_masked = xr_mask(grid_mask = shp_exp_grid,
                             netcdf_i = PISCOtx.tx)

## Verano
### Promedio
PISCOtx_season_prom = PISCOtx_masked.groupby("time.season").mean("time")
PISCOtx_prom_ver = (PISCOtx_season_prom.sel(season="DJF"))
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = PISCOtx_prom_ver.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatologiaTempPeru/outputs/PISCOtx_prom_ver.png")

###Desviación estándar
PISCOtx_season_sd = PISCOtx_masked.groupby("time.season").std("time")
PISCOtx_sd_ver = (PISCOtx_season_sd.sel(season="DJF"))
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = PISCOtx_sd_ver.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatologiaTempPeru/outputs/PISCOtx_sd_ver.png")

###Coeficiente de asimetria
PISCOtx_season_md = PISCOtx_masked.groupby("time.season").median("time")
PISCOtx_md_ver = (PISCOtx_season_sd.sel(season="DJF"))
PISCOtx_ac_ver = (3*(PISCOtx_prom_ver-PISCOtx_md_ver))/PISCOtx_sd_ver
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = PISCOtx_ac_ver.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatologiaTempPeru/outputs/PISCOtx_ac_ver.png")

###Percentil 5
PISCOtx_season_p5 = PISCOtx_masked.groupby("time.season").quantile(0.05, "time")
PISCOtx_p5_ver = (PISCOtx_season_p5.sel(season="DJF"))
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = PISCOtx_p5_ver.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatologiaTempPeru/outputs/PISCOtx_p5_ver.png")

###Percentil 95
PISCOtx_season_p95 = PISCOtx_masked.groupby("time.season").quantile(0.95, "time")
PISCOtx_p95_ver = (PISCOtx_season_p95.sel(season="DJF"))
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = PISCOtx_p95_ver.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatologiaTempPeru/outputs/PISCOtx_p95_ver.png")

##Invierno
###Promedio
PISCOtx_prom_inv = (PISCOtx_season_prom.sel(season="JJA"))
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = PISCOtx_prom_inv.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatologiaTempPeru/outputs/PISCOtx_prom_inv.png")

###Desviación estándar
PISCOtx_sd_inv = (PISCOtx_season_sd.sel(season="JJA"))
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = PISCOtx_sd_inv.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatologiaTempPeru/outputs/PISCOtx_sd_inv.png")

###Coeficiente de asimetria
PISCOtx_md_inv = (PISCOtx_season_sd.sel(season="JJA"))
PISCOtx_ac_inv = (3*(PISCOtx_prom_inv-PISCOtx_md_inv))/PISCOtx_sd_inv
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = PISCOtx_ac_inv.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatologiaTempPeru/outputs/PISCOtx_ac_inv.png")

###Percentil 5
PISCOtx_p5_inv = (PISCOtx_season_p5.sel(season="JJA"))
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = PISCOtx_p5_inv.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatologiaTempPeru/outputs/PISCOtx_p5_inv.png")

###Percentil 95
PISCOtx_p95_inv = (PISCOtx_season_p95.sel(season="JJA"))
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = PISCOtx_p95_inv.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatologiaTempPeru/outputs/PISCOtx_p95_inv.png")

#PISCOt v1.1 (tn:temperatura mínima)
PISCOtn = xr.open_dataset("C:/Users/FELIPEOH/ClimatologiaTempPeru/data/PISCO_temperature/tn/PISCOdtn_v1.1.nc")

shp_exp_grid = xr_shp_to_grid(shp_i = shp_Peru,
                              netcdf_array = PISCOtn.tn)


PISCOtn_masked = xr_mask(grid_mask = shp_exp_grid,
                             netcdf_i = PISCOtn.tn)

##Verano
###Promedio
PISCOtn_season_prom = PISCOtn_masked.groupby("time.season").mean("time")
PISCOtn_prom_ver = (PISCOtn_season_prom.sel(season="DJF"))
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = PISCOtn_prom_ver.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatologiaTempPeru/outputs/PISCOtn_prom_ver.png")

###Desviación estándar
PISCOtn_season_sd = PISCOtn_masked.groupby("time.season").std("time")
PISCOtn_sd_ver = (PISCOtn_season_sd.sel(season="DJF"))
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = PISCOtn_sd_ver.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatologiaTempPeru/outputs/PISCOtn_sd_ver.png")

###Coeficiente de asimetria
PISCOtn_season_md = PISCOtn_masked.groupby("time.season").median("time")
PISCOtn_md_ver = (PISCOtn_season_sd.sel(season="DJF"))
PISCOtn_ac_ver = (3*(PISCOtn_prom_ver-PISCOtn_md_ver))/PISCOtn_sd_ver
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = PISCOtn_ac_ver.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatologiaTempPeru/outputs/PISCOtn_ac_ver.png")

###Percentil 5
PISCOtn_season_p5 = PISCOtn_masked.groupby("time.season").quantile(0.05, "time")
PISCOtn_p5_ver = (PISCOtn_season_p5.sel(season="DJF"))
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = PISCOtn_p5_ver.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatologiaTempPeru/outputs/PISCOtn_p5_ver.png")

###Percentil 95
PISCOtn_season_p95 = PISCOtn_masked.groupby("time.season").quantile(0.95, "time")
PISCOtn_p95_ver = (PISCOtn_season_p95.sel(season="DJF"))
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = PISCOtn_p95_ver.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatologiaTempPeru/outputs/PISCOtn_p95_ver.png")

##Invierno
###Promedio
PISCOtn_prom_inv = (PISCOtn_season_prom.sel(season="JJA"))
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = PISCOtn_prom_inv.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatologiaTempPeru/outputs/PISCOtn_prom_inv.png")

###Desviación estándar
PISCOtn_sd_inv = (PISCOtn_season_sd.sel(season="JJA"))
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = PISCOtn_sd_inv.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatologiaTempPeru/outputs/PISCOtn_sd_inv.png")

###Coeficiente de asimetria
PISCOtn_md_inv = (PISCOtn_season_sd.sel(season="JJA"))
PISCOtn_ac_inv = (3*(PISCOtn_prom_inv-PISCOtn_md_inv))/PISCOtn_sd_inv
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = PISCOtn_ac_inv.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatologiaTempPeru/outputs/PISCOtn_ac_inv.png")

###Percentil 5
PISCOtn_p5_inv = (PISCOtn_season_p5.sel(season="JJA"))
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = PISCOtn_p5_inv.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatologiaTempPeru/outputs/PISCOtn_p5_inv.png")

###Percentil 95
PISCOtn_p95_inv = (PISCOtn_season_p95.sel(season="JJA"))
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = PISCOtn_p95_inv.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatologiaTempPeru/outputs/PISCOtn_p95_inv.png")

#PISCOt v1.1 (tm:temperatura media)
PISCOtm_masked = (PISCOtx_masked+PISCOtn_masked)/2

##Verano
fechas = ['1981-01-01']
for year in range(1981,2017):
  fechas.append(get_last_date_of_month(year, 2))
  fechas.append(get_first_date_of_current_month(year, 12))

fechas = fechas[:-1]

PISCOtm_mean_files = []
for i in range(0,72,2):
  PISCOtm_year = PISCOtm_masked.sel(time=slice(fechas[i], fechas[i+1])).mean("time")
  PISCOtm_mean_files.append(PISCOtm_year)

PISCOtm_mean_s_ver = xr.concat(PISCOtm_mean_files, dim="time")

###%min<p5
PISCOtn_por_files = []
for i in range(0,72,2):
  PISCOtn_year = PISCOtn_masked.sel(time=slice(fechas[i], fechas[i+1]))
  PISCOtn_por = (PISCOtn_year.where(PISCOtn_year < PISCOtn_p5_ver, drop=True).count("time"))*100/len(PISCOtn_year)
  PISCOtn_por_files.append(PISCOtn_por)

PISCOtn_por_ver = xr.concat(PISCOtn_por_files, dim="time")

fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = xr.corr(PISCOtn_por_ver, PISCOtm_mean_s_ver, dim="time").plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatologiaTempPeru/outputs/PISCOtm_%minp5_ver.png")

###%max>p95
PISCOtx_por_files = []
for i in range(0,72,2):
  PISCOtx_year = PISCOtx_masked.sel(time=slice(fechas[i], fechas[i+1]))
  PISCOtx_por = (PISCOtx_year.where(PISCOtx_year > PISCOtx_p95_ver, drop=True).count("time"))*100/len(PISCOtx_year)
  PISCOtx_por_files.append(PISCOtx_por)

PISCOtx_por_ver = xr.concat(PISCOtx_por_files, dim="time")

fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = xr.corr(PISCOtx_por_ver, PISCOtm_mean_s_ver, dim="time").plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatologiaTempPeru/outputs/PISCOtm_%maxp95_ver.png")

##Invierno
fechas = []
for year in range(1981,2017):
  fechas.append(get_first_date_of_current_month(year, 6))
  fechas.append(get_last_date_of_month(year, 8))

PISCOtm_mean_files = []
for i in range(0,72,2):
  PISCOtm_year = PISCOtm_masked.sel(time=slice(fechas[i], fechas[i+1])).mean("time")
  PISCOtm_mean_files.append(PISCOtm_year)

PISCOtm_mean_s_inv = xr.concat(PISCOtm_mean_files, dim="time")

###%min<p5
PISCOtn_por_files = []
for i in range(0,72,2):
  PISCOtn_year = PISCOtn_masked.sel(time=slice(fechas[i], fechas[i+1]))
  PISCOtn_por = (PISCOtn_year.where(PISCOtn_year < PISCOtn_p5_ver, drop=True).count("time"))*100/len(PISCOtn_year)
  PISCOtn_por_files.append(PISCOtn_por)

PISCOtn_por_inv = xr.concat(PISCOtn_por_files, dim="time")

fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = xr.corr(PISCOtn_por_inv, PISCOtm_mean_s_inv, dim="time").plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatologiaTempPeru/outputs/PISCOtm_%minp5_inv.png")

###%max>p95
PISCOtx_por_files = []
for i in range(0,72,2):
  PISCOtx_year = PISCOtx_masked.sel(time=slice(fechas[i], fechas[i+1]))
  PISCOtx_por = (PISCOtx_year.where(PISCOtx_year > PISCOtx_p95_ver, drop=True).count("time"))*100/len(PISCOtx_year)
  PISCOtx_por_files.append(PISCOtx_por)

PISCOtx_por_inv = xr.concat(PISCOtx_por_files, dim="time")

fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = xr.corr(PISCOtx_por_inv, PISCOtm_mean_s_inv, dim="time").plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatologiaTempPeru/outputs/PISCOtm_%maxp95_inv.png")
