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


shp_Peru = gpd_read_file('C:/Users/FELIPEOH/ClimatoTempPeru/data/shps/Departamentos.shp')
ERA_files = sorted(glb.glob("C:/Users/FELIPEOH/ClimatoTempPeru/data/ERA5land/2m_temperature/*nc"))

#ERA5land (tx:temperatura máxima)
ERA5_files = []
for file in ERA_files[0:-5]:
  ERA5t = xr.open_dataset(file)
  ERA5t = ERA5t.resample(time='1D').max()
  ERA5t = ERA5t - 273.15
  ERA5t = ERA5t.rename_vars({"t2m":"tx"})
  ERA5_files.append(ERA5t)

ERA5tx = xr.concat(ERA5_files, dim="time")

shp_exp_grid = xr_shp_to_grid(shp_i = shp_Peru,
                              netcdf_array = ERA5tx.tx)

ERA5tx_masked = xr_mask(grid_mask = shp_exp_grid,
                             netcdf_i = ERA5tx.tx)

##Verano
###Promedio
ERA5tx_season_prom = ERA5tx_masked.groupby("time.season").mean("time")
ERA5tx_prom_ver = (ERA5tx_season_prom.sel(season="DJF"))
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = ERA5tx_prom_ver.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatoTempPeru/outputs/ERA5tx_prom_ver.png")

###Desviación estándar
ERA5tx_season_sd = ERA5tx_masked.groupby("time.season").std("time")
ERA5tx_sd_ver = (ERA5tx_season_sd.sel(season="DJF"))
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = ERA5tx_sd_ver.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatoTempPeru/outputs/ERA5tx_sd_ver.png")

###Coeficiente de asimetria
ERA5tx_season_md = ERA5tx_masked.groupby("time.season").median("time")
ERA5tx_md_ver = (ERA5tx_season_sd.sel(season="DJF"))
ERA5tx_ac_ver = (3*(ERA5tx_prom_ver-ERA5tx_md_ver))/ERA5tx_sd_ver
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = ERA5tx_ac_ver.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatoTempPeru/outputs/ERA5tx_ac_ver.png")

###Percentil 5
ERA5tx_season_p5 = ERA5tx_masked.groupby("time.season").quantile(0.05, "time")
ERA5tx_p5_ver = (ERA5tx_season_p5.sel(season="DJF"))
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = ERA5tx_p5_ver.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatoTempPeru/outputs/ERA5tx_p5_ver.png")

###Percentil95
ERA5tx_season_p95 = ERA5tx_masked.groupby("time.season").quantile(0.95, "time")
ERA5tx_p95_ver = (ERA5tx_season_p95.sel(season="DJF"))
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = ERA5tx_p95_ver.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatoTempPeru/outputs/ERA5tx_p95_ver.png")

##Invierno
###Promedio
ERA5tx_prom_inv = (ERA5tx_season_prom.sel(season="JJA"))
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = ERA5tx_prom_inv.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatoTempPeru/outputs/ERA5tx_prom_inv.png")

###Desviación estandar
ERA5tx_sd_inv = (ERA5tx_season_sd.sel(season="JJA"))
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = ERA5tx_sd_inv.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatoTempPeru/outputs/ERA5tx_sd_inv.png")

###Coeficiente de asimetria
ERA5tx_md_inv = (ERA5tx_season_sd.sel(season="JJA"))
ERA5tx_ac_inv = (3*(ERA5tx_prom_inv-ERA5tx_md_inv))/ERA5tx_sd_inv
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = ERA5tx_ac_inv.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatoTempPeru/outputs/ERA5tx_ac_inv.png")

###Percentil 5
ERA5tx_p5_inv = (ERA5tx_season_p5.sel(season="JJA"))
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = ERA5tx_p5_inv.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatoTempPeru/outputs/ERA5tx_p5_inv.png")

###Percentil95
ERA5tx_p95_inv = (ERA5tx_season_p95.sel(season="JJA"))
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = ERA5tx_p95_inv.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatoTempPeru/outputs/ERA5tx_p95_inv.png")

#ERA5land (tn:temperatura mínima)
ERA5_files = []
for file in ERA_files[0:-5]:
  ERA5t = xr.open_dataset(file)
  ERA5t = ERA5t.resample(time='1D').min()
  ERA5t = ERA5t - 273.15
  ERA5t = ERA5t.rename_vars({"t2m":"tn"})
  ERA5_files.append(ERA5t)

ERA5tn = xr.concat(ERA5_files, dim="time")

shp_exp_grid = xr_shp_to_grid(shp_i = shp_Peru,
                              netcdf_array = ERA5tn.tn)

ERA5tn_masked = xr_mask(grid_mask = shp_exp_grid,
                             netcdf_i = ERA5tn.tn)

##Verano
###Promedio
ERA5tn_season_prom = ERA5tn_masked.groupby("time.season").mean("time")
ERA5tn_prom_ver = (ERA5tn_season_prom.sel(season="DJF"))
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = ERA5tn_prom_ver.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatoTempPeru/outputs/ERA5tn_prom_ver.png")

###Desviación estándar
ERA5tn_season_sd = ERA5tn_masked.groupby("time.season").std("time")
ERA5tn_sd_ver = (ERA5tn_season_sd.sel(season="DJF"))
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = ERA5tn_sd_ver.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatoTempPeru/outputs/ERA5tn_sd_ver.png")

###Coeficiente de asimetria
ERA5tn_season_md = ERA5tn_masked.groupby("time.season").median("time")
ERA5tn_md_ver = (ERA5tn_season_sd.sel(season="DJF"))
ERA5tn_ac_ver = (3*(ERA5tn_prom_ver-ERA5tn_md_ver))/ERA5tn_sd_ver
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = ERA5tn_ac_ver.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatoTempPeru/outputs/ERA5tn_ac_ver.png")

###Percentil 5
ERA5tn_season_p5 = ERA5tn_masked.groupby("time.season").quantile(0.05, "time")
ERA5tn_p5_ver = (ERA5tn_season_p5.sel(season="DJF"))
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = ERA5tn_p5_ver.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatoTempPeru/outputs/ERA5tn_p5_ver.png")

###Percentil 95
ERA5tn_season_p95 = ERA5tn_masked.groupby("time.season").quantile(0.95, "time")
ERA5tn_p95_ver = (ERA5tn_season_p95.sel(season="DJF"))
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = ERA5tn_p95_ver.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatoTempPeru/outputs/ERA5tn_p95_ver.png")

##Invierno
###Promedio
ERA5tn_prom_inv = (ERA5tn_season_prom.sel(season="JJA"))
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = ERA5tn_prom_inv.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatoTempPeru/outputs/ERA5tn_prom_inv.png")

###Desviación estándar
ERA5tn_sd_inv = (ERA5tn_season_sd.sel(season="JJA"))
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = ERA5tn_sd_inv.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatoTempPeru/outputs/ERA5tn_sd_inv.png")

###Coeficiente de asimetria
ERA5tn_md_inv = (ERA5tn_season_sd.sel(season="JJA"))
ERA5tn_ac_inv = (3*(ERA5tn_prom_inv-ERA5tn_md_inv))/ERA5tn_sd_inv
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = ERA5tn_ac_inv.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatoTempPeru/outputs/ERA5tn_ac_inv.png")

###Percentil 5
ERA5tn_p5_inv = (ERA5tn_season_p5.sel(season="JJA"))
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = ERA5tn_p5_inv.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatoTempPeru/outputs/ERA5tn_p5_inv.png")

###Percentil 95
ERA5tn_p95_inv = (ERA5tn_season_p95.sel(season="JJA"))
fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = ERA5tn_p95_inv.plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatoTempPeru/outputs/ERA5tn_p95_inv.png")

#ERA5land (tm:temperatura media)
ERA5tm_masked = (ERA5tx_masked+ERA5tn_masked)/2

##Verano
fechas = ['1981-01-01']
for year in range(1981,2017):
  fechas.append(get_last_date_of_month(year, 2))
  fechas.append(get_first_date_of_current_month(year, 12))

fechas = fechas[:-1]

ERA5tm_mean_files = []
for i in range(0,72,2):
  ERA5tm_year = ERA5tm_masked.sel(time=slice(fechas[i], fechas[i+1])).mean("time")
  ERA5tm_mean_files.append(ERA5tm_year)

ERA5tm_mean_s_ver = xr.concat(ERA5tm_mean_files, dim="time")

###%min < p5
ERA5tn_por_files = []
for i in range(0,72,2):
  ERA5tn_year = ERA5tn_masked.sel(time=slice(fechas[i], fechas[i+1]))
  ERA5tn_por = (ERA5tn_year.where(ERA5tn_year < ERA5tn_p5_ver, drop=True).count("time"))*100/len(ERA5tn_year)
  ERA5tn_por_files.append(ERA5tn_por)

ERA5tn_por_ver = xr.concat(ERA5tn_por_files, dim="time")

fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = xr.corr(ERA5tn_por_ver, ERA5tm_mean_s_ver, dim="time").plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatoTempPeru/outputs/ERA5tm_%minp5_ver.png")

###%max>p95
ERA5tx_por_files = []
for i in range(0,72,2):
  ERA5tx_year = ERA5tx_masked.sel(time=slice(fechas[i], fechas[i+1]))
  ERA5tx_por = (ERA5tx_year.where(ERA5tx_year > ERA5tx_p95_ver, drop=True).count("time"))*100/len(ERA5tx_year)
  ERA5tx_por_files.append(ERA5tx_por)

ERA5tx_por_ver = xr.concat(ERA5tx_por_files, dim="time")

fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = xr.corr(ERA5tx_por_ver, ERA5tm_mean_s_ver, dim="time").plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatoTempPeru/outputs/ERA5tm_%maxp95_ver.png")

##Invierno
fechas = []
for year in range(1981,2017):
  fechas.append(get_first_date_of_current_month(year, 6))
  fechas.append(get_last_date_of_month(year, 8))

ERA5tm_mean_files = []
for i in range(0,72,2):
  ERA5tm_year = ERA5tm_masked.sel(time=slice(fechas[i], fechas[i+1])).mean("time")
  ERA5tm_mean_files.append(ERA5tm_year)

ERA5tm_mean_s_inv = xr.concat(ERA5tm_mean_files, dim="time")

###%min<p5
ERA5tn_por_files = []
for i in range(0,72,2):
  ERA5tn_year = ERA5tn_masked.sel(time=slice(fechas[i], fechas[i+1]))
  ERA5tn_por = (ERA5tn_year.where(ERA5tn_year < ERA5tn_p5_ver, drop=True).count("time"))*100/len(ERA5tn_year)
  ERA5tn_por_files.append(ERA5tn_por)

ERA5tn_por_inv = xr.concat(ERA5tn_por_files, dim="time")

fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = xr.corr(ERA5tn_por_inv, ERA5tm_mean_s_inv, dim="time").plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatoTempPeru/outputs/ERA5tm_%minp5_inv.png")

###%max>p95
ERA5tx_por_files = []
for i in range(0,72,2):
  ERA5tx_year = ERA5tx_masked.sel(time=slice(fechas[i], fechas[i+1]))
  ERA5tx_por = (ERA5tx_year.where(ERA5tx_year > ERA5tx_p95_ver, drop=True).count("time"))*100/len(ERA5tx_year)
  ERA5tx_por_files.append(ERA5tx_por)

ERA5tx_por_inv = xr.concat(ERA5tx_por_files, dim="time")

fig, ax = plt.subplots(figsize=(12,12))
shp_Peru.boundary.plot(ax=ax,edgecolor='black',linewidth=0.1)
pl = xr.corr(ERA5tx_por_inv, ERA5tm_mean_s_inv, dim="time").plot.contourf()
plt.savefig("C:/Users/FELIPEOH/ClimatoTempPeru/outputs/ERA5tm_%maxp95_inv.png")