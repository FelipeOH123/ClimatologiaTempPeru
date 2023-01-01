import numpy as np

def xr_shp_to_grid(shp_i, netcdf_array):

  # get real box
  shp_i_geometry = shp_i.geometry

  # adding crs
  mask = netcdf_array.rio.set_crs(shp_i.crs)

  # "rasterizing"
  mask = mask.rio.clip(shp_i_geometry, drop = False)

  # making "True/False" values
  mask.values[~np.isnan(mask.values)] = 1

  return mask.drop(["time", "spatial_ref"])


def xr_mask(grid_mask, netcdf_i):

  # masking
  mask_netcdf_i = netcdf_i.where(grid_mask == True)

  return mask_netcdf_i