import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import ndpyramid as ndp

import sys
import zarr

if ndp.__version__ != '0.1.0':
    print(f"Error: ndpyramid version {ndp.__version__} != required 0.1.0")
    sys.exit(0)

VERSION = 2
LEVELS = 6
PIXELS_PER_TILE = 128

input_path = f"gs://carbonplan-maps/v{VERSION}/demo/raw"
save_path = f"data_demo"

input_path = '/home/artless/Documents/src/ncar/icar/website/data/data/wc2_data'
input_path = '/home/artless/Documents/src/ncar/icar/website/data/data/icar_output'
save_path = f"icar_zarr"

# input dataset
# only icar_out_2000-10-01_00-00-00.nc

ds1 = []
# months = list(map(lambda d: d + 1, range(12)))
days = list(map(lambda d: d + 1, range(9,10)))
for i in days:
    path = f"{input_path}/icar_out_2000-{i:02g}-01_00-00-00.nc"  # tavg originally
    # FOO: may need band variable??
    ds = xr.open_dataset(path, engine="netcdf4")
    if ds1:
        ds1.append(ds)
    else:
        ds1 = ds

   # .squeeze() #.reset_coords(["band"], drop=True)
    # ds = (
    #     xr.open_dataarray(path, engine="netcdf4") # this is dataset so open_dataset()
    #      .to_dataset(name="climate")
    #      .squeeze()
    #      .reset_coords(["band"], drop=True)
    # )



# --- transform to dataset for website, test ---
# transform 2d lat lon dimension to 1d
ds1['lat'] = ds1.lat[:,0]
ds1['lon'] = ds1.lon[0,:]
# rename dimensions
ds1 = ds1.rename({'time':'month',
                  'lat_y':'y',
                  'lon_x':'x',
                  'lat':'y',
                  'lon':'x',
                  'precipitation':'prec',
                  'ta2m':'tavg'})
# add climate (aka variable) dimension
var1='prec'; var2='tavg'
ds1['climate'] = xr.concat([ds1[var1], ds1[var2]],
                           dim='band')
var_names = [var1, var2]
ds1 = ds1.drop(var_names)
# add band coordinates
fixed_length = 4
var_names_U4 = [s[:fixed_length].ljust(fixed_length) for s in var_names]
ds1 = ds1.assign_coords(band=var_names_U4)


# --- clean up types
# month to int type
ds1['month'] = xr.Variable(dims=('month',),
                           data=list(range(1, ds1.month.shape[0] + 1)))
                           # attrs={'dtype': 'int32'})
ds1["month"] = ds1["month"].astype("int32")
ds1["climate"] = ds1["climate"].astype("float32")
ds1["band"] = ds1["band"].astype("str")
ds1.attrs.clear()


# --- force to be like their data
ds1 = ds1.where(ds1.month<=12, drop=True)


# sys.exit()


# --- create the pyramid
# EPSG:4326 fixes error:
#    MissingCRS: CRS not found. Please set the CRS with 'rio.write_crs()'
ds1 = ds1.rio.write_crs('EPSG:4326')
dt = ndp.pyramid_reproject(ds1,
                           levels=LEVELS,
                           pixels_per_tile=PIXELS_PER_TILE,
                           extra_dim='band')

dt = ndp.utils.add_metadata_and_zarr_encoding(dt,
                                              levels=LEVELS,
                                              pixels_per_tile=PIXELS_PER_TILE)

# write the pyramid to zarr, defaults to zarr_version 2
# consolidated=True, metadata files will have the information expected by site
dt.to_zarr(save_path + '/4d-ndp0.1/tavg-prec-month', consolidated=True)
