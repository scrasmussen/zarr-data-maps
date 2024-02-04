import xarray as xr
import pandas as pd
import ndpyramid as ndp
import rioxarray

import sys
import zarr
from numcodecs import Zlib


if ndp.__version__ != '0.1.0':
    print(f"Error: ndpyramid version {ndp.__version__} != required 0.1.0")
    sys.exit(0)


print("Setting up Compressor")

# enc = zlib.Zlib()
# compressor = zarr.Blosc(cname="zlib",clevel=1)



VERSION = 2
# LEVELS = 6
# PIXELS_PER_TILE = 128


input_path = '/glade/work/soren/src/icar/data/icar-zarr-data/data_input'
save_path = f"data_zarr/icar-noresm"

# input dataset
# only noresm_hist_exl_conv_2000_2005.nc

ds1 = []
# months = list(map(lambda d: d + 1, range(12)))
path = f"{input_path}/noresm_hist_exl_conv_2000_2005.nc"
print("Opening", path)
ds1 = xr.open_dataset(path) #, engine="netcdf4")


print("Selecting time frame")

daily_avg_precip = ds1['pcp'].mean(dim=['lat', 'lon'])
daily_avg_temp = ds1['t_mean'].mean(dim=['lat', 'lon'])
daily_min_temp = ds1['t_min'].mean(dim=['lat', 'lon'])
daily_max_temp = ds1['t_max'].mean(dim=['lat', 'lon'])

monthly_avg_precip = daily_avg_precip.resample(time='1M').mean(dim='time')
monthly_avg_temp = daily_avg_temp.resample(time='1M').mean(dim='time')
monthly_min_temp = daily_min_temp.resample(time='1M').mean(dim='time')
monthly_max_temp = daily_max_temp.resample(time='1M').mean(dim='time')

ds1 = xr.Dataset({'pcp': monthly_avg_precip,
                  't_mean': monthly_avg_temp,
                  't_min': monthly_min_temp,
                  't_max': monthly_max_temp
                  })
print("Transforming variables to match website")
# --- transform to dataset for website, test ---

# rename dimensions
ds1 = ds1.rename({'time':'month',
                  # 'lat_y':'y',
                  # 'lon_x':'x',
                  # 'lat':'y',
                  # 'lon':'x',
                  # 'precipitation':'prec',
                  'pcp':'prec',
                  # 'ta2m':'tavg'
                  't_mean':'tavg',
                  't_min':'tmin',
                  't_max':'tmax'
                  })
# add climate (aka variable) dimension
# print(" - add climate (aka variable) dimension")
# var1='prec'; var2='tavg'
# ds1['climate'] = xr.concat([ds1[var1], ds1[var2]],
#                            dim='band')
# encodings
# enc = {x: {"compressor": compressor} for x in ds1}
print("Write to Zarr")
# comp = dict(zlib=True, complevel=1)
# for var in ds1.data_vars:
#     var.encoding.update(comp)
# ds1.to_zarr(save_path + '/monthly_prec_tmps', consolidated=True, encoding=enc)
# ds1.to_zarr(save_path + '/monthly_prec_tmps', encoding=Zlib(level=1))
# ds1.to_zarr(save_path + '/monthly_prec_tmps', encoding=None)
# ds1.to_zarr(save_path + '/monthly_prec_tmps') #, consolidated=True)

ds1.to_netcdf(save_path + '/monthly_prec_tmps.nc')#, format="NETCDF3_CLASSIC")
sys.exit()


# cleanup variables
keep_vars = ['climate']
all_vars = list(ds1.data_vars)
remove_vars = [var for var in all_vars if var not in keep_vars]
ds1 = ds1.drop_vars(remove_vars)

# add band coordinates
print(" - add band coordinates")
band_var_names = ['prec','tavg']
fixed_length = 4
var_names_U4 = [s[:fixed_length].ljust(fixed_length) for s in band_var_names]
ds1 = ds1.assign_coords(band=var_names_U4)


# --- clean up types
print(" - clean up types")
# month to int type
ds1['month'] = xr.Variable(dims=('month',),
                           data=list(range(1, 12 + 1)))
                           # data=list(range(1, ds1.month.shape[0] + 1)))
                           # attrs={'dtype': 'int32'})
ds1["month"] = ds1["month"].astype("int32")
ds1["climate"] = ds1["climate"].astype("float32")
ds1["band"] = ds1["band"].astype("str")
ds1.attrs.clear()


# --- force to be like their data
print (" - force to be like their data[??]")
ds1 = ds1.where(ds1.month<=12, drop=True)


print("Write to Zarr")
# write the pyramid to zarr, defaults to zarr_version 2
# consolidated=True, metadata files will have the information expected by site
dt.to_zarr(save_path + '/4d-ndp0.1/tavg-prec-month', consolidated=True)
