import pandas as pd
import numpy as np
import sys
import xarray as xr
import zarr

from tools import dimensionNames

# --- combination of downscaling methods and climate models to create data
downscaling_methods = [
    'icar',
    'gard',
    'LOCA',
    'bcsd',]
climate_models = [
    'noresm',
    'cesm',
    'gfdl',
    'miroc5',]
time = 'time'

# main: open input data, manipulate it, write to zarr
def main():
    data_paths, method, model = get_arguments()
    ds = open_data_srcs(data_paths)
    print("Opened", method, "downscaling method and", model, "climate model")

    ds = rename_dimensions(ds, method, model)
    ds = drop_extra_dimensions(ds)
    ds = handle_time_dimension(ds)

    # ds = convert_to_zarr_format(ds) # already in single precision
    write_to_zarr(ds, method, model)

    print('fin')
    sys.exit()

# argument check
def get_arguments():
    if len(sys.argv) < 2:
        print("Usage: python script.py <dir_path> [<file1> <file2> ...]")
        sys.exit(0)
    data_path = sys.argv[1]
    data_files = sys.argv[2:]
    fullpaths = combine_path_and_files(data_path, data_files)
    method, model = get_method_and_model(fullpaths[0])
    return fullpaths, method, model

def get_method_and_model(f):
    found = False
    method = None
    model = None
    for m in downscaling_methods:
        if (m in f):
            if found:
                print("Error: multiple methods found in path")
            found = True
            method = m
    found = False
    for m in climate_models:
        if (m in f):
            if found:
                print("Error: multiple models found in path")
            found = True
            model = m

    if (method == None) or (model == None):
        print("Error: method =", method, ", model =", model)
    return method.lower(), model.lower()

def combine_path_and_files(data_path, data_files):
    combined_paths = []
    for f in data_files:
        combined_paths.append(data_path + '/' + f)
    return combined_paths

def open_data_srcs(data_srcs):
    print("OPENING ", data_srcs)
    if (len(data_srcs) == 1):
        ds = xr.open_dataset(data_srcs[0])
    else:
        datasets = []
        for f in data_srcs:
            datasets.append(xr.open_dataset(f))
        ds = xr.concat(datasets, dim='time')
    return ds

def rename_dimensions(ds, method, model):
    new_dims = dimensionNames.get_dimension_name(method, model)
    ds = ds.rename(new_dims)
    return ds

def drop_extra_dimensions(ds):
    print(ds)
    vars_to_keep = [time, 'y', 'x', 'prec', 'tavg']
    vars_to_drop = [var for var in ds.variables if var not in vars_to_keep]
    ds = ds.drop_vars(vars_to_drop)
    return ds

def handle_time_dimension(ds):
    print("Creating spatial and monthly average")
    print("   - if silent failure, run on interactive node")

    # --- smaller time slice for testing
    # print("REMOVE 1999-2001 TIME SLICE")
    # ds = ds.sel(time=slice("1999","2001"))

    spatial_avg_precip = ds['prec'].mean(dim=['y', 'x'])
    spatial_avg_temp = ds['tavg'].mean(dim=['y', 'x'])
    monthly_avg_precip = spatial_avg_precip.resample(time='MS').mean(dim='time')
    monthly_avg_temp = spatial_avg_temp.resample(time='MS').mean(dim='time')

    ds = xr.Dataset({'prec': monthly_avg_precip,
                     'tavg': monthly_avg_temp
    #                   # 't_min': monthly_min_temp,
    #                   # 't_max': monthly_max_temp
                  })

    return ds

def write_to_zarr(ds, method, model):
    print("Writing to zarr format")
    save_path = f'data/chart/' + method + '/' + model
    prec_save_path = save_path + '/prec'
    tavg_save_path = save_path + '/tavg'

    print("WARNING: NEED TO ADD LARGER TIME FRAME")
    # write precip
    z_prec = zarr.open(prec_save_path, mode='w', shape=len(ds.prec),
                  compressor=None, dtype=np.float32)
    z_prec[:] = ds.prec.data

    # write temp
    z_temp = zarr.open(tavg_save_path, mode='w', shape=len(ds.tavg),
                  compressor=None, dtype=np.float32)
    z_temp[:] = ds.tavg.data

if __name__ == "__main__":
    main()
