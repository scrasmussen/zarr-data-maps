import argparse
from collections import defaultdict
from fnmatch import fnmatch
import glob
import itertools
import json
import os
import pandas as pd
import ndpyramid as ndp
import numpy as np
import re
import rioxarray
import sys
import xarray as xr
# import xesmf as xe
import yaml
import zarr
from tools import dimensionNames, handleArgs

print("ndpyramid Version = ", ndp.__version__)

# maps_matching_str = '*CMIP5*signal*'
# maps_matching_str = '*CMIP6*signal*'
maps_matching_str = 'signal_pr_snr.nc'


# Tables of Sam's Input Files
# note, the files are not guarenteed to have the same set of variables
#
# - file name: {Downscaling}_CMIP5_signal_{Variable Signal}.nc
# | CMIP  | Downscaling | Variable Signal    | GCMs per var |
# |-------+-------------+--------------------+--------------|
# | CMIP5 | BCCA        | pr, tasmin, tasmax | 6,6,6        |
# | CMIP5 | BCSD        | pr, tasmin, tasmax | 7,7,5        |
# | CMIP5 | GARD        | pr, tasmin, tasmax | 5,5,5        |
# | CMIP5 | GCM RCP45   | pr, tasmin, tasmax | 5,5,5        |
# | CMIP5 | ICAR        | pr, tasmin, tasmax | 5,5,5        |
# | CMIP5 | LOCA        | pr, tasmin, tasmax | 7,7,6        |
# | CMIP5 | MACA        | pr, tasmin, tasmax | 6,6,6        |
# | CMIP5 | NACORDEX    | pr, tasmin, tasmax | 1,1,1        |
# | CMIP5 | NEXGDDP     | pr, tasmin, tasmax | 7,7,7        |

# CMIP6 files without signal have time dimension and not plotted.
# - file name: {Downscaling}_CMIP6_{SSP}_signal_{Variable Signal}.nc
# | CMIP  | Downscaling | SSP      | Variable Signal    | GCMS per var |
# |-------+-------------+----------+--------------------+--------------|
# | CMIP6 | DEEPSD      | 370, 585 | pr, tasmax         | 2, 2, 2, 2   |
# | CMIP6 | GARDLENS    | 370      | pr, tasmin, tasmax | 1,1,1        |
# | CMIP6 | GCM         | 370, 585 | pr, tasmax         | 2, 3, 2, 3   |
# | CMIP6 | ICAR        | 370, 585 | pr, tasmax         | 2, 2, 2, 2   |
# | CMIP6 | LOCA2       | 370, 585 | pr, tasmax         | 5, 3, 5, 3   |
# | CMIP6 | NEXGDDP     | 370, 585 | pr, tasmax         | 5, 5, 5, 5   |
# | CMIP6 | REGCM       | 585      | pr                 | 3            |
# | CMIP6 | STARESDM    | 585      | pr, tasmax         | 4, 4         |
# | CMIP6 | WUS-D3      | none     | pr, tasmax         | 2, 2         |
# | CMIP6 | WUS-D3      | 370      | pr                 | 3            |


# class NoQuotesDumper(yaml.SafeDumper):
#     def represent_str(self, data):
#         # Always use plain scalars, no quotes
#         return self.represent_scalar("tag:yaml.org,2002:str", data, style='')
# NoQuotesDumper.add_representer(str, NoQuotesDumper.represent_str)

# class PlainKeyDumper(yaml.SafeDumper):
#     pass
# def str_representer(dumper, data):
#p     # Only force plain style for dict keys
#     return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='')
# # Attach representer for str type used in dict keys
# PlainKeyDumper.add_representer(str, str_representer)
# --- quote values, leave keys plain ---
class Quoted(str):
    pass

def quoted_presenter(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')

yaml.add_representer(Quoted, quoted_presenter, Dumper=yaml.SafeDumper)

LEVELS = 4
# LEVELS = 1
PIXELS_PER_TILE = 512 # this one too high
PIXELS_PER_TILE = 256
FIXED_LENGTH = 4


# Define the global grid (covering the entire world)
# global_lon = np.linspace(-180, 180, int(360/0.125 + 1))  # 0.125-degree resolution
# global_lat = np.linspace(-90, 90, int(180/0.125 + 1))    # 0.125-degree resolution

# world_grid = xr.Dataset({
#     'lat': (['lat'], global_lat),
#     'lon': (['lon'], global_lon)
# })
test = False
debug=False

# set these somewhere else?
VERSION = 2
time = 'time'

PAST=0
FUTURE=1
CLIMATE_SIGNAL=2


class Dataset:
    def __init__(self, file_path, ds_type, era, region,
                 method=None, model=None, obs=None, ens=None,
                 dif_file_path=None, metric=None):
        if not os.path.exists(file_path):
            print("ERROR: file path does not exist:", file_path)
            sys.exit()
        self.obs = obs
        self.method = method
        self.metric = metric
        self.model = model
        self.era = era
        self.region = region
        self.ens = ens
        self.ds_type = ds_type
        self.file_path = file_path
        self.dif_file_path = dif_file_path
    def print(self):
        print("Dataset:")
        print("  file_path =", self.file_path)
        print("  ds_type =", self.ds_type)
        print("  era =", self.era)
        print("  region =", self.region)
        print("  method =", self.method)
        print("  model =", self.model)
        print("  metric =", self.metric)
        print("  obs =", self.obs)
        print("  ens =", self.ens)
        print("  dif_file_path =", self.dif_file_path)



class Options:
    def __init__(self, args):
        # default values
        # 3 requires paths
        self.input_maps_path = args.input_maps_path
        self.input_obs_path = args.input_obs_path
        self.input_metrics_path = args.input_metrics_path
        # --climate-signal
        self.write_climate_signal = False
        self.climate_signal_path = None
        # --maps
        self.write_maps = False
        self.output_maps_path = None
        # --metric-score
        self.write_metric_score = False
        self.metric_score_path = None
        # --time-series
        self.write_timeseries = False
        self.timeseries_path = None
        # -- obs
        self.write_obs = False
        self.output_obs_path = None
        # --write-yaml
        self.write_yaml = False
        # --test
        self.test = False
        if args.output_maps_path != None:
            self.write_maps = True
            self.output_maps_path = self.check_path(args.output_maps_path)
        if args.metric_score_path != None:
            self.write_metric_score = True
            self.metric_score_path = self.check_path(args.metric_score_path)
        if args.output_obs_path != None:
            self.write_obs = True
            self.output_obs_path = self.check_path(args.output_obs_path)
        if args.climate_signal_path != None:
            self.write_climate_signal = True
            self.climate_signal_path = self.check_path(args.climate_signal_path)
        if args.timeseries_path != None:
            self.write_timeseries = True
            self.timeseries_path = self.check_path(args.timeseries_path)
        if args.write_yaml == True:
            self.write_yaml = True
        if args.test != None:
            self.test_in = True
            global test
            test = True
    def check_path(self, path, trailing_slash=True):
        if isinstance(path, list):
            path = path[0]
        if trailing_slash and path[-1] != '/':
            path += '/'
        return path
    def print(self):
        print("Options:")
        print("  input maps path =", self.input_maps_path)
        print("  input obs path =", self.input_obs_path)
        print("  input metrics path =", self.input_metrics_path)
        print("  maps write =", self.write_maps,
              ", maps_output_path =", self.output_maps_path)
        print("  metric score write =", self.write_metric_score,
              ", metric_score_path =", self.metric_score_path)
        print("  climate signal write =", self.write_climate_signal,
              ", path =", self.climate_signal_path)
        print("  obs write =", self.write_obs,
              ", obs_output_path =", self.output_obs_path)
        print("  write yaml =", self.write_yaml)
        print("  test =", self.test)

# change foo: bar entries to foo: "bar"
# def quoted_presenter(dumper, data):
#     return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')
# yaml.add_representer(str, quoted_presenter)


def check_arrays(A, B, array_type):
    res = set(A).difference(set(B))
    if res:
        print(f"Testing Error: Input {array_type} missing", res)
        sys.exit()


def convertDStoDZ(ds):
    # put in format for zarr
    ds = ds.drop_vars('gcm_')
    if 'height' in ds.coords:
        ds = ds.drop_vars('height')
    print(ds)

    variables = list(ds.variables.keys())
    variables = [var for var in variables if var not in ['x', 'y']]
    concatenated_vars = []
    for var_name in variables:
        concatenated_vars.append(ds[var_name])
    ds['climate'] = xr.concat(concatenated_vars, dim='band')
    var_names_U4 = [s[:FIXED_LENGTH].ljust(FIXED_LENGTH) for s in variables]
    ds = ds.assign_coords(band=var_names_U4)
    ds = ds.drop_vars(set(ds.data_vars) - set(['climate']))
    ds["climate"] = ds["climate"].astype("float32")
    ds["band"] = ds["band"].astype("str")
    ds.attrs.clear()
    dz = convert_to_zarr_format(ds) # already in single precision

    return dz



def writeDatasetToZarr(output_path, dataset=None,
                       write_maps=False,
                       write_climate_signal=False,
                       write_metric_score=False,
                       write_obs=False,
                       write_timeseries=False):
    method = dataset.method
    model = dataset.model
    if method != None:
        method_s = method.lower().replace('-','_')
    if model != None:
        model_s = model.lower().replace('-','_')


    print("Opening file(s):")

    ds = xr.open_dataset(dataset.file_path)
    print("Opened pr ds:", dataset.file_path)

    # add and merge min and max datasets into pr one if they exist
    # ds_min = None
    # ds_max = None
    # decided not to merge
    # tasmin_path = dataset.file_path.replace("_pr.nc", "_tasmin.nc")
    # if os.path.exists(tasmin_path):
    #     ds_min = xr.open_dataset(tasmin_path)
    # tasmax_path = dataset.file_path.replace("_pr.nc", "_tasmax.nc")
    # if os.path.exists(tasmax_path):
    #     ds_max = xr.open_dataset(tasmax_path)

    # these are breaking for various reasons
    if 'GARDLENS_CMIP6_ssp370' in dataset.file_path:
        return
    elif 'DEEPSD_CMIP6_ssp585' in dataset.file_path:
        return
    # if 'LOCA2_CMIP6_ssp585' in dataset.file_path:
    #     return
    # elif 'GCM_CMIP6_ssp585' in dataset.file_path:
    #     return
    # elif 'LOCA2_CMIP6_ssp370' in dataset.file_path:
    #     return
    # elif 'GCM_CMIP6_ssp370' in dataset.file_path:
    #     return



    # if ds_min != None:
    #     print("Merging ds_min:", tasmin_path)
    #     ds = xr.merge([ds, ds_min], compat="no_conflicts", join="exact")

    # if ds_max != None:
    #     print("Merging ds_max:", tasmax_path)
    #     ds = xr.merge([ds, ds_max], compat="no_conflicts", join="exact")

    print(ds)

    # variables for zarr creation, value has to be four characters
    new_vars = {
        # --- Sam's metrics ---
        'mean_prcp':'am_p',
        'mean_jja_prcp':'jjap',
        'sum_prcp':'sump',
        'q95_prcp':'q95p',
        'std_prcp':'stdp',

        'mean_pr':'am_p',
        'mean_jja_pr':'jjap',
        'sum_pr':'sump',
        'q95_pr':'q95p',
        'std_pr':'stdp',
        '2yr_pr':'2yrp',
        '5yr_pr':'5yrp',
        'gcm':'gcm_',
        # max
        'mean_tasmax':'amtx',
        'mean_jja_tasmax':'jjax',
        'q95_tasmax':'q95x',
        'sum_tasmax': 'sumx',
        'std_tasmax': 'stdx',
        # min
        'mean_djf_tasmin':'djfi' ,
        'mean_tasmin':'am_i',
        'max_tasmin':'maxi',
        'q95_tasmin':'q95i',
        'std_tasmin':'stdi',
        '2yr_tasmin':'2yri',
        # --- original variables ---
        'lat': 'y',
        'lon': 'x',
        'n34pr':'n34p',
        'nino3.4_t':'n34t',
        'nino3.4_p':'n34p',
        'ttrend':'ttre',
        'ptrend':'ptre',
        'pr90':'pr90',
        'pr99':'pr99',
        't90':'t90_',
        't99':'t99_',
        'eli_t':'elit',
        'eli_p':'elip',
        'djf_t':'djft',
        'djf_p':'djfp',
        'mam_t':'mamt',
        'mam_p':'mamp',
        'jja_t':'jjat',
        'jja_p':'jjap',
        'son_t':'sont',
        'son_p':'sonp',
        # for time series
        'djf_pr':'djfp',
        'mam_pr':'mamp',
        'jja_pr':'jjap',
        'son_pr':'sonp',
        # ---
        'ann_t':'annt',
        'ann_p':'annp',
        'ann_snow':'anns',
        'freezethaw':'fzth',
        'tpcorr':'tpco',
        'drought_1yr':'d1yr',
        'drought_2yr':'d2yr',
        'drought_5yr':'d5yr',
        'wt_clim':'wtcl',
        'wt_day_to_day':'wtds',
        # --- new metrics:
        # {ann,seasonal}_iav_{t,p}, ann_snow_iav,
        # pr_gev_{20,50,100}yr, wet_day_frac
        'ann_p_iav':'anpi',
        'djf_p_iav':'djpi',
        'mam_p_iav':'mapi',
        'jja_p_iav':'jjpi',
        'son_p_iav':'sopi',
        'ann_t_iav':'anti',
        'djf_t_iav':'djti',
        'mam_t_iav':'mati',
        'jja_t_iav':'jjti',
        'son_t_iav':'soti',
        'ann_snow_iav':'ansi',
        'pr_gev_20yr':'g_20',
        'pr_gev_50yr':'g_50',
        'pr_gev_100yr':'g100',
        'wet_day_frac':'wdfr',
        # agreement variables
        'snr':'snr_', # signal-to-noise ratio
        # 'snr_abs':'snr_', # signal-to-noise ratio
        'disagreement':'dagr', # disagreement
    }


    if ('_snr.nc' in dataset.file_path):
        # variables needed for snr
        signal_to_noise_drop_vars = [
            'snr_abs',
            'signal_mean', 'signal_std', 'n_downscaling_models', 'conus_mask'
        ]
        if 'quantile' in list(ds.variables):
            signal_to_noise_drop_vars.append('quantile')
        if 'quan' in list(ds.variables):
            signal_to_noise_drop_vars.append('quan')
        ds['snr'] = np.abs(ds['snr'])
        ds = ds.drop_vars(signal_to_noise_drop_vars)
    if ('_agreement.nc' in dataset.file_path):
        agreement_drop_vars = [
            'agreement',
            'hatch_mask',
            'direction',
            'agree_positive',
            'agree_negative',
            'n_positive',
            'n_negative',
            'n_zero',
            'n_downscaling_models',
            'conus_mask',
        ]
        if 'quantile' in list(ds.variables):
            agreement_drop_vars.append('quantile')
        if 'quan' in list(ds.variables):
            agreement_drop_vars.append('quan')
        ds = ds.drop_vars(agreement_drop_vars)


    if 'member_id' in list(ds.dims):
        ds = ds.isel(member_id=0)

    print(ds)
    # sys.exit()


    # lowercase dimension names
    lowercase_vars = {v: v.lower() for v in ds.data_vars}
    ds = ds.rename(lowercase_vars)

    # renames - to _
    rename_vars = {v: v.replace("-", "_") for v in ds.data_vars if "-" in v}
    ds = ds.rename(rename_vars)

    valid_rename_map = {k: v for k, v in new_vars.items()
                        if k in ds.variables or k in ds.dims}
    ds = ds.rename(valid_rename_map)

    bad_names = any(len(v) != 4 for v in list(ds.data_vars))

    if (bad_names):
        print("---")
        print("ds")
        print(ds)
        print("vars:", list(ds.data_vars))
        print("---")
        print("error: some of the names are the wrong length, != 4")
        print("bad_names =", bad_names)
        sys.exit()

    print('variables renamed')

    # zarr format requires variables four characters in length
    invalid_vars = [var for var in ds.data_vars if len(var) != FIXED_LENGTH]
    if invalid_vars:
        print("Variables with names not 4 characters long:", invalid_vars)
        sys.exit()
    orig_ds = ds
    # iterate through GCMs
    for gcm in orig_ds['gcm_']:
        gcm_ds = orig_ds.sel(gcm_=gcm)
        dz = convertDStoDZ(gcm_ds)

        gcm_s = gcm.item().lower()
        write_path = output_path + method_s + '/' + \
            model_s + '/' + \
            gcm_s + '/' + \
            dataset.metric.lower()
        if method == 'CMIP6':
            era_s = dataset.era.lower().replace('-','_')
            write_path = output_path + method_s + '/' + \
                model_s + '/' + \
                gcm_s + '/' + \
                era_s + '/' + \
                dataset.metric.lower()

            # not doing era, not sure which
            # dataset.era.lower().replace('-','_')
        dataset.print()
        print("write_path =", write_path)
        # sys.exit()
        write_to_zarr(dz, write_path)
        print(f'wrote {gcm} to {write_path}')
    # print('exit writeDatasetToZarr : write_path=', write_path)
    # sys.exit()


def write_to_zarr(ds, output_path, zarr_file='data.zarr'):
    save_f = output_path + '/' + zarr_file
    if (os.path.exists(save_f)):
        print("ERROR WRITING ZARR FILE: path exists at", save_f)
    print("Writing to Zarr file", save_f)
    ds.to_zarr(save_f, consolidated=True) #, encoding={"zlib":True})
    print("Done writing to zarr format")


def addEnsembleDatasets(datasets):
    ensemble_datasets = []
    for d in datasets:
        ds = xr.open_dataset(d.file_path)[['ens']]
        for ens in ds['ens'].values:
            new_ds = Dataset(d.file_path,
                             d.ds_type,
                             'hist.'+d.era,
                             d.region,
                             method = d.method,
                             model = d.model,
                             obs = d.obs,
                             ens = ens)
            ensemble_datasets.append(new_ds)
    return ensemble_datasets

def addTimeseriesDatasets(datasets):
    ensemble_datasets = []
    for d in datasets:
        ds = xr.open_dataset(d.file_path)[['ens']]
        for ens in ds['ens'].values:
            new_ds = Dataset(d.file_path,
                             d.ds_type,
                             'hist.'+d.era,
                             d.region,
                             method = d.method,
                             model = d.model,
                             obs = d.obs,
                             ens = ens)
            ensemble_datasets.append(new_ds)
    return ensemble_datasets



def checkMetricVarsEquality(datasets):
    ds = xr.open_dataset(datasets[0].file_path)
    vars_list = list(ds.data_vars)
    print(vars_list)

    for d in datasets:
        ds = xr.open_dataset(d.file_path)
        if (vars_list != list(ds.data_vars)):
            print("ERROR: metric observations are not equal for file ", d.file_path)
            sys.exit()
    print("All metric files have same variables")
    return vars_list


def checkMetricObsEquality(datasets):
    ds = xr.open_dataset(datasets[0].file_path)
    obs = ds['obs']
    for d in datasets:
        ds = xr.open_dataset(d.file_path)
        if (not obs.equals(ds['obs'])):
            print("ERROR: metric observations are not equal for file ", d.file_path)
            sys.exit()
    print("All metric files have same observations")
    return obs


# MIROC5.ICAR.hist.1981-2004.ds.DesertSouthwest.metrics.nc
# MIROC5.ICAR.hist.1981-2004.ds.conus.metric.maps.nc
def findDatasets(input_path, matching_str, ds_type):
    datasets = []
    print(ds_type)
    print("ARGS:", input_path, matching_str, ds_type)
    for filename in os.listdir(input_path):
        # not sure how to handle these files yet
        if filename in ['WUS-D3_CMIP6_signal_pr.nc',
                        'WUS-D3_CMIP6_signal_tasmax.nc']:
            print(f"WARNING: skipping {filename}")
            continue
        if fnmatch(filename, matching_str):
            print(filename)
            parts = re.split(r'[._|]', filename)
            print("parts =", parts)
            # sys.exit()
            # filename in format of
            # [cm].[dm].[era].ds.[region].[matching_str]
            climate_model = parts[0]
            downscaling_method = parts[1]
            era = ''
            signal = parts[2]
            metric = ''
            if signal != 'signal':
                'GCM_CMIP6_ssp585_signal_tasmax.nc'
                era = parts[2]
                signal = parts[3]
                metric = parts[4]
            else:
                'BCCA_CMIP5_signal_pr.nc'
                metric = parts[3]
                era = 'historical'
            # sys.exit()
            if era == 'historical':
                file_path = input_path + '/' + \
                    climate_model + '_' + \
                    downscaling_method + '_' + \
                    signal + '_' + \
                    metric + '.nc'
            else:
                file_path = input_path + '/' + \
                    climate_model + '_' + \
                    downscaling_method + '_' + \
                    era + '_' + \
                    signal + '_' + \
                    metric + '.nc'


            if os.path.exists(file_path):
                ds = Dataset(file_path,
                             ds_type,
                             era,
                             'CONUS',
                             method=downscaling_method,
                             model=climate_model,
                             metric=metric)
                datasets.append(ds)
            else:
                print("Warning: file path doesn't exist:", file_path)
                print("Parsing failed, exiting...")
                sys.exit()
    return datasets

def findObsDatasets(input_path, suffix, ds_type):
    datasets = []

    for filename in os.listdir(input_path):
        if filename.endswith(suffix):
            parts = filename.split(".")
            if len(parts) != 6:
                print("Error: can't parse obs file, len(parts) != 6")

            # filename in format of
            # [obs].ds.[region].[suffix]
            obs = parts[0]
            region = parts[2]

            era = '1981_2004'

            # recreate file_path to check parsing
            file_path = input_path + '/' + \
                obs + '.' + \
                'ds.' + \
                region + \
                suffix

            if os.path.exists(file_path):
                ds = Dataset(file_path,
                             ds_type,
                             era,
                             region,
                             obs=obs)
                datasets.append(ds)
            else:
                print("Warning: file path doesn't exist:", file_path)
                print("Parsing failed, exiting...")
                sys.exit()
    return datasets

def findCmipObsDatasets(input_path, suffix, ds_type):
    datasets = []

    for filename in os.listdir(input_path):
        if filename.endswith(suffix):
            parts = filename.split(".")
            if len(parts) != 5:
                print("Error: can't parse obs file, len(parts) != 5")

            # filename in format of
            # [obs].ds.[region].[suffix]
            obs = parts[0]
            region = parts[1]

            era = '1850_2100'

            # recreate file_path to check parsing
            file_path = input_path + '/' + \
                'obs' + '.' + \
                region + \
                suffix

            if os.path.exists(file_path):
                d = xr.open_dataset(file_path)[['obs']]
                for ob in d['obs'].values:
                    ds = Dataset(file_path,
                                 ds_type,
                                 era,
                                 region,
                                 obs=ob)
                    datasets.append(ds)
            else:
                print("Warning: file path doesn't exist:", file_path)
                print("Parsing failed, exiting...")
                sys.exit()
    return datasets


def prep(s):
    return s.lower().replace('-','_')

def writeMetricYamlFromFile(metric_path):
    ds = xr.open_dataset(metric_path)
    print(ds)
    print("--------------------")

    regions = ds['regions'].data
    models = ds['models'].data
    methods = ds['methods'].data
    schemes = ds['normscheme'].data
    metrics = list(ds.data_vars)

    # remove null data
    # ADD BACK IN LATER AS A TEST
    methods = methods[methods != 'ICARwest']
    # ADD BACK AFTER DEV
    # regions = regions[:1]
    # models = models[:2]
    # methods = methods[:2]
    # metrics = metrics[:1]

    metrics_arr = [prep(m) for m in metrics]
    schemes_arr = {prep(s):s for s in schemes}
    for region in regions:
        r = prep(region)
        print("region = ", r)
        model_arr = []
        method_arr = []
        combination_arr = []
        for model, method in itertools.product(models, methods):
            combination_arr.append(method + ' with ' + model)
            model_arr.append(prep(model))
            method_arr.append(prep(method))

        yaml_obj = {
            'num_metrics': len(metrics),
            'num_datasets': len(combination_arr),
            'combinations': combination_arr,
            'combinations_downscaling': method_arr,
            'combinations_model': model_arr,
            'metrics': metrics_arr,
            'schemes': schemes_arr,
            'scores': {s: {m:[] for m in metrics} for s in schemes_arr},
        }
        # print(yaml_obj)
        # sys.exit()
        for scheme in schemes:
            for m in metrics:
                metric_vals = []
                for model, method, in itertools.product(models, methods):
                    val = ds.sel(models=model,
                                 methods=method,
                                 regions=region,
                                 normscheme=scheme)[m].item()
                    if (np.isnan(val)):
                        val = 9999.0
                        # print(val)
                        # sys.exit()
                    metric_vals.append(val)
                yaml_obj['scores'][prep(scheme)][m] = metric_vals

                # print(m, " metric_vals =", metric_vals)
                # print("len(metric_vals) =", len(metric_vals))
                # print("len(combination_arr) =", len(combination_arr))
                # print("metric =", m)
                # print("Scheme =", scheme)
                # sys.exit()
                # yaml_obj.update({
                #     m: metric_vals
                # })
        # print(yaml_obj)
        # sys.exit()
            # mod = ds.model.lower().replace('-','_')
            # ens = ds.ens.lower().replace('-','_')
            # by_model[mod].add(ens)
            # yaml_obj = {
            #     "ensemble": {model: list(ens_set)
            #                  for model, ens_set in by_model.items()}
            # }

            # with open("metrics.yaml", "w") as f:
            #     yaml.dump(yaml_obj, f, sort_keys=False, default_flow_style=True)
        os.makedirs("metrics", exist_ok=True)
        with open("metrics/"+prep(region.replace(' ',''))+"_metrics.js", "w", encoding="utf-8") as f:
            f.write("// auto-generated; do not edit\n")
            f.write("export const metrics_settings = ")
            json.dump(yaml_obj, f, indent=2, ensure_ascii=False, sort_keys=False)
            f.write(";\n")





def writeMetricYaml(metric_datasets, ob):
    by_model = defaultdict(set)
    count = 0

    region = metric_datasets[0].region.lower()

    metrics = ['n34t_r', 'ttrend_r', 't90_r', 't99_r', 'n34pr_r', 'ptrend_r', 'pr90_r', 'pr99_r', 'tpcorr_r', 'ann_snow_r', 'freezethaw_r', 'drought_1yr_r', 'drought_2yr_r', 'drought_5yr_r', 'n34t_rmse', 'ttrend_rmse', 't90_rmse', 't99_rmse', 'n34pr_rmse', 'ptrend_rmse', 'pr90_rmse', 'pr99_rmse', 'tpcorr_rmse', 'freezethaw_rmse', 'drought_1yr_rmse', 'drought_2yr_rmse', 'drought_5yr_rmse', 'ann_snow_std', 'djf_p_r', 'djf_p_std', 'djf_t_r', 'djf_t_std', 'mam_p_r', 'mam_p_std', 'mam_t_r', 'mam_t_std', 'jja_p_r', 'jja_p_std', 'jja_t_r', 'jja_t_std', 'son_p_r', 'son_p_std', 'son_t_r', 'son_t_std', 'ann_p_r', 'ann_p_std', 'ann_t_r', 'ann_t_std']
    # metrics = ['son_t_r', 'son_t_std']
    # print("add more metrics, only:", metrics)
    ob_name = ob.data


    model_arr = []
    method_arr = []
    combination_arr = []
    for d in metric_datasets:
        combination_arr.append(d.method + ' with ' + d.model)
        model = d.model.lower().replace('-','_')
        method = d.method.lower().replace('-','_')
        model_arr.append(model)
        method_arr.append(method)

    ds = xr.open_dataset(metric_datasets[0].file_path)
    yaml_obj = {
        'num_metrics': len(metrics),
        'num_datasets': len(metric_datasets),
        'combinations': combination_arr,
        'combinations_downscaling': method_arr,
        'combinations_model': model_arr,
                }

    for m in metrics:
        metric_vals = []
        for d in metric_datasets:
            ds = xr.open_dataset(d.file_path)
            metric_vals.append(ds.sel(obs=ob_name)[m].item())
            # print(ds.sel(obs=ob_name)[m].item())

        # print("metric_vals =", metric_vals)
        # print("metric =", m)

        yaml_obj.update({
            m: metric_vals
        })
            # mod = ds.model.lower().replace('-','_')
            # ens = ds.ens.lower().replace('-','_')
            # by_model[mod].add(ens)
            # yaml_obj = {
            #     "ensemble": {model: list(ens_set)
            #                  for model, ens_set in by_model.items()}
            # }

    # print("yaml_obj", yaml_obj)

    # with open("metrics.yaml", "w") as f:
    #     yaml.dump(yaml_obj, f, sort_keys=False, default_flow_style=True)
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/"+region+"_metrics.js", "w", encoding="utf-8") as f:
        f.write("// auto-generated; do not edit\n")
        f.write("export const metrics_settings = ")
        json.dump(yaml_obj, f, indent=2, ensure_ascii=False, sort_keys=False)
        f.write(";\n")


def writeEnsembleYaml(maps_datasets):
    by_model = defaultdict(set)
    count = 0
    for ds in maps_datasets:
        # count += 1
        # if count > 5:
        #     continue
        mod = ds.model.lower().replace('-','_')
        ens = ds.ens.lower().replace('-','_')
        by_model[mod].add(ens)
        yaml_obj = {
            "ensemble": {model: list(ens_set)
                         for model, ens_set in by_model.items()}
        }

    # sort yaml object
    yaml_obj['ensemble'] = {
        model: sorted(members)
        for model, members in sorted(yaml_obj['ensemble'].items())
    }

    with open("ensemble.yaml", "w") as f:
        yaml.dump(yaml_obj, f, sort_keys=False, default_flow_style=True)

def writeModelYaml(maps_datasets, yaml_file, top_var):
    by_model_set = defaultdict(set)
    by_model_dict = defaultdict(dict)
    ensemble = False


    # for ds in maps_datasets:
    #     mod = ds.model.lower().replace('-','_')
    #     model_key = ds.model.lower().replace('-', '_')
    #     model_label = ds.model
    #     print(mod)
    #     print(ds.era)
    #     print(ds.method)
    #     print(model_key)
    #     print(model_label)
    #     sys.exit()
    #     by_model_dict[ds.method][model_key] = model_label

    # default_flow_style = True
    # yaml_obj = {
    #     top_var: {
    #         method.lower().replace('-','_'):
    #         {k: Quoted(v) for k, v in sorted(models.items())}
    #             for method, models in by_model_dict.items()
    #     }
    # }


    by_era_dict = defaultdict(dict)

    for ds in maps_datasets:
        model_key = ds.model.lower().replace("-", "_")
        model_label = ds.model
        era_key = ds.era  # keep as-is (string/number). If needed: str(ds.era)

        print(model_key, model_label, era_key)
        sys.exit()
    #     if   in 'LOCA2_CMIP6_ssp585':

    # elif 'GCM_CMIP6_ssp585':

    # elif 'BCSD_CMIP5_signal':

    # elif 'LOCA_CMIP5_signal':

    # elif 'LOCA2_CMIP6_ssp370':

    # elif 'GCM_CMIP6_ssp370':

    # elif 'DEEPSD_CMIP6_ssp585':

    # elif 'NEXGDDP_CMIP5_signal':

    # elif 'GARDLENS_CMIP6_ssp370':



        by_era_dict[era_key][model_key] = model_label

        default_flow_style = True  # keep True if you want { ... } flow-style YAML

        yaml_obj = {
            top_var: {
                era: {k: Quoted(v) for k, v in sorted(models.items())}
                for era, models in sorted(by_era_dict.items(), key=lambda kv: str(kv[0]))
            }
        }


    with open(yaml_file, "w") as f:
        # yaml.dump(yaml_obj, f, default_flow_style=default_flow_style)#, Dumper=PlainKeyDumper,)
        yaml.dump(yaml_obj, f,          allow_unicode=True,       Dumper=yaml.SafeDumper,  sort_keys=False, default_flow_style=default_flow_style)
        print("wrote to", yaml_file)


def writeObsYaml(obs_datasets):
    by_ob = defaultdict(set)
    for ds in obs_datasets:
        ds.print()
        region = ds.region.lower().replace('-','_')
        ob = ds.obs.lower().replace('-','_')
        yaml_obj = {
            "obs_lev1": {ob: ob}
        }

    # sort yaml object
    yaml_obj['obs_lev1'] = {
        k: v
        for k, v in sorted(yaml_obj["obs_lev1"].items())
    }

    with open("obs.yaml", "w") as f:
        yaml.dump(yaml_obj, f, sort_keys=False, default_flow_style=True)



def checkCLA(args):
    if (args.write_yaml):
        return
    if (args.output_maps_path == None and
        args.metric_score_path == None and
        args.climate_signal_path == None and
        args.output_obs_path == None and
        args.timeseries_path == None):
        print("ERROR: maps, metric-score, time-series or climate-signal options are required")
        print(args)
        sys.exit(1)


def parseArgs():
    # define argument parser
    parser = argparse.ArgumentParser(
        description="Create Zarr files for ICAR Maps.")
    parser.add_argument("input_maps_path", help="Path to input map files")
    parser.add_argument("input_obs_path", help="Path to input observations")
    parser.add_argument("input_metrics_path", help="Path to input metrics")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose mode")
    group = parser.add_argument_group('Output Data',
                                      'types of output and path to write to')
    group.add_argument("--climate-signal", nargs=1, dest="climate_signal_path",
                       help="Write climate signal data to passed path")
    group.add_argument("--maps", nargs=1, dest="output_maps_path",
                       help="Write modeled data to passed path")
    group.add_argument("--metric-score", nargs=1, dest="metric_score_path",
                       help="Write (model_data - obs_data) dataset")
    group.add_argument("--obs", nargs=1, dest="output_obs_path",
                       help="Write observation dataset to passed path")
    group.add_argument("--time-series", nargs=1,
                       dest="timeseries_path",
                       help="Write time series dataset to passed path")
    parser.add_argument("--write-yaml", action="store_true", dest="write_yaml",
                        help="Write Yaml Files")
    parser.add_argument("--test", action="store_true",
                        help="Test inputs")

    return parser.parse_args()


# parse command line arguments
def parseCLA():
    args = parseArgs()
    checkCLA(args)
    options = Options(args)
    return options

def main():
    print("--- Starting Zarr Data Maps Setup ---")
    library_check()

    # parse command line arguments
    options = parseCLA()
    print("post options")
    options.print()
    # sys.exit()
    # organize this better in the future
    # process_obs()

    print('---')
    metric_score_datasets = []
    climate_signal_datasets = []
    obs_datasets = []

    if options.write_yaml:
        maps_datasets = findDatasets(options.input_maps_path,
                                     maps_matching_str,
                                     'map')
        print("NEED TO FIX THIS")
        sys.exit()
        if (maps_datasets[-1].method in ['cmip5', 'cmip6']):
            maps_datasets = addEnsembleDatasets(maps_datasets)
            writeEnsembleYaml(maps_datasets)
        writeModelYaml(maps_datasets, 'model.yaml', 'model')

        climate_signal_datasets = \
            handleClimateSignalArgs(options.input_maps_path)
        writeModelYaml(climate_signal_datasets, 'climateSignal.yaml',
                       'model_climatesignal')

        sys.exit('--- finished write yaml option ---')


    if options.write_maps:
        # maps_datasets = findDatasets(options.input_maps_path,
        #                              maps_matching_str,
        #                              'map')
        metrics = ['pr', 'tasmax']
        for metric in metrics:
            # maps_matching_f = f'signal_{metric}_snr.nc'
            maps_matching_f = f'signal_{metric}_agreement.nc'

            ds = Dataset(options.input_maps_path + '/' + maps_matching_f,
                     'map',
                     '1950-2100',
                     'CONUS',
                     method='',
                     model='',
                     metric=metric)
            maps_datasets = [ds]

            for dataset in maps_datasets:
                print(dataset.file_path)

            # writeModelYaml(maps_datasets, 'maps.yaml',
            #                'model_maps')

            for dataset in maps_datasets:
                writeDatasetToZarr(options.output_maps_path, dataset,
                                   write_maps = True)
        sys.exit('--- finished writing maps ---')

    print('---fin---')
    sys.exit()

# library check
def library_check():
    # if ndp.__version__ != '0.1.0':
    #     print(f"Error: ndpyramid version {ndp.__version__} != required 0.1.0")
    #     sys.exit(0)
    return

def convert_to_zarr_format(ds):
    # --- create the pyramid
    print("Convert to Zarr Format")
    print("-- Create Pyramid")
    # EPSG:4326 fixes error:
    #    MissingCRS: CRS not found. Please set the CRS with 'rio.write_crs()'
    print("ds3 =", ds)

    # write CRS data in Climate and Forecast (CF) Metadata Convention style
    ds.rio.write_crs('EPSG:4326', inplace=True)

    fillValue = 3.4028234663852886e38 # end result red everywhere, inf in np array??
    # fillValue = 9.969209968386869e36 # end result is? NORMAL??
    ds = ds.fillna(fillValue)

    # Pad the dataset with zeros to prevent unwanted interpolation artifacts
    pad_width = 3  # Adjust padding as needed
    # print("--PADDED--")
    # ds_padded = ds.pad(x=pad_width, y=pad_width, constant_values=0)
    # print(ds_padded)
    pixels = 512
    pixels = 13875 / 4
    pixels = 3469
    pixels = 2880 # for equidistant-cylindrical
    pixels = 1440 # test
    pixels = 720 # test
    pixels = 360 # test, this reaches 2880 for level 4

    dt = ndp.pyramid_reproject(ds,
    # dt = ndp.pyramid_reproject(ds_padded,
                               levels=LEVELS,
                               pixels_per_tile=pixels,
                               projection='equidistant-cylindrical',
                               # pixels_per_tile=PIXELS_PER_TILE,
                               extra_dim='band')
                               # levels=1, # THIS DIDN'D DO MUCH AT ALL
    print("Done Creating Pyramid")
    return dt


if __name__ == "__main__":
    main()
