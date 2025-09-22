import argparse
from collections import defaultdict
import glob
import json
import os
import pandas as pd
import ndpyramid as ndp
import numpy as np
import rioxarray
import sys
import xarray as xr
import xesmf as xe
import yaml
import zarr
from tools import dimensionNames, handleArgs

print("ndpyramid Version = ", ndp.__version__)

# class NoQuotesDumper(yaml.SafeDumper):
#     def represent_str(self, data):
#         # Always use plain scalars, no quotes
#         return self.represent_scalar("tag:yaml.org,2002:str", data, style='')
# NoQuotesDumper.add_representer(str, NoQuotesDumper.represent_str)

# class PlainKeyDumper(yaml.SafeDumper):
#     pass
# def str_representer(dumper, data):
#     # Only force plain style for dict keys
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

# Define the global grid (covering the entire world)
# global_lon = np.linspace(-180, 180, int(360/0.125 + 1))  # 0.125-degree resolution
# global_lat = np.linspace(-90, 90, int(180/0.125 + 1))    # 0.125-degree resolution

# world_grid = xr.Dataset({
#     'lat': (['lat'], global_lat),
#     'lon': (['lon'], global_lon)
# })
test = False
downscaling_methods = [
    'ICAR',
    'ICARwest',
    'GARD_r2',
    'GARD_r3',
    'LOCA_8th',
    'MACA',
    'NASA-NEX',
    ]
    # 'GARDwest',
climate_models = [
    'ACCESS1-3',
    'CanESM2',
    'CCSM4',
    'MIROC5',
    'MRI-CGCM3',
    'NorESM1-M',
#    # 'modelmean',
#    # ADD MRI-CGCM3, NorESM1 but are they there??
    ]

observation_datasets = []
observation_datasets_test = ['global', 'Midwest', 'Northeast',
                             'NorthernGreatPlains', 'Northwest', 'Southeast',
                             'SouthernGreatPlains', 'Southwest']


debug=False
if (debug):
    downscaling_methods = ['ICAR']
    climate_models = ['ACCESS1-3']
    observation_datasets = ['global']

# set these somewhere else?
VERSION = 2
time = 'time'

PAST=0
FUTURE=1
CLIMATE_SIGNAL=2


class Dataset:
    def __init__(self, file_path, ds_type, era, region,
                 method=None, model=None, obs=None, ens=None,
                 dif_file_path=None):
        if not os.path.exists(file_path):
            print("ERROR: file path does not exist:", file_path)
            sys.exit()
        self.obs = obs
        self.method = method
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
        print("  obs =", self.obs)
        print("  ens =", self.ens)
        print("  dif_file_path =", self.dif_file_path)



class Options:
    def __init__(self, args):
        # default values
        self.input_maps_path = args.input_maps_path
        self.input_obs_path = args.input_obs_path
        self.input_metrics_path = args.input_metrics_path
        self.write_maps = False
        self.output_maps_path = None
        self.write_metric_score = False
        self.metric_score_path = None
        self.write_climate_signal = False
        self.climate_signal_path = None
        self.write_obs = False
        self.output_obs_path = None
        self.write_yaml = False
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


def writeDatasetToZarr(output_path, dataset=None,
                       write_maps=False,
                       write_climate_signal=False,
                       write_metric_score=False,
                       write_obs=False):
    method = dataset.method
    model = dataset.model

    print("Opening file(s):")
    # elif (write_metric_score):
    #     print("past:", dataset.past_path)
    #     ds_past = xr.open_dataset(dataset.past_path)
    #     print("obs:", dataset.obs)
    #     ds_obs = xr.open_dataset(dataset.obs)
    #     ds = ds_obs
    #     for var in ds_obs.data_vars:
    #         ds[var] = abs(ds_past[var] - ds_obs[var])
    if (write_maps):
        print(f'writing map {dataset.file_path} to dir {output_path}')
    elif (write_metric_score):
        print(f'writing metric {dataset.file_path} to dir {output_path}')
    elif (write_obs):
        print(f'writing obs {dataset.file_path} to dir {output_path}')
    elif (write_climate_signal):
        print("past:", dataset.file_path)
        ds_past = xr.open_dataset(dataset.file_path)
        print("future:", dataset.dif_file_path)
        ds_future = xr.open_dataset(dataset.dif_file_path)
        ds = ds_future - ds_past
    else:
        print('Bad write choice')
        sys.exit()

    ens_path = ''
    if (dataset.ens != None):
        ds = xr.open_dataset(dataset.file_path).sel(ens=dataset.ens)
        ds = ds.drop_vars('ens')
        # print(ds)
        # sys.exit()
        ens_path = '/' + dataset.ens + '/'
    elif (dataset.obs != None):
        ds = xr.open_dataset(dataset.file_path).sel(obs=dataset.obs)
        ds = ds.drop_vars('obs')
        # print(ds)
        # sys.exit()
    else:
        ds = xr.open_dataset(dataset.file_path)

    if "obs" in ds.coords:
        ds = ds.drop_vars('obs')

    # variables for zarr creation, value has to be four characters
    new_vars = {#'time': 'time',
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
        'ann_t':'annt',
        'ann_p':'annp',
        'ann_snow':'anns',
        'freezethaw':'fzth',
        'tpcorr':'tpco',
        'drought_1yr':'d1yr',
        'drought_2yr':'d2yr',
        'drought_5yr':'d5yr',
    }

    # lowercase dimension names
    lowercase_vars = {v: v.lower() for v in ds.data_vars}
    ds = ds.rename(lowercase_vars)

    # renames - to _
    rename_vars = {v: v.replace("-", "_") for v in ds.data_vars if "-" in v}
    ds = ds.rename(rename_vars)

    valid_rename_map = {k: v for k, v in new_vars.items()
                        if k in ds.variables or k in ds.dims}
    ds = ds.rename(valid_rename_map)

    print("---")
    print("ds")
    print(ds)
    print("---")

    # zarr format requires variables four characters in length
    fixed_length = 4
    invalid_vars = [var for var in ds.data_vars if len(var) != fixed_length]
    if invalid_vars:
        print("Variables with names not 4 characters long:", invalid_vars)
        sys.exit()

    # put in format for zarr
    variables = list(ds.variables.keys())
    variables = [var for var in variables if var not in ['x', 'y']]
    concatenated_vars = []
    for var_name in variables:
        concatenated_vars.append(ds[var_name])
    print(ds)
    ds['climate'] = xr.concat(concatenated_vars, dim='band')
    var_names_U4 = [s[:fixed_length].ljust(fixed_length) for s in variables]
    ds = ds.assign_coords(band=var_names_U4)
    ds = ds.drop_vars(set(ds.data_vars) - set(['climate']))
    # --- clean up types
    # print(" - clean up types")
    # month to int type
    # ds['month'] = xr.Variable(dims=('month',),
    #                    data=list(range(1, 12 + 1)))
    #                    # data=list(range(1, ds.month.shape[0] + 1)))
    #                    # attrs={'dtype': 'int32'})
    # ds["month"] = ds["month"].astype("int32")
    ds["climate"] = ds["climate"].astype("float32")
    ds["band"] = ds["band"].astype("str")
    ds.attrs.clear()
    dz = convert_to_zarr_format(ds) # already in single precision

    # setup write_path
    if method != None:
        method_s = method.lower().replace('-','_')
    if model != None:
        model_s = model.lower().replace('-','_')
    if (write_climate_signal):
        write_path = output_path + method_s + '/' + model_s + '/' + dataset.era.lower().replace('-','_')
    if (write_metric_score):
        write_path = output_path + method_s + '/' + \
            model_s + '/' + dataset.era.lower().replace('-','_')
        # MORE COMPLETE ONE, OLD WAY FOR NOW
        # write_path = output_path + dataset.region + '/' + method_s + '/' + \
        #     model_s + '/' + dataset.era
    elif (write_maps):
        write_path = output_path + method_s + '/' + \
            model_s + '/' + dataset.era.lower().replace('-','_') + \
            ens_path
        # write_path = output_path + dataset.region + '/' + method_s + '/' + \
        #     model_s + '/' + dataset.era
    elif (write_obs):
        # write_obs_to_zarr(ds, ob.lower().replace('-','_'))
        # ob_filename = os.path.basename(dataset.obs).split(".ds")[0]
        # hist.[year]-[year]
        write_path = output_path + \
            dataset.region.lower() + '/' + \
            dataset.obs.lower().replace('-','_') + '/' + \
            'hist.' + \
            dataset.era + \
            ens_path
        print("Write ob to Zarr file", write_path)

    write_to_zarr(dz, write_path)
    # print("small fin", 'write_path=', write_path)
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
def findDatasets(input_path, suffix, ds_type):
    datasets = []
    # print("ARGS:", input_path, suffix, ds_type)
    for filename in os.listdir(input_path):
        if filename.endswith(suffix):
            parts = filename.split(".")
            if len(parts) not in [6,8,9]:
                print("Error: can't parse maps file, len(parts) not in [6,8,9]")
                print(parts)
                print(f"len(parts) = {len(parts)}")
                sys.exit()
            if len(parts) == 9:
                # filename in format of
                # [cm].[dm].[era].ds.[region].[suffix]
                climate_model = parts[0]
                downscaling_method = parts[1]
                era = parts[2] + '.' + parts[3]
                region = parts[5]
                file_path = input_path + '/' + \
                    climate_model + '.' + \
                    downscaling_method + '.' + \
                    era + '.' + \
                    'ds.' + \
                    region + \
                    suffix
            elif len(parts) == 8:
                # double check metrics
                metrics = parts[6]
                if (metrics != 'metrics'):
                    sys.exit(f"Error: {filename} doesn't have .metrics.")
                # print("parts", parts)
                # print("continue coding here")
                climate_model = parts[0]
                downscaling_method = parts[1]
                era = parts[2] + '.' + parts[3]
                region = parts[5]
                file_path = input_path + '/' + \
                    climate_model + '.' + \
                    downscaling_method + '.' + \
                    era + '.' + \
                    'ds.' + \
                    region + \
                    '.metrics.nc'
            elif len(parts) == 6:
                # filename in format of
                # [cm].[dm].[region].[suffix]
                climate_model = parts[0]
                downscaling_method = parts[1] # actually CMIP
                region = parts[2]
                if downscaling_method == 'cmip5':
                    era = '1850_2005'
                elif downscaling_method == 'cmip6':
                    era = '1850_2100'
                else:
                    print("Error: not cmip5 or cmip6")
                    sys.exit()
                file_path = input_path + '/' + \
                    climate_model + '.' + \
                    downscaling_method + '.' + \
                    region + \
                    suffix

            if os.path.exists(file_path):
                ds = Dataset(file_path,
                             ds_type,
                             era,
                             region,
                             method=downscaling_method,
                             model=climate_model)
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



def handleClimateSignalArgs(input_path):
    rcps = ['rcp45.2076-2099', 'rcp85.2076-2099']
    datasets = []
    era = 'hist.1981-2004'

    for cm in climate_models:
        for dm in downscaling_methods:
            past_path = input_path+'/'+cm+'.'+dm+'.'+era+ \
                '.ds.conus.metric.maps.nc'
            for rcp in rcps:
                future_path = \
                    input_path+'/'+cm+'.'+dm+\
                    '.'+rcp+'.ds.conus.metric.maps.nc'
                # if ('GARD' in dm):
                #     print(cm,"and",dm)
                #     print("past_path :", past_path)
                #     print("future_path :", future_path)

                if os.path.exists(past_path) and os.path.exists(future_path):
                    # if ('GARD' in dm):
                    #     print('exists for', rcp)
                    datasets.append(Dataset(past_path, 'maps',
                                            era=rcp,
                                            region='conus',
                                            method=dm,
                                            model=cm,
                                            dif_file_path=future_path))
                # else:
                #     print("paths not found:", past_path, "or", future_path)
    # print('handleClimateSignalArgs fin')
    # sys.exit()
    return datasets


# def findObservationDatasets(obs_path, suffix):
#     datasets = []
#     files = glob.glob(obs_path+"/obs.*"+suffix)
#     for f in files:
#         base = os.path.basename(f)   # e.g., obs.ABCD.metrics.nc
#         # Extract the dataset name between 'obs.' and '.metrics.nc'
#         if base.startswith("obs.") and base.endswith(suffix):
#             dataset_name = base[len("obs."):-len(suffix)]
#             datasets.append(dataset_name)
#     global test
#     if (test):
#         check_arrays(observation_datasets_test, datasets,
#                      'observational datasets')
#     return datasets

# def handleObsArgs(input_obs_path, suffix):
#     datasets = []
#     obs_datasets = findObservationDatasets(input_obs_path, suffix)
#     for ob in obs_datasets:
#         obs_path = input_obs_path+'/obs.'+ob+'.metric.maps.nc'
#         if os.path.exists(obs_path):
#                 datasets.append(Dataset(obs=obs_path))

#     return datasets

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

def writeModelYaml(maps_datasets):
    by_model_set = defaultdict(set)
    by_model_dict = defaultdict(dict)
    ensemble = False
    for ds in maps_datasets:
        mod = ds.model.lower().replace('-','_')
        if (ds.ens != None):
            ens = ds.ens.lower().replace('-','_')
            by_model_set[mod].add(ens)
            yaml_obj = {
                "model": {model: model
                          for model, ens_set in sorted(by_model_set.items())}
            }
            ensemble = True
        else:
            model_key = ds.model.lower().replace('-', '_')
            model_label = ds.model
            by_model_dict[ds.method][model_key] = model_label

    default_flow_style = True
    if ensemble:
        yaml_obj['model'] = {
            k: v
            for k, v in sorted(yaml_obj["model"].items())
        }
    else:
        # default_flow_style = False
        yaml_obj = {
            "model": {
                method.lower().replace('_','_'):
                {k: Quoted(v) for k, v in sorted(models.items())}
                for method, models in by_model_dict.items()
            }
        }

    with open("model.yaml", "w") as f:
        # yaml.dump(yaml_obj, f, default_flow_style=default_flow_style)#, Dumper=PlainKeyDumper,)
        yaml.dump(yaml_obj, f,          allow_unicode=True,       Dumper=yaml.SafeDumper,  sort_keys=False, default_flow_style=default_flow_style)


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
        args.output_obs_path == None):
        print("ERROR: maps, metric-score, or climate-signal options are required")
        print(parser.print_help())
        sys.exit(1)


def parseArgs():
    # define argument parser
    parser = argparse.ArgumentParser(description="Create Zarr files for ICAR Maps.")
    parser.add_argument("input_maps_path", help="Path to input map files")
    parser.add_argument("input_obs_path", help="Path to input observations")
    parser.add_argument("input_metrics_path", help="Path to input metrics")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose mode")
    group = parser.add_argument_group('Output Data', 'types of output and path to write to')
    group.add_argument("--maps", nargs=1, dest="output_maps_path",
                       help="Write modeled data to passed path")
    group.add_argument("--metric-score", nargs=1, dest="metric_score_path",
                       help="Write (model_data - obs_data) dataset")
    group.add_argument("--climate-signal", nargs=1, dest="climate_signal_path",
                       help="Write climate signal data to passed path")
    group.add_argument("--obs", nargs=1, dest="output_obs_path",
                       help="Write observation dataset to passed path")
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
                                     '.metric.maps.nc',
                                     'map')
        if (maps_datasets[-1].method in ['cmip5', 'cmip6']):
            maps_datasets = addEnsembleDatasets(maps_datasets)
            writeEnsembleYaml(maps_datasets)
        writeModelYaml(maps_datasets)


    if options.write_maps:
        maps_datasets = findDatasets(options.input_maps_path,
                                     '.metric.maps.nc',
                                     'map')

        # ensemble of datasets that need to be added
        if (maps_datasets[-1].method in ['cmip5', 'cmip6']):
            maps_datasets = addEnsembleDatasets(maps_datasets)
            writeEnsembleYaml(maps_datasets)
        writeModelYaml(maps_datasets)

        for dataset in maps_datasets:
            writeDatasetToZarr(options.output_maps_path, dataset,
                               write_maps = True)

    if options.write_obs:
        obs_datasets = findCmipObsDatasets(options.input_obs_path,
                                    '.metric.maps.nc',
                                    'map')

        # obs_datasets = findObsDatasets(options.input_obs_path,
        #                             '.metric.maps.nc',
        #                             'map')
        for dataset in obs_datasets:
            # if dataset.region != 'global':
            #     continue
            # if dataset.obs != 'UDel':
            #     continue
            dataset.print()
            writeDatasetToZarr(options.output_obs_path, dataset,
                               write_obs = True)
        # writeObsYaml(obs_datasets) # FIX THIS
        sys.exit()
        # todo: obs_metrics
        # obs_metrics = findDatasets(options.input_obs_path,
        #                             '.metric.maps.nc',
                                    # 'metric')

    if options.write_climate_signal: # todo
        climate_signal_datasets = \
            handleClimateSignalArgs(options.input_maps_path)
        for dataset in climate_signal_datasets:
            dataset.print()
            writeDatasetToZarr(options.climate_signal_path, dataset,
                               write_climate_signal = True)
        sys.exit('----debugging metric fin-----')


    # TODO
    # --- TO REFACTOR ---
    if options.write_metric_score:
        print("refactoring write metric score")

        regions = ['DesertSouthwest',  'GreatLakes',  'GulfCoast',
                   'MidAtlantic',  'MountainWest',  'NorthAtlantic',
                   'NorthernPlains',  'PacificNorthwest',  'PacificSouthwest']
        # regions = ['DesertSouthwest',  'GreatLakes']
        # print('add back regions')
        for r in regions:
            print("Writing to region", r)
            r_path = options.input_metrics_path+'/'+r
            metric_score_datasets = findDatasets(r_path,
                                                 '.metrics.nc',
                                                 'metric')
            obs = checkMetricObsEquality(metric_score_datasets)
            metric_vars = checkMetricVarsEquality(metric_score_datasets)
            ob = obs[1]
            writeMetricYaml(metric_score_datasets, ob)

        sys.exit('----debugging metric fin-----')

        # for dataset in metric_score_datasets:
                # dataset.print()
    #     writeDatasetToZarr(options.metric_score_path, dataset,
    #                        write_metric_score = True)


    # if options.write_obs: # todo
    #     obs_maps_datasets = handleObsArgs(options.input_obs_path,
    #                                       '.metric.maps.nc')
    #     obs_metrics_datasets = handleObsArgs(options.input_obs_path,
    #                                          '.metrics.nc')

    sys.exit('----debugging fin-----')

    # # --- process datasets to write to zarr

    # for dataset in metric_score_datasets:
    #     count+=1
    #     dataset.print()
    #     sys.exit()
    #     # question is how to order the datasets now?
    #     writeDatasetToZarr(options.metric_score_path, dataset,
    #                        write_metric_score = True)
    #     if (count > max_count):
    #         print(max_count, "max count reached")
    #         break

    # print('---early fin---')
    # sys.exit()

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
