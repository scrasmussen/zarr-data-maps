.PHONY:dl
lc=python3
data_script=create_global_zarr.py

all: host

host:
	$(lc) host_server.py

build: build_zarr_data

help:
	python $(data_script) --help

# download and extract data from hydro.rap.ucar.edu/hydro-climate-eval/data
dl: download
download:
	make -C data/map download

untar:
	make -C data/map/ untar

# --- create maps from NetCDF files ---
# paths to files for data creation
maps_path=data/input/maps
obs_path=data/input/obs
metric_path=data/input/metrics
output_path=data/output

maps:
        rm -rf ${output_path}/maps/*
        python3 $(data_script) \
        ${maps_path} \
        ${obs_path} \
        ${metric_path} \
        --maps ${output_path}/maps
metrics:
        python3 $(data_script) \
        ${maps_path} \
        ${obs_path} \
        ${metric_path} \
        --metric-score ${output_path}/metric
climatesignal:
        rm -rf ${output_path}/climateSignal/*
        python3 $(data_script) \
        ${maps_path}/ \
        ${obs_path} \
        ${metric_path} \
        --climate-signal ${output_path}/climateSignal/
obs:
        rm -rf ${output_path}/obs/*
        python3 $(data_script) \
        ${maps_path} \
        ${obs_path} \
        ${metric_path} \
        --obs ${output_path}/obs
yaml:
        python3 $(data_script) \
        ${maps_path} \
        ${obs_path} \
        ${metric_path} \
        --write-yaml

clean:
	rm -f *~
cleandata:
	rm -rf data/output
