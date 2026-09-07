.PHONY:dl metrics yaml agreement_signal_zarr tar_agreement_signal_zarr tar_snr tar_agreement scp_agreement_signal_zarr
lc=python3
data_script=create_global_zarr.py
# data_script=create_icar_zarr.py

all: agreement_signal_zarr

host:
	$(lc) host_server.py

build: maps

help:
	python $(data_script) --help
ml:
	@echo "$ ml conda; conda activate zarr-foo"

# download and extract data from hydro.rap.ucar.edu/hydro-climate-eval/data
dl: download
download:
	make -C data/map download

untar:
	make -C data/map/ untar

# --- create maps from NetCDF files ---
# paths to files for data creation
maps_path=/glade/work/nlybarger/downscaling_metrics/cmip5/
obs_path=/glade/work/nlybarger/downscaling_metrics/obs/
metric_path=/glade/work/nlybarger/downscaling_metrics/cmip5/Normalized_Error_Metrics_CMIP5.nc

output_path=data/
output_path=new_metrics/

# new Nick data
# dirs
# /glade/campaign/ral/hap/nlybarger/ESM_eval_postproc/cmip5/postproc/
# /glade/campaign/ral/hap/nlybarger/ESM_eval_postproc
# /glade/u/home/nlybarger/scripts/esmeval/nca_regional_analysis
# /glade/u/home/nlybarger/scripts/python_funcs
# ESM evaluation code is located at ./nicks_nca_regional_analysis
#  template_cmip_metrics_batch.pbs, template_cmip_metrics.py, gen_cmip_*_metrics.bash

# # new Nick Data, running 2026.02.16
# output_path=nick_new_data/
# # new Nick Data, running 2026.02.16
output_path=nick_new_data/
# saved in cp -r nick_new_data/maps/cmip5/ nick_new_data/maps_test_rcp45 and 85!
# maps_path=/glade/work/soren/src/icar/data/zarr-data-maps/nca_regional_analysis/global_future_metric_template_cmip5_rcp45/output/cmip5_metrics/global
# maps_path=/glade/work/soren/src/icar/data/zarr-data-maps/nca_regional_analysis/global_future_metric_template_cmip5_rcp85/output/cmip5_metrics/global
# maps_path=/glade/work/soren/src/icar/data/zarr-data-maps/nca_regional_analysis/global_future_metric_template_cmip6_ssp245/output/cmip6_metrics/global/
# maps_path=/glade/work/soren/src/icar/data/zarr-data-maps/nca_regional_analysis/global_future_metric_template_cmip6_ssp370/output/cmip6_metrics/global/
maps_path=/glade/work/soren/src/icar/data/zarr-data-maps/nca_regional_analysis/global_future_metric_template_cmip6_ssp585/output/cmip6_metrics/global/
obs_path=empty/
metric_path=empty/

# --- Sam's signal-to-noise / agreement maps (Hartke et al. 2025, fig 2 row e) ---
# Two passes, one per band. Between them, flip maps_matching_f in
# create_agreement_signal_zarr.py (~line 1074) between '_snr.nc' and '_agreement.nc' and
# switch which output_path below is last. Do NOT point both passes at the same
# output dir: write_to_zarr() will fail on the pre-existing store.
agreement_input_path=/glade/derecho/scratch/soren/src/icar/sams_stat_files/agreement
snr_output_path=output_snr
agreement_output_path=output_agreement
output_path = ${snr_output_path}
output_path = ${agreement_output_path}

agreement_signal_zarr:
	# rm -rf ${output_path}/*
	python3 create_agreement_signal_zarr.py \
	$(agreement_input_path) \
	'' \
	'' \
	--maps ${output_path}/maps

tar_agreement_signal_zarr: tar_snr tar_agreement
tar_snr:
	tar zcf signal_to_noise.tar.gz -C $(snr_output_path) \
	--transform 's,^maps,signalToNoise/map,' maps
tar_agreement:
	tar zcf agreement.tar.gz -C $(agreement_output_path) \
	--transform 's,^maps,agreement/map,' maps
scp_agreement_signal_zarr:
	scp signal_to_noise.tar.gz agreement.tar.gz soren@hydro-c1-web.rap.ucar.edu:/d1/www/html/hydro-climate-eval/data/refactor/agreement/

maps:
	# rm -rf ${output_path}/maps/*
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
	--metric-score ${output_path}/metrics
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
	empty/ empty/ empty/ \
	--write-yaml


eval: maps_path=/glade/work/nlybarger/downscaling_metrics/cmip6/
eval: obs_path=/glade/work/nlybarger/downscaling_metrics/obs/
eval: metric_path=/glade/work/nlybarger/downscaling_metrics/cmip5/Normalized_Error_Metrics_CMIP5.nc
eval: # create datasets for the hydro-climate-eval website
	python3 create_icar_zarr.py \
	$(maps_path) \
	$(obs_path) \
	$(metric_path) \
	--maps new-hydro-climate/maps

# ls /glade/work/nlybarger/downscaling_metrics/cmip6/*STAR*r1i1p1f1*
# ls /glade/work/nlybarger/downscaling_metrics/cmip6/*LOCA*r1i1p1f1*


tarmaps:
	tar zcf maps.tar.gz $(output_path)/maps
tarmetrics:
	tar zcf metrics.tar.gz $(output_path)/metrics
tarsignal:
	tar zcf signal.tar.gz $(output_path)/signal
tarobs:
	tar zcf obs.tar.gz $(output_path)/obs

scpmaps:
	scp maps.tar.gz soren@hydro-c1-web.rap.ucar.edu:/d1/www/html/hydro-climate-eval/data/refactor_metrics/
scpmetrics:
	scp metrics.20260312.tar.gz soren@hydro-c1-web.rap.ucar.edu:/d1/www/html/hydro-climate-eval/metrics/
scpsignal:
	scp  signal_to_noise.tar.gz soren@hydro-c1-web.rap.ucar.edu:/d1/www/html/hydro-climate-eval/data/refactor_metrics/
scpagreement:
	scp  agreement.tar.gz soren@hydro-c1-web.rap.ucar.edu:/d1/www/html/hydro-climate-eval/data/refactor_metrics/
scpobs:
	scp obs.tar.gz soren@hydro-c1-web.rap.ucar.edu:/d1/www/html/hydro-climate-eval/data/refactor_metrics/
scpnick:
	scp nicks_new_data.tar.gz soren@hydro-c1-web.rap.ucar.edu:/d1/www/html/hydro-climate-eval/data/global/

clean:
	rm -f *~
cleandata:
	rm -rf data/output
