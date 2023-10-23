# ICAR Zarr Data
Create Zarr data files in a format that can be access by [carbonplan/maps](https://github.com/carbonplan/maps) websites.

## Requirements
Setup and activate conda environment.
`$ conda install --file requirements.txt`

## Create Data
Setup paths in `create_icar_zarr_.py`. Will become command line arguments in the future.
`$ python3 create_icar_zarr.py`

### Notes on Data Creation
* Data variables must be dimensions of a four dimensional `climate` variable.
The `climate` variables dimensions will be `x`, `y`, `band`, `month`.
The `month` requirement can be updated in the Maps site's `components/parameter-controls.js` file.
* Note variable names must be of type `U4`, Python strings of length 4.


## Hosting Data for Local Access
Host Github Jekyll website locally with the following
 - host Map website locally
   `$ bundle exec jekyll serve` on port 4000,  Server address: http://127.0.0.1:4000

 - where data is accessible
   `const bucket_ndp = 'http://localhost:4000/'.... +'tavg-prec-month.zarr'`

  - access npm website in Chrome since for Firefox a plugin is needed to access a CORS file
	`$ flatpak run org.chromium.Chromium --disable-web-security --user-data-dir=/path/to/website/`
