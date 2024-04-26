# Host ICAR Zarr Data
## Prepare Data
Either [create data](#Create Data) or copy it over.
To copy the data edit the [Makefile](Makefile) and run `make scp untar` to copy the data from Derecho and untar it.
## Host Site
Once data is created the user can host the data locally for local development.
Running `make host` or the following command will start a local server.
```
$ python host_server.py
```
The data is available at `localhost:4000` and can be accessed from a browser.



# ICAR Zarr Data
Create Zarr data files in a format that can be accessed by [carbonplan/maps](https://github.com/carbonplan/maps) websites.
This is setup for generation of data that can be access locally by [icar/maps](https://github.com/scrasmussen/icar-maps).

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
