lc=python3
file=create_icar_zarr.py
file=create_icar_zarr_charts.py

all: build

build:
	$(lc) $(file)

clean:
	rm -f *~
cleandata:
	rm -rf icar_zarr
