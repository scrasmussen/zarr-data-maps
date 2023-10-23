lc=python3
file=create_icar_zarr.py

all: build

build: cleandata
	$(lc) $(file)

clean:
	rm -f *~
cleandata:
	rm -rf icar_zarr
