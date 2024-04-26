lc=python3
username=USERNAME
cluster=${username}@derecho.hpc.ucar.edu
clusterpath=/glade/work/${username}/path/to/data

all: host

host:
	$(lc) host_server.py

build: create_zarr_data

create_zarr_data:
	$(lc) create_icar_zarr.py

create_zarr_charts:
	$(lc) create_icar_charts.py

scp:
	scp $(cluster):$(clusterpath)/map/icar-map.tar.gz .
	scp $(cluster):$(clusterpath)/chart/icar-chart.tar.gz .

untar:
	tar zxf icar-map.tar.gz
	mv icar data/map/
	tar zxf icar-chart.tar.gz
	mv icar data/chart/

clean:
	rm -f *~
cleandata:
	rm -rf downscaling
