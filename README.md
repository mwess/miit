# MIIT (the Multi-omics Imaging Integration Toolset)

MIIT (pronounce: `meet`) is a Python framework for integratig spatially resolved multi-omcis data. The main spatial-omics technologies that we focus on are Spatial Transcriptomics 
(through Visium) and MSI (ImzML). There is additional_data support for various types of annotations (pointset valued data, geojson data, and masks).

## Installation

MIIT can be installed using pip:

```
pip install --user git+https://github.com/mwess/miit@v0.0.2
```

## Usage

We list examples of using miit in `examples/notebooks`. 

## Docker

MIIT is also available as a docker image. It comes along with GreedyFHist's external dependency 
`greedy`.

### Downloading the docker image

The docker image can be loaded as follows:

```
docker pull mwess89/miit:0.0.2
```

It should then be available to run as follows:

```
docker run -it -p 8888:8888 mwess89/miit:0.0.2
```

### Building docker image locally

The docker image can be build locally as well:

```
docker build -t miit -f Dockerfile .
```

In this case, miit can then be started similarly:

```
docker run -it -p 8888:8888 miit:latest
```

### Binding external directories to the docker instance

MIIT can be used fully in a docker environment and external directories can be easily added in the
`run` command:

```
docker run -it -p 8888:8888 \
--mount type=bind,src=/home/maxi/applications/miit,dst=/external_directory \
mwess89/miit:0.0.2
```

This example mounts the local directory `/home/maxi/applications/miit` to the path 
`/external_directory` inside the docker container.


### Running examples in docker environment

The docker container clones this repository explicitly and contains all example data. After 
starting the instance and connecting to it, the example data can be found int `/miit/examples/`.

## Citation

If you use this code, please cite: 

```
@article{wess2024spatial,
  title={Spatial Integration of Multi-Omics Data using the novel Multi-Omics Imaging Integration Toolset},
  author={Wess, Maximillian and Andersen, Maria K and Midtbust, Elise and Guillem, Juan Carlos Cabellos and Viset, Trond and St{\o}rkersen, {\O}ystein and Krossa, Sebastian and Rye, Morten Beck and Tessem, May-Britt},
  journal={bioRxiv},
  pages={2024--06},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```
