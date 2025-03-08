# MIIT (the Multi-omics Imaging Integration Toolset)

MIIT (pronounce: `meet`) is a Python framework for integrating spatially resolved multi-omics data. The main spatial-omics technologies that we focus on are Spatial Transcriptomics 
(through Visium) and MSI (ImzML). There is additional_data support for various types of annotations (pointset valued data, geojson data, and masks).

## Installation

MIIT can be installed using pip:

```
pip install --user git+https://github.com/mwess/miit@v0.0.3-rc2
```

## Docker

MIIT is also available as a docker image. It comes along with GreedyFHist's external dependency 
`greedy`.

### Downloading the docker image

The docker image can be loaded as follows:

```
docker pull mwess89/miit:0.0.3-rc2
```

It should then be available to run as follows:

```
docker run -it -p 8888:8888 mwess89/miit:0.0.3-rc2
```

### Building docker image locally

The docker image can be build locally as well:

```
docker build -t miit -f Dockerfile .
```

In this case, miit can then be started similarly:

```
docker run -it -p 8888:8888 miit:0.0.3-rc2
```

### Binding external directories to the docker instance

MIIT can be used fully in a docker environment and external directories can be easily added in the
`run` command:

```
docker run -it -p 8888:8888 \
--mount type=bind,src=/home/user/applications/miit,dst=/external_directory \
mwess89/miit:0.0.3-rc2

```

This example mounts the local directory `/home/user/applications/miit` to the path 
`/external_directory` inside the docker container.

## Running example data

We list examples of using miit in `examples/notebooks`.  These examples require test_data.

At the moment, `examples/notebooks` contains 5 notebooks:

- 04_analysis_from_paper: Repeats analysis steps from paper (registration, integration, proof-of-concept analysis). Also includes an introduction to MIIT by shortly explaining concepts and data types.
- 05_analysis_from_paper_short.ipynb: Repeats analysis steps from paper (registration, integration, proof-of-concept analysis). Explains only little about MIIT and its data types.
- 01_data_types.ipynb: Data types of MIIT.
- 02_integrate_st_and_msi.ipynb: Integrating Visium and MSI data.
- 03_integrate_msi_and_msi.ipynb: Integrating MSI and MSI data.

### Docker environment

Choose a directory where you want to store the test data (or stay where you are).

Download some test data from zenodo, extract and start the docker container and bind the test_data directory. Note: Replace `path/to/test_data` with the path that the test_data was extracted to.

```
# Download and extract test data
wget https://zenodo.org/records/14931377/files/test_data.tar.gz -P examples/notebooks/
tar xfvz examples/notebooks/test_data.tar.gz

# Load docker image and connect to 
docker run -it -p 8888:8888 \
--mount type=bind,src=/path/to/test_data,dst=/external_directory \
mwess89/miit:0.0.3-rc2
```

Important: The `ROOT_DIR` variable in the notebooks needs to be set to `/external_directory`.


After starting the docker environment has started and the jupyterlab has been opened, the example notebooks can found in `miit/examples/notebooks`.


### Local (non-docker) environment

Download some test data from zenodo (this assumes that you are at the root of the github repository):
```
wget https://zenodo.org/records/14931377/files/test_data.tar.gz
tar xfvz examples/notebooks/test_data.tar.gz -C examples/notebooks/
```

The notebooks should be runnable now.


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
