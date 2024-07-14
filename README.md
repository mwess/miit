# MIIT (the Multi-omics Imaging Integration Toolset)

MIIT (pronounce: `meet`) is a Python framework for integratig spatially resolved multi-omcis data. The main spatial-omics technologies that we focus on are Spatial Transcriptomics 
(through Visium) and MSI (ImzML). There is additional_data support for various types of annotations (pointset valued data, geojson data, and masks).

## Installation

MIIT can be installed using pip:

```
pip install --user git+https://github.com/mwess/miit@master
```

## Usage

We list examples of using miit in `examples/notebooks`. 


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