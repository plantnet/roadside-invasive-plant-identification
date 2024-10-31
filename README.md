
# Quadrat - Invasive Species Detection on Danish Roadsides with Vision Transformers

This repository provides the complete codebase for reproducing the results of the study *"From Species Identification to Species Presence Detection in High-Resolution Images: An Application to Invasive Plant Species Using Roadside Views."* This work, part of the EU-funded [MAMBO project](https://ec.europa.eu/), leverages **deep learning** to monitor the spread of invasive plants along roadsides, especially in Denmark. By combining biological data with deep learning models, specifically **Vision Transformers (ViTs)**, the project aims to improve invasive species detection without requiring extensive field resources.

For more detailed information, refer to the preprint [here](https://ssrn.com/abstract=4936442).

## Quick Start

Below are instructions to clone the repository, install dependencies, and run an example experiment.

### 1. Clone the Repository and Set Up the Environment

```bash
# Clone the repository
$ git clone https://github.com/plantnet/roadside-invasive-plant-identification.git

# Install dependencies (Python ≥ 3.11 required)
$ cd roadside-invasive-plant-identification
$ micromamba env create -f environment_pytorch2.yml
```

### 2. Prepare the Data and Models

Before running the code, download the following external resources:

- **Pretrained Models**: Download from [Zenodo Models Link](https://zenodo.org/records/13891416) and place them in `./0_datastore/30_models`.
- **Images and Deep Features**: Download images from [Zenodo Data Link](https://zenodo.org/records/14013930) and place them in the respective `10_images` and `20_deep_features` folders within `./0_datastore`.

### 3. Run an Example Experiment (XP2 - VaMIS with fine-tuning)

After setting up, you can launch Experiment 2 (XP2), focusing on the fine-tuned vamis method, by activating the environment and executing the command below:

```bash
$ conda activate pytorch2
$ cd 1_sources/3_scripts
$ bash job102_officiel_vamis_evaluate_model_plantnet_finetune.sh
```

This will trigger the pipeline, reproducing results for the fine-tuned vamis method.

## Experiment Setup and Results

The research investigates **five primary experiments** (XP1-XP5):

- **XP1-XP3**: Experiments exploring the VaMIS approach with various configurations.
- **XP4**: Recommended method (tiling without fine-tuning).
- **XP5**: Tiling method with fine-tuning.

Jobs 100 to 107 in the provided Bash scripts correspond to specific tasks:

- **Job 100**: Precomputes deep features for tiling.
- **Jobs 101-105**: Perform inference for XP1-XP5, respectively.
- **Job 106-107**: Executes tiling and VaMIS training tasks.

## Data and Model Details

### Data

The dataset includes **14,838 high-resolution images** taken along Danish roads. Images are split into train, validation, and test sets. Each image contains annotations for **six invasive plant taxa** (consolidated as meta-species based on visual similarity).

The images can be found here [Zenodo](https://zenodo.org/records/14013930).

### Models

The pre-trained [BEiT](https://arxiv.org/abs/2106.08254) Vision Transformer model forms the backbone for this image analysis. Our paper proposes 2 methods to handle high resolution images with vision transformers: Tiling the image or increasing the model input size (VaMIS for Variable Model Input Size). The model weights can be found on [Zenodo](https://zenodo.org/records/13891416).

## Pipeline Overview

The pipeline is designed to facilitate extensive testing and tuning of deep learning configurations for invasive species detection:

- **Sequential Task Execution**: Define a series of tasks in JSON to execute sequentially or repeat with different parameters.
- **Caching Mechanism**: Reuses previous outputs to avoid redundant computations, improving speed and efficiency across repeated experiments.
- **Pipeline Modularity**: Pipelines and parameters are configurable, supporting pipeline and parameter inheritance, enabling users to build complex workflows from simpler configurations.

## Project Structure

The repository is organized into directories that hold datasets, models, and scripts required for executing the pipeline:

```
./0_datastore/
    ├── 10_images/                 # High-resolution images (.jpg)
    ├── 11_annotations/            # Generated annotations per image folder structure
    ├── 20_deep_features/          # Deep features for images
    ├── 30_models/                 # Pretrained models (.pth files)
    ├── 50_probas_predictions_csv/ # generated CSVs with model predictions
    ├── 70_output_statistiques/    # Statistics based on predictions and annotations

./1_sources/
    ├── 1_pipelines/               # JSON files for specifying inference and training pipelines
    ├── 2_parameters/              # JSON files for pipeline parameters
    ├── 3_scripts/                 # Bash and Python scripts
    ├── 4_logs/                    # Logs for each pipeline run
```

## Requirements

The environment requires Python 3.11 and dependencies, including PyTorch, torchvision, timm, and more. For all dependencies, refer to the `requirements.txt`.

## Citation

If you use this code, data, or models, please cite our article:

```plaintext
@article{espitalier2024species,
  title={From Species Identification to Species Presence Detection in High Resolution Images - an Application to Invasive Plant Species Using Roadside Views},
  author={Espitalier, Vincent and Goëau, Hervé and Botella, Christophe and Dyrmann, Mads and Hoye, Toke T. and Bonnet, Pierre and Joly, Alexis},
  journal={Ecological Informatics},
  year={2024},
  ssrn={https://ssrn.com/abstract=4936442}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE.md) file for details.
