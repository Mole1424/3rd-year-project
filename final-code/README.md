# Main Code

This is the main code for the project, a single function that can train models, find the best ones, and process all the data.  
The script is designed to run on any dataset. The dataset can be in any location but needs to have the following file structure:

```
<name_of_dataset>/
|--real/
|  |--<video_1>.mp4
|--fake/
   |--<video_1>.mp4
```

## To run

1. Set up the venv
```bash
cd final-code
python3.12 -m venv venv
source venv/bin/activate
pip3.12 install -r requirements.txt
```

2. Train eye landmark model (optional)  

There are two option models for landmark detection (PFLD and HRNet), the models are trained seperately. They can be trained using the following command:
```bash
python3.12 eye_detection.py train
```
and then tested with
```bash
python3.12 eye_detection.py test
```
This is not intented for user use but for development purposes. Therefore you will need to provide your own dataset (for marking dissertation this is provided as `eyes.zip`). The two models should be saved as `.keras` files in the models directory.

2. Extract images  
These images are used for training traditional model detectors. The images are extracted from the videos using the following command:
```bash
python3.12 -u create_images.py <path_to_dataset>
```
This is not in the main loop because it does not benefit from GPU acceleration (shoulod be run as CPU) and thus should be either run locally or on a CPU partition of batch compute.

3. Main loop  
The main loop trains all the models, finds the best ones, processes the entire dataset, saving the results. It is designed to be run on a slurm batch compute system. Whilst CPU training is possible, it is reccomended to run on a gpu partition.
```bash
sbatch main.sbatch
```
The line `python3.12 -u main.py` in the `main.sbatch` should be changed to `python3.12 -u main.py <path_to_dataset> <path_to_models>`, the results will be saved to the current directory. The script has checkpointing, and so all trained models will be put and loaded from `<path_to_models>` along with the EARs dataset created, the ongoing results file will be saved to the current directory. Models will be saves as `<dataset_name>_<model_name>` to allow for easy identification. The script will also save the best model for each dataset and model type, along with the results of the analysis.

## File Structure

### Side Code

Code that is in the main-code folder but is not used as part of the main loop. Either broken code or code that runs seperately of the main loop.

#### `old-code/`

Old code for the attempted implementation of [Eye landmarks detection via weakly supervised learning](https://www.sciencedirect.com/science/article/pii/S0031320319303772). This is, unfortunately, broken, some things might work, somethings might not, who knows?!

#### `eye_detection.py`, `eyes.py` & `eye_detection.sbatch`

Code for defining and training eye landmark model. `eye_detection.*` is the main file for training the models. `eyes.py` containes the model definitions and class wrappers for the models. 

#### `create_images.py` & `create_images.sbatch`

Code and sbatch file to extract random frames from a dataset for trainind traditional models

### `transferability.py` & `transferability.sbatch`

Evaluates the transferability of blink-based DeepFake detection and traditional detectors on unkown datasets

#### `utils.py`

Random python scripts (mostly for formatting datasets). Intended for development purposes, use at your own risk!

### Main Code

Main file and helper files for the main loop

#### `ear_analysis.py`

Timeseries analysis of ear data. Trains a host of models (see file for exact ones) and selects the best one. It will also save all models for potential use and debugging later.

#### `traditional_detectors.py`

Trains and saves models for traiditional detectors, models are trained on frames extracted from `create_images.py`.

#### `main.py`

Main file for this project. Creates ear dataset, trains models, evaluates them, and then saves the results, all with checkpointing so it can pick up from where it left off.  