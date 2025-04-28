All datasets and pre-trained models can be found at https://www.dcs.warwick.ac.uk/~u2204489/CS310/, note they are quite large (50GB) it is reccomended to unzip them on a DCS large partition.

Actual code is found in the final-code/ directory. The following is a quick start guide. It assumes models the following file structure.

1. Activate the virtual environment  
`python3.12 -m venv venv`  
`source venv/bin/activate`  
`pip3.12 isntall -r requirements.txt`

2. Download a dataset into the datasets and models from the website above

3. Unpack images in the dataset  
`python3.12 create-images.py <path-to-dataset>`

4. Run the main loop
`python3.12 main.py <path-to-dataset> <path-to-models> 0.05 <hrnet|pfld>`

Note this program requires A LOT of compute resources (>12GB GPU memory and >128GB RAM) and so it is reccomended to run on a HPC system, sample SLURM files for DCS' kudu cluster are provided 
