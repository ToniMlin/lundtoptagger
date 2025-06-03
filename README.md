# lundtoptagger

Tag top and W jets using the LundNet model.


## Setup

On UChicago, samples and flat weights are here:  
`/data/jmsardain/LJPTagger/FullSplittings/SplitForTopTagger/`

The included setup script can set up the environment on several different systems:

- a system with Red Hat Enterprise Linux 9, an NVIDIA driver which supports CUDA >= 11.8, and access to CVMFS, such as `lxplus-gpu`
- a system with CentOS 7 and access to CVMFS (currently set up without CUDA)
- UCL's `gpu02` server

The script will automatically figure out which of these systems it is running on and set up the environment accordingly; just do

```bash
source setup.sh
```

On UChicago, do

```bash
source /data/jmsardain/LJPTagger/JetTagging/miniconda/bin/activate
conda activate rootenv
```

## Data preparation

To create graphs for training from ROOT files and save them to a file, first process JETM2 or FTAG1 derivations with the following code:  
<https://gitlab.cern.ch/rvinasco/jetmdatamc/-/tree/temporaryRun2>  
Then run `Make_data.py` on the output:

```bash
python Make_data.py configs/config_make_data.yaml
```

The script applies selections defined in the configuration files, creates Lund trees (graphs) for each jet, and calculates weights which make the jet $p_T$ distribution flat.
Two file files are created:
a file containing a list of graphs (`torch_geometric.data.Data` objects) that can be used for training and testing the tagging model,
and a ROOT file containing some properties of the jets passing selection:

- DSID of the dataset from the which the jet was taken
- mass
- $p_T$
- MC event weight
- weight which makes the $p_T$ distribution flat
- truth label - 0 for background, 1 for signal

Some of these are already stored as attributes of the graphs, and they could all be, but the graphs can take a long time to load,
so it can be useful to have a separate file for plots which don't require the Lund trees.

### Configuration and parameters for `Make_data.py`

The input and output file paths and a value for the optional $k_T$ cut are set in `config_make_data.yaml`.
You can also set the fraction of the data to save in a separate file for testing, if any.
The path to this config file must be given as a command-line argument, as in the example above.

The script uses another configuration file, `config_signal.yaml`, which contains parameter sets for several signal samples.
You should choose the appropriate paramter set by modifying the first line of the file.
The parameters include values for the selection cuts (mass, $p_T$, minimum number of splittings)
and paths to files with histograms of the $p_T$ distributions of the jets, which are used to calculate the $p_T$ weights
so that they are proportional to 1/(bin count).
These histograms are included in the repository; they are located in the `histos` folder.
They can be created with the `make_histos.py` script, which also applies mass and $pT$ cuts from `config_signal.yaml`.

## Training

For the training, the main changes one should do are in the configuration file: `config_ONLY_TRAIN.yaml`.
In this file you will define the learning rate, batch size, the input files, the model to use, the location to save your checkpoints.

To run the training:

```bash
python weight_ONLY_TRAINS.py configs/config_ONLY_TRAIN.yaml
```

You can override the $k_T$ cut in the configuration file using a command line argument:
There are two optional arguments which can be used to override the values in the config file:

- `--ln_kT_cut`: float
- `--do_combined_training`: value can be true/false, yes/no, 0/1, case insensitive

For example:

```bash
python weight_ONLY_TRAINS.py configs/config_ONLY_TRAIN.yaml --ln_kT_cut 0 --do_combined_training true
```

## Testing

For the testing, you should run the final_makescores notebook. The only changes you should do are under the conditions in the for loop. The different variables should point to your test files, the ckpt you want to use and the repo to save your output root files 

At the end, when you are done with the testing, make sure you hadd all the root files together: 
```
hadd -f tree.root user.*root
```

## Plotting

In a clean and new terminal, go to the plotting repo and source the setup file. 
It will get the version of the libraries you want to use from /cvmfs/. 
Go to plotting.py and check that you are using the root file you just created with hadd after the testing of the model. 
Plot! 
```
source setup.sh
python -b plotting.py 
```

## To do list: 
- [ ] Cut on ln(kt): prepare multiple graphs with different values of ln(kT) cuts 
- [ ] Make a bkg rej vs ln(kT) plot
- [ ] Make the LundJetPlane plot with the prediction to see where the modeling uncertainties impact the most
- [ ] Apply a shift of 5% to mean pT of the constituent, and test on that sample
- [ ] Apply a shift of 5% to resolution pT of the constituent, and test on that sample
