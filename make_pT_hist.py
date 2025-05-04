import argparse
import os
import glob
import time
from datetime import timedelta
import gc

import uproot
import awkward as ak

from tools.GNN_model_weight.utils_newdata import load_yaml, GetPtWeight_2, create_train_dataset_fulld_new_Ntrk_pt_weight_file
import numpy as np
from ROOT import TH1F, TFile

print("Libraries loaded!")

def main():
    parser = argparse.ArgumentParser(description="Prepare data for classifier input")
    add_arg = parser.add_argument
    add_arg("--infile", default=None, help="Input file path")
    add_arg("--outfile", default=None, help="Output file path")
    args = parser.parse_args()
    infile = args.infile
    outfile = args.outfile

    # path_to_files = infile if infile is not None else '/eos/home-t/tmlinare/Lund/jetetmiss/JETMDataMC/jpierre/run/submitDir-2025-01-06-1052-fee0 files 1-50/data-ANALYSIS/mc20_13TeV.802017.Py8EG_A14NNPDF23LO_WprimeWZ_flatpT_wideWmass.deriv.DAOD_JETM2.e8482_s3797_r13145_p5548.root'
    path_to_files = infile if infile is not None else '/eos/home-t/tmlinare/Lund/jetetmiss/JETMDataMC/jpierre/run/submitDir-2025-01-12-1919-bc4c W files 1-50 (not flat mass)/data-ANALYSIS/mc20_13TeV.801859.Py8EG_A14NNPDF23LO_WprimeWZ_flatpT.deriv.DAOD_JETM2.e8482_s3681_r13145_p5548.root'
    files = glob.glob(path_to_files)

    intreename = "AnalysisTree"

    print(f"Processing {len(files)} files")
    t_start = time.time()

    for file_number, file in enumerate(files, start=1):
        print("\nLoading file", file)

        with uproot.open(file) as infile:
            tree = infile[intreename]

            truth_labels = ak.flatten(tree["LRJ_truthLabel"].array(library="ak"))
            jet_pts = ak.flatten(tree["LRJ_pt"].array(library="ak"))[truth_labels==5]  # but sample also includes some QCD jets which are included in training
    
    # Create histogram
    hist = TH1F("pt", "Jet pT Histogram", 100, 100, 3000)

    # Fill histogram
    for pt in jet_pts:
        hist.Fill(pt)

    # Save histogram to a ROOT file
    # outfile_path = outfile if outfile is not None else "pTs/Wwidemass.root"
    # outfile_path = outfile if outfile is not None else "pTs/Wrealmass.root"
    outfile_path = outfile if outfile is not None else "pTs/Z.root"
    output_file = TFile(outfile_path, "RECREATE")
    hist.Write()
    output_file.Close()

    print(f"Histogram saved to {outfile_path}")

    
    delta_t_fileax = timedelta(seconds=round(time.time() - t_start))
    print(f"Time taken (hh:mm:ss): {delta_t_fileax}")

if __name__ == "__main__":
    main()