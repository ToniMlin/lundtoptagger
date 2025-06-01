import argparse
import glob
import time
from datetime import timedelta

import uproot
import awkward as ak
from ROOT import TH1F, TFile
import os

from tools.GNN_model_weight.utils_newdata import load_yaml

print("Libraries loaded!")

# parameters
config_signal_path = "configs/config_signal.yaml"
infiles_path = "/eos/home-t/tmlinare/Lund/jetetmiss/JETMDataMC/jpierre/run/submitDir-2025-01-06-1052-fee0 files 1-50/data-ANALYSIS/mc20_13TeV.802017.Py8EG_A14NNPDF23LO_WprimeWZ_flatpT_wideWmass.deriv.DAOD_JETM2.e8482_s3797_r13145_p5548.root"
# infiles_path = "/eos/home-t/tmlinare/Lund/jetetmiss/JETMDataMC/jpierre/run/submitDir-2025-01-12-1919-bc4c W files 1-50 (not flat mass)/data-ANALYSIS/mc20_13TeV.801859.Py8EG_A14NNPDF23LO_WprimeWZ_flatpT.deriv.DAOD_JETM2.e8482_s3681_r13145_p5548.root"
outfile_path = "histos/mass_40-300/WBSMP8.root"
# outfile_path = "histos/mass_40-300/WBSMP8_flat.root"
nbins = 100


def main():
    parser = argparse.ArgumentParser(description="Prepare data for classifier input")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", help="signal configuration file")
    add_arg("--infile", default=None, help="Input file path")
    add_arg("--outfile", default=None, help="Output file path")
    args = parser.parse_args()

    global config_signal_path, infiles_path, outfile_path
    config_signal_path = args.config if args.config is not None else config_signal_path
    infiles_path = args.infile if args.infile is not None else infiles_path
    outfile_path = args.outfile if args.outfile is not None else outfile_path

    config_signal = load_yaml(config_signal_path)
    signal = config_signal["signal"]
    pt_min, pt_max = config_signal[signal]["pt_range"]
    m_min, m_max = config_signal[signal]["mass_range"]

    files = glob.glob(infiles_path)

    intreename = "AnalysisTree"

    print(f"Processing {len(files)} files")
    t_start = time.time()

    # Create histograms
    hist_pt = TH1F("pt", "Jet pT histogram", nbins, pt_min, pt_max)
    hist_m = TH1F("mass", "Jet mass histogram", nbins, m_min, m_max)
    hist_truth_label = TH1F("truth_label", "Jet truth label histogram", 11, -0.5, 10.5)
    

    for file_number, file in enumerate(files, start=1):
        print(f"\nLoading file {file_number}/{len(files)}: {file}")

        with uproot.open(file) as infile:
            tree = infile[intreename]

            dsid_test = tree["dsid"].array(library="np")[0]
            jet_truth_label = config_signal[signal]["signal_jet_truth_label"] if dsid_test==config_signal[signal]["dsid"] else 10
            truth_labels = ak.flatten(tree["LRJ_truthLabel"].array(library="ak"))
            jet_masses = ak.flatten(tree["LRJ_mass"].array(library="ak"))

            selection = (truth_labels == jet_truth_label) & (jet_masses >= m_min) & (jet_masses <= m_max)
            jet_pts = ak.flatten(tree["LRJ_pt"].array(library="ak"))[selection]
            jet_masses = jet_masses[selection]
            truth_labels = truth_labels[selection]

        # Fill histograms
        for pt in jet_pts:
            hist_pt.Fill(pt)
        for m in jet_masses:
            hist_m.Fill(m)
        for label in truth_labels:
            hist_truth_label.Fill(label)

    # Save histogram to a ROOT file
    os.makedirs(os.path.dirname(outfile_path), exist_ok=True)
    output_file = TFile(outfile_path, "RECREATE")
    hist_pt.Write()
    hist_m.Write()
    hist_truth_label.Write()
    output_file.Close()

    print(f"Histograms saved to {outfile_path}")

    
    delta_t_fileax = timedelta(seconds=round(time.time() - t_start))
    print(f"Time taken (hh:mm:ss): {delta_t_fileax}")


if __name__ == "__main__":
    main()