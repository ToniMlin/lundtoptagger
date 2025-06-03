import argparse
import os
import glob
import time
from datetime import timedelta
import gc

import uproot
import awkward as ak
import numpy as np
import torch

from tools.GNN_model_weight.utils_newdata import load_yaml, GetPtWeight, create_train_dataset_fulld_new_Ntrk_pt_weight_file

print("Libraries loaded!")

def main():
    parser = argparse.ArgumentParser(description="Prepare data for classifier input")
    add_arg = parser.add_argument
    add_arg("config", help="job configuration file")
    args = parser.parse_args()
    config_file = args.config
    config = load_yaml(config_file)
    config_signal = load_yaml("configs/config_signal.yaml") # TODO: make this an optional argument, but then the same file needs to be used in utils_newdata.py
    signal = config_signal["signal"]

    path_to_files = config["path_to_trainfiles"]
    files = glob.glob(path_to_files)[:config["n_files"]]

    intreename = "AnalysisTree"

    n_files = len(files)
    print(f"Processing {n_files} files")
    t_start = time.time()

    dataset = []
    primary_Lund_only_one_arr = []

    out_tree_dict = {
        "dsids": ak.Array([]),
        "EventInfo_mcEventWeight": ak.Array([]),
        "fjet_m": ak.Array([]),
        "fjet_pt": ak.Array([]),
        "fjet_weight_pt": ak.Array([]),
        "labels": ak.Array([])
    }

    for file_number, file in enumerate(files, start=1):
        print("\nLoading file", file)

        with uproot.open(file) as infile:
            tree = infile[intreename]

            dsids = tree["dsid"].array(library="np")
            dsid_test = dsids[0]                                 # check the first DSID, they should all be the same
            if dsid_test in config_signal[signal]["skip_dsids"]: # don't lose time with jets that don't pass pt cut or wrong signal sample
                continue

            truth_labels_unflattened = tree["LRJ_truthLabel"].array(library="ak")
            truth_labels = ak.flatten(truth_labels_unflattened)

            numbers_of_jets_per_event = ak.num(truth_labels_unflattened)

            mcEventWeights = tree["mcEventWeight"].array(library="np")
            mcEventWeights = np.repeat(mcEventWeights, numbers_of_jets_per_event) # expand out the array so it has same length as flattened array
            dsids = np.repeat(dsids, numbers_of_jets_per_event)            # TODO: can I do this without numpy? expand out the array so it has same length as flattened array

            print(f"length dataset: {len(dataset)}, file number: {file_number}/{n_files}")
            parent1 = ak.flatten(tree["jetLundIDParent1"].array(library="ak"))
            parent2 = ak.flatten(tree["jetLundIDParent2"].array(library="ak"))
            jet_ms = ak.flatten(tree["LRJ_mass"].array(library="ak"))
            jet_pts = ak.flatten(tree["LRJ_pt"].array(library="ak"))
            all_lund_zs = ak.flatten(tree["jetLundZ"].array(library="ak"))
            all_lund_kts = ak.flatten(tree["jetLundKt"].array(library="ak"))
            all_lund_drs = ak.flatten(tree["jetLundDeltaR"].array(library="ak"))
            N_tracks = ak.flatten(tree["LRJ_Nconst_Charged"].array(library="ak"))
            # N_tracks = ak.flatten(tree["LRJ_Ntrk500"].array(library="ak"))
            # N_tracks = ak.flatten(tree["LRJ_Nconst"].array(library="ak"))

            print("\nCalculating weights:")
            flat_weights = GetPtWeight(jet_pts, truth_labels, dsid_test, 5)
            kT_selection = config["kT_cut"]

            passed_selection = []   # will be a boolean array, True if jet passes selection

            print("\nCreating PyTorch graphs:")
            dataset = create_train_dataset_fulld_new_Ntrk_pt_weight_file(
                dataset, all_lund_zs, all_lund_kts, all_lund_drs,
                parent1, parent2, flat_weights, truth_labels, dsids,
                N_tracks, jet_pts, jet_ms, kT_selection,
                primary_Lund_only_one_arr,
                passed_selection,
                config_signal[signal]["signal_jet_truth_label"],
                signal_dsid=config_signal[signal]["dsid"],
                pt_range=config_signal[signal]["pt_range"],
                mass_range=config_signal[signal]["mass_range"],
                include_pt=config["include_pt"],
            )

            out_tree_dict["dsids"] = ak.concatenate([out_tree_dict["dsids"], dsids[passed_selection]])
            out_tree_dict["EventInfo_mcEventWeight"] = ak.concatenate([out_tree_dict["EventInfo_mcEventWeight"], mcEventWeights[passed_selection]])
            out_tree_dict["fjet_m"] = ak.concatenate([out_tree_dict["fjet_m"], jet_ms[passed_selection]])
            out_tree_dict["fjet_pt"] = ak.concatenate([out_tree_dict["fjet_pt"], jet_pts[passed_selection]])
            out_tree_dict["fjet_weight_pt"] = ak.concatenate([out_tree_dict["fjet_weight_pt"], flat_weights[passed_selection]])
            out_tree_dict["labels"] = ak.concatenate([out_tree_dict["labels"], truth_labels[passed_selection]])

            gc.collect()

    print("\nDataset created! len():", len(dataset))
    delta_t_fileax = timedelta(seconds=round(time.time() - t_start))
    print(f"Time taken (hh:mm:ss): {delta_t_fileax}")

    out_file_name_graphs = config["out_file_name_graphs"]
    outfile_name_root = config["out_file_name_root"]
    filepath_placeholder_vals = dict(
        id = config["id"],
        kT_cut = kT_selection,
        include_pt = "_with_pt" if config["include_pt"] else ""
    )
    out_dir = config["out_dir"].format(**filepath_placeholder_vals)
    os.makedirs(out_dir, exist_ok=True)

    test_frac = config["test_frac"]
    if test_frac is not None:
        print("Splitting dataset into train and test sets")
        test_num = int(len(dataset) * test_frac)
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)
        dataset = [dataset[i] for i in indices]
        dataset_test = dataset[:test_num]
        dataset = dataset[test_num:]

        out_file_name_graphs_test = out_file_name_graphs.format(
            **filepath_placeholder_vals,
            test_frac = f"_{int(test_frac*100)}percent"
        )
        output_path_graphs_test = os.path.join(out_dir, out_file_name_graphs_test)
        torch.save(dataset_test, output_path_graphs_test)
        print("Test graphs saved to:", output_path_graphs_test)

        outfile_name_root_test = outfile_name_root.format(
            **filepath_placeholder_vals,
            test_frac = f"_{int(test_frac*100)}percent"
        )
        output_path_root_test = os.path.join(out_dir, outfile_name_root_test)
        out_tree_dict_test = {}
        for key in out_tree_dict:
            out_tree_dict_test[key] = out_tree_dict[key][indices][:test_num]
            out_tree_dict[key] = out_tree_dict[key][indices][test_num:]
        with uproot.recreate(output_path_root_test) as outfile:
            outfile["FlatSubstructureJetTree"] = out_tree_dict_test
        print("Test dataset written to ROOT file:", output_path_root_test)

    out_file_name_graphs = out_file_name_graphs.format(
        **filepath_placeholder_vals,
        test_frac = f"_{int((1-test_frac)*100)}percent" if test_frac is not None else "",
    )
    output_path_graphs = os.path.join(out_dir, out_file_name_graphs)

    torch.save(dataset, output_path_graphs)
    print("Training graphs saved to:", output_path_graphs)

    outfile_name_root = outfile_name_root.format(
        **filepath_placeholder_vals,
        test_frac = f"_{int((1-test_frac)*100)}percent" if test_frac is not None else "",
    )
    output_path_root = os.path.join(out_dir, outfile_name_root)
    with uproot.recreate(os.path.join(output_path_root)) as outfile:
        outfile["FlatSubstructureJetTree"] = out_tree_dict
    print("Training dataset written to ROOT file:", output_path_root)


if __name__ == "__main__":
    main()
