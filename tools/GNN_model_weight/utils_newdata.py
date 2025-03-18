import os
from typing import Union

import yaml
import uproot
import awkward as ak
import numpy as np
from tqdm import trange
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data

with open("configs/config_signal.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
signal = config["signal"]
weights_file = uproot.open(config[signal]["weights_file"])

flatweights_bg = weights_file["bg_inv"].to_numpy()
flatweights_sig = weights_file["h_sig_inv"].to_numpy()

def GetPtWeight( dsid , pt, SF):

    lenght_sig = len(flatweights_bg[0])
    lenght_bkg = len(flatweights_bg[0])
    scale_factor = 1
    weight_out = []

    for i in range ( 0,len(dsid) ):
        pt_bin = int( ((pt[i]-200)/3000)*lenght_sig )
        if pt_bin==lenght_sig :
            pt_bin = lenght_sig-1
        if dsid[i] < 370000 :
            #weight_out.append( (flatweights_bg[0][pt_bin]*scale_factor)*10**4 ) ## used for W tagging
            weight_out.append( (flatweights_bg[0][pt_bin])*1 )
        if dsid[i] > 370000 :
            #weight_out.append( (flatweights_sig[0][pt_bin])*10**4 ) ## used for W tagging
            weight_out.append( (flatweights_sig[0][pt_bin])*1 )
    return np.array(weight_out)

def GetPtWeight_2(truth_labels, pts, SF):

    ## PT histograms of all qcd and top jets in dataset
    filename1 = config[signal]["pt_hist_file_bkg"]
    filename2 = config[signal]["pt_hist_file_signal"]
    weights_file1 = uproot.open(filename1)
    flatweights_bg = weights_file1["pt"].to_numpy()
    weights_file2 = uproot.open(filename2)
    flatweights_sig = weights_file2["pt"].to_numpy()
    
    lenght_sig = len(flatweights_sig[0])
    lenght_bkg = len(flatweights_bg[0])
    #print("lenght_sig:", lenght_sig, "  lenght_bkg:", lenght_bkg)
    
    sig_bkg_proportion = 5  ## if is taked 5% of signal and 1% of qcd for training then sig_bkg_proportion=5
    scale_factor = (lenght_bkg/lenght_sig) / sig_bkg_proportion #1
    #print("scale_factor", scale_factor)
    
    weight_out = []
    Inv_hist_bg = []#flatweights_bg[0]
    Inv_hist_sig = []#flatweights_sig[0]
    
    ## it's time to calcuate the 1/hist
    for i in range (0,lenght_bkg):
        if flatweights_bg[0][i]==0:
            Inv_hist_bg.append(0)
            continue
        else:
            Inv_hist_bg.append(np.sum(flatweights_bg[0]) / (lenght_bkg * flatweights_bg[0][i]))
            
    for i in range (0,lenght_sig):
        if flatweights_sig[0][i]==0:
            Inv_hist_sig.append(0)
            continue
        else:
            Inv_hist_sig.append(np.sum(flatweights_sig[0]) / (lenght_sig * flatweights_sig[0][i]))

    for i in trange(len(truth_labels)):
        pt_bin = int( ((pts[i]-100)/3000)*lenght_sig )
        if pt_bin>=lenght_sig : # ==
            pt_bin = lenght_sig-1
        if truth_labels[i]==10: # background could also be identified by DSIS, which would be < 370000
            #print("pt[i] ->", pt[i])
            #print("bin_pt->", pt_bin)
            weight_out.append( (Inv_hist_bg[pt_bin])*1  )
        if truth_labels[i]!=10: # this could also contain some non-signal jets which must be removed when creating the training dataset
            weight_out.append( (Inv_hist_sig[pt_bin]*scale_factor)*1 ) #*10**2 )
    return np.array(weight_out)


def load_yaml(file_name):
    assert(os.path.exists(file_name))
    with open(file_name) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def create_train_dataset_fulld_new_Ntrk_pt_weight_file(
    graphs: list[Data],
    z, k, d, edge1, edge2, weight, label, Ntracks, jet_pts, jet_ms,
    kT_selection: Union[float, None],
    primary_Lund_only_one_arr: list,
    signal_jet_truth_label: int,
    include_pt: bool = False
) -> list[Data]:
    """
    Create a list of graphs for tagging.

    Args:
        graphs (list[Data]): List to which the generated torch_geometric.data.Data objects will be appended.
        z (array): 2D array, with an array of z values for each jet.
        k (array): 2D array, with an array of kT values for each jet.
        d (array): 2D array, with an array of ΔR values for each jet.
        edge1 (array): Array of edge1 values.
        edge2 (array): Array of edge2 values.
        weight (array): Array of jet weights.
        label (array): Array of jet truth labels.
        Ntracks (array): Array of Ntracks values.
        jet_pts (array): Array of jet pT values.
        jet_ms (array): Array of jet mass values.
        kT_selection (float | None): kT selection threshold.
        primary_Lund_only_one_arr (list): List to keep track of how many jets have only 1 splitting.
        signal_jet_truth_label (int): Truth label for signal jets.
        include_pt (bool): Whether to include pT as a graph attribute.

    Returns:
        list[Data]: List of torch_geometric.data.Data objects.
    """
    test_bool = 1
    buildID_from_graphs = 0
    Primary_Lund_Plane = 0
    extra_node = 0

    # loop over jets
    for i in trange(len(z)):  
        '''
        label_np = ak.to_numpy(label[i])
        jet_pts_np = ak.to_numpy(jet_pts[i])
        jet_ms_np = ak.to_numpy(jet_ms[i])
        label_np = label_np.astype(float)
        jet_pts_np = jet_pts_np.astype(float)
        jet_ms_np = jet_ms_np.astype(float)
        '''
        # skip jets with less than 3 splittings
        if len(z[i])<3: 
            continue
        #print(label[i])
        # skip jets which are not signal (1 for top and 2 for W) or background (10)
        if (label[i]!=signal_jet_truth_label) and (label[i]!=10):
            continue

        # label signal as 1 and background as 0
        label_out = label[i] # label_np
        if label[i] == 10:
            label_out = 0
        if label[i] == signal_jet_truth_label:
            label_out = 1

        if jet_pts[i] > 3200: continue
        if jet_pts[i] < 350: continue # . ./run.txt
        
        z_out = ak.to_numpy(z[i])
        k_out = ak.to_numpy(k[i])
        d_out = ak.to_numpy(d[i])
        
        z_out += 1e-4 
        k_out += 1e-4 
        d_out += 1e-4 
        
        z_out = np.log(1/z_out)
        k_out = np.log(k_out)
        d_out = np.log(1/d_out)
        
        
        ## lets go to do kt cut; to do this first we need to recover parentID1 and parentID2 (the ones that have a lot of -1) 
        if buildID_from_graphs==1:
            edges1 = ak.to_numpy(edge1[i]) ## it's not necesary edge2[i], it has the same information
            #print(len(edges1)/2)
            len_edges = int(len(edges1)/2)
            edges_A = edges1[:len_edges] # sons
            edges_B = edges1[len_edges:] # parents ; then edges_B[i] > edges_A[i]
            '''
            for x in range(len(edges1)):
                print(edges1[x])
            '''
            id1_id2_edge = 0
            '''
            print("edges_A len()->",len(edges_A))
            print("edges_B len()->",len(edges_B))
            print("edges_A",edges_A)
            print("edges_B",edges_B)
            '''
            for j in range(0,len(edges_A)):
                if j == len(edges_A)-1:
                    id1_id2_edge = j + 1
                    break
                if edges_A[j+1] < edges_A[j]:
                    id1_id2_edge = j + 1
                    break
            
            edges_A_1 = edges_A[id1_id2_edge:] 
            edges_A_2 = edges_A[:id1_id2_edge] 
            edges_B_1 = edges_B[id1_id2_edge:] 
            edges_B_2 = edges_B[:id1_id2_edge]
            '''
            print("edges_A_1",edges_A_1)
            print("edges_A_2",edges_A_2)
            print("edges_B_1",edges_B_1)
            print("edges_B_2",edges_B_2)
            '''
            ## it's time to recover parentID1 (using edges_A_1 and edges_B_1) and parentID2
            parentID1 = []
            parentID2 = []
            for j in range (0,len(z[i]) ):
                if len(edges_B_1) == 0:
                    parentID1.append(-1)
                elif j == edges_A_1[0]:
                    parentID1.append(edges_B_1[0])
                    edges_A_1 = np.delete(edges_A_1,0)
                    edges_B_1 = np.delete(edges_B_1,0)
                else:
                    parentID1.append(-1)
                    
                if len(edges_B_2) == 0:
                    parentID2.append(-1)
                elif j == edges_A_2[0]:
                    parentID2.append(edges_B_2[0])
                    edges_A_2 = np.delete(edges_A_2,0)
                    edges_B_2 = np.delete(edges_B_2,0)
                else:
                    parentID2.append(-1)
            
            ## Now using parentID1 and parentID1 let's go and do kT cut 
            ## I found both parentID because I think in this way code run faster, I don't want to do 
            ## extra loops or complex functions in a data sample with millions of graphs
            ### previous steps can be deleted if we take parentID1 and parentID2 from previous code
            #print("ID1   :",parentID1)
            #print("ID2   :",parentID2)
    
            
            ## here ID2 is the HARDEST branch!!

            # this fix should be not necessary anymore
            for j in range(0, len(parentID1)):
                if parentID1[j] == j :
                    #print("warning!")
                    parentID1[j] = -1
                if parentID2[j] == j :
                    #print("warning!")
                    parentID2[j] = -1
                    
        ## I just don't want to change some lines, this mix between 1 and 2 should be remove in next version
        if buildID_from_graphs != 1:
            parentID1 = ak.to_numpy(edge2[i]) #edge2
            parentID2 = ak.to_numpy(edge1[i]) #edge1
        
        # python3 weight_class_train-Copy1.py configs/config_class_train_top.yaml        
        index_count = []
        selected_nodes = []
        index_count_out = []
        kT_Cut = kT_selection if kT_selection is not None else -np.inf # 0.0 , 0.4 0.9, 2, 2.8 
        nodes_pass_KT = []
        node_kt_step = 0 ## used to renamed edges properly ()
        node_index = 0
        prev_cur_index = 0

        '''
        if i!=1061:
            continue
        print("i",i)
        print("edges_A",edges_A)
        print("edges_B",edges_B)
        print("parentID1",parentID1)
        print("parentID2",parentID2)
        print("k_out[0]",k_out[0])
        '''
        
        nodes_selected = []
        if Primary_Lund_Plane == 1:
            #print("ONLY PRIMARY LUND WILL BE USED!")
            nodes_primary_count = 0
            for j in range(0 , len(z[i])):
                if nodes_primary_count==0:  #j == 0 :
                    #j_ID1_next = parentID1[j]
                    j_ID1_next = parentID2[j]
                #selected_nodes = []
                #if k_out[j] <= kT_Cut : 
                if (k_out[j] <= kT_Cut): #  or j==0 or (j in parentID1) : 
                    node_kt_step += 1
                    nodes_pass_KT.append( int(node_kt_step) ) 
                    nodes_selected.append(False)
                    continue
                if nodes_primary_count>0 and j != j_ID1_next: # j>0
                    #print("222222")
                    node_kt_step += 1
                    nodes_pass_KT.append( int(node_kt_step) ) 
                    nodes_selected.append(False)
                    continue
                nodes_selected.append(True)
                nodes_primary_count +=1;
                #j_ID1_next = parentID1[j]
                j_ID1_next = parentID2[j]
                index_count.append(j)
                nodes_pass_KT.append( int(node_kt_step) ) 
                while len(index_count) > 0:
                    cur_index = index_count[-1]
                    prev_cur_index = cur_index
                    #index_1 = parentID1[cur_index]
                    index_2 = parentID2[cur_index]
                    index_count.pop()
                    '''
                    if len(graphs)==520:
                        #print("k_out[0]:", k_out[0], "  k_out[1]:", k_out[1])
                        print("cur_index:", cur_index)
                        print("index_1:", index_1, "kt(index_1)", k_out[index_1])
                        print("index_2:", index_2, "kt(index_2)", k_out[index_2])
                    '''
                    '''
                    if index_1 != -1:
                        if k_out[index_1] > kT_Cut:
                            selected_nodes.append( int(index_1) )
                            index_count_out.append( int(j))
                            node_index += 1
                            #if len(graphs)==520:
                            #    print("len(selected_nodes)inside  1:", len(selected_nodes))
                            #    print("len(index_count_out)inside 1:", len(index_count_out))
                        else:
                            index_count.append(index_1)
                    '''
                    if index_2 != -1:
                        if k_out[index_2] > kT_Cut:
                            selected_nodes.append( int(index_2) )
                            index_count_out.append( int(j))
                            node_index += 1                             
                        else:
                            index_count.append(index_2)
                    #'''
        ######################################################################################
        else:
            for j in range(0 , len(z[i])):
                #index_count.append(j) # this line here is an error!
                #selected_nodes = []
                if k_out[j] <= kT_Cut : 
                    node_kt_step += 1
                    nodes_pass_KT.append( int(node_kt_step) ) 
                    continue
                index_count.append(j)
                nodes_pass_KT.append( int(node_kt_step) ) 
                while len(index_count) > 0:
                    cur_index = index_count[-1]
                    prev_cur_index = cur_index
                    index_1 = parentID1[cur_index]
                    index_2 = parentID2[cur_index]
                    index_count.pop()
    
                    '''
                    if len(graphs)==520:
                        #print("k_out[0]:", k_out[0], "  k_out[1]:", k_out[1])
                        print("cur_index:", cur_index)
                        print("index_1:", index_1, "kt(index_1)", k_out[index_1])
                        print("index_2:", index_2, "kt(index_2)", k_out[index_2])
                    '''
                    if index_1 != -1:
                        if k_out[index_1] > kT_Cut:
                            selected_nodes.append( int(index_1) )
                            index_count_out.append( int(j))
                            node_index += 1
                            '''
                            if len(graphs)==520:
                                print("len(selected_nodes)inside  1:", len(selected_nodes))
                                print("len(index_count_out)inside 1:", len(index_count_out))
                            '''
                        else:
                            index_count.append(index_1)
                    if index_2 != -1:
                        if k_out[index_2] > kT_Cut:
                            selected_nodes.append( int(index_2) )
                            index_count_out.append( int(j))
                            node_index += 1 
                            '''
                            if len(graphs)==520:
                                print("len(selected_nodes)inside  2:", len(selected_nodes))
                                print("len(index_count_out)inside 2:", len(index_count_out))
                            '''
                        else:
                            index_count.append(index_2)
        
        ##transform edges numeration and avoid isolated nodes 
        '''
        print("nodes before kT slection: ", len(k_out) )
        print("nodes after kT slection: ", len(k_out[k_out > kT_Cut]) )
        print("len(index_count_out)",len(index_count_out))
        print("len(selected_nodes)",len(selected_nodes))
        #print("len(nodes_pass_KT)",len(nodes_pass_KT))
        '''
        if len(k_out[k_out > kT_Cut]) < 1:
            continue
        
        #print("index_count_out  :",index_count_out)
        #print("selected_nodes   :",selected_nodes)

        for j in range(0,len(index_count_out)):
            ## 1+ in order to add an extra node
            if extra_node==1:
                index_count_out[j] = int(1 + index_count_out[j] - nodes_pass_KT[index_count_out[j]] )
                selected_nodes[j] = int(1 + selected_nodes[j] - nodes_pass_KT[selected_nodes[j]] )
            else:
                index_count_out[j] = int( index_count_out[j] - nodes_pass_KT[index_count_out[j]] )
                selected_nodes[j] = int( selected_nodes[j] - nodes_pass_KT[selected_nodes[j]] )
        if extra_node==1:
            index_count_out.insert(0,0)
            selected_nodes.insert(0,1)
        
        
        #print("index_count_out Af:",index_count_out)
        #print("selected_nodes Af:",selected_nodes)
        #if i > 264:
        #    break
        #print(len(j))
            
        index_count_out = np.array(index_count_out, dtype=int )
        selected_nodes = np.array(selected_nodes, dtype=int)
        #index_count_out = index_count_out.astype(int)
        #selected_nodes = selected_nodes.astype(int)
        
        #print("1", selected_nodes)
        #print("1.5", selected_nodes[1])
        #print("2", type(selected_nodes[1]) )

        ## kt mask for feature
        if Primary_Lund_Plane==1:
            k_mask = np.array(nodes_selected)
        if Primary_Lund_Plane==0:
            k_mask = k_out > kT_Cut
        z_out = z_out[k_mask]
        k_out = k_out[k_mask]
        d_out = d_out[k_mask]
        
        
        mean_z, std_z = 2.0568479032747313, 1.4450598054504056
        mean_dr, std_dr = 3.8597358364389427, 2.2748462855901073
        mean_kt, std_kt = -2.379904791478249, 2.940813577366582
        #mean_ntrks, std_ntrks = 26.556999184747827, 16.53733685428723 #only qcd good partition
        #mean_ntrks, std_ntrks = 39.81133623360089, 10.99193693271175
        mean_ntrks, std_ntrks = 57.588158609500134, 23.900100132781983
        
        z_out = (z_out - mean_z) / std_z
        k_out = (k_out - mean_kt) / std_kt
        d_out = (d_out - mean_dr) / std_dr
        Ntrk = (Ntracks[i] - mean_ntrks) / std_ntrks

        #print("3", z_out)
        #print("3.5", z_out[1])
        #print("4", type(z_out[1]) )

        z_out = z_out.astype(float)
        k_out = k_out.astype(float)
        d_out = d_out.astype(float)

        #edge = torch.tensor(np.array([edge1[i], edge2[i]]) , dtype=torch.long)
        edge_ID1 = np.concatenate((index_count_out, selected_nodes))
        edge_ID2 = np.concatenate((selected_nodes, index_count_out))
        edge = torch.tensor(np.array([edge_ID1, edge_ID2]) , dtype=torch.int64)
        #edge = np.array([edge_ID1, edge_ID2]).astype(int)
        

        vec = []
        ## in order to add an extra node
        if extra_node==1:
            #print(index_count_out)
            #print(d_out)
            d_out = np.append(0, d_out)
            z_out = np.append(0, z_out)
            k_out = np.append(0, k_out)

        vec.append(np.array([d_out, z_out, k_out]).T)
        vec = np.array(vec)
        vec = np.squeeze(vec)
        vec=torch.tensor(vec, dtype=torch.float).detach()

        graph_size = 1
        if len(k_out) == 1:
            primary_Lund_only_one_arr.append(1)
            #continue
            #edge = torch.tensor([[0,0], [0,0]], dtype=torch.int64)
            #edge = torch.tensor([[0], [0]], dtype=torch.int64)
            edge = torch.tensor([[], []], dtype=torch.int64)
            vec = torch.unsqueeze(vec, dim=0)
            graph_size = 0

        '''
        if len(k_out) == 2 and len(graphs)==520:
            print("graph number:", len(graphs))
            print("ID1_f:",parentID1)
            print("ID2_f:",parentID2)
            print("edge_index", edge)
        '''

        #print("5", edge)
        #print("5.3", edge[0,1])
        #print("5.8", edge[0].dtype )
        #print("6", edge[0,1].dtype )

        #print("weights", weight[i])
        #print("weights type:", type(weight[i]) )
        #print("pt", jet_pts[i])
        #print("mass", jet_ms[i])
        #print("mass type:", type(jet_ms[i]) ) # <class 'numpy.float64'>

        
        if len(edge_ID1)<1:
            primary_Lund_only_one_arr.append(1)
            #print("k_out",k_out , "  edge_ID1:", edge_ID1)
            continue
            #print("x",vec)
            #print("edge",edge)
        
        #print("edge",edge)
        #print("edge1",edge[0])
        #print("edge2",edge[1])

        graph = Data(
            x = vec.detach(),
            #edge_index = torch.tensor(edge, dtype=torch.int64).detach(),
            edge_index = edge.detach(),
            #Ntrk=torch.tensor(Ntracks[i], dtype=torch.int).detach(),
            Ntrk = torch.tensor(Ntrk, dtype=torch.float).detach(),
            weights = torch.tensor(weight[i], dtype=torch.float).detach(),
            #graph_size = torch.tensor(graph_size, dtype=torch.float).detach(),
            mass =  float(jet_ms[i]), #torch.tensor(jet_ms[i], dtype=torch.float).detach(),
            y = float(label_out) #torch.tensor(label_out, dtype=torch.float).detach() ))
        )
        if include_pt:
            graph["pt"] = float(jet_pts[i]) #torch.tensor(jet_pts[i] , dtype=torch.float).detach()

        graphs.append(graph)
        '''
        graphs.append(Data(x=torch.tensor(vec, dtype=torch.float).detach(),
                           edge_index = torch.tensor(edge, dtype=torch.int64).detach(),
                           #Ntrk=torch.tensor(Ntracks[i], dtype=torch.int).detach(),
                           Ntrk=torch.tensor(Ntrk, dtype=torch.float).detach(),
                           weights =torch.tensor(weight[i], dtype=torch.float).detach(),
                           pt=torch.tensor(jet_pts[i], dtype=torch.float).detach(),
                           mass=torch.tensor(jet_ms[i], dtype=torch.float).detach(),
                           y=torch.tensor(label_out, dtype=torch.float).detach() ))
        '''
        #print(graphs[-1])
        #print(graphs[-1].x)
        #print(graphs[-1].edge_index)
        '''
        if len(k_out) == 1:
            print("1111111111")
            print(graphs[-1])
        if len(k_out) == 2 and len(graphs)%2==0 :
            print("2222222222")
            print(graphs[-1])
        '''

    print("all_graphs_count_graphs:", len(graphs))
    print("primary_Lund_only_one:", np.sum(primary_Lund_only_one_arr))
    print("percent_graphs:", 1 - np.sum(primary_Lund_only_one_arr) / len(graphs) )
        
    return graphs


#def create_train_dataset_fulld_new_Ntrk_pt_weight_file_test(graphs, graph_small_example, z, k, d, edge1, edge2, weight, label, Ntracks, jet_pts, jet_ms):

#def create_train_dataset_fulld_new_Ntrk_pt_weight_file_test(graphs, graph_small_example, z, k, d, edge1, edge2, weight, label, Ntracks, jet_pts, jet_ms, kT_selection, primary_Lund_only_one_arr):
def create_train_dataset_fulld_new_Ntrk_pt_weight_file_test(graphs, graph_small_example, z, k, d, edge1, edge2, weight, label, Ntracks, jet_pts, jet_ms, kT_selection, mcweights, mcweights_out, Good_jets, signal_jet_truth_label):
#create_train_dataset_fulld_new_Ntrk_pt_weight_file_test( dataset, graph_small_example , all_lund_zs, all_lund_kts, all_lund_drs, parent1, parent2, flat_weights, labels ,N_tracks,jet_pts, jet_ms, kT_selection, mcweights,mcweights_out, Good_jets)

    
    test_bool = 1
    buildID_from_graphs = 0
    Primary_Lund_Plane = 0
    extra_node = 0
    print("extra_node condition-", extra_node)

    # loop over jets
    for i in range(len(z)):  
        #print("len(z)", len(z))
        label_out = label[i]
        mc_weight_event = mcweights[i]

        #print("label_first",label_out)

        # skip jets with less than 3 splittings
        if len(z[i])<3: 
            graphs.append(graph_small_example)
            Good_jets.append(0)
            mcweights_out.append(mc_weight_event)
            label_out = 6
            #print("label_second_1",label_out)
            continue

        #print(label[i])
        # skip jets which are not signal (1 for top and 2 for W) or background (10)
        if (label[i]!=signal_jet_truth_label) and (label[i]!=10) :
            label_out = 6
            graphs.append(graph_small_example)
            Good_jets.append(0)
            mcweights_out.append(mc_weight_event)
            #print("label_second_2",label_out)
            continue

        # label signal as 1 and background as 0
        label_out = label[i] # label_np
        if label[i]== 10:
            label_out = 0
        if label[i] == signal_jet_truth_label:
            label_out = 1

        '''
        if jet_pts[i] > 3200: continue
        if jet_pts[i] < 350: continue # . ./run.txt
        '''
        
        z_out = ak.to_numpy(z[i])
        k_out = ak.to_numpy(k[i])
        d_out = ak.to_numpy(d[i])
        
        z_out += 1e-4 
        k_out += 1e-4 
        d_out += 1e-4 
        
        z_out = np.log(1/z_out)
        k_out = np.log(k_out)
        d_out = np.log(1/d_out)
        
        
        ## lets go to do kt cut; to do this first we need to recover parentID1 and parentID2 (the ones that have a lot of -1)
        if buildID_from_graphs==1:
            edges1 = ak.to_numpy(edge1[i]) ## it's not necesary edge2[i], it has the same information
            len_edges = int(len(edges1)/2)
            edges_A = edges1[:len_edges] # sons
            edges_B = edges1[len_edges:] # parents ; then edges_B[i] > edges_A[i]
            id1_id2_edge = 0
            
            for j in range(0,len(edges_A)):
                if j == len(edges_A)-1:
                    id1_id2_edge = j + 1
                    break
                if edges_A[j+1] < edges_A[j]:
                    id1_id2_edge = j + 1
                    break
            
            edges_A_1 = edges_A[id1_id2_edge:] 
            edges_A_2 = edges_A[:id1_id2_edge] 
            edges_B_1 = edges_B[id1_id2_edge:] 
            edges_B_2 = edges_B[:id1_id2_edge]
            '''
            print("edges_A_1",edges_A_1)
            print("edges_A_2",edges_A_2)
            print("edges_B_1",edges_B_1)
            print("edges_B_2",edges_B_2)
            '''
            ## it's time to recover parentID1 (using edges_A_1 and edges_B_1) and parentID2
            parentID1 = []
            parentID2 = []
            for j in range (0,len(z[i]) ):
                if len(edges_B_1) == 0:
                    parentID1.append(-1)
                elif j == edges_A_1[0]:
                    parentID1.append(edges_B_1[0])
                    edges_A_1 = np.delete(edges_A_1,0)
                    edges_B_1 = np.delete(edges_B_1,0)
                else:
                    parentID1.append(-1)
                    
                if len(edges_B_2) == 0:
                    parentID2.append(-1)
                elif j == edges_A_2[0]:
                    parentID2.append(edges_B_2[0])
                    edges_A_2 = np.delete(edges_A_2,0)
                    edges_B_2 = np.delete(edges_B_2,0)
                else:
                    parentID2.append(-1)
                        
            ## here ID2 is the HARDEST branch!!

            # this fix should be not necessary anymore
            for j in range(0, len(parentID1)):
                if parentID1[j] == j :
                    #print("warning!")
                    parentID1[j] = -1
                if parentID2[j] == j :
                    #print("warning!")
                    parentID2[j] = -1
           
        ## I just don't want to change some lines, this mix between 1 and 2 should be remove in next version
        if buildID_from_graphs != 1:
            parentID1 = ak.to_numpy(edge2[i]) #edge2
            parentID2 = ak.to_numpy(edge1[i]) #edge1
        
        # python3 weight_class_train-Copy1.py configs/config_class_train_top.yaml        
        index_count = []
        selected_nodes = []
        index_count_out = []
        kT_Cut = kT_selection # 0.0 , 0.4 0.9, 2, 2.8 
        nodes_pass_KT = []
        node_kt_step = 0 ## used to renamed edges properly ()
        node_index = 0
        prev_cur_index = 0

        nodes_selected = []
        if Primary_Lund_Plane == 1:
            #print("ONLY PRIMARY LUND WILL BE USED!")
            nodes_primary_count = 0
            for j in range(0 , len(z[i])):
                if nodes_primary_count==0:  #j == 0 :
                    #j_ID1_next = parentID1[j]
                    j_ID1_next = parentID2[j]
                #selected_nodes = []
                #if k_out[j] <= kT_Cut : 
                if (k_out[j] <= kT_Cut): #  or j==0 or (j in parentID1) : 
                    node_kt_step += 1
                    nodes_pass_KT.append( int(node_kt_step) ) 
                    nodes_selected.append(False)
                    continue
                if nodes_primary_count>0 and j != j_ID1_next: # j>0
                    #print("222222")
                    node_kt_step += 1
                    nodes_pass_KT.append( int(node_kt_step) ) 
                    nodes_selected.append(False)
                    continue
                nodes_selected.append(True)
                nodes_primary_count +=1;
                #j_ID1_next = parentID1[j]
                j_ID1_next = parentID2[j]
                index_count.append(j)
                nodes_pass_KT.append( int(node_kt_step) ) 
                while len(index_count) > 0:
                    cur_index = index_count[-1]
                    prev_cur_index = cur_index
                    #index_1 = parentID1[cur_index]
                    index_2 = parentID2[cur_index]
                    index_count.pop()
                    '''
                    if len(graphs)==520:
                        #print("k_out[0]:", k_out[0], "  k_out[1]:", k_out[1])
                        print("cur_index:", cur_index)
                        print("index_1:", index_1, "kt(index_1)", k_out[index_1])
                        print("index_2:", index_2, "kt(index_2)", k_out[index_2])
                    '''
                    '''
                    if index_1 != -1:
                        if k_out[index_1] > kT_Cut:
                            selected_nodes.append( int(index_1) )
                            index_count_out.append( int(j))
                            node_index += 1
                            #if len(graphs)==520:
                            #    print("len(selected_nodes)inside  1:", len(selected_nodes))
                            #    print("len(index_count_out)inside 1:", len(index_count_out))
                        else:
                            index_count.append(index_1)
                    '''
                    if index_2 != -1:
                        if k_out[index_2] > kT_Cut:
                            selected_nodes.append( int(index_2) )
                            index_count_out.append( int(j))
                            node_index += 1                             
                        else:
                            index_count.append(index_2)
                    #'''
        ######################################################################################
        else:
            for j in range(0 , len(z[i])):
                if k_out[j] <= kT_Cut : 
                    node_kt_step += 1
                    nodes_pass_KT.append( int(node_kt_step) ) 
                    continue
                index_count.append(j)
                nodes_pass_KT.append( int(node_kt_step) ) 
                while len(index_count) > 0:
                    cur_index = index_count[-1]
                    prev_cur_index = cur_index
                    index_1 = parentID1[cur_index]
                    index_2 = parentID2[cur_index]
                    index_count.pop()
    
                    if index_1 != -1:
                        if k_out[index_1] > kT_Cut:
                            selected_nodes.append( int(index_1) )
                            index_count_out.append( int(j))
                            node_index += 1
                        else:
                            index_count.append(index_1)
                    if index_2 != -1:
                        if k_out[index_2] > kT_Cut:
                            selected_nodes.append( int(index_2) )
                            index_count_out.append( int(j))
                            node_index += 1 
                        else:
                            index_count.append(index_2)
        
        if len(k_out[k_out > kT_Cut]) < 1:
            graphs.append(graph_small_example)
            Good_jets.append(2)
            mcweights_out.append(mc_weight_event)
            #print("label_second_3",label_out)
            continue

        
        for j in range(0,len(index_count_out)):
            ## 1+ in order to add an extra node
            if extra_node==1:
                index_count_out[j] = int(1 + index_count_out[j] - nodes_pass_KT[index_count_out[j]] )
                selected_nodes[j] = int(1 + selected_nodes[j] - nodes_pass_KT[selected_nodes[j]] )
            else:
                index_count_out[j] = int( index_count_out[j] - nodes_pass_KT[index_count_out[j]] )
                selected_nodes[j] = int( selected_nodes[j] - nodes_pass_KT[selected_nodes[j]] )
        if extra_node==1:
            index_count_out.insert(0,0)
            selected_nodes.insert(0,1)
                
        index_count_out = np.array(index_count_out, dtype=int )
        selected_nodes = np.array(selected_nodes, dtype=int)
        
        ## kt mask for feature
        if Primary_Lund_Plane==1:
            k_mask = np.array(nodes_selected)
        if Primary_Lund_Plane==0:
            k_mask = k_out > kT_Cut
        z_out = z_out[k_mask]
        k_out = k_out[k_mask]
        d_out = d_out[k_mask]
        
        mean_z, std_z = 2.0568479032747313, 1.4450598054504056
        mean_dr, std_dr = 3.8597358364389427, 2.2748462855901073
        mean_kt, std_kt = -2.379904791478249, 2.940813577366582
        #mean_ntrks, std_ntrks = 26.556999184747827, 16.53733685428723
        mean_ntrks, std_ntrks = 57.588158609500134, 23.900100132781983
        
        z_out = (z_out - mean_z) / std_z
        k_out = (k_out - mean_kt) / std_kt
        d_out = (d_out - mean_dr) / std_dr
        Ntrk = (Ntracks[i] - mean_ntrks) / std_ntrks

        z_out = z_out.astype(float)
        k_out = k_out.astype(float)
        d_out = d_out.astype(float)
        Ntrk = Ntrk.astype(float)

        #edge = torch.tensor(np.array([edge1[i], edge2[i]]) , dtype=torch.long)
        edge_ID1 = np.concatenate((index_count_out, selected_nodes))
        edge_ID2 = np.concatenate((selected_nodes, index_count_out))
        edge = torch.tensor(np.array([edge_ID1, edge_ID2]) , dtype=torch.int64)
        #edge = np.array([edge_ID1, edge_ID2]).astype(int)


        vec = []
        ## in order to add an extra node
        #'''
        if extra_node==1:
            #print(index_count_out)
            #print(d_out)
            d_out = np.append(0, d_out)
            z_out = np.append(0, z_out)
            k_out = np.append(0, k_out)
        #'''
        vec.append(np.array([d_out, z_out, k_out]).T)
        vec = np.array(vec)
        vec = np.squeeze(vec)
        vec=torch.tensor(vec, dtype=torch.float).detach()

        graph_size = 1
        if len(k_out) == 1:
            #primary_Lund_only_one_arr.append(1)
            #continue
            #edge = torch.tensor([[0,0], [0,0]], dtype=torch.int64)
            #edge = torch.tensor([[0], [0]], dtype=torch.int64)
            edge = torch.tensor([[], []], dtype=torch.int64)
            vec = torch.unsqueeze(vec, dim=0)
            graph_size = 0


        
        if len(edge_ID1)<1:
            #primary_Lund_only_one_arr.append(1)
            #print("k_out",k_out , "  edge_ID1:", edge_ID1)
            print("2122122")
            graphs.append(graph_small_example)
            Good_jets.append(2)
            mcweights_out.append(mc_weight_event)
            #print("label_second_4",label_out)
            continue

        #print(vec.detach())
        graphs.append(Data(x= vec.detach() ,
                           #edge_index = torch.tensor(edge, dtype=torch.int64).detach(),
                           edge_index = edge.detach() ,
                           #Ntrk=torch.tensor(Ntracks[i], dtype=torch.int).detach(),
                            Ntrk=torch.tensor(Ntrk, dtype=torch.float).detach(),
                           weights= torch.tensor(weight[i], dtype=torch.float).detach(),
                           graph_size = torch.tensor(graph_size, dtype=torch.float).detach(),
                           #pt= float(jet_pts[i]) ,#torch.tensor(jet_pts[i] , dtype=torch.float).detach(),
                           mass= float(jet_ms[i]) ,#torch.tensor(jet_ms[i], dtype=torch.float).detach(),
                           y= float(label_out) ))#torch.tensor(label_out, dtype=torch.float).detach() ))
        
        Good_jets.append(1)
        mcweights_out.append(mc_weight_event)

    #print(graphs)
    return graphs



def train(loader, model, device, optimizer):
    print ("dataset size:",len(loader.dataset))
    model.train()
    loss_all = 0
    batch_counter = 0
    for data in loader:
        batch_counter+=1

        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        new_y = torch.reshape(data.y, (int(list(data.y.shape)[0]),1))
        new_w = torch.reshape(data.weights, (int(list(data.weights.shape)[0]),1)) ## add weights

        # loss = F.binary_cross_entropy(output, new_y, weight = new_w)
        loss = F.binary_cross_entropy(output, new_y, weight = new_w)
        l2_lambda = 0.01 # regularization strength
        for param in model.parameters():
            if param.dim() > 1:
                # apply L2 regularization to all parameters except biases
                loss = loss + l2_lambda * nn.MSELoss()(param, torch.zeros_like(param))

        loss.backward()

        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(loader.dataset)


def train_clas(loader, model, device, optimizer1, optimizer2, optimizer3, epoch):
    print ("dataset size:",len(loader.dataset))
    model.train()
    loss_all = 0
    batch_counter = 0
    for data in loader:
        if len(data)<1024:
            continue
        batch_counter+=1
        data = data.to(device)
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()

        output = model(data)
        new_y = torch.reshape(data.y, (int(list(data.y.shape)[0]),1))
        new_w = torch.reshape(data.weights, (int(list(data.weights.shape)[0]),1)) ## add weights

        loss = F.binary_cross_entropy(output, new_y, weight = new_w)
        loss.backward()
        loss_all += data.num_graphs * loss.item()

        if epoch < 8:
            optimizer3.step()
        elif epoch < 16:
            optimizer2.step()
        else:
            optimizer1.step()

    return loss_all / len(loader.dataset)




@torch.no_grad()
def get_accuracy(loader, model, device):
    #remember to change this when evaluating combined model
    model.eval()
    correct = 0
    for data in loader:
        cl_data = data.to(device)
        new_y = torch.reshape(cl_data.y, (int(list(cl_data.y.shape)[0]),1))
        pred = model(cl_data).max(dim=1)[1]
        correct += pred.eq(new_y[0,:]).sum().item()
    return correct / len(loader.dataset)

@torch.no_grad()
def my_test (loader, model, device):
    model.eval()
    loss_all = 0
    for data in loader:
        data = data.to(device)
        output = model(data)
        new_y = torch.reshape(data.y, (int(list(data.y.shape)[0]),1))
        new_w = torch.reshape(data.weights, (int(list(data.weights.shape)[0]),1))
        loss = F.binary_cross_entropy(output, new_y, weight=new_w)
        loss_all += data.num_graphs * loss.item()
    return loss_all/len(loader.dataset)

@torch.no_grad()
def get_scores(loader, model, device):
    model.eval()
    total_output = np.array([[1]])
    batch_counter = 0
    for data in loader:
        batch_counter+=1
        # print ("Processing batch", batch_counter, "of",len(loader))
        data = data.to(device)
        pred = model(data)
        total_output = np.append(total_output, pred.cpu().detach().numpy(), axis=0)

    return total_output[1:]
