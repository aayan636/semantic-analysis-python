import jraph
import numpy as np
from pathlib import Path

partitions = 15
samples_per_partition = None
batch_size = None

nodes = []
edges = []
index = []

############################################################################
# This function is called to load the training dataset into memory as well 
# as set some hyperparameters used during training
############################################################################
def read_train_data(data_path, bsize):
    global nodes, edges, index, samples_per_partition, batch_size
    batch_size = bsize
    path = Path(data_path if data_path is not None else "")
    for i in range(partitions):
        nodes.append(np.load(path / 'train' / str(i+1) / 'nodestrain.npy'))
        edges.append(np.load(path / 'train' / str(i+1) / 'edgestrain.npy'))
        index.append(np.load(path / 'train' / str(i+1) / 'indextrain.npy'))
    samples_per_partition = index[dataset_i].shape[0]-1
    return samples_per_partition * partitions

nodes_ = []
edges_ = []
index_ = []
############################################################################
# This function is called to load the selected testing dataset into memory 
# as well as set some hyperparameters used during testing
############################################################################
def read_eval_data(testMode, data_path):
    global nodes_, edges_, index_, samples_per_partition
    path = Path(data_path if data_path is not None else "")
    for i in range(partitions):
        nodes_.append(np.load(path / testMode / str(i+1) / ('nodes%s.npy'%testMode)))
        edges_.append(np.load(path / testMode / str(i+1) / ('edges%s.npy'%testMode)))
        index_.append(np.load(path / testMode / str(i+1) / ('index%s.npy'%testMode)))
    samples_per_partition = index_[dataset_i].shape[0]-1

    

i = 0
dataset_i = 0
evalMode = False
epoch = 0

############################################################################
# These functions are used to set some counters which keep track of how 
# many epochs have passed, which dataset partition is currently being 
# iterated over.
############################################################################
def reset():
    global i, dataset_i
    i = 0
    dataset_i = 0

def setEval():
    global evalMode
    evalMode = True
############################################################################
# These functions are some basic tensor manipulations required to ensure the
# dataset tensors are always in the shape as expected by the ML model.
############################################################################
def getnparray(raw):
    ret = np.atleast_2d(np.asarray(raw))
    if ret.size == 0:
        return np.zeros(shape=(0, 1))
    else:
        return ret

def getnparray2(raw):
    ret = np.asarray(raw, dtype=np.int64)
    if ret.size == 0:
        return np.zeros(shape=(0), dtype=np.int64)
    else:
        return ret
############################################################################
# These functions fetch a batch of graphs from the dataset for training or
# for testing purposes.
############################################################################
def get_graph():
    global i, evalMode, dataset_i, epoch
    graphs = []
    labels = []
    for b in range(batch_size):
        # The index tensor of the dataset contains which indices
        # to extract from the nodes and the edges tensors for one 
        # particular datapoint.
        # The index tensor also has stored the ground truth label.
        sln = slice(index[dataset_i][i,0], index[dataset_i][i+1,0])
        sle = slice(index[dataset_i][i,1], index[dataset_i][i+1,1])
        label = np.asarray([index[dataset_i][i+1][2]])
        curGraphNodes = nodes[dataset_i][sln]
        curGraphEdges = edges[dataset_i][sle]
        # Used if evalMode is true. Deprecated currently and during evaluation,
        # `get_graph_test()` is used to fetch the data.
        if evalMode and not (i + 1) % (index[dataset_i].shape[0]-1):
            return None
        # Some tensor manipulation to make the data compatible with the format
        # expected by the JRAPH library. The edge connectivity data is modified
        # into the variables `all_receivers` and `all_senders`, while other
        # features are kept as they are (features containing IDs are removed 
        # for edges and nodes) 
        all_nodes = curGraphNodes[:,2:].astype('float64')
        all_edges = curGraphEdges[:,3:].astype('float64')
        all_senders = np.atleast_1d(curGraphEdges.astype('int64')[:,0].squeeze())
        all_receivers = np.atleast_1d(curGraphEdges.astype('int64')[:,1].squeeze())
        red_nodes = np.asarray([i for i in all_nodes]) # if i[-1] == 0])
        red_edges, red_sender, red_receiver = [], [], []
        for edge, sender, receiver in zip(all_edges, all_senders, all_receivers):
                red_edges.append(edge)
                red_sender.append(sender)
                red_receiver.append(receiver)
        # Edge case tensor manipulations to ensure compatibilty with degenrate tensors
        # (example graphs with no edges)
        red_edges = getnparray(red_edges)
        red_sender = getnparray2(red_sender)
        red_receiver = getnparray2(red_receiver)
        n_node = np.array([red_nodes.shape[0]]).astype('int64')
        n_edge = np.array([red_edges.shape[0]]).astype('int64')
        # Creating the object which can be processed by the JRAPH library
        gr = jraph.GraphsTuple(
            nodes = red_nodes,
            edges = red_edges,
            n_node = n_node,
            n_edge = n_edge,
            senders = red_sender,
            receivers = red_receiver,
            globals = {}
        )
        # Counters which contain information of current dataset partition, epoch, etc.
        i = (i + 1) % (index[dataset_i].shape[0]-1)
        if i == 0:
            dataset_i = (dataset_i + 1) % partitions
            if dataset_i == 0:
                epoch += 1
                # When all datapoints are finished, one epoch is completed,
                # we roll over to the beginning and start a new epoch
                print("######################\nNEW EPOCH: ", epoch)
            print("Starting New Dataset Partition ", dataset_i)
        graphs.append(gr)
        labels.append(label)
    return jraph.batch(graphs), np.concatenate(labels, axis=0)

def get_graph_test():
    global i, evalMode, dataset_i
    graphs = []
    labels = []
    for b in range(1):
        # The index tensor of the dataset contains which indices
        # to extract from the nodes and the edges tensors for one 
        # particular datapoint.
        # The index tensor also has stored the ground truth label.
        sln = slice(index_[dataset_i][i,0], index_[dataset_i][i+1,0])
        sle = slice(index_[dataset_i][i,1], index_[dataset_i][i+1,1])
        label = np.asarray([index_[dataset_i][i+1][2]])
        curGraphNodes = nodes_[dataset_i][sln]
        curGraphEdges = edges_[dataset_i][sle]
        # Some tensor manipulation to make the data compatible with the format
        # expected by the JRAPH library. The edge connectivity data is modified
        # into the variables `all_receivers` and `all_senders`, while other
        # features are kept as they are (features containing IDs are removed 
        # for edges and nodes)
        all_nodes = curGraphNodes[:,2:].astype('float64')
        all_edges = curGraphEdges[:,3:].astype('float64')
        all_senders = np.atleast_1d(curGraphEdges.astype('int64')[:,0].squeeze())
        all_receivers = np.atleast_1d(curGraphEdges.astype('int64')[:,1].squeeze())
        red_nodes = np.asarray([i for i in all_nodes]) # if i[-1] == 0])
        red_edges, red_sender, red_receiver = [], [], []
        for edge, sender, receiver in zip(all_edges, all_senders, all_receivers):
            #if edge[-1] == 0:
                red_edges.append(edge)
                red_sender.append(sender)
                red_receiver.append(receiver)
        # Edge case tensor manipulations to ensure compatibilty with degenrate tensors
        # (example graphs with no edges)
        red_edges = getnparray(red_edges)
        red_sender = getnparray2(red_sender)
        red_receiver = getnparray2(red_receiver)
        n_node = np.array([red_nodes.shape[0]]).astype('int64')
        n_edge = np.array([red_edges.shape[0]]).astype('int64')
        # Creating the object which can be processed by the JRAPH library
        gr = jraph.GraphsTuple(
            nodes = red_nodes,
            edges = red_edges,
            n_node = n_node,
            n_edge = n_edge,
            senders = red_sender,
            receivers = red_receiver,
            globals = {}
        )
        # Counters which contain information of current dataset partition, epoch, etc.
        i = (i + 1) % (index_[dataset_i].shape[0]-1)
        if i == 0:
            dataset_i = dataset_i + 1
            if dataset_i == partitions:
                # For the testing dataset, unlike training dataset, once all the 
                # partitions have been explored, we do not need to roll over and can
                # finish the loop
                print("Finished all partitions")
                return None, None
            print("Starting New Dataset Partition ", dataset_i)
        graphs.append(gr)
        labels.append(label)
    return jraph.batch(graphs), np.concatenate(labels, axis=0)

