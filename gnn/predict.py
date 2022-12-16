import os
import haiku as hk
import jax.numpy as jnp
import jraph
import jax
from jraph._src import utils
import functools
import pathlib
import pickle
import numpy as np

num_classes = 46
enable_jit = False
hidden_dim = 512

############################################################################
# This file contains the implementation of the Graph Convolutional Network
# which is used for prediction. This file contains the `evaluate` method 
# which is called by the `predict()` methods in `run_custom.py`, 
# `Evaluation/run_instrumented_wild.py`, `Evaluation/run_llm_baseline.py`.
# This file is not used by the training procedure `gnn/train_or_eval.py` or
# the experiments on scaling across input size.
############################################################################

############################################################################
# The function for updating the hidden state of each edge
############################################################################
def edge_update_fn(edges: jnp.ndarray, sent: jnp.ndarray, receive: jnp.ndarray, glob: jnp.ndarray) -> jnp.ndarray:
  """Edge update function for graph net."""
  net = hk.Sequential(
      [hk.Linear(hidden_dim), jax.nn.relu,
       hk.Linear(hidden_dim)])
  return net(jnp.concatenate([edges, sent, receive, glob], axis=1))

############################################################################
# The function for updating the hidden state of each node
############################################################################
def node_update_fn(nodes: jnp.ndarray, sent: jnp.ndarray, receive: jnp.ndarray, glob: jnp.ndarray) -> jnp.ndarray:
  """Node update function for graph net."""
  net = hk.Sequential(
      [hk.Linear(hidden_dim), jax.nn.relu,
       hk.Linear(hidden_dim)])
  return net(jnp.concatenate([nodes, sent, receive, glob], axis=1))

############################################################################
# The function for updating the hidden state of the graph global hidden state
############################################################################
def update_global_fn(nodes: jnp.ndarray, edges: jnp.ndarray, glob: jnp.ndarray) -> jnp.ndarray:
  """Global update function for graph net."""
  global num_classes
  # Molhiv is a binary classification task, so output pos neg logits.
  net = hk.Sequential(
      [hk.Linear(hidden_dim), jax.nn.relu,
       hk.Linear(num_classes)])
  return net(jnp.concatenate([nodes, edges, glob], axis=1))

############################################################################
# The function for performing one step of message passing on the graph which
# combines all 3 functions above.
############################################################################
def net_fn(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
  """Graph net function."""
  global num_classes
  # Add a global paramater for graph classification.
  graph = graph._replace(globals=jnp.zeros([graph.n_node.shape[0], num_classes]))
  embedder = jraph.GraphMapFeatures(
      hk.Linear(hidden_dim), hk.Linear(hidden_dim), hk.Linear(hidden_dim))
  net = jraph.GraphNetwork(
      update_node_fn=node_update_fn,
      update_edge_fn=edge_update_fn,
      update_global_fn=update_global_fn,
      aggregate_nodes_for_globals_fn=utils.segment_mean,
      aggregate_edges_for_globals_fn=utils.segment_mean
  )
  return net(embedder(graph))

############################################################################
# Each graph during inference is padded to the next higher power of 2 as is
# required by the JAX framework. For each different graph size during 
# inference, a new executable has to be jitted. This padding process helps
# amortize the cost of compilation, else inference and training would take
# a very long time.
############################################################################
def _nearest_bigger_power_of_two(x: int) -> int:
  """Computes the nearest power of two greater than x for padding."""
  y = 2
  while y < x:
    y *= 2
  return y


def pad_graph_to_nearest_power_of_two(
    graphs_tuple: jraph.GraphsTuple) -> jraph.GraphsTuple:
  """Pads a batched `GraphsTuple` to the nearest power of two."""
  # Add 1 since we need at least one padding node for pad_with_graphs.
  pad_nodes_to = _nearest_bigger_power_of_two(jnp.sum(graphs_tuple.n_node)) + 1
  pad_edges_to = _nearest_bigger_power_of_two(jnp.sum(graphs_tuple.n_edge))
  # Add 1 since we need at least one padding graph for pad_with_graphs.
  # We do not pad to nearest power of two because the batch size is fixed.
  pad_graphs_to = graphs_tuple.n_node.shape[0] + 1
  return jraph.pad_with_graphs(graphs_tuple, pad_nodes_to, pad_edges_to,
                               pad_graphs_to)

############################################################################
# Computing the loss of the prediction
############################################################################
def compute_loss(params, graph, label, net):
  """Computes loss."""
  global num_classes
  pred_graph = net.apply(params, graph)

  preds = jax.nn.log_softmax(pred_graph.globals, axis=1)
  raw_preds = jax.nn.softmax(pred_graph.globals, axis=1)
  label_pos = num_classes - jnp.where(jnp.argsort(preds, axis=1)[0] == label[0])[0]
  return jnp.argmax(preds, axis=1)[0], jnp.max(raw_preds, axis=1)[0], label_pos

############################################################################
# This function is called by the `predict` methods mentioned above, and acts
# as the inferace to all other functions in this file. It returns the 
# prediction (class with highest probability score), the probability of the
# predicted class, as well as the rank of the grouth truth class.
############################################################################
def evaluate(nodes, edges, index):
  """Evaluation Script."""
  filedir = os.path.dirname(os.path.realpath(__file__))
  save_dir = "model"
  model_name = "model.pkl"
  net = hk.without_apply_rng(hk.transform(net_fn))
  with pathlib.Path(filedir, save_dir, model_name).open('rb') as fp:
    params = pickle.load(fp)

  compute_loss_fn = (
  # jax.jit
  functools.partial(compute_loss, net=net))
  if enable_jit:
    compute_loss_fn = jax.jit(compute_loss_fn)
  
 
  s = get_graph_test(nodes, edges, index)
    # print(s)
  graph, label = s

  graph = pad_graph_to_nearest_power_of_two(graph)

    # Since padding is implemented with pad_with_graphs, an extra graph has
    # been added to the batch, which means there should be an extra label.
  label = jnp.concatenate([label, jnp.array([-1])])
  prediction, confidence, label_pos = compute_loss_fn(params, graph, label)
  return prediction.item(), confidence.item(), label_pos.item()

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
# This function transforms a graph into a form understandable by JAX for 
# inference.
############################################################################
def get_graph_test(nodes, edges, index):
    graphs = []
    labels = []
    i = 0
    for b in range(1):
        sln = slice(index[i,0], index[i+1,0])
        sle = slice(index[i,1], index[i+1,1])
        label = np.asarray([index[i+1][2]])
        curGraphNodes = nodes[sln]
        curGraphEdges = edges[sle]
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
        graphs.append(gr)
        labels.append(label)
    return jraph.batch(graphs), np.concatenate(labels, axis=0)

