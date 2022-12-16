from instrumentation.data_tracing_receiver import DataTracingReceiver
from instrumentation.module_loader import PatchingPathFinder
from gnn.predict import evaluate

from instrumentation.helper import IntToClassMapping

patcher = PatchingPathFinder()
patcher.install()

receiver = DataTracingReceiver()

############################################################################
# This function should be called after a generator `testWildImpl` is 
# invoked. `predict` extracts the graph and feeds it to the GNN for 
# prediction.
############################################################################
def predict():
  global receiver
  import numpy as np
  #`predict` then proceeds to extract the graph from the `receiver`
  # and captures the details in `allNodeDetails`, `allEdgeDetails` and 
  # `nodeEdgeCounts`. Some tensor transformations are carried out to make 
  # the data compatible with what the ML model expects
  (allNodeDetails, allEdgeDetails, nodeEdgeCounts), times = receiver.receiverData
  # Some tensor reshaping to make the data compatible with what the GCN expects.
  def flatten(lol):
    return [i for l in lol for i in l]
  allNodeDetails = np.asarray(flatten(allNodeDetails))
  allEdgeDetails = np.reshape(np.asarray(flatten(allEdgeDetails)), (-1, 4))
  # The GCN also expects the labels in order to compare the predicted label
  # with the actual label, and evaluate the ranking of the actual label
  # Since we do not know the label beforehand, we just give a dummy here and
  # ignore it's score
  labels = [-1, 0]
  nodeEdgeCounts = np.concatenate([np.asarray(nodeEdgeCounts), np.expand_dims(np.asarray(labels), axis=1)], axis=1)
  # Calling patcher.uninstall() here so that `evaluate` does not get instrumented
  # when it is invoked.
  patcher.uninstall()
  print("GNN Predicts this as: ", IntToClassMapping[evaluate(allNodeDetails, allEdgeDetails, nodeEdgeCounts)[0]])
  # Calling patcher.install() here so that subsequent queries 
  # have their imported code snippets (code under test) instrumented.
  patcher.install()
  

############################################################################
# Consider the generator function `testWildImpl` below.
# We generate a random input (variable `a`) which is then fed into the 
# function under test `custom.my_function`. Executing the function within the 
# context manager `receiver` results in the creation of the graph from the
# program execution, which is stored within the `receiver` variable, and 
# is processed by the `predict` function below.
############################################################################
def testWildImpl():
  import custom
  import numpy as np

  # Generate an input
  length = np.random.randint(3, 10)
  a = np.random.random(size=length).tolist()

  with receiver:
    # Call the custom function within the context manager
    ans = custom.my_function(a)
  predict()

testWildImpl()

patcher.uninstall()
