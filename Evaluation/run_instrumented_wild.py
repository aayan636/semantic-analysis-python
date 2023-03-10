import os
import sys

parent_directory = filedir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_directory)

from absl import app
from absl import flags


from instrumentation.data_tracing_receiver import DataTracingReceiver
from instrumentation.module_loader import PatchingPathFinder
from gnn.predict import evaluate

from instrumentation.helper import IntToClassMapping, ClassToIntMapping

flags.DEFINE_integer('length', None, 'Length of the input with which to run the code snippets')
flags.DEFINE_integer('repeats', None, 'Number of times to query one code snippet (numbers can vary due to randomness)')
flags.mark_flags_as_required(['length', 'repeats'])
FLAGS = flags.FLAGS

patcher = PatchingPathFinder()
############################################################################
# After patcher.install() is called, all subsequently imported libraries  
# would be instrumented to collect the dynamic dataflow and control flow
# dependency information to which would enable the DataTracingReceiver()
# below to construct the execution graph.
############################################################################
patcher.install() 

receiver = DataTracingReceiver()

############################################################################
# This file contains input generators which would generate inputs for 
# executing codes in the Datasets/Wild folder. This dataset contains code
# which is used for testing the performance of our framework on code 
# snippets which were not observed during the training process and hence
# tests how well the generated graphs generalize to different 
# implementations, not just the ones seen during training.
############################################################################


############################################################################
# As an example, consider the generator function `testWildImplSum` below.
# We generate a random input (variable `a`) which is then fed into the 
# function under test `totest.func1`. Executing the function within the 
# context manager `receiver` results in the creation of the graph from the
# program execution, which is stored within the `receiver` variable, and 
# is processed by the `predict` function below.
############################################################################
def testWildImplSum(length):
  import Datasets.Wild.sum as totest
  import numpy as np
  a = np.random.uniform(-10, 10, length).tolist()
  with receiver:
    ans = totest.func1(a)
  return "Sum"

def testWildImplSum2(length):
  import Datasets.Wild.sum as totest
  import numpy as np
  a = np.random.uniform(-10, 10, (length, length)).tolist()
  with receiver:
    ans = totest.func2(a)
  return "Sum"

def testWildImplProd(length):
  import Datasets.Wild.prod as totest
  import numpy as np
  a = np.random.uniform(-10, 10, length).tolist()
  with receiver:
    ans = totest.func1(a)
  return "Prod"

def testWildImplProd2(length):
  import Datasets.Wild.prod as totest
  import numpy as np
  a = np.random.uniform(-10, 10, (length, length)).tolist()
  with receiver:
    ans = totest.func2(a)
  return "Prod"

def testWildImplMean(length):
  import Datasets.Wild.mean as totest
  import numpy as np
  a = np.random.uniform(-10, 10, length).tolist()
  with receiver:
    ans = totest.func1(a)
  return "Mean"

def testWildImplMean2(length):
  import Datasets.Wild.mean as totest
  import numpy as np
  a = np.random.uniform(-10, 10, (length, length)).tolist()
  with receiver:
    ans = totest.func2(a)
  return "Mean"

def testWildImplMax(length):
  import Datasets.Wild.max as totest
  import numpy as np
  a = np.random.uniform(-10, 10, length).tolist()
  with receiver:
    ans = totest.func1(a)
  return "Max"

def testWildImplMax2(length):
  import Datasets.Wild.max as totest
  import numpy as np
  a = np.random.uniform(-10, 10, (length, length)).tolist()
  with receiver:
    ans = totest.func2(a, 1)
  return "Max"

def testWildImplMin(length):
  import Datasets.Wild.min as totest
  import numpy as np
  a = np.random.uniform(-10, 10, length).tolist()
  with receiver:
    ans = totest.func1(a)
  return "Min"

def testWildImplMin2(length):
  import Datasets.Wild.min as totest
  import numpy as np
  a = np.random.uniform(-10, 10, (length, length)).tolist()
  with receiver:
    ans = totest.func2(a, 1)
  return "Min"

def testWildImplAll(length):
  import Datasets.Wild.all as totest
  import numpy as np
  a = np.random.uniform(-10, 10, length).tolist()
  a = np.asarray(a).astype(int).tolist()
  with receiver:
    ans = totest.func2(a)
  return "All"

def testWildImplAll2(length):
  import Datasets.Wild.all as totest
  import numpy as np
  a = np.random.uniform(-10, 10, length).tolist()
  a = np.asarray(a).astype(int).tolist()
  with receiver:
    ans = totest.func3(a)
  return "All"

def testWildImplAny(length):
  import Datasets.Wild.any as totest
  import numpy as np
  a = np.random.uniform(-10, 10, length).tolist()
  a = np.asarray(a).astype(int).tolist()
  with receiver:
    ans = totest.func1(a)
  return "Any"


def testWildImplAny2(length):
  import Datasets.Wild.any as totest
  import numpy as np
  a = np.random.uniform(-10, 10, length).tolist()
  a = np.asarray(a).astype(int).tolist()
  with receiver:
    ans = totest.func2(a)
  return "Any"

def testWildImplCountNonzero(length):
  import Datasets.Wild.count_nonzero as totest
  import numpy as np
  a = np.random.uniform(-10, 10, length).tolist()
  with receiver:
    ans = totest.func1(a)
  return "Count_Nonzero"

def testWildImplCountNonzero2(length):
  import Datasets.Wild.count_nonzero as totest
  import numpy as np
  a = np.random.uniform(-10, 10, (length, length)).tolist()
  with receiver:
    ans = totest.func2(a)
  return "Count_Nonzero"

def testWildImplCountNonzero3(length):
  import Datasets.Wild.count_nonzero as totest
  import numpy as np
  a = np.random.uniform(-10, 10, (length, length)).tolist()
  with receiver:
    ans = totest.func3(a)
  return "Count_Nonzero"

def testWildImplStd(length):
  import Datasets.Wild.std as totest
  import numpy as np
  a = np.random.uniform(-10, 10, length).tolist()
  with receiver:
    ans = totest.func1(a)
  return "Std"

def testWildImplCumsum(length):
  import Datasets.Wild.cumsum as totest
  import numpy as np
  a = np.random.uniform(-10, 10, length).tolist()
  with receiver:
    ans = totest.func1(a)
  return "Cumsum"

def testWildImplCumsum2(length):
  import Datasets.Wild.cumsum as totest
  import numpy as np
  a = np.random.uniform(-10, 10, length).tolist()
  with receiver:
    ans = totest.func2(a)
  return "Cumsum"

def testWildImplArgmax(length):
  import Datasets.Wild.argmax as totest
  import numpy as np
  a = np.random.uniform(-10, 10, length).tolist()
  with receiver:
    ans = totest.func1(a)
  return "Argmax"

def testWildImplArgmin(length):
  import Datasets.Wild.argmin as totest
  import numpy as np
  a = np.random.uniform(-10, 10, length).tolist()
  with receiver:
    ans = totest.func1(a)
  return "Argmin"

def testWildImplAdd(length):
  import Datasets.Wild.add as totest
  import numpy as np
  a = np.random.uniform(-10, 10, length).tolist()
  b = np.random.uniform(-10, 10, length).tolist()
  with receiver:
    ans = totest.func1(a, b)
  return "Add"

def testWildImplAdd2(length):
  import Datasets.Wild.add as totest
  import numpy as np
  a = np.random.uniform(-10, 10, (length, length)).tolist()
  b = np.random.uniform(-10, 10, (length, length)).tolist()
  with receiver:
    ans = totest.func2(a, b)
  return "Add"

def testWildImplWhere(length):
  import Datasets.Wild.where as totest
  import numpy as np
  a = np.random.choice([0,1], length).tolist()
  b = np.random.uniform(-10, 10, length).tolist()
  c = np.random.uniform(-10, 10, length).tolist()
  with receiver:
    ans = totest.func1(a, b, c)
  return "Where"

def testWildImplWher2(length):
  import Datasets.Wild.where as totest
  import numpy as np
  a = np.random.choice([0,1], length).tolist()
  b = np.random.uniform(-10, 10, length).tolist()
  c = np.random.uniform(-10, 10, length).tolist()
  with receiver:
    ans = totest.func2(a, b, c)
  return "Where"

def testWildImplDot(length):
  import Datasets.Wild.dot as totest
  import numpy as np
  a = np.random.uniform(-10, 10, length).tolist()
  b = np.random.uniform(-10, 10, length).tolist()
  with receiver:
    ans = totest.func1(a, b)
  return "Dot"

def testWildImplMatmul(length):
  import Datasets.Wild.matmul as totest
  import numpy as np
  a = np.random.uniform(-10, 10, (length, length)).tolist()
  b = np.random.uniform(-10, 10, (length, length)).tolist()
  with receiver:
    ans = totest.func1(a, b)
  return "Matmul"

def testWildImplMatmul2(length):
  import Datasets.Wild.matmul as totest
  import numpy as np
  length = length // 2
  a = np.random.uniform(-10, 10, (length, length, length)).tolist()
  b = np.random.uniform(-10, 10, (length, length, length)).tolist()
  with receiver:
    ans = totest.func2(a, b)
  return "Matmul"

def testWildImplOuter(length):
  import Datasets.Wild.outer as totest
  import numpy as np
  a = np.random.uniform(-10, 10, length).tolist()
  b = np.random.uniform(-10, 10, length).tolist()
  with receiver:
    ans = totest.func1(a, b)
  return "Outer"

def testWildImplClip(length):
  import Datasets.Wild.clip as totest
  import numpy as np
  a = np.random.uniform(-10, 10, length).tolist()
  lim1 = -5
  lim2 = 5
  with receiver:
    ans = totest.func1(a, lim1, lim2)
  return "Clip"

def testWildImplClip2(length):
  import Datasets.Wild.clip as totest
  import numpy as np
  a = np.random.uniform(-10, 10, length).tolist()
  lim1 = -5
  lim2 = 5
  with receiver:
    ans = totest.func2(a, lim1, lim2)
  return "Clip"

def testWildImplDiff(length):
  import Datasets.Wild.diff as totest
  import numpy as np
  a = np.random.uniform(-10, 10, length).tolist()
  with receiver:
    ans = totest.func1(a)
  return "Diff"

def testWildImplDiff2(length):
  import Datasets.Wild.diff as totest
  import numpy as np
  a = np.random.uniform(-10, 10, length).tolist()
  with receiver:
    ans = totest.func2(a)
  return "Diff"

def testWildImplAllclose(length):
  import Datasets.Wild.allclose as totest
  import numpy as np
  a = np.random.uniform(-10, 10, length).tolist()
  b = np.random.uniform(-10, 10, length).tolist()
  if np.random.choice([0,2]):
    b = a
  with receiver:
    ans = totest.func1(a, b)
  return "Allclose"

def testWildImplTranspose(length):
  import Datasets.Wild.transpose as totest
  import numpy as np
  a = np.random.uniform(-10, 10, size=(length, length)).tolist()
  with receiver:
    ans = totest.func1(a)
  return "Transpose"

def testWildImplTranspose2(length):
  import Datasets.Wild.transpose as totest
  import numpy as np
  a = np.random.uniform(-10, 10, size=(length, length)).tolist()
  with receiver:
    ans = totest.func2(a)
  return "Transpose"

def testWildImplRavel(length):
  import Datasets.Wild.ravel as totest
  import numpy as np
  a = np.random.uniform(-10, 10, size=(length,length)).tolist()
  with receiver:
    ans = totest.func1(a)
  return "Ravel"

def testWildImplReshape(length):
  import Datasets.Wild.reshape as totest
  import numpy as np
  a = np.random.uniform(-10, 10, size=length**2).tolist()
  if length % 2:
    shp = (length, length)
  else:
    shp = (length * 2, length // 2) 
  with receiver:
    ans = totest.func1(a, shp)
  return "Reshape"

def testWildImplSqueeze(length):
  import Datasets.Wild.squeeze as totest
  import numpy as np
  a = np.random.uniform(-10, 10, size=(1, length,1, length, 1)).tolist()
  with receiver:
    ans = totest.func1(a)
  return "Squeeze"

def testWildImplAtleast1D(_):
  import Datasets.Wild.atleast_1d as totest
  import numpy as np
  a = np.random.uniform(-10, 10, size=(1,)).tolist()
  with receiver:
    ans = totest.func1(a)
  return "Atleast_1D"

def testWildImplArange(start, stop, step):
  import Datasets.Wild.arange as totest
  import numpy as np
  with receiver:
    ans = totest.func1(start, stop, step)
  return "Arange"

def testWildImplArange2(start, stop, step):
  import Datasets.Wild.arange as totest
  import numpy as np
  with receiver:
    ans = totest.func2(start, stop, step)
  return "Arange"

def testWildImplConcatenate(length1, length2):
  import Datasets.Wild.concatenate as totest
  import numpy as np
  a = np.random.uniform(-10, 10, size=(length1, length2)).tolist()
  b = np.random.uniform(-10, 10, size=(length1, length2)).tolist()
  with receiver:
    ans = totest.func1(a, b)
  return "Concatenate"

def testWildImplVstack(length):
  import Datasets.Wild.vstack as totest
  import numpy as np
  a = np.random.uniform(-10, 10, size=(length, length)).tolist()
  b = np.random.uniform(-10, 10, size=(length, length)).tolist()
  with receiver:
    ans = totest.func1(a, b)
  return "Vstack"

def testWildImplHstack(length):
  import Datasets.Wild.hstack as totest
  import numpy as np
  a = np.random.uniform(-10, 10, size=(length, length)).tolist()
  b = np.random.uniform(-10, 10, size=(length, length)).tolist()
  with receiver:
    ans = totest.func1(a, b)
  return "Hstack"

def testWildImplTile(length):
  import Datasets.Wild.tile as totest
  import numpy as np
  a = np.random.uniform(-10, 10, size=(length, length)).tolist()
  b = np.random.randint(2, 4)
  with receiver:
    ans = totest.func1(a, b)
  return "Tile"

def testWildImplRepeat(length):
  import Datasets.Wild.repeat as totest
  import numpy as np
  a = np.random.uniform(-10, 10, size=(length, length)).tolist()
  b = np.random.randint(2, 4)
  with receiver:
    ans = totest.func1(a, b)
  return "Repeat"

def testWildImplAppend(length):
  import Datasets.Wild.append as totest
  import numpy as np
  a = np.random.uniform(-10, 10, size=(length)).tolist()
  b = np.random.randint(2, 5)
  with receiver:
    ans = totest.func1(a, b)
  return "Append"

def testWildImplInsert(length):
  import Datasets.Wild.insert as totest
  import numpy as np
  a = np.random.uniform(-10, 10, size=(length)).tolist()
  ind = np.random.randint(length)
  b = np.random.randint(2, 5)
  with receiver:
    ans = totest.func1(a, ind, b)
  return "Insert"

def testWildImplEye(_):
  import Datasets.Wild.eye as totest
  import numpy as np
  n = np.random.randint(2, 4)
  with receiver:
    ans = totest.func1(n)
  return "Eye"

def testWildImplDiag(length):
  import Datasets.Wild.diag as totest
  import numpy as np
  a = np.random.uniform(-10, 10, size=(length, length)).tolist()
  with receiver:
    ans = totest.func1(a)
  return "Diag"

def testWildImplFull(length):
  import Datasets.Wild.full as totest
  import numpy as np
  a = np.random.randint(-5, 5)
  with receiver:
    ans = totest.func1((length, length), a)
  return "Full"

def testWildImplSort(length):
  import Datasets.Wild.sort as totest
  import numpy as np
  a = np.random.random(size=length).tolist()
  with receiver:
    ans = totest.func1(a)
  return "Sort"

def testWildImplSort2(length):
  import Datasets.Wild.sort as totest
  import numpy as np
  a = np.random.random(size=length).tolist()
  with receiver:
    ans = totest.func2(a)
  return "Sort"

def testWildImplArgsort(length):
  import Datasets.Wild.argsort as totest
  import numpy as np
  a = np.random.random(size=length).tolist()
  with receiver:
    ans = totest.func1(a)
  return "Argsort"

def testWildImplPercentile(length):
  import Datasets.Wild.percentile as totest
  import numpy as np
  a = np.random.random(size=length).tolist()
  p = 25
  with receiver:
    ans = totest.func1(a, p)
  return "Percentile"

def testWildImplMedian(length):
  import Datasets.Wild.median as totest
  import numpy as np
  a = np.random.random(size=length).tolist()
  with receiver:
    ans = totest.func1(a)
  return "Median"

def testWildImplUnique(length):
  import Datasets.Wild.unique as totest
  import numpy as np
  a = np.random.randint(0, 10, size=length).tolist()
  with receiver:
    ans = totest.func1(a)
  return "Unique"

############################################################################
# This function should be called after a generator `testWildImpl*` is 
# invoked. `predict` extracts the graph and feeds it to the GNN for 
# prediction.
############################################################################
def predict(grouth_truth):
  global receiver
  import numpy as np
  #`predict` then proceeds to extract the graph from the `receiver`
  # and captures the details in `allNodeDetails`, `allEdgeDetails` and 
  # `nodeEdgeCounts`. Some tensor transformations are carried out to make 
  # the data compatible with what the ML model expects
  (allNodeDetails, allEdgeDetails, nodeEdgeCounts), times = receiver.receiverData
  # If multiple generators are called, the receiver 
  # accumulates the data of all generated graphs, which should be cleaned up
  # by calling `receiver.clear_cumulative_data()
  receiver.clear_cumulative_data()
  # Some tensor reshaping to make the data compatible with what the GCN expects.
  def flatten(lol):
    return [i for l in lol for i in l]
  allNodeDetails = np.asarray(flatten(allNodeDetails))
  allEdgeDetails = np.reshape(np.asarray(flatten(allEdgeDetails)), (-1, 4))
  # The GCN also expects the labels in order to compare the predicted label
  # with the actual label, and evaluate the ranking of the actual label
  labels = [-1, ClassToIntMapping[grouth_truth]]
  nodeEdgeCounts = np.concatenate([np.asarray(nodeEdgeCounts), np.expand_dims(np.asarray(labels), axis=1)], axis=1)
  # Calling patcher.uninstall() here so that `evaluate` does not get instrumented
  # when it is invoked.
  patcher.uninstall()
  # The evaluate function returns the predicted class, probability of the predicted
  # class, as well as the rank of the grouth truth label (ideally the ground truth
  # rank should be 0 or as close to 0 as possible.
  pred, conf, label_pos = evaluate(allNodeDetails, allEdgeDetails, nodeEdgeCounts)
  prediction = IntToClassMapping[pred]
  # Calling patcher.install() here so that subsequent `testWildImpl*` calls 
  # have their imported code snippets (code under test) instrumented.
  patcher.install()
  return prediction, conf, label_pos

############################################################################
# `main` below is a driver function which calls all of the input generators
# defined above, which populates `receiver` with the execution graph. The
# `main` function then calls `predict` which extracts the graph from 
# `receiver` and feeds it to the ML model for prediction.
############################################################################
def main(_):
  # globals() refers to all globally defined names, which includes all 
  # function definitions above. By design, all generators are defined
  # using the prefix `testWildImpl`, which gives us a way to obtain all
  # generators which need to be executed.
  tests = [i for i in globals() if "testWildImpl" in i]
  count = 0
  correct = 0
  top2_correct = 0
  top3_correct = 0
  for rep in range(FLAGS.repeats):
      for i in tests:
          # Few generators have special signatures which we define below.
          # Generally, most generators require just one field: length
          # which is the length of the input which has to be generated
          if "testWildImplArange" in i:
              start = 10
              stop = 26
              step = 3
              gt = globals()[i](start, stop, step)
              pr, confidence, label_pos = predict(gt)
          elif i == "testWildImplConcatenate":
              length1 = 10
              length2 = 15
              gt = globals()[i](length1, length2)
              pr, confidence, label_pos = predict(gt)
          else:
              length = FLAGS.length
              gt = globals()[i](length)
              pr, confidence, label_pos = predict(gt)
          print("Ground truth: ", gt)
          print("GNN prediction: ", pr)
          print("Prediction Confidence: ", confidence)
          print("True label rank:", label_pos)
          count += 1
          if gt == pr:
              correct += 1
          if label_pos <= 2:
              top2_correct += 1
          if label_pos <= 3:
              top3_correct += 1

  print("Top 1 Correct: ", correct, " Total: ", count, " Accuracy: ", correct / count)
  print("Top 2 Correct: ", top2_correct, " Total: ", count, " Accuracy: ", top2_correct / count)
  print("Top 3 Correct: ", top3_correct, " Total: ", count, " Accuracy: ", top3_correct / count)

if __name__ == '__main__':
  app.run(main)


############################################################################
# After calling patcher.uninstall, any subsequently imported libraries would
# not be instrumented, which would save on the execution time of libraries 
# not under test.
############################################################################
patcher.uninstall()
