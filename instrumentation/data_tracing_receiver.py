
from dis import opname, opmap
from sys import version
from types import FrameType
from bytecode import Bytecode
from bytecode.instr import Compare
import inspect
from time import time
import math

from networkx.algorithms.operators import unary

from .event_receiver import EventReceiver
from .heap_object_tracking import HeapObjectTracker
from .instrument import binary_ops, unary_ops
from .util import ObjectId, get_instrumented_program_frame

from typing import Any, Dict, List, Set, Tuple, Union, Optional
from typing_extensions import Literal

from .data_tracing_variables import *

from .memory_graph_generator import *

from .helper import printDebug

############################################################################
# `arg` contains the concrete result of the comparision which was just 
# performed. If we find out that the comparision was false, we return
# the negation of the established fact.
# Example: suppose the expression compared was a < b
# Suppose the result is false
# In this case, we know that a >= b, and if the comparision was a >= b, it
# would have returned true
############################################################################
def negCompare(arg):
  negList = {
    Compare.EQ: Compare.NE,
    Compare.GE: Compare.LT,
    Compare.GT: Compare.LE,
    Compare.IN: Compare.NOT_IN,
    Compare.IS: Compare.IS_NOT
  }
  negListRev = {a: b for b, a in negList.items()}
  if arg in negList:
    return negList[arg]
  elif arg in negListRev:
    return negListRev[arg]
  else:
    raise NotImplementedError("Unknown Compare Op")
  
############################################################################
# When a function call is made, we store whether the symbolic return value 
# would be stored on the stack or not (in case the called function cannot
# be instrumented, it's symbolic value would not be on the stack after we
# return to a instrumented frame.
############################################################################
class FunctionCallHandled(object):
  return_on_stack: bool
  arg_mapping: Dict[str, Union[StackElement, List[StackElement]]]
  closure_mapping: Dict[str, SymbolicElement]

  def __init__(self, arg_mapping: Dict[str, StackElement], closure_mapping: Dict[str, SymbolicElement]) -> None:
    self.return_on_stack = False
    self.arg_mapping = arg_mapping
    self.closure_mapping = closure_mapping


class DataTracingReceiver(EventReceiver):
  # List of instrumented function call frames which are currently on the compute stack.
  function_call_stack: List[Any]
  # Store of objects assigned on the heap.
  heap_object_tracking: HeapObjectTracker
  # Store of function call frame objects.
  frame_tracking: HeapObjectTracker
  # Deprecated.
  cell_to_frame: Dict[Union[int, str, ObjectId], int]
  # Tracks whether another event is already being handled.
  already_in_receiver: bool = False
  # Stack of Symbolic Values, which is a symbolic execution counterpart to the concrete execution stack
  symbolic_stack: List[StackElement]
  # Map of function call frames which maps to the corresponding Symbolic Value
  frame_variables: Dict[Union[FrameType, int], Dict[str, Union[SymbolicElement, List[StackElement]]]]
  # Map of function call frames to Symbolic value of a cell variable in the said frame 
  # Cell variables are variables which can be accessed by an inner scope, like a nested function
  # These variables are encountered in list comprehensions, lambdas, etc
  cell_variables: Dict[Union[FrameType, int], Dict[str, SymbolicElement]]
  # Map of function call frames to Symbolic value of a free variable in the said frame 
  # Free variables are variables which are defined in a outer scope, like the parent of a nested function
  # These variables are encountered in list comprehensions, lambdas, etc
  free_variables: Dict[Union[FrameType, int], Dict[str, SymbolicElement]]
  # Map of a frame variable to closure cell (closures are created when a inner function is associated)
  # with its free variables
  closure_cells: Dict[Union[FrameType, int], Dict[str, SymbolicElement]]
  # Map of heap values of closure elements to their corresponding symbolic elements.
  # This map is used to trace the symbolic values when executing inside the scope of a nested function
  closure_heap_to_symb: Dict[HeapElement, Tuple[SymbolicElement, str]]
  # Corresponds to the block stack of the execution stack. This is currently not used
  block_stack: Dict[Union[FrameType, int], List[int]]
  # Map from variable name to Symbolic Value of any global variable. Similar to frame_variables, but for
  # global variables
  global_variables: Dict[str, SymbolicElement]
  # When a Binary_Op is called, the symbolic input variables are popped from symbolic_stack and pushed onto 
  # pre_op_stack. After the operation is completed, the pre_op_stack is popped to trace dependencies between
  # the symbolic operands and the symbolic result
  pre_op_stack: List[Union[FunctionCallHandled, Tuple[StackElement, StackElement], Tuple[StackElement, StackElement, StackElement]]]
  # Stack of the concrete function call elements
  frame_stack: List[FrameType]
  # Used to disambiguate states when a for loop is in iteration, vs when the loop ends and the 
  # symbolic value of the loop generator needs to be popped off
  pre_instrument_state_for_iter: bool = False
  # Used for debugging, time taken in the symbolic operations
  timetaken: List[float]
  # Used to trace if the symbolic object on top of stack actually corresponds to a method or a module
  set_methods: Set[HeapElement]
  # Keep track of first frame
  first_frame: Optional[FrameType]
  # Keep track of the operation executed in the previous execution of the Main event handler
  prev_op: str
  # Whether to track comparisions or not
  trace_comparisions: bool = True

  def reset_receiver(self) -> None:
    self.function_call_stack = []
    self.heap_object_tracking = HeapObjectTracker()
    self.frame_tracking = HeapObjectTracker()
    self.cell_to_frame = {}
    self.symbolic_stack = []
    self.frame_variables = {}
    self.cell_variables = {}
    self.free_variables = {}
    self.closure_cells = {}
    self.closure_heap_to_symb = {}
    self.block_stack = {}
    self.global_variables = {}
    self.pre_op_stack = []
    self.frame_stack = []
    self.timetaken = [0.0 for i in self.timetaken]
    self.set_methods = set()
    self.first_frame = None
    self.prev_op = ""
    # We do not set self.trace_comparisions here as that is controlled in the outermost loop
    set_current_heap_object_tracker(self.heap_object_tracking)

  def __init__(self) -> None:
    self.function_call_stack = []
    self.heap_object_tracking = HeapObjectTracker()
    self.frame_tracking = HeapObjectTracker()
    self.cell_to_frame = {}
    self.symbolic_stack = []
    self.frame_variables = {}
    self.cell_variables = {}
    self.free_variables = {}
    self.closure_cells = {}
    self.closure_heap_to_symb = {}
    self.block_stack = {}
    self.global_variables = {}
    self.pre_op_stack = []
    self.frame_stack = []
    self.timetaken = [0.0, 0.0]
    self.set_methods = set()
    self.first_frame = None
    self.prev_op = ""
    self.trace_comparisions = True
    set_current_heap_object_tracker(self.heap_object_tracking)
    super().__init__()

  ############################################################################
  # The following are a set of helper methods used by the main event handler
  ############################################################################

  def set_trace_comparisions(self, trace):
    self.trace_comparisions = trace

  def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    super().__exit__(exc_type, exc_val, exc_tb)
    self.receiverData = generate_memory_graph(self.trace_comparisions), self.timetaken
    set_current_heap_object_tracker(None)

  def clear_cumulative_data(self) -> None:
    clear_cumulative_graph_data()

  def stringify_maybe_object_id(self, maybe_id: Union[int, ObjectId]) -> str:
    if isinstance(maybe_id, ObjectId):
      return "obj #" + str(maybe_id.id) + " (" + str(self.heap_object_tracking.get_by_id(maybe_id.id)) + ")"
    else:
      return str(maybe_id)

  def stringify_frame_id(self, frame_id: Union[FrameType, int]) -> str:
    return "frame #" + str(frame_id)

  def convert_stack_to_heap_id(self, stack: List[Any]) -> List[HeapElement]:
    object_id_stack = []
    for elem in stack:
      object_id_stack.append(getHeapElement(elem, self.heap_object_tracking))
    return object_id_stack

  def get_var_reference_frame(self, cur_frame: FrameType, arg: Any) -> Union[FrameType, int]:
    assert False, "Not tested"
    if "cell" in arg:
      return self.frame_tracking.get_object_id(cur_frame)
    else:
      fn_object = self.heap_object_tracking.get_by_id(self.function_call_stack[-1].id)
      cell_vars = fn_object.__code__.co_cellvars
      free_vars = fn_object.__code__.co_freevars
      var_index = free_vars.index(arg["free"])
      cell = fn_object.__closure__[var_index]
      return self.cell_to_frame[self.heap_object_tracking.get_object_id(cell)]

  ##########################
  # DYNAMIC STACK OBSERVER #
  ##########################
  def on_stack_observe_event(self, stack: List[Any], opcode: Union[Literal["JUMP_TARGET"], int], is_post: bool) -> bool:
    object_id_stack = self.convert_stack_to_heap_id(stack)

    if opname[opcode] == "LOAD_METHOD":
      assert is_post
      callable_or_self_heapelem = object_id_stack[0]
      self_symbolic = self.symbolic_stack[-1]   # Stack Observer called Before Main Event Handler. 
      # In LOAD_METHOD's case, the symbolic stack only has SELF on TOS, before Main Event Handler fires.
      if self_symbolic.heap_elem == callable_or_self_heapelem: # Case where topmost element of stack is self,
        # which means this LOAD_METHOD indeed results in a function call.
        self.set_methods.add(callable_or_self_heapelem)
        return True
      else:
        return False
    if opname[opcode] == "CALL_METHOD":
      assert not is_post
      callable_or_self = object_id_stack[0]
      return callable_or_self in self.set_methods
    else:
      raise NotImplementedError("Opcode %s does not have Stack Observer Implemented"%opname[opcode])


  ######################
  # MAIN EVENT HANDLER #
  ######################
  def on_event(self, stack: List[Any], opcode: Union[Literal["JUMP_TARGET"], int], arg: Any, opindex: int, code_id: int, is_post: bool, id_to_orig_bytecode: Dict[int, Bytecode]) -> None:
    if self.already_in_receiver:
      return
    self.already_in_receiver = True
    
    st = time()
    cur_frame = get_instrumented_program_frame()
    en = time()
    self.timetaken[0] += (en - st)

    if len(self.frame_stack) == 0 or not self.frame_stack[-1] == cur_frame:
      # first time entering this instrumented frame

      # No frames observed till now; this frame is the entry instrumented frame, from which we infer program inputs/outputs
      if len(self.frame_variables.keys()) == 0:
        self.first_frame = cur_frame

      self.frame_stack.append(cur_frame)
      assert cur_frame not in self.frame_variables, "Cannot reenter a previously exited context"
      self.frame_variables[cur_frame] = {}
      self.cell_variables[cur_frame] = {}
      self.free_variables[cur_frame] = {}
      self.closure_cells[cur_frame] = {}
      self.block_stack[cur_frame] = [0]

      # frameId used for visualization only
      frameId = self.frame_tracking.get_object_id(cur_frame)

      if len(self.frame_stack) == 1 or not self.frame_stack[-2] == cur_frame.f_back:
        # this frame was called from a non-instrumented frame, or is the top-level frame,
        # so we have to populate locals without symbolic traces
        function_object = cur_frame.f_globals[cur_frame.f_code.co_name]
        parameters = dict(inspect.signature(function_object).parameters.items())
        for local, value in cur_frame.f_locals.items():
          valueHeap = getHeapElement(value, self.heap_object_tracking)
          self.frame_variables[cur_frame][local] = SymbolicElement("\'\'\'%s|%s"%("frame%d"%frameId,local), valueHeap)
          if self.first_frame == cur_frame:
            if local in parameters:
              try:
                if not parameters[local].default == value:
                  if parameters[local].kind == inspect.Parameter.VAR_POSITIONAL:
                    for el in self.frame_variables[cur_frame][local].heap_elem.collection_heap_elems:
                      set_input(el)
                  else:
                    set_input(self.frame_variables[cur_frame][local])
              except:
                pass

        assert len(cur_frame.f_code.co_freevars) == 0, "Not handled free variable function called from non-instrumented or top frame"
      else:
        # handle closures
        for free_var in cur_frame.f_code.co_freevars:
          assert free_var not in self.pre_op_stack[-1].arg_mapping.keys(), "Variable cant be free variable AND function argument"
          self.free_variables[cur_frame][free_var] = self.pre_op_stack[-1].closure_mapping[free_var]
          assert isinstance(self.pre_op_stack[-1].closure_mapping[free_var], SymbolicElement), "Expected symbolic element to handle closures"

        for name, value in self.pre_op_stack[-1].arg_mapping.items():
          if isinstance(value, StackElement):
            self.frame_variables[cur_frame][name] = SymbolicElement("\'\'\'%s|%s"%("frame%d"%frameId,name), value)
            add_dependency(frameId, self.frame_variables[cur_frame][name], value)
          else:
            # We currently set the frame variable to the raw value, when we encounter a LOAD_* instruction
            # we will have access to the list object and can lazily set the required metadata
            self.frame_variables[cur_frame][name] = value

       
        # handle default arguments
        for local, value in cur_frame.f_locals.items():
          if local not in self.frame_variables[cur_frame]:
            # TODO(shadaj): handle mutable default arguments
            valueHeap = getHeapElement(value, self.heap_object_tracking)
            self.frame_variables[cur_frame][local] = SymbolicElement("\'\'\'%s|%s"%("frame%d"%frameId,local), valueHeap)

      for cell_var in cur_frame.f_code.co_cellvars:
        if cell_var not in self.frame_variables[cur_frame]:
          self.cell_variables[cur_frame][cell_var] = SymbolicElement("\'\'\'%s|%s"%("frame%d"%frameId,cell_var), None) #Uninitialized
        else:
          self.cell_variables[cur_frame][cell_var] = self.frame_variables[cur_frame][cell_var]

    # frameId used for visualization only
    frameId = self.frame_tracking.get_object_id(cur_frame)

    if self.pre_instrument_state_for_iter == True:
      if opcode == "JUMP_TARGET" or not (opname[opcode] == "FOR_ITER" and is_post):
        self.symbolic_stack.pop() #Popping off the iterator of the for loop
      self.pre_instrument_state_for_iter = False

    if opcode == "JUMP_TARGET":
      pass
    elif opname[opcode] == "JUMP_FORWARD" or opname[opcode] == "JUMP_ABSOLUTE":
      pass
    elif opname[opcode] == "CALL_FUNCTION" or opname[opcode] == "CALL_FUNCTION_KW" or opname[opcode] == "CALL_METHOD" or opname[opcode] == "CALL_FUNCTION_EX":
      if not is_post:
        symbolic_stack_args = self.symbolic_stack[len(self.symbolic_stack) - len(stack) + 1:]
        
        if opname[opcode] == "CALL_FUNCTION_KW":
          keys = stack[-1]
          symbolic_stack_args.pop()
        else:
          keys = ()

        if opname[opcode] == "CALL_FUNCTION_EX":
          assert not(arg & 1), "Keyword arguments not supported for functions with variadic arguments"
          arguments_iterable = symbolic_stack_args.pop()

          for symbVal in arguments_iterable.heap_elem.collection_heap_elems:
            # The iterable is unpacked by the interpreter. We model the same behavior by unpacking
            # the iterable StackElements into StackElements of individual items, and push it directly
            # onto symbolic_stack_args instead of self.symbolic_stack as they get popped anyway when 
            # the function is called.
            stackElem = StackElement(symbVal)         
            symbolic_stack_args.append(stackElem)   
            add_dependency(frameId, stackElem, symbVal)

        self.symbolic_stack = self.symbolic_stack[:len(self.symbolic_stack) - len(stack)]

        function_args_id_stack = self.convert_stack_to_heap_id(stack)

        args_mapping = {}
        function_object = self.heap_object_tracking.get_by_id(function_args_id_stack[0].object_id.id)

        if hasattr(function_object, "__code__"):
          all_argument_names = list(inspect.signature(function_object).parameters.values())
          # Handling keyword arguments
          for i, key in enumerate(keys[::-1]):
            args_mapping[key] = symbolic_stack_args[len(symbolic_stack_args) - i - 1]
          # Handling positional arguments
          args_to_handle = len(symbolic_stack_args) - len(keys)
          
          arg: Optional[inspect.Parameter] = None
          for i in range(args_to_handle):
            # Compute next argument to be assigned
            if arg is None:
              arg = all_argument_names[i]
            if arg.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD or arg.kind == inspect.Parameter.POSITIONAL_ONLY:
              arg = all_argument_names[i]

            # Fill values for current argument
            if arg.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD or arg.kind == inspect.Parameter.POSITIONAL_ONLY:
              args_mapping[arg.name] = symbolic_stack_args[i]
            else:
              assert arg.kind == inspect.Parameter.VAR_POSITIONAL, "Variadic keyword arguments not supported, Keyword only arguments not expected here"
              if arg.name not in args_mapping:
                args_mapping[arg.name] = [symbolic_stack_args[i]]
              else:
                args_mapping[arg.name].append(symbolic_stack_args[i])
        else:
          for i, el in enumerate(symbolic_stack_args):
            args_mapping[i] = el
            
        if hasattr(function_args_id_stack[0], "metadata"):
          closure_dict = function_args_id_stack[0].metadata
        else:
          closure_dict = {}

        self.function_call_stack.append(function_args_id_stack[0])
        self.pre_op_stack.append(FunctionCallHandled(args_mapping, closure_dict))

        # TODO: Uninstrumentable function dependencies where None is returned:
        if function_object == list.append:
          assert len(symbolic_stack_args) == 2, "Unexpected number of function call parameters" #self and object to append
          # When list append is called, we need to trace dependency between the symbolic value of the value to be
          # appended and also the symbolic value of the collection element after the append is completed.
          mainList = symbolic_stack_args[0]
          toAppend = symbolic_stack_args[1]
          mainList.heap_elem.collection_counter += 1
          currentCount = mainList.heap_elem.collection_counter
          currentPrefix = mainList.heap_elem.collection_prefix
          toAppendSymbolic = SymbolicElement(currentPrefix%(currentCount), toAppend)
          mainList.heap_elem.collection_heap_elems.append(toAppendSymbolic)
          add_dependency(frameId, toAppendSymbolic, toAppend)
        elif function_object == list.pop:
          # Here we simply need to pop off the symbolic element of the collection corresponding to the popped value
          mainList = symbolic_stack_args[0]
          mainList.heap_elem.collection_counter -= 1
          mainList.heap_elem.collection_heap_elems.pop()
      else:
        function_args_id_stack = self.convert_stack_to_heap_id(stack)
        assert len(stack) == 1, "Expect one return value for any function"
        called_function = self.function_call_stack.pop()
        pre_op_stack_last_element = self.pre_op_stack.pop()
        return_on_stack = pre_op_stack_last_element.return_on_stack
        if return_on_stack:
          return_value_stack_el = self.symbolic_stack[0]
          assert isinstance(return_value_stack_el, StackElement), "Expected type mismatch"
        else:
          # TODO: Uninstrumentable function dependencies where return value helps infer dependencies
          if False: # Some method
            pass
          else: # Generic function
            return_value_heap = getHeapElement(stack[0],self.heap_object_tracking)
            return_value_stack_el = StackElement(return_value_heap)
            parents = []
            for el in pre_op_stack_last_element.arg_mapping.values():
              parents.append(el)
            conc_func_obj = self.heap_object_tracking.get_by_id(called_function.object_id.id)
            if conc_func_obj == bool:
              if len(parents) > 0:
                add_dependencyN(frameId, return_value_stack_el, parents, str(len(parents)) + (conc_func_obj.__name__))
            #StackElementVersion(StackElementFactory.getStackElement(function_ret_stack[0], opcode))
            self.symbolic_stack.append(return_value_stack_el)
          
    elif opname[opcode] == "RETURN_VALUE":
      self.frame_stack.pop()
      # If the first instrumented frame returns, it means the value on stack corresponds to the output of the entire program
      if cur_frame == self.first_frame:
        if self.prev_op == "BUILD_TUPLE":
          for el in self.symbolic_stack[-1].heap_elem.collection_heap_elems:
            set_output(el)
        else:
          set_output(self.symbolic_stack[-1])
      # if there is no frame on the stack, then we are at the top level, so we don't drop the return value
      if len(self.frame_stack) > 0:
        if not self.frame_stack[-1] == cur_frame.f_back:
          # this frame was called from a non-instrumented frame, so we drop the return value
          self.symbolic_stack.pop()
        else:
          self.pre_op_stack[-1].return_on_stack = True
    else:
      object_id_stack = self.convert_stack_to_heap_id(stack)

      if not is_post:
        self.check_symbolic_stack(object_id_stack, opcode)
      
      if opname[opcode] == "POP_TOP" or opname[opcode] == "POP_JUMP_IF_FALSE" or opname[opcode] == "POP_JUMP_IF_TRUE":
        if not is_post:
          self.symbolic_stack = self.symbolic_stack[:len(self.symbolic_stack) - 1]
      elif opname[opcode] == "ROT_TWO":
        tos = self.symbolic_stack.pop()
        tos1 = self.symbolic_stack.pop()
        self.symbolic_stack.append(tos)
        self.symbolic_stack.append(tos1)
      elif opname[opcode] == "ROT_THREE":
        tos = self.symbolic_stack.pop()
        tos1 = self.symbolic_stack.pop()
        tos2 = self.symbolic_stack.pop()
        self.symbolic_stack.append(tos)
        self.symbolic_stack.append(tos2)
        self.symbolic_stack.append(tos1)
      elif opname[opcode] == "ROT_FOUR":
        tos = self.symbolic_stack.pop()
        tos1 = self.symbolic_stack.pop()
        tos2 = self.symbolic_stack.pop()
        tos3 = self.symbolic_stack.pop()
        self.symbolic_stack.append(tos)
        self.symbolic_stack.append(tos3)
        self.symbolic_stack.append(tos2)
        self.symbolic_stack.append(tos1)
      elif opname[opcode] == "ROT_N":
        count = int(arg)
        assert count > 1, "Should have multiple elements for stack rotation"
        tosses = []
        for i in range(count):
          tosses.append(self.symbolic_stack.pop())
        self.symbolic_stack.append(tosses[0])
        for i in range(count-1, 0, -1):
          self.symbolic_stack.append(tosses[i])
      elif opname[opcode] == "DUP_TOP_TWO":
        tos = self.symbolic_stack.pop()
        tos2 = self.symbolic_stack.pop()
        tos_copy = tos.duplicate()
        tos2_copy = tos2.duplicate()
        self.symbolic_stack.append(tos2_copy)
        self.symbolic_stack.append(tos_copy)
        self.symbolic_stack.append(tos2)
        self.symbolic_stack.append(tos)
      elif opname[opcode] == "DUP_TOP":
        tos = self.symbolic_stack.pop()
        tos_copy = tos.duplicate()
        self.symbolic_stack.append(tos_copy)
        self.symbolic_stack.append(tos)
      elif opname[opcode] == "LOAD_CONST":
        assert is_post
        assert len(stack) == 1, "Only one const loaded at a time"
        newStackElement = StackElement(object_id_stack[0])
        newStackElement.set_const()
        self.symbolic_stack.append(newStackElement)
        # NO REAL NEED FOR A NAME HERE AS IT IS JUST A CONST
      elif opname[opcode] == "LOAD_GLOBAL":
        # TODO(shadaj): implement correctly
        
        assert is_post
        if arg not in self.global_variables:
          self.global_variables[arg] = SymbolicElement("\'%s"%(arg), object_id_stack[0])
        assert isinstance(self.global_variables[arg], SymbolicElement), "Type mismatch"
        stackVal = StackElement(self.global_variables[arg])
        self.symbolic_stack.append(stackVal)
        if stack[0] == math.inf or stack[0] == math.e or stack[0] == math.tau or stack[0] == math.pi or stack[0] == math.nan:
          stackVal.set_const()
        # StackElement has been freshly created from SymbolicElement, so the heapelem will be there
        # To ensure the const fact gets propagated, we set_const above, and do not use updating
        # function because the old value overwrites the new facts in add_dependency()
        add_dependency_nonupdating(frameId, stackVal, self.global_variables[arg])
      elif opname[opcode] == "LOAD_METHOD":
        assert is_post
        if len(stack) == 1:   # Not an object method, Stack looks like nullptr | callable. 
          # We do not model the bottom most element as Python crashes when trying to incorporate C's nullptr
          methodHeapId = object_id_stack[-1]
          selfStackElement = self.symbolic_stack.pop()
          methodStackElement = StackElement(methodHeapId)
          self.symbolic_stack.append(methodStackElement)
        else:                 # Object method, Stack looks like callable | self
          methodHeapId = object_id_stack[-2]
          selfStackElement = self.symbolic_stack.pop()
          assert isinstance(methodHeapId, HeapElement), "Type mismatch"
          methodStackElement = StackElement(methodHeapId)
          self.symbolic_stack.append(methodStackElement)
          self.symbolic_stack.append(selfStackElement)
          #TODO: Dependency between dereferenced and parent obj?
      elif opname[opcode] == "LOAD_ATTR":
        assert is_post
        selfStackElement = self.symbolic_stack.pop()
        loadedAttrHeapId = object_id_stack[-1]
        attrStackElement = StackElement(loadedAttrHeapId)
        self.symbolic_stack.append(attrStackElement)
        #TODO: Dependency between dereferenced and parent obj?
      elif opname[opcode] == "LOAD_NAME" or opname[opcode] == "LOAD_FAST":
        assert is_post
        if isinstance(self.frame_variables[cur_frame][arg], SymbolicElement):
          stackVal = StackElement(self.frame_variables[cur_frame][arg])
          self.symbolic_stack.append(stackVal)
          add_dependency(frameId, stackVal, self.frame_variables[cur_frame][arg])
          assert object_id_stack[0] == stackVal.heap_elem, "This variable got modified at an unknown position"
        else:
          # Lazy creation of SymbolicElement, StackElement for Variadic function arguments
          raw_val = self.frame_variables[cur_frame][arg]
          assert isinstance(raw_val, list)
          assert object_id_stack[0].collection_counter == len(raw_val)
          for i in range(len(raw_val)):
            assert object_id_stack[0].collection_heap_elems[i].heap_elem == raw_val[i].heap_elem, "This variable got modified at an unknown position"

          self.frame_variables[cur_frame][arg] = SymbolicElement("\'\'\'%s|%s"%("frame%d"%frameId,arg), object_id_stack[0])
          for i in range(len(raw_val)):
            add_dependency_nonupdating(frameId, self.frame_variables[cur_frame][arg], raw_val[i])
          stackVal = StackElement(self.frame_variables[cur_frame][arg])
          self.symbolic_stack.append(stackVal)
          add_dependency(frameId, stackVal, self.frame_variables[cur_frame][arg])
      elif opname[opcode] == "LOAD_DEREF":
        assert is_post
        if "cell" in arg:
          varName = arg["cell"]
          assert isinstance(self.cell_variables[cur_frame][varName], SymbolicElement), "Type mismatch"
          symbVal = self.cell_variables[cur_frame][varName]
          stackVal = StackElement(symbVal)
        elif "free" in arg:
          varName = arg["free"]
          assert isinstance(self.free_variables[cur_frame][varName], SymbolicElement), "Type mismatch"
          symbVal = self.free_variables[cur_frame][varName]
          stackVal = StackElement(symbVal)
        else:
          raise Exception("Unexpected arg type")
        self.symbolic_stack.append(stackVal)
        add_dependency(frameId, stackVal, symbVal)
        assert object_id_stack[0] == stackVal.heap_elem, "This variable got modified at an unknown position"
      elif opname[opcode] == "LOAD_CLOSURE":
        assert is_post
        closureHeapElem = object_id_stack[0]
        if "cell" in arg:
          varName = arg["cell"]
          derefSymbVal = self.cell_variables[cur_frame][varName]
        elif "free" in arg:
          varName = arg["free"]
          derefSymbVal = self.free_variables[cur_frame][varName]
        else:
          raise Exception("Unexpected arg type")
        if varName not in self.closure_cells[cur_frame]:
          self.closure_cells[cur_frame][varName] = SymbolicElement("\'\'\'%s|%s"%("frame%d"%frameId,varName), closureHeapElem)
        closureSymbVal = self.closure_cells[cur_frame][varName]
        self.closure_heap_to_symb[closureSymbVal.heap_elem] = (derefSymbVal, varName)
        closureStackEl = StackElement(closureSymbVal)
        self.symbolic_stack.append(closureStackEl)
        add_dependency(frameId, closureStackEl, closureSymbVal)
      elif opname[opcode] == "LIST_APPEND":
        assert not is_post
        appendedStackElement = self.symbolic_stack.pop()
        listStackElement = self.symbolic_stack[-arg]
        assert hasattr(listStackElement.heap_elem, "collection_heap_elems"), "Expected a symbolic collection"
        symbVal = listStackElement.heap_elem.list_append(appendedStackElement, self.heap_object_tracking)
        add_dependency(frameId, symbVal, appendedStackElement)
      elif opname[opcode] == "GET_ITER":
        assert is_post
        orig_collection = self.symbolic_stack.pop() #Popping the original collection from symbolic stack
        iteratorHeapValue = object_id_stack[0]
        iteratorStackElement = StackElement(iteratorHeapValue)
        iteratorStackElement.heap_elem.collection_heap_elems = iter(orig_collection.heap_elem.collection_heap_elems)
        self.symbolic_stack.append(iteratorStackElement)
      elif opname[opcode] == "FOR_ITER":
        if not is_post:
          self.pre_instrument_state_for_iter = True
        else:
          iteratorStackElement = self.symbolic_stack[-1]
          try:
            # THIS MAY BE PRESENT WHEN WE ITERATE ACROSS A COLLECTION (for i in list)
            iterateSymbolicValue = iteratorStackElement.heap_elem.collection_heap_elems.__next__()
            iterateStackElement = StackElement(iterateSymbolicValue)
            add_dependency(frameId, iterateStackElement, iterateSymbolicValue)
          except:
            iterateHeapValue = object_id_stack[-1]
            iterateStackElement = StackElement(iterateHeapValue)
          self.symbolic_stack.append(iterateStackElement)
          # See GET_ITER notes
      elif opname[opcode] == "BUILD_LIST" or opname[opcode] == "BUILD_SLICE" or opname[opcode] == "BUILD_TUPLE":
        assert is_post
        newListHeap = object_id_stack[0]
        newListSymStack = StackElement(newListHeap)
        # TODO TODO TODO: Constructor dependencies
        # for i in range(int(arg)):
        #   add_dependency(newListSymStack, self.symbolic_stack[- i - 1])
        toBePopped = self.symbolic_stack[len(self.symbolic_stack) - int(arg):]
        for i, stackElement in enumerate(toBePopped):
          assert newListSymStack.heap_elem.collection_heap_elems[i].heap_elem == stackElement.heap_elem, "Heap object modified unexpectedly"
          add_dependency(frameId, newListSymStack.heap_elem.collection_heap_elems[i], stackElement)
        self.symbolic_stack = self.symbolic_stack[:len(self.symbolic_stack) - int(arg)]
        self.symbolic_stack.append(newListSymStack)  
      elif opname[opcode] == "STORE_NAME" or opname[opcode] == "STORE_FAST":
        #######
        # TODO: AAYAN: IF LIST + SLICE, NEW STACKELEMENTS MAY BE INTRODUCED
        # ELSE: JUST ADD DEPENDENCY FROM LOADED SYMBOLIC LHS TO SYMBOLIC RHS
        #######
        #### IF A COLLECTION, CHANGE THE THING IT IS POINTING TO
        assert not is_post
        stackVal = self.symbolic_stack.pop()
        if arg not in self.frame_variables[cur_frame]:
          self.frame_variables[cur_frame][arg] = SymbolicElement("\'\'\'%s|%s"%("frame%d"%frameId,arg), stackVal)
        symbVal = self.frame_variables[cur_frame][arg]
        assert isinstance(symbVal, SymbolicElement), "Type mismatch"
        assert isinstance(stackVal, StackElement), "Type mismatch"
        # symbVal.heap_elem = stackVal.heap_elem ALREADY HANDLED INSIDE ADD_DEPENDENCY
        add_dependency(frameId, symbVal, stackVal)
      elif opname[opcode] == "STORE_DEREF":
        assert not is_post
        stackVal = self.symbolic_stack.pop()
        if "cell" in arg:
          varName = arg["cell"]
          assert varName in self.cell_variables[cur_frame], "Symbolic Value should be initialized when entered into a context"
          if self.cell_variables[cur_frame][varName].heap_elem is None:
            self.cell_variables[cur_frame][varName].populate(stackVal)
          symbVal = self.cell_variables[cur_frame][varName]
        elif "free" in arg:
          varName = arg["free"]
          assert varName in self.free_variables[cur_frame], "Symbolic Value should be initialized when entered into a context"
          symbVal = self.free_variables[cur_frame][varName]
          assert symbVal.heap_elem is not None, "Free variable unbounded"
        else:
          raise Exception("Unexpected arg type")
        assert isinstance(symbVal, SymbolicElement), "Type mismatch"
        assert isinstance(stackVal, StackElement), "Type mismatch"
        add_dependency(frameId, symbVal, stackVal)
      elif opname[opcode] == "BINARY_SUBSCR":
        if not is_post:
          index = self.symbolic_stack.pop()
          collection = self.symbolic_stack.pop()
          self.pre_op_stack.append((collection, index))
        else:
          collection, index = self.pre_op_stack.pop()

          #if collection.is_cow_pointer and collection.cow_latest_value and collection.cow_latest_value.collection_elems:
          if hasattr(collection.heap_elem, "collection_heap_elems"):
            #TODO: Handle case if there may be side effects caused by custom __index__ for custom objects
            if not self.heap_object_tracking.is_heap_object(index.heap_elem.object_id):
              index_reified = index.heap_elem.object_id
              nameless_symbolic_element = collection.heap_elem.collection_heap_elems[index_reified]
              #loaded_heap_element_at_index = collection.heap_elem.collection_heap_elems[index_reified]
              assert isinstance(nameless_symbolic_element, SymbolicElement)
              stackElem = StackElement(nameless_symbolic_element)
              self.symbolic_stack.append(stackElem)
              add_dependency(frameId, stackElem, nameless_symbolic_element)
              # trying out index depeendencies add_dependency_nonupdating(frameId, stackElem, index)
            else:
              index_reified = self.heap_object_tracking.get_by_id(index.heap_elem.object_id.id)
              nameless_symbolic_elements = collection.heap_elem.collection_heap_elems[index_reified]
              sliceHeap = object_id_stack[0]
              sliceStackEl = StackElement(sliceHeap)
              assert len(nameless_symbolic_elements) == len(sliceStackEl.heap_elem.collection_heap_elems), "Concrete and symbolic arrays should be of the same size"
              self.symbolic_stack.append(sliceStackEl)
              for i in range(len(nameless_symbolic_elements)):
                child = sliceStackEl.heap_elem.collection_heap_elems[i]
                parent = nameless_symbolic_elements[i]
                if not child == parent:
                  add_dependency(frameId, child, parent)
            
          else:
            raise Exception("expected collection")
            #Exception(f"Cannot store into non-cow collection: {self.stringify_maybe_object_id(collection.concrete)}")
      elif opname[opcode] == "STORE_SUBSCR":
        index = self.symbolic_stack.pop()
        collection = self.symbolic_stack.pop()
        value = self.symbolic_stack.pop()
        
        # TODO:Handle the splice case for list when list size changes. splice intoduces a new array symbolically
        if not self.heap_object_tracking.is_heap_object(index.heap_elem.object_id):
          index_reified = index.heap_elem.object_id # TODO(shadaj): handle non-integer indices
          nameless_symbolic_element = collection.heap_elem.collection_heap_elems[index_reified]
          #loaded_heap_element_at_index = collection.heap_elem.collection_heap_elems[index_reified]
          assert isinstance(nameless_symbolic_element, SymbolicElement)
          add_dependency(frameId, nameless_symbolic_element, value)
        else:
          index_reified = self.heap_object_tracking.get_by_id(index.heap_elem.object_id.id)
          nameless_symbolic_elements = collection.heap_elem.collection_heap_elems[index_reified]
          assert len(nameless_symbolic_elements) == len(value.heap_elem.collection_heap_elems), "Concrete and symbolic arrays should be of the same size"
          for i in range(len(nameless_symbolic_elements)):
            add_dependency(frameId, nameless_symbolic_elements[i], value.heap_elem.collection_heap_elems[i])
        #TODO: Store the Symbolic element in ollection?
        #symbElem = SymbolicElement(collection.heap_elem.object_id.__str__(), loaded_heap_element_at_index)
        # collection.heap_elem.collection_heap_elems[index_reified] = value.heap_elem
        
        
        #nameless_symbolic_element.heap_elem = value.heap_elem ALREADY HANDLED INSIDE ADD_DEPENDENCY
        
        
        # if collection.is_cow_pointer and collection.cow_latest_value:
        # orig_collection = collection.cow_latest_value
        # new_collection = orig_collection.collection_updated(index_reified, value)
        # collection.cow_latest_value = new_collection
        #######
        #AAYAN: IF LIST + SLICE, NEW STACKELEMENTS MAY BE INTRODUCED
        # ELSE: JUST ADD DEPENDENCY FROM LOADED SYMBOLIC LHS TO SYMBOLIC RHS
        #######
        # else:
        #   raise Exception("Cannot store into non-cow collection")
      elif opname[opcode] == "SETUP_LOOP":
        pass
      elif opname[opcode] == "UNPACK_SEQUENCE":
        assert not is_post
        numElements = int(arg)
        sequenceSym = self.symbolic_stack.pop()
        assert numElements == len(sequenceSym.heap_elem.collection_heap_elems)
        for i in range(numElements):
          symbVal = sequenceSym.heap_elem.collection_heap_elems[- i - 1]
          stackElem = StackElement(symbVal)
          assert symbVal.heap_elem == stackElem.heap_elem, "Heap object modified unexpectedly"
          self.symbolic_stack.append(stackElem)
          add_dependency(frameId, stackElem, symbVal)
      elif opname[opcode] in binary_ops:
        if not is_post:
          tos = self.symbolic_stack.pop()
          tos1 = self.symbolic_stack.pop()
          self.pre_op_stack.append((tos1, tos))
          ########
          # ONLY  IF the __op__ method is not defined on 1st argument
          ########
          # TODO: Update Unary_ops too if changed
        else:
          cur_inputs = self.pre_op_stack.pop()

          ########
          # How to do this
          ########
          # TODO: Update Unary_ops too if changed
          stackEl = StackElement(object_id_stack[0])
          self.symbolic_stack.append(stackEl)
          if isinstance(stack[-1], list) or (isinstance(stack[-1], tuple)) and (opname[opcode] == "BINARY_ADD" or opname[opcode] == "INPLACE_ADD"):
            inp_coll = cur_inputs[0].heap_elem.collection_heap_elems + cur_inputs[1].heap_elem.collection_heap_elems
            for i in range(len(inp_coll)):
              add_dependency_nonupdating(frameId, stackEl.heap_elem.collection_heap_elems[i], inp_coll[i])
          else:
            if opname[opcode] == "COMPARE_OP":
              add_dependency2(frameId, stackEl, cur_inputs[0], cur_inputs[1], binary_ops[opname[opcode]], arg if stackEl.heap_elem.object_id else negCompare(arg))
            else:
              add_dependency2(frameId, stackEl, cur_inputs[0], cur_inputs[1], binary_ops[opname[opcode]])

          
      elif opname[opcode] in unary_ops:
        if not is_post:
          tos = self.symbolic_stack.pop()
          self.pre_op_stack.append((tos,))
        else:
          cur_inputs = self.pre_op_stack.pop()
          stackEl = StackElement(object_id_stack[0])
          self.symbolic_stack.append(stackEl)
          add_dependency1(frameId, stackEl, cur_inputs[0], unary_ops[opname[opcode]])
      elif opname[opcode] == "MAKE_FUNCTION":
        if not is_post:
          tos = self.symbolic_stack.pop()
          tos1 = self.symbolic_stack.pop()
          if int(arg) != 0:
            assert int(arg) == 8, "Unhnadled case"
            tos2 = self.symbolic_stack.pop()
            self.pre_op_stack.append((tos2, tos1, tos))
          else:
            self.pre_op_stack.append((tos1, tos))
        else:
          cur_inputs = self.pre_op_stack.pop()
          stackEl = StackElement(object_id_stack[0])
          self.symbolic_stack.append(stackEl)
          if int(arg) != 0:
            assert int(arg) == 8, "Unhnadled case"
            closureListStackEl = cur_inputs[-3]
            closureCellSymbVal = closureListStackEl.heap_elem.collection_heap_elems
            assert isinstance(closureCellSymbVal, list)
            
            closure_dict = {}
            for cellSymbVal in closureCellSymbVal:
              symbVal, varName = self.closure_heap_to_symb[cellSymbVal.heap_elem]
              closure_dict[varName] = symbVal

            stackEl.heap_elem.metadata = closure_dict
            
            add_dependency3(frameId, stackEl, cur_inputs[0], cur_inputs[1], cur_inputs[2])
          else:
            add_dependency2(frameId, stackEl, cur_inputs[0], cur_inputs[1])
      else:
        raise NotImplementedError(opname[opcode])
      if is_post:
        self.check_symbolic_stack(object_id_stack, opcode)
    self.already_in_receiver = False
    self.prev_op = opname[opcode] if opcode != "JUMP_TARGET" else self.prev_op 

  ############################################################################
  # This function ensures that the symbolic stack and the concrete stack are
  # Always in agreement (the concrete value of the symbolic stack must 
  # always point to the same concrete object as on the concrete stack)
  ############################################################################

  def check_symbolic_stack(self, object_id_stack: List[Any], opcode: int) -> None:
    
    printDebug(opname[opcode])
    printDebug("symbolic:", [self.stringify_maybe_object_id(e.heap_elem.object_id) for e in self.symbolic_stack])
    printDebug("concrete:", [self.stringify_maybe_object_id(e.object_id) for e in object_id_stack])
    for i, e in enumerate(object_id_stack):
      index_from_end = i - len(object_id_stack)
      try:
        if not self.symbolic_stack[index_from_end].heap_elem.object_id == e.object_id:
          printDebug(opname[opcode])
          printDebug("symbolic:", [self.stringify_maybe_object_id(e.heap_elem.object_id) for e in self.symbolic_stack])
          printDebug("concrete:", [self.stringify_maybe_object_id(e.object_id) for e in object_id_stack])
          raise Exception(
            "Stack element " + str(i) + " is symbolically " + \
              self.stringify_maybe_object_id(self.symbolic_stack[index_from_end].heap_elem.object_id) + \
                " but concretely " + self.stringify_maybe_object_id(e.object_id))
      except IndexError:
        print(opname[opcode])
        print("symbolic:", [self.stringify_maybe_object_id(e.heap_elem.object_id) for e in self.symbolic_stack])
        print("concrete:", [self.stringify_maybe_object_id(e.object_id) for e in object_id_stack])
        raise Exception("Stack element at index " + str(i) + " is not in symbolic stack")
    
    
