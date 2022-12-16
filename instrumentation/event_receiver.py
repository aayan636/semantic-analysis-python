from bytecode import Bytecode

from typing import Any, Callable, Dict, List, Optional, Union
from typing_extensions import Literal

############################################################################
# Interface object for an Event Receiver (of which DataTracingReceiver is an
# instance.
############################################################################

class EventReceiver(object):
  current_exit_func: Optional[Callable[[], None]] = None
  def on_event(self, stack: List[Any], opcode: Union[Literal["JUMP_TARGET"], int], arg: Any, opindex: int, code_id: int, is_post: bool, id_to_orig_bytecode: Dict[int, Bytecode]) -> None:
    pass

  def on_stack_observe_event(self, stack: List[Any], opcode: Union[Literal["JUMP_TARGET"], int], is_post: bool) -> bool:
    pass

  # This method is called when a context manager block with the receiver is initialized
  # Look out for `with receiver` in run_instrumented.py, run_custom.py etc
  # This method may be redefined inside the actual Receiver implementation
  def __enter__(self) -> None:
    assert self.current_exit_func is None
    self.reset_receiver()
    self.current_exit_func = add_receiver(self)
  
  # This method is called when a context manager block with the receiver exits
  # Look out for `with receiver` in run_instrumented.py, run_custom.py etc
  # This method may be redefined inside the actual Receiver implementation
  def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    assert self.current_exit_func is not None
    self.current_exit_func()
    self.current_exit_func = None

  # This method has to be called to reset all variables which the user may not want persisting across multiple calls
  # to receiver (basically when generating multiple graphs while generating the training/testing dataset
  def reset_receiver(self) -> None:
    pass

_active_receivers: List[EventReceiver] = []

def add_receiver(receiver: EventReceiver) -> Callable[[], None]:
  _active_receivers.insert(0, receiver)
  return lambda: _active_receivers.remove(receiver)

def call_all_stack_observers(stack: List[Any], opcode: Union[Literal["JUMP_TARGET"], int], is_post: bool) -> None:
  assert len(_active_receivers) == 1, "Currently only one receiver with a stack observer supported."
  receiver = _active_receivers[0]
  return receiver.on_stack_observe_event(stack, opcode, is_post)


def call_all_receivers(stack: List[Any], opcode: Union[Literal["JUMP_TARGET"], int], arg: Any, opindex: int, code_id: int, is_post: bool, id_to_orig_bytecode: Dict[int, Bytecode]) -> None:
  for receiver in _active_receivers:
    receiver.on_event(stack, opcode, arg, opindex, code_id, is_post, id_to_orig_bytecode)
