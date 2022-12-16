from .instrument_nested import extract_all_codeobjects, instrument_extracted
from .event_receiver import call_all_receivers

from typing import Any, List, Union
from typing_extensions import Literal

############################################################################
# This function executes a given instrumented function by setting the 
# `py_instrument_receiver` as the function `call_all_receivers`. That means
# Whenever the instrumented code encounters a call to the function
# `py_instrument_receiver`, the interpreter would call `call_all_receivers`
# with the appropriate arguemtns.
# This function is currently not in use, please go to module_loader.py
############################################################################

def exec_instrumented(source: str) -> None:
  root_codeobject = compile(source, "<string>", "exec")
  id_to_bytecode, code_to_id = extract_all_codeobjects(root_codeobject)
  id_to_bytecode_new_codeobjects = instrument_extracted(id_to_bytecode, code_to_id)

  instrumented = id_to_bytecode_new_codeobjects[code_to_id[root_codeobject]]

  def py_instrument_receiver(stack: List[Any], opcode: Union[Literal["JUMP_TARGET"], int], arg: Any, opindex: int, code_id: int, is_post: bool) -> None:
    call_all_receivers(stack, opcode, arg, opindex, code_id, is_post, id_to_bytecode)

  exec(instrumented.to_code(), {
    "py_instrument_receiver": py_instrument_receiver
  })
