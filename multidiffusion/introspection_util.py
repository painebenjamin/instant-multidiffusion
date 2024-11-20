# Developed by Benjamin Paine, 2024
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import sys

from typing import Any, Dict, Optional, Type, List, Callable, TypeGuard, Sequence, TYPE_CHECKING
from typing_extensions import TypedDict, NotRequired

if TYPE_CHECKING:
    import numpy as np
    import torch
    from PIL import Image

__all__ = [
    "CallSignatureParameter",
    "CallSignature",
    "get_signature",
    "realize_kwargs",
]

class NOTSET:
    """
    A sentinel value to indicate that a parameter has no default value.
    """
    pass

class CallSignatureParameter(TypedDict):
    """
    A parameter of a call signature
    """
    parameter_type: type
    required: bool
    default: Any
    description: NotRequired[str]

class CallSignature(TypedDict):
    """
    Model of a task that can be executed by a server.
    """
    parameters: Dict[str, CallSignatureParameter]
    return_type: Optional[Type[Any]]
    short_description: NotRequired[str]
    long_description: NotRequired[str]

def get_signature(method: Callable[..., Any]) -> CallSignature:
    """
    Get the signature of a method.

    Combines runtime introspection via `inspect` with
    docstring parsing via `docstring_parser`.
    """
    import inspect
    from docstring_parser import parse as parse_docstring
    from docstring_parser.common import DocstringParam

    signature = inspect.signature(method)
    docstring = parse_docstring(method.__doc__ or "")
    parameters: Dict[str, CallSignatureParameter] = {}

    for parameter_name in signature.parameters:
        if signature.parameters[parameter_name].kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            # Skip variadic arguments (e.g. *args, **kwargs)
            continue

        param_docstring: Optional[DocstringParam] = None

        for param in docstring.params:
            if param.arg_name == parameter_name:
                param_docstring = param
                break

        parameter_default = signature.parameters[parameter_name].default
        if parameter_default is inspect._empty:
            parameter_default = NOTSET

        parameter_type = signature.parameters[parameter_name].annotation
        if parameter_type is inspect._empty:
            if param_docstring is not None and param_docstring.type_name is not None:
                parameter_type = param_docstring.type_name
            else:
                parameter_type = Any

        parameters[parameter_name] = {
            "parameter_type": parameter_type,
            "required": parameter_default is NOTSET,
            "default": parameter_default,
        }

        if (
            isinstance(parameters[parameter_name]["parameter_type"], str) and
            parameters[parameter_name]["parameter_type"].startswith("Optional") # type: ignore[attr-defined]
        ):
            parameters[parameter_name]["required"] = False
            parameters[parameter_name]["parameter_type"] = parameters[parameter_name]["parameter_type"][9:-1] # type: ignore[index]

        if param_docstring is not None and param_docstring.description is not None:
            parameters[parameter_name]["description"] = param_docstring.description

    return_type = signature.return_annotation
    if return_type is inspect._empty:
        if docstring.returns is not None and docstring.returns.type_name is not None:
            return_type = docstring.returns.type_name
        else:
            return_type = Any

    signature_dict: CallSignature = {
        "parameters": parameters,
        "return_type": return_type,
    }

    if docstring.short_description:
        signature_dict["short_description"] = docstring.short_description
    if docstring.long_description:
        signature_dict["long_description"] = docstring.long_description

    return signature_dict

def realize_kwargs(
    method: Callable[..., Any],
    args: Sequence[Any],
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Takes a method and a set of arguments and keyword arguments,
    and returns them as a dictionary of keyword arguments - i.e.,
    all positional arguments are converted to their corresponding
    keyword argument names.
    """
    signature = get_signature(method)
    parameters = signature["parameters"]
    parameter_names = list(parameters.keys())
    num_parameters = len(parameters)
    realized_kwargs: Dict[str, Any] = {}

    for i, arg in enumerate(args):
        if i >= num_parameters:
            break
        parameter_name = parameter_names[i]
        realized_kwargs[parameter_name] = arg

    for parameter_name, parameter_info in parameters.items():
        if parameter_name in kwargs:
            realized_kwargs[parameter_name] = kwargs[parameter_name]

    return realized_kwargs
