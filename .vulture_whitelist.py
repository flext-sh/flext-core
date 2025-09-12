"""Vulture whitelist for flext-core project.

This file contains definitions for intentionally unused code that Vulture
should not report as dead code.
"""

from typing import Any

# Protocol interface parameters - these are part of interface contracts
# and are intentionally unused in abstract method definitions

# Foundation Layer Protocols
args: Any = None  # *args in Protocol __call__ methods
kwargs: Any = None  # **kwargs in Protocol __call__ methods
other: Any = None  # comparison protocol parameters
data: Any = None  # validator protocol parameter
error: Any = None  # error handler protocol parameter
name: Any = None  # attribute protocol parameters
value: Any = None  # attribute protocol parameters
obj: Any = None  # validation protocol parameters
cls: Any = None  # classmethod parameters in protocols

# Domain Layer Protocols
entity_id: Any = None  # repository protocol parameters
entity: Any = None  # repository protocol parameters
aggregate_id: Any = None  # event store protocol parameters
events: Any = None  # event store protocol parameters
expected_version: Any = None  # event store protocol parameters

# Application Layer Protocols
input_data: Any = None  # handler protocol parameters
message: Any = None  # message handler protocol parameters
message_type: Any = None  # handler type checking parameters
context: Any = None  # authorization protocol parameters
event: Any = None  # event processor protocol parameters
event_type: Any = None  # event processor type checking parameters

# Infrastructure Layer Protocols
config: Any = None  # configurable protocol parameters
service_name: Any = None  # plugin context parameters
request: Any = None  # middleware protocol parameters
operation_name: Any = None  # observability protocol parameters
exc_info: Any = None  # logger protocol parameters

# Exception handling parameters (contextlib protocols)
exc_type: Any = None  # exception type in context managers
exc_val: Any = None  # exception value in context managers
exc_tb: Any = None  # exception traceback in context managers

# Testing framework parameters
test_module: Any = None  # test module parameter in matchers
