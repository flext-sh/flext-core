"""Vulture whitelist for flext-core project.

This file contains definitions for intentionally unused code that Vulture
should not report as dead code.
"""

# Protocol interface parameters - these are part of interface contracts
# and are intentionally unused in abstract method definitions

# Foundation Layer Protocols
args  # *args in Protocol __call__ methods
kwargs  # **kwargs in Protocol __call__ methods
other  # comparison protocol parameters
data  # validator protocol parameter
error  # error handler protocol parameter
name  # attribute protocol parameters
value  # attribute protocol parameters
obj  # validation protocol parameters
cls  # classmethod parameters in protocols

# Domain Layer Protocols
entity_id  # repository protocol parameters
entity  # repository protocol parameters
aggregate_id  # event store protocol parameters
events  # event store protocol parameters
expected_version  # event store protocol parameters

# Application Layer Protocols
input_data  # handler protocol parameters
message  # message handler protocol parameters
message_type  # handler type checking parameters
context  # authorization protocol parameters
event  # event processor protocol parameters
event_type  # event processor type checking parameters

# Infrastructure Layer Protocols
config  # configurable protocol parameters
service_name  # plugin context parameters
request  # middleware protocol parameters
operation_name  # observability protocol parameters
exc_info  # logger protocol parameters

# Logger Protocol Parameters - these are standard logging interface
# All logging methods use message and **kwargs
message  # logging method parameter
