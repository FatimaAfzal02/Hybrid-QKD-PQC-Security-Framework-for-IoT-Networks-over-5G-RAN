"""
src/iot/device.py
-----------------
IoT device abstraction layer.

Individual device configuration and state is managed directly within
IoTNetwork (src/iot/network.py) using per-device dicts for simplicity.
This module is reserved for future expansion if device-level logic
(e.g. firmware versioning, battery state, sensor calibration) needs
to be encapsulated into a dedicated Device class.

No class is defined here intentionally — device properties are
currently represented as plain dicts in network.py.
"""
