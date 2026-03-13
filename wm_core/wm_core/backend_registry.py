from __future__ import annotations

from typing import Any

from wm_core.dummy_backend import DummyPointMassBackend
from wm_core.oracle_gym_backend import OracleGymBackend

def create_backend(backend_type: str, **kwargs: Any):
    """
    Factory from wm_runtime backends. 

    Supported backend types:
        - point_mass
        - oracle_gym
    """

    if backend_type == "point_mass":
        return DummyPointMassBackend()

    if backend_type == "oracle_gym":

        env_id = kwargs.get("env_id")
        adapter = kwargs.get("adapter")

        if env_id is None:
            raise ValueError("oracle_gym backend requires env_id")

        if adapter is None:
            raise ValueError("oracle_gym backend requires adapter")

        return OracleGymBackend(
            env_id=env_id,
            adapter=adapter,
        )

    raise ValueError(f"Unsupported backend_type: {backend_type}")

