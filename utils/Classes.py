# This file contains the classes used in the project.
from typing import Optional
import numpy as np
from pydantic import BaseModel


class TransportResults(BaseModel):
    transport_plan: np.ndarray  # shape=(dim_source, dim_target)
    source_distribution: np.ndarray  # shape=(dim_source,)
    target_distribution: np.ndarray  # shape=(dim_target,)
    transported_mass: float  # transported mass, sum of transport_plan, the same as cost.
    lift_parameter: Optional[float] = None  # lift, the amount by which we lifted the distributions
    Pos_plan: Optional[np.ndarray] = None  # shape=(dim_source, dim_target)
    Neg_plan: Optional[np.ndarray] = None  # shape=(dim_source, dim_target)

    class Config:
        arbitrary_types_allowed = True
