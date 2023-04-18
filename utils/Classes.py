from typing import List, Any

import numpy
from pydantic import BaseModel, validator

class TransportResults(BaseModel):
    transport_plan: numpy.ndarray #shape=(dim_source, dim_target)
    source_distribution: numpy.ndarray #shape=(dim_source,)
    target_distribution: numpy.ndarray #shape=(dim_target,)
    transported_mass: float #transported mass, sum of transport_plan

