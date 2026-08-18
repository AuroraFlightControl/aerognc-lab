from dataclasses import dataclass, field
from typing import Dict
import pandas as pd


@dataclass
class ControllerStatus:
    timestamp_s: float
    name: str
    is_healthy: bool
    active_mode: str
    telemetry: Dict[str, float] = field(default_factory=dict) 


def package_Status(history: list[ControllerStatus]) -> pd.DataFrame:

    flat_data = []
    
    for msg in history:
        # Create the standard header columns
        row = {
            "time_s": msg.timestamp_s,
            "controller_name": msg.name,
            "is_healthy": msg.is_healthy,
            "active_mode": msg.active_mode
        }
        # Merge the generic telemetry dictionary directly into the row
        row.update(msg.telemetry)
        flat_data.append(row)
        
    df = pd.DataFrame(flat_data)
    df.set_index("time_s", inplace=True)
    return df