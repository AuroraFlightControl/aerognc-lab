from tqdm import tqdm
from src.aerognc.core.state import VehicleState

class SimulationProgressBar:
    def __init__(self, total_steps: int, description: str = "Simulating"):
        self.pbar = tqdm(total=total_steps, desc=description, unit="steps")

    def __call__(self, time_s: float, state: VehicleState) -> str | None:
        self.pbar.update(1)
        return None

    def close(self) -> None:
        self.pbar.close()