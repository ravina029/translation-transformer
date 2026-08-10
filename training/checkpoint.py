from pathlib import Path
import torch
from modelling.transformer import Transformer

def save_checkpoint(
    path: str,
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    training_loss: float,
    validation_loss: float,
) -> None:
    """
    Save the model and training state.
    """
    checkpoint_path = Path(path)

    checkpoint_path.parent.mkdir( parents=True, exist_ok=True, )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_loss": training_loss,
            "validation_loss": validation_loss,
        },
        checkpoint_path,
    )