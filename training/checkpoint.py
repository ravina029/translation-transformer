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

def load_checkpoint(
    path: str,
    model: Transformer,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, int | float]:
    """
    Load a saved Transformer checkpoint. The optimizer is restored only when it is provided.
    """
    checkpoint_path = Path(path)

    if not checkpoint_path.exists():
        raise FileNotFoundError( f"checkpoint not found: {checkpoint_path}" )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict( checkpoint["model_state_dict"] )
    model.to(device)

    if optimizer is not None:
        optimizer.load_state_dict( checkpoint["optimizer_state_dict"] )

    return {
        "epoch": checkpoint["epoch"],
        "training_loss": checkpoint["training_loss"],
        "validation_loss": checkpoint["validation_loss"],
    }


