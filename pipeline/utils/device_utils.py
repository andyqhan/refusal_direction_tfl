"""Device utility functions for cross-platform GPU support (CUDA/MPS)."""

import torch
import logging

logger = logging.getLogger(__name__)


def get_optimal_device() -> str:
    """
    Automatically detect the best available device.

    Priority:
    1. CUDA (NVIDIA GPUs)
    2. MPS (Apple Silicon)
    3. Raise exception if neither available

    Returns:
        str: Device string ("cuda" or "mps")

    Raises:
        RuntimeError: If neither CUDA nor MPS is available
    """
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Using CUDA device: {device_name}")
        return device
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using MPS (Metal Performance Shaders) device")
        return device
    else:
        raise RuntimeError(
            "No GPU acceleration available. This codebase requires either:\n"
            "  - CUDA (NVIDIA GPU) on Linux/Windows\n"
            "  - MPS (Apple Silicon) on macOS\n"
            "Please ensure you have the appropriate hardware and PyTorch installation."
        )


def clear_device_cache(device: torch.device) -> None:
    """
    Clear memory cache for the given device type.

    Args:
        device: PyTorch device object (can be from model.device or tensor.device)
    """
    device_type = device.type if hasattr(device, 'type') else str(device).split(':')[0]

    if device_type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif device_type == 'mps':
        torch.mps.empty_cache()
        # Note: MPS doesn't have synchronize() as of PyTorch 2.x


def get_computation_dtype(device: torch.device) -> torch.dtype:
    """
    Get appropriate high-precision dtype for the given device.

    MPS (Apple Silicon) doesn't support float64, so we use float32 instead.
    CUDA and CPU can use float64 for better numerical precision.

    Args:
        device: PyTorch device object (can be from model.device or tensor.device)

    Returns:
        torch.float64 for CUDA/CPU (full precision)
        torch.float32 for MPS (MPS limitation)
    """
    device_type = device.type if hasattr(device, 'type') else str(device).split(':')[0]

    if device_type == 'mps':
        logger.warning(
            "MPS device detected: using float32 instead of float64 for high-precision computations. "
            "This may slightly affect numerical precision compared to CUDA devices."
        )
        return torch.float32
    else:
        return torch.float64
