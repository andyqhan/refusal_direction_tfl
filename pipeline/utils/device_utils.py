"""Device utility functions for cross-platform GPU support (CUDA/MPS)."""

import torch
import logging
import subprocess
from contextlib import contextmanager
from typing import Optional

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


def get_gpu_utilization() -> Optional[float]:
    """
    Get current GPU utilization percentage using nvidia-smi.

    Returns:
        GPU utilization as a float (0-100), or None if unable to query
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split('\n')[0])
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return None


class GPUMonitor:
    """
    Monitor GPU memory and utilization during execution.

    Usage:
        monitor = GPUMonitor()
        monitor.start()
        # ... run your code ...
        monitor.print_stats()
    """

    def __init__(self):
        self.is_cuda = torch.cuda.is_available()
        self.max_memory_allocated = 0
        self.max_memory_reserved = 0
        self.max_utilization = 0
        self.initial_memory = 0

    def start(self):
        """Start monitoring GPU stats."""
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats()
            self.initial_memory = torch.cuda.memory_allocated()

    def update(self):
        """Update peak statistics."""
        if self.is_cuda:
            self.max_memory_allocated = max(
                self.max_memory_allocated,
                torch.cuda.max_memory_allocated()
            )
            self.max_memory_reserved = max(
                self.max_memory_reserved,
                torch.cuda.max_memory_reserved()
            )

            utilization = get_gpu_utilization()
            if utilization is not None:
                self.max_utilization = max(self.max_utilization, utilization)

    def print_stats(self):
        """Print GPU statistics summary."""
        if not self.is_cuda:
            return

        # Update one final time
        self.update()

        print("\n" + "="*60)
        print("GPU STATISTICS SUMMARY")
        print("="*60)

        # Memory stats
        max_mem_gb = self.max_memory_allocated / (1024**3)
        max_reserved_gb = self.max_memory_reserved / (1024**3)

        print(f"Max GPU Memory Allocated: {max_mem_gb:.2f} GB")
        print(f"Max GPU Memory Reserved:  {max_reserved_gb:.2f} GB")

        # Utilization
        if self.max_utilization > 0:
            print(f"Max GPU Utilization:      {self.max_utilization:.1f}%")
        else:
            print("Max GPU Utilization:      N/A (nvidia-smi not available)")

        # Device info
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"GPU Device:               {device_name}")

        print("="*60 + "\n")


@contextmanager
def monitor_gpu():
    """
    Context manager for monitoring GPU usage.

    Usage:
        with monitor_gpu():
            # your code here
            pass
        # GPU stats will be printed automatically
    """
    monitor = GPUMonitor()
    monitor.start()

    try:
        yield monitor
    finally:
        monitor.print_stats()
