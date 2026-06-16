from __future__ import annotations

import logging


logger = logging.getLogger(__name__)


def preferred_torch_device(cuda_enabled: bool, component_name: str) -> str:
    if not cuda_enabled:
        logger.info("%s using CPU because CUDA=FALSE.", component_name)
        return "cpu"

    try:
        import torch
    except Exception:
        logger.warning(
            "%s using CPU because PyTorch is unavailable.",
            component_name,
        )
        return "cpu"

    try:
        if torch.cuda.is_available():
            device_name = _cuda_device_name(torch)
            logger.info(
                "%s using CUDA%s.",
                component_name,
                f" device {device_name}" if device_name else "",
            )
            return "cuda"
    except Exception:
        logger.exception(
            "%s CUDA availability check failed; falling back to CPU.",
            component_name,
        )
        return "cpu"

    logger.warning(
        "%s using CPU because CUDA=TRUE but no CUDA device is visible. "
        "If this is running in Docker, check the NVIDIA driver, NVIDIA "
        "Container Toolkit, and GPU exposure for the container.",
        component_name,
    )
    return "cpu"


def _cuda_device_name(torch_module: object) -> str:
    try:
        cuda = getattr(torch_module, "cuda")
        return str(cuda.get_device_name(0))
    except Exception:
        return ""
