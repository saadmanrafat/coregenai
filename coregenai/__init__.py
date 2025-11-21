import platform

if platform.python_version_tuple() < ("3", "12"):
    raise RuntimeError("coregenai requires Python 3.12 or higher")

from coregenai.config import GenConfig
from coregenai.retry import with_retry
from coregenai.interfaces import GenInterceptor
from coregenai.core import CoreGenAI, SafetyViolationError

__version__ = "0.3.2"


__all__ = [
    "GenConfig",
    "with_retry",
    "GenInterceptor",
    "CoreGenAI",
    "SafetyViolationError",
]
