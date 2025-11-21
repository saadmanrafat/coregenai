import platform
import shutil
from pathlib import Path

if platform.python_version_tuple() < ("3", "12"):
    raise RuntimeError("coregenai requires Python 3.12 or higher")

if not (Path(__file__).parent.parent / ".env").exists():
    shutil.copy(
        Path(__file__).parent.parent / ".env.example",
        Path(__file__).parent.parent / ".env",
    )
    print(
        "A default .env file has been automatically created. \n"
        "Please review and update it with your configuration."
    )


from coregenai.config import GenConfig
from coregenai.retry import with_retry
from coregenai.interfaces import GenInterceptor
from coregenai.core import CoreGenAI, SafetyViolationError

__version__ = "0.3.1"


__all__ = [
    "GenConfig",
    "with_retry",
    "GenInterceptor",
    "CoreGenAI",
    "SafetyViolationError",
]
