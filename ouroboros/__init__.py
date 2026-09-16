"""Ouroboros model-dialect adaptive Agent evolution primitives."""

__version__ = "0.1.0"

from .profile import RuntimeProfile, bind_genome, branch_key
from .pydantic_adapter import PydanticAIAgentAdapter

__all__ = ["RuntimeProfile", "bind_genome", "branch_key", "PydanticAIAgentAdapter"]
