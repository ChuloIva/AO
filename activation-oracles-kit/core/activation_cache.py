"""
Activation Cache - Manages storage and retrieval of captured activations
"""

from dataclasses import dataclass, asdict
from typing import Dict, Optional
from pathlib import Path
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class CapturedActivation:
    """Single captured activation with metadata"""
    layer: int
    token_position: int  # Absolute position in sequence
    activation: np.ndarray  # Shape: [hidden_dim]
    token_text: str
    message_idx: int  # Which message this token belongs to

    def to_dict(self) -> Dict:
        """Convert to dictionary (excluding large activation array)"""
        return {
            "layer": self.layer,
            "token_position": self.token_position,
            "token_text": self.token_text,
            "message_idx": self.message_idx,
            "activation_shape": self.activation.shape if self.activation is not None else None
        }


class ActivationCache:
    """Manages activation storage and retrieval using NPZ format"""

    def __init__(self, cache_dir: Path):
        """
        Initialize activation cache.

        Args:
            cache_dir: Directory to store activation files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, trace_id: str, layer: int) -> Path:
        """Get cache file path for a trace and layer"""
        return self.cache_dir / f"{trace_id}_layer{layer}.npz"

    def _get_metadata_path(self, trace_id: str, layer: int) -> Path:
        """Get metadata file path for a trace and layer"""
        return self.cache_dir / f"{trace_id}_layer{layer}_meta.json"

    def save_activations(
        self,
        trace_id: str,
        activations: Dict[int, CapturedActivation],
        layer: int
    ) -> Path:
        """
        Save activations to NPZ file with metadata.

        Args:
            trace_id: Unique trace identifier
            activations: Dict mapping token position -> CapturedActivation
            layer: Layer number (for filename)

        Returns:
            Path to saved NPZ file
        """
        cache_path = self._get_cache_path(trace_id, layer)
        metadata_path = self._get_metadata_path(trace_id, layer)

        # Prepare arrays for NPZ
        # Format: position_0, position_1, etc.
        arrays_dict = {}
        metadata = {}

        for pos, captured in activations.items():
            # Save activation array
            arrays_dict[f"pos_{pos}"] = captured.activation

            # Save metadata
            metadata[str(pos)] = captured.to_dict()

        # Save NPZ file
        np.savez_compressed(cache_path, **arrays_dict)

        # Save metadata as JSON
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved {len(activations)} activations to {cache_path}")
        return cache_path

    def load_activations(
        self,
        trace_id: str,
        layer: int
    ) -> Dict[int, CapturedActivation]:
        """
        Load activations from NPZ file.

        Args:
            trace_id: Unique trace identifier
            layer: Layer number

        Returns:
            Dict mapping token position -> CapturedActivation
        """
        cache_path = self._get_cache_path(trace_id, layer)
        metadata_path = self._get_metadata_path(trace_id, layer)

        if not cache_path.exists():
            logger.debug(f"Cache miss: {cache_path}")
            return {}

        try:
            # Load NPZ file
            npz_data = np.load(cache_path)

            # Load metadata
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                logger.warning(f"Metadata file missing: {metadata_path}")
                metadata = {}

            # Reconstruct CapturedActivation objects
            activations = {}

            for key in npz_data.files:
                # Extract position from key (pos_123 -> 123)
                pos = int(key.split('_')[1])
                activation_array = npz_data[key]

                # Get metadata if available
                meta = metadata.get(str(pos), {})

                activations[pos] = CapturedActivation(
                    layer=meta.get('layer', layer),
                    token_position=pos,
                    activation=activation_array,
                    token_text=meta.get('token_text', ''),
                    message_idx=meta.get('message_idx', -1)
                )

            logger.info(f"Loaded {len(activations)} activations from {cache_path}")
            return activations

        except Exception as e:
            logger.error(f"Failed to load cache from {cache_path}: {e}")
            return {}

    def has_activations(
        self,
        trace_id: str,
        positions: list[int],
        layer: int
    ) -> bool:
        """
        Check if activations exist for given positions and layer.

        Args:
            trace_id: Unique trace identifier
            positions: List of token positions
            layer: Layer number

        Returns:
            True if all positions are cached, False otherwise
        """
        cache_path = self._get_cache_path(trace_id, layer)

        if not cache_path.exists():
            return False

        try:
            # Quick check: load and verify all positions exist
            npz_data = np.load(cache_path)

            for pos in positions:
                key = f"pos_{pos}"
                if key not in npz_data.files:
                    return False

            return True

        except Exception as e:
            logger.error(f"Error checking cache: {e}")
            return False

    def clear_cache(self, trace_id: Optional[str] = None):
        """
        Clear activation cache.

        Args:
            trace_id: If provided, clear only this trace. Otherwise clear all.
        """
        if trace_id:
            # Clear specific trace
            pattern = f"{trace_id}_*.npz"
            for path in self.cache_dir.glob(pattern):
                path.unlink()
                logger.info(f"Deleted cache file: {path}")
        else:
            # Clear all
            for path in self.cache_dir.glob("*.npz"):
                path.unlink()
            for path in self.cache_dir.glob("*_meta.json"):
                path.unlink()
            logger.info("Cleared all activation cache")
