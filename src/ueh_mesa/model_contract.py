"""Input-contract checks for pretrained models; never performs inference."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping


EXPECTED_ESI_CLASSES = {f"C{i}": i - 1 for i in range(1, 6)}


def audit_model_input_contract(
    metadata: Mapping[str, object],
    available_features: Iterable[str],
) -> dict[str, object]:
    """Compare model metadata features with fields available to a generator.

    The function reports compatibility only. It does not load serialized
    artifacts, transform inputs, or call a model's prediction methods.
    """
    raw_features = metadata.get("features")
    if not isinstance(raw_features, list) or not raw_features:
        raise ValueError("Los metadatos deben declarar una lista no vacía de features.")
    features = [str(feature) for feature in raw_features]
    if len(features) != len(set(features)):
        raise ValueError("Los metadatos contienen features duplicadas.")
    available = {str(feature) for feature in available_features}
    missing = [feature for feature in features if feature not in available]
    groups = Counter(
        "beto_clinico_*" if feature.startswith("beto_clinico_") else feature
        for feature in missing
    )
    class_map = metadata.get("class_to_int")
    classes_valid = isinstance(class_map, Mapping) and dict(class_map) == EXPECTED_ESI_CLASSES
    return {
        "model_version": metadata.get("model_version"),
        "selected_model": metadata.get("selected_model"),
        "target": metadata.get("target"),
        "required_feature_count": len(features),
        "available_feature_count": len(available),
        "missing_feature_count": len(missing),
        "missing_feature_preview": missing[:10],
        "missing_feature_groups": dict(sorted(groups.items())),
        "class_mapping_valid": classes_valid,
        "classes": list(class_map) if isinstance(class_map, Mapping) else [],
        "compatible": not missing and classes_valid,
    }
