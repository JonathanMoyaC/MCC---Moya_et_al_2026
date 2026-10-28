from ueh_mesa.model_contract import audit_model_input_contract


def test_model_contract_blocks_inference_when_embeddings_are_unavailable():
    metadata = {
        "model_version": "prioria_esi_triage_v1.0",
        "selected_model": "LightGBM",
        "target": "niv_cod",
        "class_to_int": {f"C{i}": i - 1 for i in range(1, 6)},
        "features": ["AÑO", "beto_clinico_000", "beto_clinico_001"],
    }
    audit = audit_model_input_contract(
        metadata,
        ["patient_id", "arrival_time", "triage_observed"],
    )
    assert audit["compatible"] is False
    assert audit["required_feature_count"] == 3
    assert audit["missing_feature_count"] == 3
    assert audit["missing_feature_groups"] == {"AÑO": 1, "beto_clinico_*": 2}
    assert audit["class_mapping_valid"] is True


def test_model_contract_rejects_unexpected_class_mapping():
    metadata = {
        "features": ["age"],
        "class_to_int": {f"C{i}": i for i in range(1, 6)},
    }
    audit = audit_model_input_contract(metadata, ["age"])
    assert audit["missing_feature_count"] == 0
    assert audit["class_mapping_valid"] is False
    assert audit["compatible"] is False
