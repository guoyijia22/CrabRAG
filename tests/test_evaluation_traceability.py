from services.rag_api.evaluation.traceability import evaluation_configuration_fingerprint


def _profiles() -> list[dict]:
    return [
        {
            "id": "baseline",
            "settings": {"top_k": 2, "rerank_enabled": False},
            "summary": {"recall_at_5": 0.8},
        },
        {
            "id": "candidate",
            "settings": {"top_k": 2, "rerank_enabled": True},
            "summary": {"recall_at_5": 0.9},
        },
    ]


def test_evaluation_configuration_fingerprint_is_stable_and_excludes_results():
    first = evaluation_configuration_fingerprint(
        generation_id="gen-1",
        permission_fingerprint="permission-1",
        question_generation={"dataset_fingerprint": "dataset-1", "generated_at": "now"},
        profiles=_profiles(),
    )
    profiles = _profiles()
    profiles[0]["summary"]["recall_at_5"] = 0.1
    second = evaluation_configuration_fingerprint(
        generation_id="gen-1",
        permission_fingerprint="permission-1",
        question_generation={"generated_at": "later", "dataset_fingerprint": "dataset-1"},
        profiles=profiles,
    )

    assert first == second
    assert len(first) == 64


def test_evaluation_configuration_fingerprint_changes_with_traceable_inputs():
    base = dict(
        generation_id="gen-1",
        permission_fingerprint="permission-1",
        question_generation={"dataset_fingerprint": "dataset-1"},
        profiles=_profiles(),
    )
    original = evaluation_configuration_fingerprint(**base)

    changed_generation = evaluation_configuration_fingerprint(**{**base, "generation_id": "gen-2"})
    changed_dataset = evaluation_configuration_fingerprint(
        **{**base, "question_generation": {"dataset_fingerprint": "dataset-2"}}
    )
    profiles = _profiles()
    profiles[1]["settings"]["top_k"] = 3
    changed_settings = evaluation_configuration_fingerprint(**{**base, "profiles": profiles})

    assert len({original, changed_generation, changed_dataset, changed_settings}) == 4
