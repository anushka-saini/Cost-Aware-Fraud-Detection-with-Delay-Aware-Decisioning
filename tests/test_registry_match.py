"""The API must serve exactly what the registry says is deployed: the same
models, the same calibrator and the same thresholds."""

import joblib
import numpy as np
import pandas as pd
from conftest import MODEL_NAME, needs_model

from fraud.models import predict_fraud_proba
from fraud.registry import _registry_root, list_versions
from fraud.service.core import get_scorer

pytestmark = needs_model


def load_bundle_file(version):
    return joblib.load(_registry_root() / MODEL_NAME / version / "bundle.joblib")


def test_api_loads_the_registry_bundle_exactly(client):
    scorer = get_scorer()
    assert scorer.version in list_versions(MODEL_NAME)
    on_disk = load_bundle_file(scorer.version)

    # Thresholds and feature list: the same values.
    assert scorer.t_review == on_disk["thresholds"]["t_review"]
    assert scorer.t_block == on_disk["thresholds"]["t_block"]
    assert scorer.feature_columns == list(on_disk["feature_columns"])

    # Calibrator: the same output for a spread of raw scores.
    raw = np.geomspace(1e-7, 1 - 1e-7, 50)
    assert np.array_equal(scorer.calibrator.transform(raw), on_disk["calibrator"].transform(raw))

    # Models: the same number of them, and the same score for a typical row.
    models = on_disk["model"] if isinstance(on_disk["model"], list) else [on_disk["model"]]
    assert len(scorer.models) == len(models)
    row = scorer.model_frame(pd.DataFrame([on_disk["reference_row"]]))
    expected = np.mean([predict_fraud_proba(m, row) for m in models], axis=0)
    assert np.allclose(scorer.raw_score(row), expected, rtol=0, atol=1e-12)


def test_model_info_reports_the_loaded_version_and_thresholds(client):
    response = client.get("/model-info")
    assert response.status_code == 200, response.text
    body = response.json()
    scorer = get_scorer()
    assert body["version"] == scorer.version
    assert body["thresholds"] == {"t_review": scorer.t_review, "t_block": scorer.t_block}
