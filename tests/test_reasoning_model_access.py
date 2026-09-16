"""## Executive summary (read this first)

Synthetic transport checks prove the reasoning example uses the organizer's proxy and token,
preserves local API-key experiments, and labels failures without exposing credentials.
No test makes a network request.
"""

from __future__ import annotations

import base64
import http.client
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from baselines import reasoning_agent as agent


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "MODEL_ENDPOINT",
        "MODEL_NAME",
        "MODEL_TOKEN",
        "MODEL_API_KEY",
        "MODEL_MAX_TOKENS",
        "MODEL_THINKING",
        "http_proxy",
        "https_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MODEL_NAME", "synthetic-model")
    monkeypatch.setenv("MODEL_ENDPOINT", "http://model.invalid")
    monkeypatch.setenv("MODEL_TOKEN", "synthetic-grant")
    monkeypatch.setenv("MODEL_API_KEY", "synthetic-legacy")
    monkeypatch.setenv("http_proxy", "http://synthetic-user:pass%3Aword@proxy.invalid:3129")
    # An inherited bypass must not send an organizer request directly to its destination.
    monkeypatch.setenv("no_proxy", "*")
    monkeypatch.setenv("NO_PROXY", "*")


def response_bytes() -> bytes:
    return json.dumps({"choices": [{"message": {"content": '{"assets": {}}'}}]}).encode()


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {"requests": [], "connections": [], "status": 200}

    class Connection:
        def __init__(self, host: str, port: int, *, timeout: int) -> None:
            seen["connections"].append((host, port, timeout))

        def request(self, method: str, target: str, body: bytes, headers: dict[str, str]) -> None:
            seen["requests"].append((method, target, json.loads(body), headers))

        def getresponse(self) -> Any:
            response = io.BytesIO(response_bytes())
            response.status = seen["status"]  # type: ignore[attr-defined]
            return response

        def close(self) -> None:
            seen["closed"] = True

    def no_legacy(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("House request reached the legacy/direct path")

    monkeypatch.setattr(http.client, "HTTPConnection", Connection)
    monkeypatch.setattr(agent.urllib.request, "urlopen", no_legacy)
    return seen


@pytest.mark.parametrize("suffix", ["", "/", "/v1", "/v1/"])
def test_house_token_root_and_v1_use_explicit_proxy_once(
    monkeypatch: pytest.MonkeyPatch, transport: dict[str, Any], suffix: str
) -> None:
    monkeypatch.setenv("MODEL_ENDPOINT", "http://model.invalid" + suffix)
    monkeypatch.setenv("MODEL_MAX_TOKENS", "8192")
    assert agent.call_model("synthetic prompt") == ({"assets": {}}, "", "")
    assert transport["connections"] == [("proxy.invalid", 3129, 60)]
    assert len(transport["requests"]) == 1
    method, target, body, headers = transport["requests"][0]
    assert method == "POST"
    assert target == "http://model.invalid/v1/chat/completions"
    assert body["model"] == "synthetic-model"
    assert body["max_tokens"] == 4000
    assert headers["Authorization"] == "Bearer synthetic-grant"
    assert headers["Proxy-Authorization"] == (
        "Basic " + base64.b64encode(b"synthetic-user:pass:word").decode()
    )
    assert transport["closed"] is True


@pytest.mark.parametrize(
    "name,value",
    [
        ("http_proxy", ""),
        ("http_proxy", "http://proxy.invalid:3129"),
        ("MODEL_TOKEN", ""),
        ("MODEL_TOKEN", "bad\nheader"),
        ("MODEL_ENDPOINT", "http://user:password@model.invalid/v1"),
    ],
)
def test_bad_house_configuration_never_downgrades_to_legacy(
    monkeypatch: pytest.MonkeyPatch, transport: dict[str, Any], name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)
    parsed, reason, _ = agent.call_model("synthetic prompt")
    assert parsed is None
    assert reason == "model request failed (ValueError)"
    assert transport["connections"] == []


@pytest.mark.parametrize("status", [302, 407, 429])
def test_refusal_or_redirect_is_one_request_without_secret_output(
    transport: dict[str, Any], status: int
) -> None:
    transport["status"] = status
    parsed, reason, _ = agent.call_model("synthetic prompt")
    assert parsed is None
    assert reason == "model request failed (ValueError)"
    assert len(transport["requests"]) == 1
    assert transport["closed"] is True


def test_local_api_key_and_larger_local_budget_remain_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MODEL_TOKEN")
    monkeypatch.setenv("MODEL_ENDPOINT", "https://local.invalid/v1")
    monkeypatch.setenv("MODEL_MAX_TOKENS", "8192")
    seen = []

    def local(request: Any, *, timeout: int) -> io.BytesIO:
        seen.append(request)
        assert timeout == 60
        return io.BytesIO(response_bytes())

    monkeypatch.setattr(agent.urllib.request, "urlopen", local)
    assert agent.call_model("synthetic prompt") == ({"assets": {}}, "", "")
    assert len(seen) == 1
    assert seen[0].full_url == "https://local.invalid/v1/chat/completions"
    assert seen[0].get_header("Authorization") == "Bearer synthetic-legacy"
    assert json.loads(seen[0].data)["max_tokens"] == 8192


def test_no_endpoint_stays_an_explicit_offline_fallback(
    monkeypatch: pytest.MonkeyPatch, transport: dict[str, Any]
) -> None:
    monkeypatch.delenv("MODEL_ENDPOINT")
    assert agent.call_model("synthetic prompt") == (None, "MODEL_ENDPOINT is unset", "")
    assert transport["connections"] == []


def test_transport_error_text_does_not_enter_the_failure_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise OSError("synthetic-grant and proxy pass:word must not enter output")

    monkeypatch.setattr(http.client, "HTTPConnection", fail)
    assert agent.call_model("synthetic prompt") == (None, "model request failed (OSError)", "")


def test_cli_labels_a_refused_house_request_and_keeps_the_statistical_forecast(
    tmp_path: Path, transport: dict[str, Any]
) -> None:
    from qfbench2_track_forecasting.cli import _draw

    unit = tmp_path / "unit"
    unit.mkdir()
    dates = pd.bdate_range("2020-01-01", periods=80)
    asof = dates[-1].strftime("%Y-%m-%d")
    panel = pd.DataFrame({"date": dates, "asset": "SYN-A", "value": np.tile([1.0, 2.0], 40)})
    panel.to_parquet(unit / "synthetic.parquet", index=False)
    (unit / "card.toml").write_text(
        '[task]\nid = "synthetic"\n[targets]\nasset_ids = ["SYN-A"]\n'
        'horizons = [1]\ntarget_type = "level"\n'
    )
    corpus = unit / "text"
    corpus.mkdir()
    (corpus / "synthetic.txt").write_text("Synthetic evidence for a transport test.")
    (corpus / "corpus_index.json").write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "doc_id": "synthetic",
                        "timestamp": asof,
                        "file": "synthetic.txt",
                        "doc_type": "test",
                    }
                ]
            }
        )
    )
    output = tmp_path / "output" / "forecast.parquet"
    transport["status"] = 429
    assert (
        agent.main(
            [
                "--panels",
                str(unit),
                "--text",
                str(corpus),
                "--asof",
                asof,
                "--card",
                str(unit / "card.toml"),
                "--out",
                str(output),
                "--seed",
                "17",
            ]
        )
        == 0
    )
    metadata = json.loads((output.parent / "forecast_meta.json").read_text())
    assert metadata["reasoning_applied"] is False
    assert metadata["reasoning_skipped_reason"] == "model request failed (ValueError)"
    assert len(transport["requests"]) == 1
    samples, _ = _draw({"synthetic": panel}, ["SYN-A"], [1], asof, 500, 17)
    np.testing.assert_array_equal(pd.read_parquet(output)["value"], samples[:, 0, 0])
