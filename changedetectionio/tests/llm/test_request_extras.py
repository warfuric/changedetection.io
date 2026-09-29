"""
Extra request headers / body for LLM calls.

Some OpenAI-compatible gateways need a header on every request (for example a
session or routing id) or a provider-specific body field (for example turning a
model's default reasoning off). They can be set in the UI (stored under
settings.application.llm.extra_headers / extra_body) or, when the connection is
configured via LLM_MODEL, with the LLM_EXTRA_HEADERS / LLM_EXTRA_BODY env vars.

All LLM calls are mocked — no real API key needed.
"""
import os
from unittest.mock import patch

import pytest

from changedetectionio.tests.llm.test_evaluator import _make_datastore, _make_watch

HEADERS = {'x-session-id': 'cdio'}
BODY = {'thinking': {'type': 'disabled'}}


@pytest.fixture(autouse=True)
def _no_llm_env(monkeypatch):
    for name in ('LLM_MODEL', 'LLM_API_KEY', 'LLM_API_BASE', 'LLM_EXTRA_HEADERS', 'LLM_EXTRA_BODY'):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# get_llm_config — env var connection
# ---------------------------------------------------------------------------

class TestEnvConfig:
    def test_env_extra_headers_and_body_parsed(self, monkeypatch):
        from changedetectionio.llm.evaluator import get_llm_config
        monkeypatch.setenv('LLM_MODEL', 'openai/some-model')
        monkeypatch.setenv('LLM_EXTRA_HEADERS', '{"x-session-id": "cdio"}')
        monkeypatch.setenv('LLM_EXTRA_BODY', '{"thinking": {"type": "disabled"}}')
        cfg = get_llm_config(_make_datastore())
        assert cfg['extra_headers'] == HEADERS
        assert cfg['extra_body'] == BODY

    def test_env_header_values_are_coerced_to_strings(self, monkeypatch):
        from changedetectionio.llm.evaluator import get_llm_config
        monkeypatch.setenv('LLM_MODEL', 'openai/some-model')
        monkeypatch.setenv('LLM_EXTRA_HEADERS', '{"x-retries": 3}')
        assert get_llm_config(_make_datastore())['extra_headers'] == {'x-retries': '3'}

    @pytest.mark.parametrize('raw', ['not json', '["a", "b"]', '"string"', '42'])
    def test_env_invalid_or_non_object_json_is_ignored(self, monkeypatch, raw):
        from changedetectionio.llm.evaluator import get_llm_config
        monkeypatch.setenv('LLM_MODEL', 'openai/some-model')
        monkeypatch.setenv('LLM_EXTRA_HEADERS', raw)
        monkeypatch.setenv('LLM_EXTRA_BODY', raw)
        cfg = get_llm_config(_make_datastore())
        assert cfg['model'] == 'openai/some-model'
        assert cfg['extra_headers'] == {}
        assert cfg['extra_body'] == {}

    def test_env_non_ascii_header_value_is_ignored(self, monkeypatch):
        from changedetectionio.llm.evaluator import get_llm_config
        monkeypatch.setenv('LLM_MODEL', 'openai/some-model')
        monkeypatch.setenv('LLM_EXTRA_HEADERS', '{"x-name": "caf\u00e9"}')
        assert get_llm_config(_make_datastore())['extra_headers'] == {}

    @pytest.mark.parametrize('raw', ['{"x-a": null}', '{"x-a": true}', '{"x-a": {"b": 1}}', '{"x-a": [1]}'])
    def test_env_non_scalar_header_values_are_ignored(self, monkeypatch, raw):
        from changedetectionio.llm.evaluator import get_llm_config
        monkeypatch.setenv('LLM_MODEL', 'openai/some-model')
        monkeypatch.setenv('LLM_EXTRA_HEADERS', raw)
        assert get_llm_config(_make_datastore())['extra_headers'] == {}

    def test_env_body_cannot_override_core_request_keys(self, monkeypatch):
        from changedetectionio.llm.evaluator import get_llm_config
        monkeypatch.setenv('LLM_MODEL', 'openai/some-model')
        monkeypatch.setenv('LLM_EXTRA_BODY', '{"stream": true}')
        assert get_llm_config(_make_datastore())['extra_body'] == {}

    def test_env_malformed_value_is_logged_once(self, monkeypatch):
        from changedetectionio.llm import extras
        extras._parse_env.cache_clear()
        monkeypatch.setenv('LLM_EXTRA_BODY', '{malformed-logged-once')
        with patch.object(extras.logger, 'error') as mock_error:
            for _ in range(3):
                assert extras.env_json_object('LLM_EXTRA_BODY') == {}
        assert mock_error.call_count == 1

    def test_env_extras_win_over_stored_ones(self, monkeypatch):
        from changedetectionio.llm.evaluator import get_llm_config
        monkeypatch.setenv('LLM_MODEL', 'openai/some-model')
        monkeypatch.setenv('LLM_EXTRA_HEADERS', '{"x-session-id": "env"}')
        cfg = get_llm_config(_make_datastore(llm_cfg={'model': 'x', 'extra_headers': HEADERS, 'extra_body': BODY}))
        assert cfg['extra_headers'] == {'x-session-id': 'env'}
        assert cfg['extra_body'] == {}

    def test_env_extras_ignored_without_llm_model(self, monkeypatch):
        """They belong to the env-configured connection, like LLM_API_KEY / LLM_API_BASE."""
        from changedetectionio.llm.evaluator import get_llm_config
        monkeypatch.setenv('LLM_EXTRA_HEADERS', '{"x-session-id": "env"}')
        cfg = get_llm_config(_make_datastore(llm_cfg={'model': 'gpt-4o-mini', 'extra_headers': HEADERS}))
        assert cfg['extra_headers'] == HEADERS


# ---------------------------------------------------------------------------
# request_extras — what gets handed to llm_client.completion()
# ---------------------------------------------------------------------------

class TestRequestExtras:
    def test_empty_config_gives_none(self):
        from changedetectionio.llm.evaluator import request_extras
        assert request_extras({'model': 'gpt-4o-mini'}, thinking_budget=0) == {
            'extra_headers': None, 'extra_body': None}

    def test_configured_values_pass_through(self):
        from changedetectionio.llm.evaluator import request_extras
        cfg = {'model': 'openai/x', 'extra_headers': HEADERS, 'extra_body': BODY}
        assert request_extras(cfg, thinking_budget=0) == {'extra_headers': HEADERS, 'extra_body': BODY}

    def test_merges_with_gemini_thinking_body_and_user_keys_win(self):
        from changedetectionio.llm.evaluator import request_extras
        thinking = {'generationConfig': {'thinkingConfig': {'thinkingBudget': 512}}}
        cfg = {'model': 'gemini/gemini-2.5-flash', 'extra_body': {'custom': 1}}
        with patch('changedetectionio.llm.evaluator._thinking_extra_body', return_value=thinking):
            assert request_extras(cfg, thinking_budget=512)['extra_body'] == {**thinking, 'custom': 1}
        override = {'generationConfig': {'responseMimeType': 'application/json'}}
        with patch('changedetectionio.llm.evaluator._thinking_extra_body', return_value=thinking):
            assert request_extras({**cfg, 'extra_body': override}, thinking_budget=512)['extra_body'] == override

    def test_nested_body_is_not_shared_with_stored_config(self):
        """The Gemini retry pops generationConfig.thinkingConfig in place; that must not reach the store."""
        from changedetectionio.llm.evaluator import request_extras
        stored = {'generationConfig': {'thinkingConfig': {'thinkingBudget': 0}, 'temperature': 0}}
        cfg = {'model': 'gemini/x', 'extra_body': stored}
        extras = request_extras(cfg, thinking_budget=0)
        extras['extra_body']['generationConfig'].pop('thinkingConfig')
        assert stored['generationConfig']['thinkingConfig'] == {'thinkingBudget': 0}

    def test_returns_fresh_dicts(self):
        """client.completion() may pop keys from extra_body on a retry; stored config must not change."""
        from changedetectionio.llm.evaluator import request_extras
        cfg = {'model': 'openai/x', 'extra_headers': dict(HEADERS), 'extra_body': dict(BODY)}
        extras = request_extras(cfg, thinking_budget=0)
        extras['extra_body'].clear()
        extras['extra_headers'].clear()
        assert cfg['extra_body'] == BODY
        assert cfg['extra_headers'] == HEADERS


# ---------------------------------------------------------------------------
# Every LLM call site forwards them
# ---------------------------------------------------------------------------

_CFG = {'model': 'openai/some-model', 'api_key': 'sk-test', 'api_base': 'https://gateway.example/v1',
        'extra_headers': HEADERS, 'extra_body': BODY}


def _assert_forwarded(mock_llm):
    mock_llm.assert_called_once()
    kwargs = mock_llm.call_args.kwargs
    assert kwargs['extra_headers'] == HEADERS
    assert kwargs['extra_body'] == BODY


class TestExtrasParsing:
    def test_header_lines(self):
        from changedetectionio.llm.extras import parse_header_lines
        assert parse_header_lines('x-a: 1\r\n\r\n  X-B :  two words \n') == {'x-a': '1', 'X-B': 'two words'}
        assert parse_header_lines('Authorization: Bearer a:b') == {'Authorization': 'Bearer a:b'}
        assert parse_header_lines('') == {}

    @pytest.mark.parametrize('bad', ['no separator', 'bad name: v', ': v', 'x-a:', 'x-a: caf\u00e9',
                                     'x-a: a\x00b', 'x-a: 1\nX-A: 2'])
    def test_header_lines_rejects(self, bad):
        from changedetectionio.llm.extras import parse_header_lines
        with pytest.raises(ValueError):
            parse_header_lines(bad)


def test_stored_extras_reach_litellm():
    """Whole path: stored settings -> evaluate_change -> client.completion -> litellm kwargs."""
    from unittest.mock import MagicMock
    from changedetectionio.llm.evaluator import evaluate_change
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content='{"important": true, "summary": "x"}', parts=None),
                              finish_reason='stop')]
    resp.usage = MagicMock(total_tokens=5, prompt_tokens=3, completion_tokens=2)
    ds = _make_datastore(llm_cfg=dict(_CFG))
    with patch('litellm.completion', return_value=resp) as mock_litellm:
        evaluate_change(_make_watch(llm_intent='flag price drops'), ds, diff='- $5\n+ $4')
    kwargs = mock_litellm.call_args.kwargs
    assert kwargs['extra_headers'] == HEADERS
    assert kwargs['extra_body'] == BODY
    assert kwargs['api_base'] == 'https://gateway.example/v1'


class TestCallSitesForwardExtras:
    def test_evaluate_change(self):
        from changedetectionio.llm.evaluator import evaluate_change
        watch = _make_watch(llm_intent='flag price drops')
        with patch('changedetectionio.llm.client.completion',
                   return_value=('{"important": true, "summary": "x"}', 10)) as mock_llm:
            evaluate_change(watch, _make_datastore(llm_cfg=dict(_CFG)), diff='- $5\n+ $4')
        _assert_forwarded(mock_llm)

    def test_summarise_change(self):
        from changedetectionio.llm.evaluator import summarise_change
        watch = _make_watch(llm_change_summary='List what changed')
        with patch('changedetectionio.llm.client.completion', return_value=('summary', 10)) as mock_llm:
            summarise_change(watch, _make_datastore(llm_cfg=dict(_CFG)), diff='- old\n+ new')
        _assert_forwarded(mock_llm)

    def test_preview_extract(self):
        from changedetectionio.llm.evaluator import preview_extract
        watch = _make_watch(llm_intent='what is the price')
        with patch('changedetectionio.llm.client.completion',
                   return_value=('{"found": true, "answer": "$4"}', 10)) as mock_llm:
            preview_extract(watch, _make_datastore(llm_cfg=dict(_CFG)), content='Price: $4')
        _assert_forwarded(mock_llm)

    def test_run_setup(self):
        from changedetectionio.llm.evaluator import run_setup
        watch = _make_watch(llm_intent='flag price drops')
        with patch('changedetectionio.llm.client.completion',
                   return_value=('{"needs_prefilter": false}', 10)) as mock_llm:
            run_setup(watch, _make_datastore(llm_cfg=dict(_CFG)), snapshot_text='Price: $4')
        _assert_forwarded(mock_llm)

    def test_restock_fallback(self):
        from changedetectionio.processors.restock_diff.plugins import llm_restock
        from changedetectionio.tests.llm.test_llm_restock_plugin import _make_datastore as _restock_ds
        ds = _restock_ds()
        ds.data['settings']['application']['llm'].update(extra_headers=HEADERS, extra_body=BODY)
        llm_restock._LLM_RESULT_CACHE.clear()
        llm_restock.datastore = ds
        with patch('changedetectionio.llm.client.completion',
                   return_value=('{"price": 4, "currency": "USD", "availability": "instock"}', 50, 40, 10)) as mock_llm:
            llm_restock.get_itemprop_availability_override(
                content='<html><body>$4 In stock</body></html>',
                fetcher_name='html_requests', fetcher_instance=None,
                url='https://example.com/p', llm_intent=None,
            )
        llm_restock._LLM_RESULT_CACHE.clear()
        _assert_forwarded(mock_llm)
