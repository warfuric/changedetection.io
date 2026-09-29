#!/usr/bin/env python3
"""
Settings UI + /settings/llm/test for LLM extra request headers / body.

extra_headers is entered as "Name: value" lines, extra_body as a JSON object.
Both are provider-connection fields: they can carry credentials (a gateway token
in a header), so the stored values get the same treatment as the stored api_key —
wiped by "Remove provider", and never sent by /llm/test to an api_base other than
the saved one.
"""

import json

from flask import url_for

from changedetectionio.tests.util import delete_all_watches

GATEWAY = 'https://gateway.example/v1'
HEADERS = {'x-session-id': 'cdio', 'X-Other': 'two words'}
BODY = {'thinking': {'type': 'disabled'}}

_BASE_FORM = {
    'llm-model': 'openai/some-model',
    'llm-api_key': '',
    'llm-api_base': GATEWAY,
    'application-pager_size': '50',
    'application-notification_format': 'html',
    'application-fetch_backend': 'html_requests',
    'application-rss_diff_length': '5',
    'application-filter_failure_notification_threshold_attempts': '0',
    'requests-time_between_check-days': '0',
    'requests-time_between_check-hours': '0',
    'requests-time_between_check-minutes': '5',
    'requests-time_between_check-seconds': '0',
    'requests-time_between_check-weeks': '0',
    'requests-jitter_seconds': '0',
    'requests-workers': '10',
    'requests-timeout': '60',
}


def _seed(ds, **extra):
    ds.data['settings']['application']['llm'] = {
        'model': 'openai/some-model',
        'api_key': 'sk-stored',
        'api_base': GATEWAY,
        **extra,
    }


def _llm(ds):
    return ds.data['settings']['application'].get('llm') or {}


def _post_settings(client, **fields):
    return client.post(url_for('settings.settings_page'), data={**_BASE_FORM, **fields}, follow_redirects=True)


# ---------------------------------------------------------------------------
# Settings form
# ---------------------------------------------------------------------------

def test_settings_form_saves_and_renders_extra_headers_and_body(
        client, live_server, measure_memory_usage, datastore_path, monkeypatch):
    monkeypatch.setenv('ALLOW_IANA_RESTRICTED_ADDRESSES', 'true')
    ds = client.application.config.get('DATASTORE')
    _seed(ds)

    res = _post_settings(client, **{
        'llm-extra_headers': 'x-session-id: cdio\r\n\r\nX-Other: two words\r\n',
        'llm-extra_body': json.dumps(BODY),
    })
    assert b'Settings updated.' in res.data
    assert _llm(ds)['extra_headers'] == HEADERS
    assert _llm(ds)['extra_body'] == BODY
    assert _llm(ds)['api_key'] == 'sk-stored', "Saving extras must not touch the stored key"

    page = client.get(url_for('settings.settings_page')).data.decode()
    assert 'x-session-id: cdio' in page
    assert '&#34;thinking&#34;' in page or '"thinking"' in page
    delete_all_watches(client)


def test_settings_form_rejects_invalid_extra_body(
        client, live_server, measure_memory_usage, datastore_path, monkeypatch):
    monkeypatch.setenv('ALLOW_IANA_RESTRICTED_ADDRESSES', 'true')
    ds = client.application.config.get('DATASTORE')
    _seed(ds, extra_body=BODY)

    for bad in ('{not json', '["a list"]', '"a string"', '{"model": "other"}', '{"stream": true}'):
        res = _post_settings(client, **{'llm-extra_headers': '', 'llm-extra_body': bad})
        assert b'Settings updated.' not in res.data, f"{bad!r} must not validate"
        assert _llm(ds)['extra_body'] == BODY, f"{bad!r} must not overwrite the stored body"
    delete_all_watches(client)


def test_settings_form_rejects_invalid_header_lines(
        client, live_server, measure_memory_usage, datastore_path, monkeypatch):
    monkeypatch.setenv('ALLOW_IANA_RESTRICTED_ADDRESSES', 'true')
    ds = client.application.config.get('DATASTORE')
    _seed(ds, extra_headers={'x-session-id': 'cdio'})

    for bad in ('no-separator-here', 'bad name: value', ': no name', 'x-empty:',
                'x-accent: caf\u00e9', 'x-dup: 1\r\nX-Dup: 2'):
        res = _post_settings(client, **{'llm-extra_headers': bad, 'llm-extra_body': ''})
        assert b'Settings updated.' not in res.data, f"{bad!r} must not validate"
        assert _llm(ds)['extra_headers'] == {'x-session-id': 'cdio'}
    delete_all_watches(client)


def test_blank_fields_clear_and_absent_fields_keep(
        client, live_server, measure_memory_usage, datastore_path, monkeypatch):
    monkeypatch.setenv('ALLOW_IANA_RESTRICTED_ADDRESSES', 'true')
    ds = client.application.config.get('DATASTORE')

    # Absent from the POST (e.g. a partial form submission): stored values are kept.
    _seed(ds, extra_headers=HEADERS, extra_body=BODY)
    res = _post_settings(client)
    assert b'Settings updated.' in res.data
    assert _llm(ds)['extra_headers'] == HEADERS
    assert _llm(ds)['extra_body'] == BODY

    # Submitted blank: cleared.
    res = _post_settings(client, **{'llm-extra_headers': '', 'llm-extra_body': ''})
    assert b'Settings updated.' in res.data
    assert _llm(ds).get('extra_headers') == {}
    assert _llm(ds).get('extra_body') == {}
    delete_all_watches(client)


def test_remove_provider_wipes_extra_headers_and_body(
        client, live_server, measure_memory_usage, datastore_path):
    ds = client.application.config.get('DATASTORE')
    _seed(ds, extra_headers=HEADERS, extra_body=BODY)
    client.post(url_for('settings.llm.llm_clear'), follow_redirects=True)
    assert 'extra_headers' not in _llm(ds)
    assert 'extra_body' not in _llm(ds)
    delete_all_watches(client)


def test_env_configured_note_mentions_extras(
        client, live_server, measure_memory_usage, datastore_path, monkeypatch):
    monkeypatch.setenv('LLM_MODEL', 'openai/some-model')
    monkeypatch.setenv('LLM_EXTRA_HEADERS', '{"x-gateway-token": "hdr-CANARY-value"}')
    page = client.get(url_for('settings.settings_page')).data.decode()
    assert 'LLM_EXTRA_HEADERS' in page
    assert 'hdr-CANARY-value' not in page, "Header values can be credentials and must not be echoed"


# ---------------------------------------------------------------------------
# /settings/llm/test
# ---------------------------------------------------------------------------

def _capture_completion(monkeypatch):
    calls = []
    import changedetectionio.llm.client as llm_client
    monkeypatch.setattr(llm_client, 'completion', lambda **kw: calls.append(kw) or ('ready', 3, 2, 1))
    return calls


def test_llm_test_forwards_request_supplied_extras(
        client, live_server, measure_memory_usage, datastore_path, monkeypatch):
    monkeypatch.setenv('ALLOW_IANA_RESTRICTED_ADDRESSES', 'true')
    ds = client.application.config.get('DATASTORE')
    _seed(ds)
    calls = _capture_completion(monkeypatch)

    res = client.post(url_for('settings.llm.llm_test'),
                      query_string={'model': 'openai/some-model', 'api_base': GATEWAY},
                      data={'extra_headers': 'x-session-id: cdio\nX-Other: two words\n',
                            'extra_body': json.dumps(BODY)})
    assert res.status_code == 200, res.get_json()
    assert calls[0]['extra_headers'] == HEADERS
    assert calls[0]['extra_body'] == BODY
    delete_all_watches(client)


def test_llm_test_uses_stored_extras_for_the_saved_api_base(
        client, live_server, measure_memory_usage, datastore_path, monkeypatch):
    monkeypatch.setenv('ALLOW_IANA_RESTRICTED_ADDRESSES', 'true')
    ds = client.application.config.get('DATASTORE')
    _seed(ds, extra_headers=HEADERS, extra_body=BODY)
    calls = _capture_completion(monkeypatch)

    res = client.post(url_for('settings.llm.llm_test'), query_string={'api_base': GATEWAY})
    assert res.status_code == 200, res.get_json()
    assert calls[0]['extra_headers'] == HEADERS
    assert calls[0]['extra_body'] == BODY
    delete_all_watches(client)


def test_llm_test_never_sends_stored_extras_to_a_different_api_base(
        client, live_server, measure_memory_usage, datastore_path, monkeypatch):
    """Same threat as GHSA-g36r-fm2p-87xm, for headers: a stored gateway token must not be
    shipped to an attacker-chosen api_base, even when the caller supplies its own api_key."""
    monkeypatch.setenv('ALLOW_IANA_RESTRICTED_ADDRESSES', 'true')
    ds = client.application.config.get('DATASTORE')
    _seed(ds, extra_headers={'Authorization': 'Bearer gateway-CANARY'}, extra_body=BODY)
    calls = _capture_completion(monkeypatch)

    res = client.post(url_for('settings.llm.llm_test'), query_string={
        'model': 'openai/some-model',
        'api_base': 'https://attacker.example/v1',
        'api_key': 'sk-attacker',
    })
    assert res.status_code == 200, res.get_json()
    assert not calls[0].get('extra_headers')
    assert not calls[0].get('extra_body')
    delete_all_watches(client)


def test_llm_test_blank_form_values_override_stored_extras(
        client, live_server, measure_memory_usage, datastore_path, monkeypatch):
    """The UI always sends the fields, so a blank textarea means "test without extras"."""
    monkeypatch.setenv('ALLOW_IANA_RESTRICTED_ADDRESSES', 'true')
    ds = client.application.config.get('DATASTORE')
    _seed(ds, extra_headers=HEADERS, extra_body=BODY)
    calls = _capture_completion(monkeypatch)

    res = client.post(url_for('settings.llm.llm_test'), query_string={'api_base': GATEWAY},
                      data={'extra_headers': '', 'extra_body': ''})
    assert res.status_code == 200, res.get_json()
    assert not calls[0].get('extra_headers')
    assert not calls[0].get('extra_body')
    delete_all_watches(client)


def test_llm_test_ignores_extras_in_the_query_string(
        client, live_server, measure_memory_usage, datastore_path, monkeypatch):
    """Extras go in the POST body: a query string ends up in access logs."""
    monkeypatch.setenv('ALLOW_IANA_RESTRICTED_ADDRESSES', 'true')
    ds = client.application.config.get('DATASTORE')
    _seed(ds)
    calls = _capture_completion(monkeypatch)
    res = client.post(url_for('settings.llm.llm_test'),
                      query_string={'api_base': GATEWAY, 'extra_headers': 'x-session-id: cdio'})
    assert res.status_code == 200, res.get_json()
    assert not calls[0].get('extra_headers')
    delete_all_watches(client)


def test_llm_test_does_not_mutate_stored_extra_body(
        client, live_server, measure_memory_usage, datastore_path, monkeypatch):
    """completion() pops generationConfig.thinkingConfig in place on a retry; the stored copy must survive."""
    monkeypatch.setenv('ALLOW_IANA_RESTRICTED_ADDRESSES', 'true')
    ds = client.application.config.get('DATASTORE')
    stored_body = {'generationConfig': {'thinkingConfig': {'thinkingBudget': 0}, 'temperature': 0}}
    _seed(ds, extra_body=stored_body)

    import changedetectionio.llm.client as llm_client
    def popping_completion(**kw):
        kw['extra_body']['generationConfig'].pop('thinkingConfig', None)
        return ('ready', 3, 2, 1)
    monkeypatch.setattr(llm_client, 'completion', popping_completion)

    res = client.post(url_for('settings.llm.llm_test'), query_string={'api_base': GATEWAY})
    assert res.status_code == 200, res.get_json()
    assert _llm(ds)['extra_body']['generationConfig']['thinkingConfig'] == {'thinkingBudget': 0}
    delete_all_watches(client)


def test_llm_test_rejects_malformed_extras(
        client, live_server, measure_memory_usage, datastore_path, monkeypatch):
    monkeypatch.setenv('ALLOW_IANA_RESTRICTED_ADDRESSES', 'true')
    ds = client.application.config.get('DATASTORE')
    _seed(ds)
    calls = _capture_completion(monkeypatch)

    for field, bad in (('extra_headers', 'no separator'), ('extra_headers', 'bad name: v'),
                       ('extra_headers', 'x-a: caf\u00e9'), ('extra_body', '[1, 2]'),
                       ('extra_body', '{not json'), ('extra_body', '{"stream": true}')):
        res = client.post(url_for('settings.llm.llm_test'), query_string={'api_base': GATEWAY}, data={field: bad})
        assert res.status_code == 400, f"{field}={bad!r} should be refused"
        assert res.get_json()['ok'] is False
    assert calls == []
    delete_all_watches(client)
