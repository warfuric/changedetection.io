"""
Parsing/validation for LLM extra request headers and body.

For OpenAI-compatible gateways that need a header on every request (a session or
routing id, a gateway token) or a provider-specific body field (e.g. turning a
model's default reasoning off). Configured in the UI or via LLM_EXTRA_HEADERS /
LLM_EXTRA_BODY, see evaluator.request_extras().
"""

import functools
import json
import os
import re

from loguru import logger

# RFC 9110 field-name token.
HEADER_NAME_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
# httpx sends header values as ASCII; tab is the only control character allowed.
HEADER_VALUE_RE = re.compile(r"^[\t\x20-\x7e]*$")
# Set by the client itself; overriding them breaks the call or its response parsing.
RESERVED_BODY_KEYS = frozenset({'model', 'messages', 'stream', 'stream_options'})


def parse_json_object(text: str) -> dict:
    """Parse text as a JSON object; blank is {}. Raises ValueError."""
    text = (text or '').strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except ValueError as e:
        raise ValueError(f"Not valid JSON: {e}") from None
    if not isinstance(value, dict):
        raise ValueError("Must be a JSON object, e.g. {\"key\": \"value\"}")
    return value


def validate_headers(headers: dict) -> dict:
    """Check names and values, coerce values to str. Raises ValueError."""
    result = {}
    seen = set()
    for name, value in headers.items():
        name = str(name).strip()
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ValueError(f"Header {name!r} value must be a string or number")
        value = str(value).strip()
        if not HEADER_NAME_RE.match(name):
            raise ValueError(f"Invalid header name: {name!r}")
        if not value:
            raise ValueError(f"Header {name!r} has an empty value")
        if not HEADER_VALUE_RE.match(value):
            raise ValueError(f"Header {name!r} value must be printable ASCII")
        if name.lower() in seen:
            raise ValueError(f"Header {name!r} is set more than once")
        seen.add(name.lower())
        result[name] = value
    return result


def validate_body(body: dict) -> dict:
    """Refuse keys the client sets itself. Raises ValueError."""
    reserved = sorted(RESERVED_BODY_KEYS & set(body))
    if reserved:
        raise ValueError(f"Cannot override {', '.join(reserved)}")
    return body


def parse_body(text: str) -> dict:
    """Parse an extra request body (JSON object). Raises ValueError."""
    return validate_body(parse_json_object(text))


def parse_header_lines(text: str) -> dict:
    """'Name: value' lines -> {name: value}. Blank lines are skipped. Raises ValueError."""
    headers = {}
    for idx, line in enumerate((text or '').splitlines(), start=1):
        if not line.strip():
            continue
        if ':' not in line:
            raise ValueError(f"Line {idx} is missing a ':' separator")
        name, value = line.split(':', 1)
        if name.strip() in headers:
            raise ValueError(f"Header {name.strip()!r} is set more than once")
        headers[name.strip()] = value
    return validate_headers(headers)


@functools.lru_cache(maxsize=8)
def _parse_env(name: str, raw: str, headers: bool) -> dict:
    # Cached on the raw value so a malformed setting is logged once, not on every check.
    try:
        value = parse_json_object(raw)
        return validate_headers(value) if headers else validate_body(value)
    except ValueError as e:
        logger.error(f"Ignoring {name}: {e}")
        return {}


def env_headers(name: str) -> dict:
    """Headers from an env var holding a JSON object; malformed is logged and ignored."""
    return dict(_parse_env(name, os.getenv(name, ''), True))


def env_json_object(name: str) -> dict:
    """JSON object from an env var; malformed is logged and ignored."""
    return json.loads(json.dumps(_parse_env(name, os.getenv(name, ''), False)))
