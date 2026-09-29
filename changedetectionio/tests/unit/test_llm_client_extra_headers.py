import unittest
from unittest.mock import MagicMock, patch

from changedetectionio.llm import client as m


def _mock_success_response(text="ok"):
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content=text, parts=None), finish_reason="stop")]
    mock_resp.usage = MagicMock(total_tokens=5, prompt_tokens=3, completion_tokens=2)
    return mock_resp


class TestClientExtraHeaders(unittest.TestCase):
    def test_extra_headers_are_forwarded_to_litellm(self):
        with patch("litellm.completion", return_value=_mock_success_response()) as mock_call:
            m.completion(
                model="openai/some-model",
                messages=[{"role": "user", "content": "hi"}],
                api_base="https://gateway.example/v1",
                extra_headers={"x-session-id": "cdio"},
            )
        self.assertEqual(mock_call.call_args.kwargs.get("extra_headers"), {"x-session-id": "cdio"})

    def test_no_extra_headers_kwarg_when_none_or_empty(self):
        for value in (None, {}):
            with patch("litellm.completion", return_value=_mock_success_response()) as mock_call:
                m.completion(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "hi"}],
                    extra_headers=value,
                )
            self.assertNotIn("extra_headers", mock_call.call_args.kwargs)

    def test_extra_headers_survive_bad_request_retry(self):
        import litellm

        exc = litellm.BadRequestError(
            message="temperature is not supported",
            model="openai/some-model",
            llm_provider="openai",
            response=MagicMock(status_code=400),
        )
        with patch("litellm.completion", side_effect=[exc, _mock_success_response()]) as mock_call:
            m.completion(
                model="openai/some-model",
                messages=[{"role": "user", "content": "hi"}],
                extra_headers={"x-session-id": "cdio"},
                extra_body={"thinking": {"type": "disabled"}},
            )
        retry_kwargs = mock_call.call_args_list[1].kwargs
        self.assertNotIn("temperature", retry_kwargs)
        self.assertEqual(retry_kwargs.get("extra_headers"), {"x-session-id": "cdio"})
        # Only Gemini's thinkingConfig is stripped on retry; other user-supplied body keys stay.
        self.assertEqual(retry_kwargs.get("extra_body"), {"thinking": {"type": "disabled"}})

    def test_debug_logging_redacts_extra_header_values(self):
        """litellm's debug output prints extra_headers unmasked; values must not reach our log sink."""
        import logging
        from loguru import logger

        captured = []
        sink_id = logger.add(lambda msg: captured.append(str(msg)), level="DEBUG")

        def fake_completion(**kwargs):
            logging.getLogger("LiteLLM").debug(f"POST Request Sent from LiteLLM: -d '{kwargs.get('extra_headers')}'")
            return _mock_success_response()

        try:
            with patch("litellm.completion", side_effect=fake_completion):
                m.completion(
                    model="openai/some-model",
                    messages=[{"role": "user", "content": "hi"}],
                    extra_headers={"x-api-key": "KEY-CANARY-123", "x-opencode-session": "SESSION-CANARY"},
                    debug=True,
                )
        finally:
            logger.remove(sink_id)
        text = "".join(captured)
        self.assertIn("POST Request Sent from LiteLLM", text)
        self.assertNotIn("KEY-CANARY-123", text)
        self.assertNotIn("SESSION-CANARY", text)


if __name__ == "__main__":
    unittest.main()
