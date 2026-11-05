"""A stand-in for the backend's HTTP API, for tests that patch requests.get / requests.post (no server runs)."""

import json
from unittest import mock
from urllib.parse import urlsplit

VALIDATED = {"args": {}, "dataset": {}, "tags": {}}
STARTED = {"detail": "Training Started", "training": True}
IDLE = {"training": False, "errored": False}
RUNNING = {"training": True, "errored": False}


class Answers:
    """Successive answers of one route; the last one repeats."""

    def __init__(self, *answers):
        self.answers = list(answers)

    def next(self):
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


class FakeBackend:
    """Answers by route, records (method, route, params, headers) of every call."""

    def __init__(self, routes: dict):
        self.routes = routes  # route -> (status, body), an exception to raise, or Answers of those
        self.calls = []

    def _answer(self, method: str, url: str, **kwargs):
        route = urlsplit(url).path
        self.calls.append((method, route, dict(kwargs.get("params") or {}), dict(kwargs.get("headers") or {})))
        answer = self.routes[route]
        if isinstance(answer, Answers):
            answer = answer.next()
        if isinstance(answer, Exception):
            raise answer
        status, body = answer
        if isinstance(body, str):  # a page that is not JSON (a tunnel's error page)
            response = mock.Mock(status_code=status, text=body)
            response.json.side_effect = ValueError("not JSON")
            return response
        response = mock.Mock(status_code=status, text=json.dumps(body))
        response.json.return_value = body
        return response

    def get(self, url, **kwargs):
        return self._answer("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self._answer("POST", url, **kwargs)

    def routes_called(self) -> list[str]:
        return [call[1] for call in self.calls]
