"""TigerGraph client for GSQL installation, detached feature jobs, and exports."""

from collections.abc import Mapping
from functools import partial
import time
from typing import Any, cast

import requests
from pyTigerGraph import TigerGraphConnection

from tfgnn.tigergraph.settings import Settings


class ClientQueryTimeoutError(requests.exceptions.ReadTimeout):
    """A TigerGraph query exceeded its configured deadline."""


class DetachedQueryError(RuntimeError):
    """A detached TigerGraph query ended unsuccessfully."""


def _find_value(payload: object, keys: set[str]) -> object | None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if str(key).lower() in keys:
                return value
        for value in payload.values():
            found = _find_value(value, keys)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_value(value, keys)
            if found is not None:
                return found
    return None


class Client:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.conn = TigerGraphConnection(
            host=settings.host,
            graphname=settings.graphname,
            gsqlSecret=settings.secret.get_secret_value(),
        )
        _ = self.conn.getToken(settings.secret.get_secret_value())
        self._install_default_timeout()

    @property
    def settings(self) -> Settings:
        """The resolved settings, for callers that need the same timeouts."""
        return self._settings

    def _install_default_timeout(self) -> None:
        session = cast(object, getattr(self.conn, "_session", None))
        if not isinstance(session, requests.Session):
            return
        if getattr(session.request, "_tfgnn_has_default_timeout", False):
            return
        wrapped = partial(
            session.request,
            timeout=(
                self._settings.connect_timeout_s,
                self._settings.read_timeout_s,
            ),
        )
        setattr(wrapped, "_tfgnn_has_default_timeout", True)
        setattr(session, "request", wrapped)

    def run_installed_with_timeout(
        self,
        query_name: str,
        params: dict[str, object],
        timeout_s: float | None = None,
        size_limit: int | None = None,
    ) -> list[object]:
        """Run an installed query synchronously.

        Python settings are seconds. pyTigerGraph's timeout argument is
        milliseconds, so conversion occurs exactly once here.
        """
        query_timeout_s = timeout_s or self._settings.query_timeout_s
        response_limit = size_limit or self._settings.response_size_limit_bytes
        try:
            return cast(
                list[object],
                self.conn.runInstalledQuery(
                    query_name,
                    params,
                    timeout=int(query_timeout_s * 1000),
                    sizeLimit=response_limit,
                    usePost=True,
                ),
            )
        except requests.exceptions.ReadTimeout as exc:
            raise ClientQueryTimeoutError(
                f"installed query {query_name!r} exceeded the HTTP read deadline "
                f"({self._settings.read_timeout_s:.0f}s)"
            ) from exc

    def run_installed_detached(
        self,
        query_name: str,
        params: dict[str, object],
        timeout_s: float | None = None,
        size_limit: int | None = None,
        poll_interval_s: float | None = None,
    ) -> list[object]:
        """Run a long query in TigerGraph detached mode and poll its status."""
        query_timeout_s = timeout_s or self._settings.query_timeout_s
        response_limit = size_limit or self._settings.response_size_limit_bytes
        poll_s = poll_interval_s or self._settings.poll_interval_s

        submitted = self.conn.runInstalledQuery(
            query_name,
            params,
            timeout=int(query_timeout_s * 1000),
            sizeLimit=response_limit,
            usePost=True,
            runAsync=True,
        )
        if isinstance(submitted, str):
            request_id = submitted
        else:
            raw_id = _find_value(
                submitted,
                {"request_id", "requestid", "request-id"},
            )
            if raw_id is None:
                raise DetachedQueryError(
                    f"{query_name!r} did not return a detached request ID: "
                    f"{submitted!r}"
                )
            request_id = str(raw_id)

        print(f"detached request_id={request_id}")
        deadline = time.monotonic() + query_timeout_s + 600.0
        last_status = ""
        # BACKOFF, not a flat poll_interval_s. The old loop checked status
        # once and then slept the full interval (15 s by default), so EVERY
        # detached query that took longer than one status round trip paid a
        # 15 s floor no matter how fast it really was. Measured on the live
        # run: card_merchant_degree_stats recorded 16.08 / 15.87 / 15.88 /
        # 15.97 / 15.91 / 16.10 s across six builds -- FLAT while its input
        # edge set grew 1.4x, which no compute-bound query does. The control
        # is wcc_merchant on the same plan: 0.68 / 15.91 / 0.80 / 0.67 /
        # 0.74 / 0.64 s, one build crossing the line and paying 15.2 s for
        # it. Roughly eight such calls x six builds was ~10 minutes of pure
        # client sleep per pipeline.
        #
        # Starting at 0.25 s and backing off to poll_interval_s keeps a
        # genuinely long query's status churn exactly where it was (a 5-hour
        # PageRank still polls every 15 s) while letting a sub-second query
        # return in sub-second time. Fixed here rather than by flipping
        # individual calls to detached=False: this is one place, and it
        # cannot drift out of step with the call list.
        poll_wait = min(0.25, poll_s)

        while True:
            status_payload = self.conn.checkQueryStatus(request_id)
            raw_status = _find_value(status_payload, {"status"})
            status = str(raw_status or "").strip().lower()

            if status and status != last_status:
                print(f"detached status={status}")
                last_status = status

            if status in {"success", "completed", "complete", "finished"}:
                return cast(list[object], self.conn.getQueryResult(request_id))

            if status in {
                "timeout",
                "aborted",
                "failed",
                "error",
                "cancelled",
                "canceled",
            }:
                raise DetachedQueryError(
                    f"{query_name!r} ended with status {status!r}: {status_payload!r}"
                )

            if time.monotonic() >= deadline:
                raise ClientQueryTimeoutError(
                    f"{query_name!r} did not reach a terminal state within "
                    f"{query_timeout_s + 600:.0f}s"
                )

            time.sleep(poll_wait)
            poll_wait = min(poll_wait * 2.0, poll_s)

    def gsql(self, statement: str) -> str | dict[str, Any]:
        """Execute one or more GSQL statements.

        pyTigerGraph returns text on some TigerGraph versions and a dictionary
        on others; both are valid.
        """
        result = self.conn.gsql(statement)
        if isinstance(result, str):
            return result
        # A duck check rather than isinstance: pyTigerGraph's return
        # annotation is str | dict, but that is an annotation and not a
        # guarantee across versions, and the annotation is what makes an
        # isinstance narrowing here look redundant to the type checker.
        if not hasattr(result, "items"):
            raise RuntimeError(
                f"expected text or mapping from gsql(), got {type(result).__name__}"
            )
        return cast("dict[str, Any]", result)

    @property
    def graphname(self) -> str:
        return self._settings.graphname
