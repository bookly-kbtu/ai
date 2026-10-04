"""HTTP client for the Go backend. Every call acts as the end user:
the user's own JWT is forwarded, so authorization stays in one place."""

from typing import Any

import httpx

from app.core.exceptions import UpstreamError


class BooklyClient:
    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    async def _request(
        self,
        method: str,
        path: str,
        token: str | None = None,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        response = await self._http.request(method, path, json=json, params=params, headers=headers)
        if response.status_code >= 400:
            try:
                detail = response.json().get("error", response.text)
            except ValueError:
                detail = response.text
            raise UpstreamError(response.status_code, detail)
        if response.status_code == 204:
            return None
        return response.json()

    async def categories(self) -> list[dict[str, Any]]:
        return await self._request("GET", "/categories")

    async def create_assistant_request(
        self, token: str, transcript: str, intent: dict[str, Any]
    ) -> dict[str, Any]:
        body = {"transcript": transcript, "intent": intent}
        return await self._request("POST", "/assistant/requests", token, json=body)

    async def candidates(self, token: str, request_id: str, limit: int = 5) -> list[dict[str, Any]]:
        return await self._request(
            "GET", f"/assistant/requests/{request_id}/candidates", token, params={"limit": limit}
        )

    async def locations(self, master_id: str) -> list[dict[str, Any]]:
        return await self._request("GET", f"/masters/{master_id}/locations")

    async def slots(
        self, master_id: str, service_id: str, location_id: str, date: str
    ) -> list[dict[str, Any]]:
        return await self._request(
            "GET",
            f"/masters/{master_id}/slots",
            params={"service_id": service_id, "location_id": location_id, "date": date},
        )

    async def select_service(
        self, token: str, request_id: str, master_id: str, service_id: str
    ) -> dict[str, Any]:
        body = {"master_id": master_id, "service_id": service_id}
        return await self._request(
            "PUT", f"/assistant/requests/{request_id}/selection", token, json=body
        )

    async def book(
        self,
        token: str,
        request_id: str,
        location_id: str,
        starts_at: str,
        comment: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"location_id": location_id, "starts_at": starts_at}
        if comment:
            body["client_comment"] = comment
        return await self._request("POST", f"/assistant/requests/{request_id}/book", token, json=body)
