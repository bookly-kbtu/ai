"""Dialog orchestration: Redis-backed history, LLM tool loop, Bookly calls.

State per conversation (Redis, TTL): OpenAI-format message history plus meta
(assistant request id, selection, location). The frontend gets structured
side effects of the turn (candidates/slots/booking) next to the text reply."""

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

import structlog
from redis.asyncio import Redis

from app.clients.bookly import BooklyClient
from app.core.config import Settings
from app.core.exceptions import UpstreamError
from datetime import datetime

from app.services.llm import ALMATY, LLMClient, system_prompt


def _almaty_time(ts: str) -> str:
    """Go returns UTC timestamps; the model must talk local time."""
    try:
        return (
            datetime.fromisoformat(ts.replace("Z", "+00:00"))
            .astimezone(ALMATY)
            .strftime("%Y-%m-%d %H:%M")
        )
    except (ValueError, AttributeError):
        return ts

log = structlog.get_logger()


@dataclass
class ChatResult:
    conversation_id: str
    reply: str
    state: str
    candidates: list[dict[str, Any]] = field(default_factory=list)
    slots: list[dict[str, Any]] = field(default_factory=list)
    booking: dict[str, Any] | None = None
    bookings: list[dict[str, Any]] = field(default_factory=list)
    favorites_add: list[dict[str, Any]] = field(default_factory=list)
    favorites_show: bool = False



class DialogService:
    def __init__(self, bookly: BooklyClient, llm: LLMClient, redis: Redis, settings: Settings):
        self._bookly = bookly
        self._llm = llm
        self._redis = redis
        self._settings = settings

    def _key(self, user_id: str, conversation_id: str) -> str:
        return f"ai:conv:{user_id}:{conversation_id}"

    async def chat(
        self,
        user_id: str,
        token: str,
        conversation_id: str | None,
        message: str,
        favorites: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        conversation_id = conversation_id or uuid.uuid4().hex
        key = self._key(user_id, conversation_id)

        stored = await self._redis.get(key)
        session = json.loads(stored) if stored else {"messages": [{}], "meta": {}}
        # The system prompt embeds "today": refresh it every turn, otherwise a
        # long-lived conversation keeps resolving «завтра» against a stale date.
        categories = await self._bookly.categories()
        session["messages"][0] = {"role": "system", "content": system_prompt(categories)}

        messages: list[dict[str, Any]] = session["messages"]
        meta: dict[str, Any] = session["meta"]
        messages.append({"role": "user", "content": message})

        turn = _TurnEffects()
        turn.client_favorites = favorites or []
        reply = await self._run_llm_loop(messages, meta, token, message, turn)

        session["messages"] = _trim_history(messages)
        await self._redis.set(key, json.dumps(session, ensure_ascii=False),
                              ex=self._settings.conversation_ttl_seconds)

        return ChatResult(
            conversation_id=conversation_id,
            reply=reply,
            state=turn.state(meta),
            candidates=turn.candidates,
            slots=turn.slots,
            booking=turn.booking,
            bookings=turn.bookings,
            favorites_add=turn.favorites_add,
            favorites_show=turn.favorites_show,
        )

    async def _run_llm_loop(
        self,
        messages: list[dict[str, Any]],
        meta: dict[str, Any],
        token: str,
        user_message: str,
        turn: "_TurnEffects",
    ) -> str:
        for _ in range(self._settings.max_tool_rounds):
            response = await self._llm.complete(messages)
            if not response.tool_calls:
                return response.content or ""

            messages.append({
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in response.tool_calls
                ],
            })
            for tc in response.tool_calls:
                result = await self._execute_tool(
                    tc.function.name, tc.function.arguments, meta, token, user_message, turn
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                })
        return "Что-то я запуталась. Давай ещё раз: какая услуга и на когда?"

    async def _execute_tool(
        self,
        name: str,
        raw_args: str,
        meta: dict[str, Any],
        token: str,
        user_message: str,
        turn: "_TurnEffects",
    ) -> Any:
        try:
            args = json.loads(raw_args or "{}")
        except ValueError:
            return {"error": "invalid tool arguments"}
        log.info("tool_call", tool=name, args=args)
        try:
            return await self._dispatch(name, args, meta, token, user_message, turn)
        except UpstreamError as exc:
            # Let the LLM recover: slot taken (409), validation errors etc.
            log.warning("tool_upstream_error", tool=name, status=exc.status_code, detail=exc.detail)
            return {"error": exc.detail, "status": exc.status_code}

    async def _dispatch(
        self,
        name: str,
        args: dict[str, Any],
        meta: dict[str, Any],
        token: str,
        user_message: str,
        turn: "_TurnEffects",
    ) -> Any:
        if name == "search_candidates":
            intent: dict[str, Any] = {"query": args.get("query", "")}
            if args.get("max_price_kzt"):
                intent["max_price"] = int(args["max_price_kzt"]) * 100  # KZT -> tiyn
            request = await self._bookly.create_assistant_request(token, user_message, intent)
            candidates = await self._bookly.candidates(token, request["id"])
            # The backend matches the whole phrase with ILIKE and the model
            # loves narrow categories; top up a thin result with a keyword
            # search across all categories and merge the two.
            query = intent["query"].strip()
            if len(candidates) < 5 and (" " in query or "category_id" in intent):
                fallback = dict(intent)
                fallback.pop("category_id", None)
                fallback["query"] = max(query.split(), key=len) if query else query
                retry = await self._bookly.create_assistant_request(token, user_message, fallback)
                wider = await self._bookly.candidates(token, retry["id"])
                seen_ids = {c.get("service_id") for c in candidates}
                merged = candidates + [c for c in wider if c.get("service_id") not in seen_ids]
                if len(merged) > len(candidates):
                    request, candidates = retry, merged[:8]
            meta["request_id"] = request["id"]
            turn.candidates = candidates
            # The model is bad at tiyn arithmetic: hand it prices in tenge.
            return {"candidates": [
                {
                    "master_id": c.get("master_id"),
                    "service_id": c.get("service_id"),
                    "display_name": c.get("display_name"),
                    "service_name": c.get("service_name"),
                    "price_kzt": (c.get("price_amount") or 0) // 100,
                    "duration_minutes": c.get("duration_minutes"),
                }
                for c in candidates
            ]}

        if name == "get_slots":
            locations = await self._bookly.locations(args["master_id"])
            active = [l for l in locations if l.get("is_active", True)]
            if not active:
                return {"error": "у мастера нет активного адреса"}
            location = next((l for l in active if l.get("is_primary")), active[0])
            meta["location_id"] = location["id"]
            meta["location_address"] = location.get("address_text", "")
            slots = await self._bookly.slots(
                args["master_id"], args["service_id"], location["id"], args["date"]
            )
            turn.slots = slots[:24]
            return {"address": location.get("address_text"), "slots": slots[:24]}

        if name == "select_service":
            if not meta.get("request_id"):
                return {"error": "сначала вызови search_candidates"}
            selected = await self._bookly.select_service(
                token, meta["request_id"], args["master_id"], args["service_id"]
            )
            meta["selected"] = {"master_id": args["master_id"], "service_id": args["service_id"]}
            return {"status": selected.get("status", "selected")}

        if name == "my_bookings":
            items = await self._bookly.my_bookings(token)
            slim = [
                {
                    "service": b.get("service_name_snapshot"),
                    "starts_at_almaty": _almaty_time(b.get("starts_at", "")),
                    "status": b.get("status"),
                    "price_amount": b.get("price_amount"),
                    "currency": b.get("currency"),
                }
                for b in items
            ]
            turn.bookings = items[:10]
            meta["booking_ids"] = [b.get("id") for b in items]
            meta["booking_refs"] = {
                b.get("id"): {
                    "master_id": b.get("master_id"),
                    "service_id": b.get("master_service_id"),
                    "location_id": b.get("master_location_id"),
                    "service_name": b.get("service_name_snapshot"),
                }
                for b in items
            }
            return {"bookings": [
                {**entry, "id": b.get("id")} for entry, b in zip(slim, items)
            ]}

        if name in ("booking_slots", "reschedule_booking", "book_again"):
            ref = (meta.get("booking_refs") or {}).get(args.get("booking_id"))
            if not ref:
                return {"error": "id не из списка: сначала вызови my_bookings"}
            if name == "booking_slots":
                slots = await self._bookly.slots(
                    ref["master_id"], ref["service_id"], ref["location_id"], args["date"]
                )
                turn.slots = slots[:24]
                return {
                    "service": ref.get("service_name"),
                    "slots": slots[:24],
                    "next_step": (
                        "после подтверждения клиента вызови reschedule_booking (перенос) "
                        "или book_again (повтор) с этим же booking_id; book и "
                        "select_service здесь НЕ используются"
                    ),
                }
            created = await self._bookly.create_booking(
                token, ref["master_id"], ref["service_id"], ref["location_id"], args["starts_at"]
            )
            cancelled_old = False
            if name == "reschedule_booking":
                try:
                    await self._bookly.cancel_booking(token, args["booking_id"])
                    cancelled_old = True
                except UpstreamError as exc:
                    log.warning("reschedule_cancel_failed", detail=exc.detail)
            turn.booking = created
            return {"booking": created, "old_cancelled": cancelled_old}

        if name == "cancel_booking":
            booking_id = args.get("booking_id")
            # Only ids the model just saw via my_bookings are cancellable.
            if booking_id not in (meta.get("booking_ids") or []):
                return {"error": "id не из списка: сначала вызови my_bookings"}
            cancelled = await self._bookly.cancel_booking(token, booking_id)
            return {"status": cancelled.get("status", "cancelled")}

        if name == "list_favorites":
            turn.favorites_show = True
            return {"favorites": turn.client_favorites}

        if name == "add_favorite":
            firms = await self._bookly.market_firms(args.get("name", ""))
            if not firms:
                return {"error": "такой салон в каталоге не нашёлся"}
            firm = firms[0]
            slim = {"id": firm.get("id"), "name": firm.get("name")}
            turn.favorites_add.append(slim)
            return {"added": slim}

        if name == "book":
            # Code-level guard on top of the prompt rule: booking requires an
            # explicit earlier selection and a known location from get_slots.
            if not meta.get("request_id") or not meta.get("selected"):
                return {"error": "нельзя бронировать без выбора услуги: вызови select_service"}
            if not meta.get("location_id"):
                return {"error": "сначала покажи слоты через get_slots"}
            booking = await self._bookly.book(
                token, meta["request_id"], meta["location_id"], args["starts_at"], args.get("comment")
            )
            # The Go booking payload has no address; the frontend card wants one.
            booking = {**booking, "address": meta.get("location_address", "")}
            turn.booking = booking
            meta["booked"] = True
            return {"booking": booking, "address": meta.get("location_address", "")}

        return {"error": f"unknown tool {name}"}


def _trim_history(messages: list[dict[str, Any]], limit: int = 40) -> list[dict[str, Any]]:
    """Keep the system prompt and a recent tail. Cut only at a user message:
    a tool result without its assistant tool_calls pair breaks the OpenAI API."""
    if len(messages) <= limit:
        return messages
    tail = messages[-(limit - 1):]
    for i, m in enumerate(tail):
        if m["role"] == "user":
            return [messages[0]] + tail[i:]
    return [messages[0], messages[-1]]


class _TurnEffects:
    """Structured side effects of one turn, for frontend cards."""

    def __init__(self) -> None:
        self.candidates: list[dict[str, Any]] = []
        self.slots: list[dict[str, Any]] = []
        self.booking: dict[str, Any] | None = None
        self.bookings: list[dict[str, Any]] = []
        self.favorites_add: list[dict[str, Any]] = []
        self.favorites_show = False
        self.client_favorites: list[dict[str, Any]] = []

    def state(self, meta: dict[str, Any]) -> str:
        if self.booking:
            return "booked"
        if self.slots:
            return "pick_slot"
        if self.candidates:
            return "recommend"
        if meta.get("booked"):
            return "booked"
        return "clarify"
