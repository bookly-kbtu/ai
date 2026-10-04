"""DialogService with fake LLM and fake Bookly: the tool loop, the booking
guard and turn state are our logic — OpenAI and the Go backend are not."""

import json
from types import SimpleNamespace
from typing import Any

import pytest

from app.core.config import Settings
from app.services.dialog_service import DialogService, _trim_history


def tool_call(call_id: str, name: str, **args: Any) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


class FakeLLM:
    """Returns scripted responses in order."""

    def __init__(self, responses: list[SimpleNamespace]):
        self._responses = list(responses)
        self.seen_messages: list[list[dict]] = []

    async def complete(self, messages):
        self.seen_messages.append([dict(m) for m in messages])
        return self._responses.pop(0)


class FakeBookly:
    def __init__(self):
        self.booked: list[dict] = []

    async def categories(self):
        return [{"id": "cat-1", "name": "Ногтевой сервис", "is_active": True}]

    async def create_assistant_request(self, token, transcript, intent):
        return {"id": "req-1", "status": "parsed"}

    async def candidates(self, token, request_id, limit=5):
        return [{"master_id": "m-1", "service_id": "s-1", "display_name": "Аружан",
                 "service_name": "Маникюр", "price_amount": 800000, "currency": "KZT",
                 "duration_minutes": 60}]

    async def locations(self, master_id):
        return [{"id": "loc-1", "is_primary": True, "is_active": True,
                 "address_text": "пр. Достык, 91"}]

    async def slots(self, master_id, service_id, location_id, date):
        return [{"starts_at": "2026-10-06T10:00:00+05:00", "ends_at": "2026-10-06T11:00:00+05:00"}]

    async def select_service(self, token, request_id, master_id, service_id):
        return {"status": "selected"}

    async def book(self, token, request_id, location_id, starts_at, comment=None):
        self.booked.append({"request_id": request_id, "location_id": location_id,
                            "starts_at": starts_at})
        return {"id": "booking-1", "starts_at": starts_at}


class FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value


def make_service(llm: FakeLLM, bookly: FakeBookly | None = None, redis: FakeRedis | None = None):
    settings = Settings(jwt_secret="x", openai_api_key="x")
    return DialogService(bookly or FakeBookly(), llm, redis or FakeRedis(), settings)


async def test_search_turn_returns_candidates_and_recommend_state():
    llm = FakeLLM([
        SimpleNamespace(content=None, tool_calls=[tool_call("1", "search_candidates", query="маникюр")]),
        SimpleNamespace(content="Рекомендую Аружан за 8000 тг.", tool_calls=None),
    ])
    service = make_service(llm)

    result = await service.chat("user-1", "jwt", None, "хочу маникюр")

    assert result.state == "recommend"
    assert result.candidates[0]["display_name"] == "Аружан"
    assert "Аружан" in result.reply
    assert result.booking is None


async def test_book_rejected_without_selection():
    bookly = FakeBookly()
    llm = FakeLLM([
        SimpleNamespace(content=None, tool_calls=[
            tool_call("1", "book", starts_at="2026-10-06T10:00:00+05:00"),
        ]),
        SimpleNamespace(content="Сначала выберем услугу.", tool_calls=None),
    ])
    service = make_service(llm, bookly)

    result = await service.chat("user-1", "jwt", None, "запиши меня")

    assert bookly.booked == []
    assert result.state == "clarify"
    # The guard's message went back to the LLM as the tool result.
    tool_messages = [m for m in llm.seen_messages[-1] if m["role"] == "tool"]
    assert "select_service" in tool_messages[-1]["content"]


async def test_full_flow_books_after_selection_and_slots():
    bookly = FakeBookly()
    redis = FakeRedis()
    llm = FakeLLM([
        # turn 1: search
        SimpleNamespace(content=None, tool_calls=[tool_call("1", "search_candidates", query="маникюр")]),
        SimpleNamespace(content="Есть Аружан.", tool_calls=None),
        # turn 2: select + slots
        SimpleNamespace(content=None, tool_calls=[
            tool_call("2", "select_service", master_id="m-1", service_id="s-1"),
            tool_call("3", "get_slots", master_id="m-1", service_id="s-1", date="2026-10-06"),
        ]),
        SimpleNamespace(content="Есть 10:00. Записать?", tool_calls=None),
        # turn 3: book
        SimpleNamespace(content=None, tool_calls=[
            tool_call("4", "book", starts_at="2026-10-06T10:00:00+05:00"),
        ]),
        SimpleNamespace(content="Записала на 10:00!", tool_calls=None),
    ])
    service = make_service(llm, bookly, redis)

    r1 = await service.chat("user-1", "jwt", None, "хочу маникюр")
    r2 = await service.chat("user-1", "jwt", r1.conversation_id, "давай к Аружан во вторник")
    r3 = await service.chat("user-1", "jwt", r2.conversation_id, "да, на 10 утра")

    assert r2.state == "pick_slot"
    assert r3.state == "booked"
    assert r3.booking["id"] == "booking-1"
    assert bookly.booked[0]["location_id"] == "loc-1"


def test_trim_history_cuts_at_user_boundary():
    messages = [{"role": "system", "content": "s"}]
    for i in range(30):
        messages += [
            {"role": "user", "content": f"u{i}"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": str(i)}]},
            {"role": "tool", "tool_call_id": str(i), "content": "{}"},
            {"role": "assistant", "content": f"a{i}"},
        ]
    trimmed = _trim_history(messages, limit=40)

    assert trimmed[0]["role"] == "system"
    assert len(trimmed) <= 40
    assert trimmed[1]["role"] == "user"  # never starts mid tool exchange


async def test_my_bookings_tool_returns_agenda():
    bookly = FakeBookly()
    bookly.my_bookings = lambda token, limit=20: _async([
        {"id": "b1", "service_name_snapshot": "Маникюр", "starts_at": "2026-10-06T10:00:00Z",
         "status": "confirmed", "price_amount": 800000, "currency": "KZT"},
    ])
    llm = FakeLLM([
        SimpleNamespace(content=None, tool_calls=[tool_call("1", "my_bookings")]),
        SimpleNamespace(content="У вас маникюр во вторник в 15:00.", tool_calls=None),
    ])
    service = make_service(llm, bookly)

    result = await service.chat("user-1", "jwt", None, "какие у меня записи?")

    assert result.bookings[0]["id"] == "b1"
    assert "маникюр" in result.reply.lower()


async def _async(value):
    return value
