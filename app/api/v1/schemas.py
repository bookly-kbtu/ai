from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    conversation_id: str | None = Field(default=None, max_length=64)
    message: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    state: Literal["clarify", "recommend", "pick_slot", "confirm", "booked", "failed"]
    candidates: list[dict[str, Any]]
    slots: list[dict[str, Any]]
    booking: dict[str, Any] | None
