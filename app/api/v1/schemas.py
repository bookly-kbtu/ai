from typing import Any, Literal

from pydantic import BaseModel, Field


class FavoriteItem(BaseModel):
    id: str = Field(max_length=64)
    name: str = Field(max_length=200)


class ChatRequest(BaseModel):
    conversation_id: str | None = Field(default=None, max_length=64)
    message: str = Field(min_length=1, max_length=2000)
    favorites: list[FavoriteItem] = Field(default_factory=list, max_length=50)


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    state: Literal["clarify", "recommend", "pick_slot", "confirm", "booked", "failed"]
    candidates: list[dict[str, Any]]
    slots: list[dict[str, Any]]
    booking: dict[str, Any] | None
    bookings: list[dict[str, Any]]
    favorites_add: list[dict[str, Any]]
    favorites_show: bool
