import httpx
from dishka import Provider, Scope, make_async_container, provide
from openai import AsyncOpenAI
from redis.asyncio import Redis

from app.clients.bookly import BooklyClient
from app.core.config import Settings
from app.services.dialog_service import DialogService
from app.services.llm import LLMClient


class AppProvider(Provider):
    scope = Scope.APP

    @provide
    def settings(self) -> Settings:
        return Settings()

    @provide
    async def http_client(self, settings: Settings) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=settings.bookly_api_url.rstrip("/") + "/api/v1",
            timeout=settings.request_timeout_seconds,
        )

    @provide
    def bookly(self, http: httpx.AsyncClient) -> BooklyClient:
        return BooklyClient(http)

    @provide
    def openai_client(self, settings: Settings) -> AsyncOpenAI:
        # An empty key must not crash DI: the chat call then fails with a
        # clean 502 LLMError instead of a 500 at container resolution.
        return AsyncOpenAI(api_key=settings.openai_api_key or "missing")

    @provide
    def llm(self, client: AsyncOpenAI, settings: Settings) -> LLMClient:
        return LLMClient(client, settings.openai_model)

    @provide
    async def redis(self, settings: Settings) -> Redis:
        return Redis.from_url(settings.redis_url, decode_responses=True)

    @provide
    def dialog_service(
        self, bookly: BooklyClient, llm: LLMClient, redis: Redis, settings: Settings
    ) -> DialogService:
        return DialogService(bookly, llm, redis, settings)


def build_container():
    return make_async_container(AppProvider())
