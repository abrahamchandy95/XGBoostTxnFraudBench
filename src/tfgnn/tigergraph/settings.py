from typing import ClassVar

from pydantic import (
    AliasChoices,
    Field,
    SecretStr,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from common.config import project_root


class Settings(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=project_root() / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    host: str = Field(
        default="",
        validation_alias=AliasChoices("TG_HOST", "HOST", "host"),
    )
    graphname: str = Field(
        default="",
        validation_alias=AliasChoices("TG_GRAPHNAME", "GRAPHNAME", "graphname"),
    )
    secret: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("TG_SECRET", "SECRET", "secret"),
    )
    connect_timeout_s: float = Field(
        default=30.0,
        validation_alias=AliasChoices("TG_CONNECT_TIMEOUT_S", "CONNECT_TIMEOUT_S"),
    )
    # Seconds in this Python configuration. Client converts to milliseconds.
    query_timeout_s: float = Field(
        default=86_400.0,
        validation_alias=AliasChoices("TG_QUERY_TIMEOUT_S", "QUERY_TIMEOUT_S"),
    )
    # Must exceed query_timeout_s for synchronous export calls.
    read_timeout_s: float = Field(
        default=86_520.0,
        validation_alias=AliasChoices("TG_READ_TIMEOUT_S", "READ_TIMEOUT_S"),
    )
    poll_interval_s: float = Field(
        default=15.0,
        validation_alias=AliasChoices("TG_POLL_INTERVAL_S", "POLL_INTERVAL_S"),
    )
    response_size_limit_bytes: int = Field(
        default=500_000_000,
        validation_alias=AliasChoices(
            "TG_RESPONSE_SIZE_LIMIT_BYTES", "RESPONSE_SIZE_LIMIT_BYTES"
        ),
    )

    @field_validator("host", "graphname")
    @classmethod
    def _str_required(cls, value: str, info: ValidationInfo) -> str:
        if not value:
            raise ValueError(f"{info.field_name or 'field'} must be set in .env")
        return value

    @field_validator("secret")
    @classmethod
    def _secret_required(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value():
            raise ValueError("secret must be set in .env")
        return value

    @field_validator(
        "connect_timeout_s",
        "query_timeout_s",
        "read_timeout_s",
        "poll_interval_s",
    )
    @classmethod
    def _positive_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("timeout values must be positive")
        return value

    @model_validator(mode="after")
    def _timeouts_are_consistent(self) -> "Settings":
        if self.read_timeout_s <= self.query_timeout_s:
            raise ValueError(
                "TG_READ_TIMEOUT_S must be greater than TG_QUERY_TIMEOUT_S. "
                "Use 86520 and 86400 respectively for the 24-hour profile."
            )
        return self
