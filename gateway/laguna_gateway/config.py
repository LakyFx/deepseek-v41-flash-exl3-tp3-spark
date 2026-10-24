"""Environment-backed configuration for the DGX Botio model gateway."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None else float(raw)


def _env_optional_float(name: str) -> float | None:
    raw = os.getenv(name)
    return None if raw is None or not raw.strip() else float(raw)


def _env_csv(name: str) -> tuple[str, ...]:
    raw = os.getenv(name, "")
    return tuple(part.strip() for part in raw.split(",") if part.strip())


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings.

    Defaults remain tied to the qualified one-DGX-Spark Laguna profile for
    rollback compatibility. Model-specific production overlays can select a
    different reviewed backend without changing the bearer or LAN endpoint.
    """

    upstream_base_url: str = "http://127.0.0.1:8003"
    vision_upstream_base_url: str = ""
    vision_max_model_len: int = 131_072
    vision_max_images_per_prompt: int = 4
    chat_image_overflow_mode: str = "describe"
    chat_image_note_tokens: int = 1536
    chat_image_prepare_timeout_seconds: float = 600.0
    vision_image_token_reserve: int = 4_096
    vision_video_token_reserve: int = 32_768
    api_key: str = ""
    served_model: str = "laguna-s-2.1"
    served_model_aliases: tuple[str, ...] = ()
    backend_family: str = "laguna"
    qwen_reasoning_default: str = "medium"
    qwen_reasoning_policy: str = "client_or_default"
    max_model_len: int = 262_144

    local_tokenizer_path: str = ""
    local_tokenizer_overhead_tokens: int = 1_024
    local_tokenizer_per_message_tokens: int = 64

    max_concurrency: int = 5
    max_queue: int = 64
    default_max_tokens: int = 8_192
    xhigh_default_max_tokens: int = 16_384
    xhigh_max_tokens: int = 16_384
    max_profile_default_max_tokens: int = 32_768
    max_profile_max_tokens: int = 32_768
    max_admitted_tokens: int = 600_000
    max_max_tokens: int = 16_384

    default_temperature: float = 0.7
    default_top_p: float = 0.95
    default_top_k: int = 20
    default_presence_penalty: float | None = None
    apply_sampling_defaults: bool = True
    measure_reasoning_tokens: bool = True

    compact_trigger_tokens: int = 220_000
    compact_target_tokens: int = 160_000
    compact_recent_tokens: int = 60_000
    compact_safety_tokens: int = 8_192
    compact_chunk_chars: int = 120_000
    compact_summary_tokens: int = 3_072
    compact_cache_entries: int = 256
    compact_cache_ttl_seconds: int = 21_600
    responses_tool_output_max_chars: int = 40_000
    # Zero keeps the legacy shared Chat/Responses context contract.  Qwen
    # Codex sets an explicit Responses-only physical window so Botio Chat can
    # retain its independently qualified operational envelope.
    responses_context_window: int = 0
    responses_min_output_tokens: int = 0
    responses_compact_target_tokens: int = 0
    responses_context_retry: bool = False
    chat_context_window: int = 0
    chat_min_output_tokens: int = 0
    chat_compact_target_tokens: int = 0
    chat_checkpoint_enforce: bool = False
    chat_context_retry: bool = False
    qwen_thinking_budget: bool = False
    qwen_responses_thinking_budget: bool = False
    qwen_native_vllm: bool = False
    # TensorFold detects disconnects on the generation's own socket. Its
    # vLLM-compatible generation API does not implement vLLM abort endpoints.
    upstream_abort_mode: str = "legacy"
    deepseek_v41_vllm: bool = False
    responses_checkpoint_mode: str = "off"
    responses_checkpoint_shadow_sample_rate: float = 1.0
    responses_checkpoint_max_tokens: int = 4_096
    responses_checkpoint_task_chars: int = 120_000
    responses_checkpoint_min_literal_recall: float = 1.0
    responses_checkpoint_timeout_seconds: float = 900.0
    responses_checkpoint_extract_concurrency: int = 1

    reasoning_cache_entries: int = 512
    reasoning_cache_ttl_seconds: int = 21_600

    enforce_czech: bool = True
    language_repair_attempts: int = 2
    language_repair_max_tokens: int = 8_192

    connect_timeout_seconds: float = 10.0
    upstream_connect_recovery_seconds: float = 0.0
    read_timeout_seconds: float = 3_600.0
    write_timeout_seconds: float = 120.0
    pool_timeout_seconds: float = 30.0
    health_timeout_seconds: float = 5.0
    stream_heartbeat_seconds: float = 15.0
    max_request_bytes: int = 8 * 1024 * 1024

    @classmethod
    def from_env(cls) -> Settings:
        settings = cls(
            upstream_base_url=os.getenv(
                "LAGUNA_GATEWAY_UPSTREAM", "http://127.0.0.1:8003"
            ).rstrip("/"),
            vision_upstream_base_url=os.getenv(
                "LAGUNA_GATEWAY_VISION_UPSTREAM", ""
            ).rstrip("/"),
            vision_max_model_len=_env_int(
                "LAGUNA_GATEWAY_VISION_MAX_MODEL_LEN", 131_072
            ),
            vision_max_images_per_prompt=_env_int(
                "LAGUNA_GATEWAY_VISION_MAX_IMAGES_PER_PROMPT", 4
            ),
            chat_image_overflow_mode=os.getenv(
                "LAGUNA_GATEWAY_CHAT_IMAGE_OVERFLOW_MODE", "describe"
            )
            .strip()
            .lower(),
            chat_image_note_tokens=_env_int(
                "LAGUNA_GATEWAY_CHAT_IMAGE_NOTE_TOKENS", 1536
            ),
            chat_image_prepare_timeout_seconds=_env_float(
                "LAGUNA_GATEWAY_CHAT_IMAGE_PREPARE_TIMEOUT_SECONDS", 600.0
            ),
            vision_image_token_reserve=_env_int(
                "LAGUNA_GATEWAY_VISION_IMAGE_TOKEN_RESERVE", 4_096
            ),
            vision_video_token_reserve=_env_int(
                "LAGUNA_GATEWAY_VISION_VIDEO_TOKEN_RESERVE", 32_768
            ),
            api_key=os.getenv("LAGUNA_GATEWAY_API_KEY", ""),
            served_model=os.getenv("LAGUNA_GATEWAY_MODEL", "laguna-s-2.1"),
            served_model_aliases=_env_csv("LAGUNA_GATEWAY_MODEL_ALIASES"),
            backend_family=os.getenv("LAGUNA_GATEWAY_BACKEND_FAMILY", "laguna")
            .strip()
            .lower(),
            qwen_reasoning_default=os.getenv(
                "LAGUNA_GATEWAY_QWEN_REASONING_DEFAULT", "medium"
            )
            .strip()
            .lower(),
            qwen_reasoning_policy=os.getenv(
                "LAGUNA_GATEWAY_QWEN_REASONING_POLICY", "client_or_default"
            )
            .strip()
            .lower(),
            max_model_len=_env_int("LAGUNA_GATEWAY_MAX_MODEL_LEN", 262_144),
            local_tokenizer_path=os.getenv(
                "LAGUNA_GATEWAY_LOCAL_TOKENIZER_PATH", ""
            ).strip(),
            local_tokenizer_overhead_tokens=_env_int(
                "LAGUNA_GATEWAY_LOCAL_TOKENIZER_OVERHEAD_TOKENS", 1_024
            ),
            local_tokenizer_per_message_tokens=_env_int(
                "LAGUNA_GATEWAY_LOCAL_TOKENIZER_PER_MESSAGE_TOKENS", 64
            ),
            max_concurrency=_env_int("LAGUNA_GATEWAY_MAX_CONCURRENCY", 5),
            max_queue=_env_int("LAGUNA_GATEWAY_MAX_QUEUE", 64),
            default_max_tokens=_env_int("LAGUNA_GATEWAY_DEFAULT_MAX_TOKENS", 8_192),
            xhigh_default_max_tokens=_env_int(
                "LAGUNA_GATEWAY_XHIGH_DEFAULT_MAX_TOKENS", 16_384
            ),
            xhigh_max_tokens=_env_int("LAGUNA_GATEWAY_XHIGH_MAX_TOKENS", 16_384),
            max_profile_default_max_tokens=_env_int(
                "LAGUNA_GATEWAY_MAX_PROFILE_DEFAULT_MAX_TOKENS", 32_768
            ),
            max_profile_max_tokens=_env_int(
                "LAGUNA_GATEWAY_MAX_PROFILE_MAX_TOKENS", 32_768
            ),
            max_admitted_tokens=_env_int("LAGUNA_GATEWAY_MAX_ADMITTED_TOKENS", 600_000),
            max_max_tokens=_env_int("LAGUNA_GATEWAY_MAX_MAX_TOKENS", 16_384),
            default_temperature=_env_float("LAGUNA_GATEWAY_DEFAULT_TEMPERATURE", 0.7),
            default_top_p=_env_float("LAGUNA_GATEWAY_DEFAULT_TOP_P", 0.95),
            default_top_k=_env_int("LAGUNA_GATEWAY_DEFAULT_TOP_K", 20),
            default_presence_penalty=_env_optional_float(
                "LAGUNA_GATEWAY_DEFAULT_PRESENCE_PENALTY"
            ),
            apply_sampling_defaults=_env_bool(
                "LAGUNA_GATEWAY_APPLY_SAMPLING_DEFAULTS", True
            ),
            measure_reasoning_tokens=_env_bool(
                "LAGUNA_GATEWAY_MEASURE_REASONING_TOKENS", True
            ),
            compact_trigger_tokens=_env_int(
                "LAGUNA_GATEWAY_COMPACT_TRIGGER_TOKENS", 220_000
            ),
            compact_target_tokens=_env_int(
                "LAGUNA_GATEWAY_COMPACT_TARGET_TOKENS", 160_000
            ),
            compact_recent_tokens=_env_int(
                "LAGUNA_GATEWAY_COMPACT_RECENT_TOKENS", 60_000
            ),
            compact_safety_tokens=_env_int(
                "LAGUNA_GATEWAY_COMPACT_SAFETY_TOKENS", 8_192
            ),
            compact_chunk_chars=_env_int("LAGUNA_GATEWAY_COMPACT_CHUNK_CHARS", 120_000),
            compact_summary_tokens=_env_int(
                "LAGUNA_GATEWAY_COMPACT_SUMMARY_TOKENS", 3_072
            ),
            compact_cache_entries=_env_int("LAGUNA_GATEWAY_COMPACT_CACHE_ENTRIES", 256),
            compact_cache_ttl_seconds=_env_int(
                "LAGUNA_GATEWAY_COMPACT_CACHE_TTL_SECONDS", 21_600
            ),
            responses_tool_output_max_chars=_env_int(
                "LAGUNA_GATEWAY_RESPONSES_TOOL_OUTPUT_MAX_CHARS", 40_000
            ),
            responses_context_window=_env_int(
                "LAGUNA_GATEWAY_RESPONSES_CONTEXT_WINDOW", 0
            ),
            responses_min_output_tokens=_env_int(
                "LAGUNA_GATEWAY_RESPONSES_MIN_OUTPUT_TOKENS", 0
            ),
            responses_compact_target_tokens=_env_int(
                "LAGUNA_GATEWAY_RESPONSES_COMPACT_TARGET_TOKENS", 0
            ),
            responses_context_retry=_env_bool(
                "LAGUNA_GATEWAY_RESPONSES_CONTEXT_RETRY", False
            ),
            chat_context_window=_env_int("LAGUNA_GATEWAY_CHAT_CONTEXT_WINDOW", 0),
            chat_min_output_tokens=_env_int("LAGUNA_GATEWAY_CHAT_MIN_OUTPUT_TOKENS", 0),
            chat_compact_target_tokens=_env_int(
                "LAGUNA_GATEWAY_CHAT_COMPACT_TARGET_TOKENS", 0
            ),
            chat_checkpoint_enforce=_env_bool(
                "LAGUNA_GATEWAY_CHAT_CHECKPOINT_ENFORCE", False
            ),
            chat_context_retry=_env_bool("LAGUNA_GATEWAY_CHAT_CONTEXT_RETRY", False),
            qwen_thinking_budget=_env_bool(
                "LAGUNA_GATEWAY_QWEN_THINKING_BUDGET", False
            ),
            qwen_responses_thinking_budget=_env_bool(
                "LAGUNA_GATEWAY_QWEN_RESPONSES_THINKING_BUDGET", False
            ),
            deepseek_v41_vllm=_env_bool("LAGUNA_GATEWAY_DEEPSEEK_V41_VLLM", False),
            qwen_native_vllm=_env_bool("LAGUNA_GATEWAY_QWEN_NATIVE_VLLM", False),
            upstream_abort_mode=os.getenv(
                "LAGUNA_GATEWAY_UPSTREAM_ABORT_MODE", "legacy"
            ).strip().lower(),
            responses_checkpoint_mode=os.getenv(
                "LAGUNA_GATEWAY_RESPONSES_CHECKPOINT_MODE", "off"
            )
            .strip()
            .lower(),
            responses_checkpoint_shadow_sample_rate=_env_float(
                "LAGUNA_GATEWAY_RESPONSES_CHECKPOINT_SHADOW_SAMPLE_RATE", 1.0
            ),
            responses_checkpoint_max_tokens=_env_int(
                "LAGUNA_GATEWAY_RESPONSES_CHECKPOINT_MAX_TOKENS", 4_096
            ),
            responses_checkpoint_task_chars=_env_int(
                "LAGUNA_GATEWAY_RESPONSES_CHECKPOINT_TASK_CHARS", 120_000
            ),
            responses_checkpoint_min_literal_recall=_env_float(
                "LAGUNA_GATEWAY_RESPONSES_CHECKPOINT_MIN_LITERAL_RECALL", 1.0
            ),
            responses_checkpoint_timeout_seconds=_env_float(
                "LAGUNA_GATEWAY_RESPONSES_CHECKPOINT_TIMEOUT_SECONDS", 900.0
            ),
            responses_checkpoint_extract_concurrency=_env_int(
                "LAGUNA_GATEWAY_RESPONSES_CHECKPOINT_EXTRACT_CONCURRENCY", 1
            ),
            reasoning_cache_entries=_env_int(
                "LAGUNA_GATEWAY_REASONING_CACHE_ENTRIES", 512
            ),
            reasoning_cache_ttl_seconds=_env_int(
                "LAGUNA_GATEWAY_REASONING_CACHE_TTL_SECONDS", 21_600
            ),
            enforce_czech=_env_bool("LAGUNA_GATEWAY_ENFORCE_CZECH", True),
            language_repair_attempts=_env_int(
                "LAGUNA_GATEWAY_LANGUAGE_REPAIR_ATTEMPTS", 2
            ),
            language_repair_max_tokens=_env_int(
                "LAGUNA_GATEWAY_LANGUAGE_REPAIR_MAX_TOKENS", 8_192
            ),
            connect_timeout_seconds=_env_float(
                "LAGUNA_GATEWAY_CONNECT_TIMEOUT_SECONDS", 10.0
            ),
            upstream_connect_recovery_seconds=_env_float(
                "LAGUNA_GATEWAY_UPSTREAM_CONNECT_RECOVERY_SECONDS", 0.0
            ),
            read_timeout_seconds=_env_float(
                "LAGUNA_GATEWAY_READ_TIMEOUT_SECONDS", 3_600.0
            ),
            write_timeout_seconds=_env_float(
                "LAGUNA_GATEWAY_WRITE_TIMEOUT_SECONDS", 120.0
            ),
            pool_timeout_seconds=_env_float(
                "LAGUNA_GATEWAY_POOL_TIMEOUT_SECONDS", 30.0
            ),
            health_timeout_seconds=_env_float(
                "LAGUNA_GATEWAY_HEALTH_TIMEOUT_SECONDS", 5.0
            ),
            stream_heartbeat_seconds=_env_float(
                "LAGUNA_GATEWAY_STREAM_HEARTBEAT_SECONDS", 15.0
            ),
            max_request_bytes=_env_int(
                "LAGUNA_GATEWAY_MAX_REQUEST_BYTES", 8 * 1024 * 1024
            ),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.upstream_base_url.startswith(("http://", "https://")):
            raise ValueError("LAGUNA_GATEWAY_UPSTREAM must be an HTTP(S) URL")
        if (
            self.vision_upstream_base_url
            and not self.vision_upstream_base_url.startswith(("http://", "https://"))
        ):
            raise ValueError("LAGUNA_GATEWAY_VISION_UPSTREAM must be an HTTP(S) URL")
        if self.vision_max_model_len <= (
            self.max_profile_max_tokens + self.compact_safety_tokens + 2
        ):
            raise ValueError(
                "vision model context must exceed maximum output and safety reserve"
            )
        if self.vision_image_token_reserve < 1:
            raise ValueError("vision image-token reserve must be positive")
        if self.chat_image_overflow_mode not in {"off", "describe"}:
            raise ValueError("chat_image_overflow_mode must be off or describe")
        if not 256 <= self.chat_image_note_tokens <= 8192:
            raise ValueError("chat_image_note_tokens must be between 256 and 8192")
        if (
            not math.isfinite(self.chat_image_prepare_timeout_seconds)
            or self.chat_image_prepare_timeout_seconds <= 0
        ):
            raise ValueError("chat_image_prepare_timeout_seconds must be positive")
        if self.vision_max_images_per_prompt < 1:
            raise ValueError("vision image limit must be positive")
        if self.vision_video_token_reserve < 1:
            raise ValueError("vision video-token reserve must be positive")
        if not self.served_model:
            raise ValueError("LAGUNA_GATEWAY_MODEL must not be empty")
        if self.backend_family not in {
            "agents_a1",
            "deepseek_ds4",
            "deepseek_pulsar",
            "laguna",
            "minimax_m3",
            "qwen38_llamacpp",
            "qwen38_vllm",
        }:
            raise ValueError(
                "LAGUNA_GATEWAY_BACKEND_FAMILY must be agents_a1, "
                "deepseek_ds4, deepseek_pulsar, laguna, minimax_m3, "
                "qwen38_llamacpp or qwen38_vllm"
            )
        if self.qwen_reasoning_default not in {"none", "low", "medium", "xhigh"}:
            raise ValueError(
                "LAGUNA_GATEWAY_QWEN_REASONING_DEFAULT must be none, low, "
                "medium or xhigh"
            )
        if self.qwen_reasoning_policy not in {
            "client_or_default",
            "fixed_medium",
            "fixed_xhigh",
        }:
            raise ValueError(
                "LAGUNA_GATEWAY_QWEN_REASONING_POLICY must be "
                "client_or_default, fixed_medium or fixed_xhigh"
            )
        if self.served_model in self.served_model_aliases:
            raise ValueError("served model must not also be configured as an alias")
        if len(set(self.served_model_aliases)) != len(self.served_model_aliases):
            raise ValueError("served model aliases must be unique")
        if any(":" in alias for alias in self.served_model_aliases):
            raise ValueError("served model aliases must not contain profile separators")
        if self.local_tokenizer_overhead_tokens < 0:
            raise ValueError("local tokenizer overhead must not be negative")
        if self.local_tokenizer_per_message_tokens < 0:
            raise ValueError("local tokenizer per-message margin must not be negative")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        if not 0 <= self.upstream_connect_recovery_seconds <= 1800:
            raise ValueError(
                "Upstream connection recovery must be between 0 and 1800 seconds"
            )
        if self.max_queue < self.max_concurrency:
            raise ValueError("max_queue must be at least max_concurrency")
        if not 1 <= self.default_max_tokens <= self.max_max_tokens:
            raise ValueError("standard output limits are outside the allowed range")
        if not (
            self.default_max_tokens
            <= self.max_max_tokens
            <= self.xhigh_default_max_tokens
            <= self.xhigh_max_tokens
            <= self.max_profile_default_max_tokens
            <= self.max_profile_max_tokens
        ):
            raise ValueError("effort profile output limits are not monotonic")
        if self.compact_safety_tokens < 1:
            raise ValueError("compaction safety reserve must be positive")
        if self.max_model_len <= (
            self.max_profile_max_tokens + self.compact_safety_tokens
        ):
            raise ValueError(
                "model context must exceed maximum output and safety reserve"
            )
        if self.max_admitted_tokens < self.max_profile_max_tokens:
            raise ValueError("admission token budget is below one max-profile output")
        if self.default_temperature < 0:
            raise ValueError("default temperature must be non-negative")
        if not 0 < self.default_top_p <= 1:
            raise ValueError("default top_p must be in (0, 1]")
        if self.default_top_k < 0:
            raise ValueError("default top_k must be non-negative")
        if (
            self.default_presence_penalty is not None
            and not -2 <= self.default_presence_penalty <= 2
        ):
            raise ValueError("default presence_penalty must be in [-2, 2]")
        if not (
            0
            < self.compact_target_tokens
            < self.compact_trigger_tokens
            < self.max_model_len
        ):
            raise ValueError("invalid compaction target/trigger/model limits")
        if self.compact_recent_tokens >= self.compact_target_tokens:
            raise ValueError("recent-token budget must be below compact target")
        if (
            self.responses_tool_output_max_chars != 0
            and self.responses_tool_output_max_chars < 1_000
        ):
            raise ValueError("Responses tool-output text limit is unreasonably small")
        if self.responses_context_window < 0:
            raise ValueError(
                "LAGUNA_GATEWAY_RESPONSES_CONTEXT_WINDOW must not be negative"
            )
        if not 0 <= self.responses_min_output_tokens <= self.max_profile_max_tokens:
            raise ValueError(
                "LAGUNA_GATEWAY_RESPONSES_MIN_OUTPUT_TOKENS must be between 0 "
                "and the maximum output cap"
            )
        responses_context = self.responses_context_limit
        if responses_context <= (
            self.responses_output_floor(self.max_profile_max_tokens)
            + self.compact_safety_tokens
        ):
            raise ValueError(
                "Responses context must exceed its minimum output and safety reserve"
            )
        if self.responses_compact_target_tokens and not (
            0
            < self.responses_compact_target_tokens
            < responses_context
            - self.responses_output_floor(self.max_profile_max_tokens)
            - self.compact_safety_tokens
        ):
            raise ValueError(
                "LAGUNA_GATEWAY_RESPONSES_COMPACT_TARGET_TOKENS must fit "
                "below the Responses emergency input ceiling"
            )
        if self.responses_checkpoint_mode not in {"off", "shadow", "enforce"}:
            raise ValueError(
                "LAGUNA_GATEWAY_RESPONSES_CHECKPOINT_MODE must be off, shadow "
                "or enforce"
            )
        if not 1 <= self.responses_checkpoint_extract_concurrency <= 4:
            raise ValueError(
                "Checkpoint extraction concurrency must be between 1 and 4"
            )
        if self.chat_context_window < 0 or self.chat_min_output_tokens < 0:
            raise ValueError("Chat context and output floor must not be negative")
        chat_context = self.chat_context_window or self.max_model_len
        if self.chat_min_output_tokens > self.max_profile_max_tokens:
            raise ValueError("Chat output floor exceeds maximum output")
        chat_floor = self.chat_min_output_tokens or self.max_profile_max_tokens
        if chat_context <= chat_floor + self.compact_safety_tokens:
            raise ValueError(
                "Chat context must exceed its output floor and safety reserve"
            )
        if self.chat_compact_target_tokens and not (
            0
            < self.chat_compact_target_tokens
            < chat_context - chat_floor - self.compact_safety_tokens
        ):
            raise ValueError("Chat checkpoint target must fit below its input ceiling")
        if self.deepseek_v41_vllm and (self.backend_family != "qwen38_vllm" or self.qwen_thinking_budget or self.qwen_responses_thinking_budget):
            raise ValueError("DeepSeek V4.1 compatibility requires vLLM family and disabled Qwen thinking budgets")
        if self.qwen_thinking_budget and self.backend_family != "qwen38_vllm":
            raise ValueError("Thinking budget requires the qualified Qwen vLLM backend")
        if self.qwen_responses_thinking_budget and not (
            self.qwen_thinking_budget and self.qwen_native_vllm
        ):
            raise ValueError(
                "Responses thinking budget requires native vLLM and the shared budget policy"
            )
        if self.qwen_native_vllm and self.backend_family != "qwen38_vllm":
            raise ValueError(
                "Native vLLM request ownership requires the Qwen vLLM backend"
            )
        if self.upstream_abort_mode not in {"legacy", "tensorfold_disconnect"}:
            raise ValueError("Unsupported upstream_abort_mode")
        if self.upstream_abort_mode == "tensorfold_disconnect" and not (
            self.backend_family == "qwen38_vllm" and self.qwen_native_vllm
        ):
            raise ValueError("TensorFold disconnect mode requires native compatibility transport")
        if not 0 <= self.responses_checkpoint_shadow_sample_rate <= 1:
            raise ValueError(
                "LAGUNA_GATEWAY_RESPONSES_CHECKPOINT_SHADOW_SAMPLE_RATE "
                "must be in [0, 1]"
            )
        if not 512 <= self.responses_checkpoint_max_tokens <= 32_768:
            raise ValueError(
                "LAGUNA_GATEWAY_RESPONSES_CHECKPOINT_MAX_TOKENS must be "
                "between 512 and 32768"
            )
        if self.responses_checkpoint_task_chars < 4_000:
            raise ValueError(
                "LAGUNA_GATEWAY_RESPONSES_CHECKPOINT_TASK_CHARS must be at least 4000"
            )
        if not 0 <= self.responses_checkpoint_min_literal_recall <= 1:
            raise ValueError(
                "LAGUNA_GATEWAY_RESPONSES_CHECKPOINT_MIN_LITERAL_RECALL "
                "must be in [0, 1]"
            )
        if not 1 <= self.responses_checkpoint_timeout_seconds <= 3_600:
            raise ValueError(
                "LAGUNA_GATEWAY_RESPONSES_CHECKPOINT_TIMEOUT_SECONDS must be "
                "between 1 and 3600"
            )
        if self.max_request_bytes < 1024:
            raise ValueError("max_request_bytes is unreasonably small")
        if not 0 < self.stream_heartbeat_seconds <= 60:
            raise ValueError("stream heartbeat must be in (0, 60] seconds")

    @property
    def reasoning_message_field(self) -> str:
        """Canonical assistant reasoning field expected by the upstream."""

        if self.backend_family in {
            "deepseek_pulsar",
            "minimax_m3",
            "qwen38_llamacpp",
            "qwen38_vllm",
        }:
            return "reasoning_content"
        return "reasoning"

    @property
    def effective_reasoning_policy(self) -> str:
        if self.deepseek_v41_vllm:
            return "client_or_default"
        return self.qwen_reasoning_policy

    @property
    def native_thinking_modes(self) -> tuple[str, ...]:
        if self.deepseek_v41_vllm:
            return ("none", "low", "medium", "max")
        if self.backend_family == "minimax_m3":
            return ("disabled", "adaptive", "enabled")
        if self.backend_family == "agents_a1":
            return ("disabled", "enabled")
        if self.backend_family in {"deepseek_ds4", "deepseek_pulsar"}:
            return ("none", "high", "max")
        if self.backend_family in {"qwen38_llamacpp", "qwen38_vllm"}:
            return ("none", "low", "medium", "xhigh")
        return ("off", "max")

    @property
    def vision_enabled(self) -> bool:
        return bool(self.vision_upstream_base_url)

    def safe_input_limit(self, max_tokens: int) -> int:
        """Largest prompt that still reserves output and the safety envelope."""

        return self.max_model_len - int(max_tokens) - self.compact_safety_tokens

    def chat_context_limit(self, *, vision: bool = False) -> int:
        """Use the Chat window without enlarging a separate vision backend."""

        context = self.chat_context_window or self.max_model_len
        if vision:
            return (
                min(self.chat_context_window, self.vision_max_model_len)
                if (self.chat_context_window)
                else self.vision_max_model_len
            )
        return context

    @property
    def responses_context_limit(self) -> int:
        """Physical context used only by the stateless Responses ingress."""

        return self.responses_context_window or self.max_model_len

    def responses_output_floor(self, requested_max_tokens: int) -> int:
        """Minimum useful output retained before emergency checkpointing."""

        requested = max(1, int(requested_max_tokens))
        if self.responses_min_output_tokens <= 0:
            return requested
        return min(requested, self.responses_min_output_tokens)

    def responses_safe_input_limit(self, requested_max_tokens: int) -> int:
        """Largest Responses input that can still produce a useful next step."""

        return (
            self.responses_context_limit
            - self.responses_output_floor(requested_max_tokens)
            - self.compact_safety_tokens
        )
