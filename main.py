from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star
from astrbot.core.message.components import Image, Reply


class ForceImageCaption(Star):
    """Ensure images are converted to text before a text-only main model sees them."""

    CAPTION_RE = re.compile(
        r"<image_caption>(.*?)</image_caption>",
        re.IGNORECASE | re.DOTALL,
    )
    FAILURE_RE = re.compile(
        r"\[\s*Image\s+Captioning\s+Failed\s*\]",
        re.IGNORECASE,
    )

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        # session_key -> {"ts": float, "images": list[str], "message_id": str}
        # Only lightweight image references are kept in memory; no image bytes are copied.
        self._recent_images: dict[str, dict[str, Any]] = {}
        self._recent_images_lock = asyncio.Lock()
        self._persist_lock = asyncio.Lock()
        self._last_persist_cleanup_ts = 0.0

        # v1.2.6: keep the image-chat experience while avoiding duplicate vision calls and group-chat catch-up replies.
        # Cache entries contain caption text only; no image bytes are retained here.
        self._caption_cache: dict[str, dict[str, Any]] = {}
        self._caption_cache_lock = asyncio.Lock()
        self._caption_inflight: dict[str, asyncio.Task[str]] = {}
        self._caption_inflight_lock = asyncio.Lock()
        self._caption_request_lock = asyncio.Lock()

        # provider_id -> unix timestamp until which new vision calls are suppressed.
        # This circuit breaker is primarily for RPM/TPM exhaustion and billing/auth
        # failures, so repeated chat messages do not keep hammering a blocked API.
        self._provider_cooldown_until: dict[str, float] = {}
        self._provider_cooldown_lock = asyncio.Lock()

    def _provider_settings(self, event: AstrMessageEvent) -> dict[str, Any]:
        try:
            cfg = self.context.get_config(umo=event.unified_msg_origin)
            settings = cfg.get("provider_settings", {})
            return settings if isinstance(settings, dict) else {}
        except Exception:
            return {}

    def _caption_provider_id(self, event: AstrMessageEvent) -> str:
        custom = str(self.config.get("caption_provider_id", "") or "").strip()
        if custom:
            return custom

        return str(
            self._provider_settings(event).get(
                "default_image_caption_provider_id", ""
            )
            or ""
        ).strip()

    def _caption_prompt(self, event: AstrMessageEvent) -> str:
        custom = str(self.config.get("caption_prompt", "") or "").strip()
        if custom:
            return custom

        prompt = str(
            self._provider_settings(event).get("image_caption_prompt", "") or ""
        ).strip()
        if prompt:
            return prompt

        return (
            "把自己当成刚刚看了一眼朋友发来的图片，理解它想表达什么。"
            "不要逐项描述画面，不要使用‘图片中’‘画面里’‘可以看到’‘这是一张’等解说式表达。"
            "优先提取后续聊天真正需要的信息：情绪、动作、梗、态度、关键文字或截图中的重点。"
            "表情包重点理解情绪和意思；截图重点提取与当前聊天有关的信息。"
            "只输出1～2句简短自然的理解结果，不要直接回复用户。"
        )

    def _plugin_retries_enabled(self) -> bool:
        # New guard in v1.2.4. Existing v1.2.2 installations may already have
        # max_retries=2 persisted in their config; this opt-in switch ensures an
        # upgrade does not silently keep double-retrying on AstrBot 4.28+.
        return bool(self.config.get("enable_plugin_retries", False))

    def _max_retries(self) -> int:
        if not self._plugin_retries_enabled():
            return 0
        try:
            return max(0, min(int(self.config.get("max_retries", 0)), 10))
        except (TypeError, ValueError):
            return 0

    def _retry_delay(self) -> float:
        try:
            return max(0.0, min(float(self.config.get("retry_delay_seconds", 1.0)), 30.0))
        except (TypeError, ValueError):
            return 1.0

    def _retry_backoff(self) -> float:
        try:
            return max(1.0, min(float(self.config.get("retry_backoff", 1.8)), 5.0))
        except (TypeError, ValueError):
            return 1.8

    def _caption_cache_ttl(self) -> int:
        try:
            return max(10, min(int(self.config.get("caption_cache_ttl_seconds", 300)), 3600))
        except (TypeError, ValueError):
            return 300

    def _rate_limit_cooldown(self) -> int:
        try:
            return max(5, min(int(self.config.get("rate_limit_cooldown_seconds", 60)), 1800))
        except (TypeError, ValueError):
            return 60

    def _reuse_caption_for_natural_followup(self) -> bool:
        return bool(self.config.get("reuse_caption_for_natural_followup", True))

    def _serialize_caption_requests(self) -> bool:
        return bool(self.config.get("serialize_caption_requests", True))

    def _caption_timeout(self) -> int:
        try:
            return max(5, min(int(self.config.get("caption_timeout_seconds", 60)), 300))
        except (TypeError, ValueError):
            return 60

    def _recent_memory_enabled(self) -> bool:
        return bool(self.config.get("remember_recent_images", True))

    def _recent_image_ttl(self) -> int:
        try:
            return max(10, min(int(self.config.get("recent_image_ttl_seconds", 300)), 3600))
        except (TypeError, ValueError):
            return 300

    def _recent_image_max_count(self) -> int:
        try:
            return max(1, min(int(self.config.get("recent_image_max_count", 4)), 12))
        except (TypeError, ValueError):
            return 4

    def _natural_followup_enabled(self) -> bool:
        return bool(self.config.get("natural_followup_enabled", True))

    def _natural_followup_window(self) -> int:
        try:
            return max(5, min(int(self.config.get("natural_followup_window_seconds", 90)), 600))
        except (TypeError, ValueError):
            return 90

    def _natural_followup_same_sender_only(self) -> bool:
        return bool(self.config.get("natural_followup_same_sender_only", True))

    def _group_reply_scope_guard_enabled(self) -> bool:
        """Keep image-triggered group replies focused on the current turn.

        AstrBot may provide recent group messages as conversation context even when
        those messages did not themselves trigger an LLM call.  When a later image
        message does trigger the model, some models may try to "catch up" and
        answer those older messages.  This guard keeps the history available for
        context while marking it as background-only unless the current speaker
        explicitly refers back to it.
        """
        return bool(self.config.get("group_reply_scope_guard", True))

    def _persist_recent_images_enabled(self) -> bool:
        return bool(self.config.get("persist_recent_images", True))

    def _persistent_cache_root(self) -> Path:
        configured = str(self.config.get("persistent_cache_dir", "") or "").strip()
        if configured:
            return Path(configured).expanduser().resolve()
        # AstrBot normally runs with /AstrBot as CWD. Keeping the cache under
        # data/plugin_data makes it survive the framework's data/temp cleanup.
        return (
            Path.cwd()
            / "data"
            / "plugin_data"
            / "astrbot_plugin_force_image_caption"
            / "recent_images"
        ).resolve()

    @staticmethod
    def _ref_to_local_path(ref: str) -> Path | None:
        if not isinstance(ref, str):
            return None
        value = ref.strip()
        if not value or value.lower().startswith(("http://", "https://", "data:")):
            return None
        if value.lower().startswith("file:///"):
            value = value[8:]
            # Windows file:///C:/... compatibility.
            if len(value) > 2 and value[0] == "/" and value[2] == ":":
                value = value[1:]
        try:
            return Path(value).expanduser().resolve()
        except Exception:
            return None

    def _session_persistent_dir(self, event: AstrMessageEvent) -> Path:
        key = self._session_key(event).encode("utf-8", errors="ignore")
        digest = hashlib.sha1(key).hexdigest()[:16]
        return self._persistent_cache_root() / digest

    async def _cleanup_persistent_cache(self, *, force: bool = False) -> None:
        if not self._persist_recent_images_enabled():
            return
        now = time.time()
        if not force and now - self._last_persist_cleanup_ts < 60:
            return
        self._last_persist_cleanup_ts = now
        root = self._persistent_cache_root()
        if not root.exists():
            return
        # Keep a grace period beyond the in-memory TTL so a tool call that starts
        # near the TTL boundary does not lose its file mid-flight.
        max_age = max(self._recent_image_ttl() + 120, 300)

        def _cleanup() -> None:
            try:
                for file in root.rglob("*"):
                    if not file.is_file():
                        continue
                    try:
                        age = now - file.stat().st_mtime
                    except OSError:
                        continue
                    if age > max_age:
                        try:
                            file.unlink()
                        except OSError:
                            pass
                # Remove empty session directories bottom-up.
                for folder in sorted(
                    (x for x in root.rglob("*") if x.is_dir()),
                    key=lambda x: len(x.parts),
                    reverse=True,
                ):
                    try:
                        folder.rmdir()
                    except OSError:
                        pass
            except Exception:
                pass

        await asyncio.to_thread(_cleanup)

    async def _persist_local_image_ref(
        self,
        event: AstrMessageEvent,
        ref: str,
    ) -> str:
        """Copy an AstrBot temp image to plugin_data and keep its basename.

        Keeping the basename is deliberate: astrbot_plugin_stealer resolves an
        LLM-provided stale temp path against the current Image component by
        basename. Rewriting the component to this persistent copy therefore lets
        that tool recover instead of failing with "图片文件不存在".
        """
        if not self._persist_recent_images_enabled():
            return ref
        source = self._ref_to_local_path(ref)
        if source is None or not source.exists() or not source.is_file():
            return ref
        root = self._persistent_cache_root()
        try:
            source.relative_to(root)
            return str(source)
        except ValueError:
            pass

        target_dir = self._session_persistent_dir(event)
        target = target_dir / source.name

        def _copy() -> str:
            target_dir.mkdir(parents=True, exist_ok=True)
            # If the same temp filename is observed twice, replacing the cache copy
            # is safe and keeps basename-based recovery deterministic.
            shutil.copy2(source, target)
            return str(target.resolve())

        try:
            async with self._persist_lock:
                value = await asyncio.to_thread(_copy)
            await self._cleanup_persistent_cache()
            return value
        except Exception as exc:
            if self.config.get("debug_log", False):
                logger.debug(
                    "[ForceImageCaption] failed to persist temp image %s: %s",
                    ref,
                    exc,
                )
            return ref

    async def _image_components(
        self,
        components: Any,
        *,
        depth: int = 0,
        seen: set[int] | None = None,
    ) -> list[Image]:
        if depth > 3 or not isinstance(components, (list, tuple)):
            return []
        if seen is None:
            seen = set()
        result: list[Image] = []
        for comp in components:
            obj_id = id(comp)
            if obj_id in seen:
                continue
            seen.add(obj_id)
            if isinstance(comp, Image):
                result.append(comp)
                continue
            if isinstance(comp, Reply):
                result.extend(
                    await self._image_components(
                        getattr(comp, "chain", None),
                        depth=depth + 1,
                        seen=seen,
                    )
                )
            for attr in ("chain", "message", "content", "nodes"):
                nested = getattr(comp, attr, None)
                if isinstance(nested, (list, tuple)):
                    result.extend(
                        await self._image_components(
                            nested,
                            depth=depth + 1,
                            seen=seen,
                        )
                    )
        return result

    async def _persist_event_images(self, event: AstrMessageEvent) -> list[str]:
        """Persist local Image components and rewrite them to stable paths.

        AstrBot may delete files in data/temp before an LLM tool executes. The
        rewrite is best-effort and intentionally happens on the same event object,
        allowing downstream plugins (notably stealer's steal_meme tool) to resolve
        the stale basename back to a still-existing file.
        """
        if not self._persist_recent_images_enabled():
            return []
        try:
            chain = event.message_obj.message
        except Exception:
            return []
        comps = await self._image_components(chain)
        persisted: list[str] = []
        for comp in comps:
            candidates: list[str] = []
            for attr in ("url", "file", "path"):
                value = getattr(comp, attr, None)
                if isinstance(value, str) and value.strip():
                    candidates.append(value.strip())
            try:
                local = await comp.convert_to_file_path()
                if isinstance(local, str) and local.strip():
                    candidates.append(local.strip())
            except Exception:
                pass

            local_ref = next(
                (
                    ref
                    for ref in candidates
                    if (lambda p: p is not None and p.exists() and p.is_file())(
                        self._ref_to_local_path(ref)
                    )
                ),
                "",
            )
            if not local_ref:
                continue
            stable = await self._persist_local_image_ref(event, local_ref)
            stable_path = self._ref_to_local_path(stable)
            if stable == local_ref or stable_path is None or not stable_path.exists():
                continue
            persisted.append(stable)

            original_path = self._ref_to_local_path(local_ref)
            original_name = original_path.name if original_path else Path(local_ref).name
            wrote = False
            for attr in ("url", "file", "path"):
                try:
                    current = getattr(comp, attr, None)
                except Exception:
                    continue
                current_path = self._ref_to_local_path(current) if isinstance(current, str) else None
                same_original = bool(
                    current_path
                    and original_path
                    and os.path.normcase(str(current_path)) == os.path.normcase(str(original_path))
                )
                same_basename = bool(
                    current_path
                    and original_name
                    and current_path.name == original_name
                )
                if same_original or same_basename:
                    try:
                        setattr(comp, attr, stable)
                        wrote = True
                    except Exception:
                        pass
            if not wrote:
                # Many AstrBot Image components are mutable dataclasses. Even when
                # the original local path came only from convert_to_file_path(),
                # adding file/path gives downstream plugins a stable candidate.
                for attr in ("file", "path"):
                    try:
                        setattr(comp, attr, stable)
                        wrote = True
                        break
                    except Exception:
                        continue

            if self.config.get("debug_log", False):
                logger.info(
                    "[ForceImageCaption] persisted temp image for downstream tools old=%s new=%s rewritten=%s",
                    local_ref,
                    stable,
                    wrote,
                )
        return self._dedupe(persisted)

    def _followup_context_max_chars(self) -> int:
        try:
            return max(0, min(int(self.config.get("followup_context_max_chars", 300)), 1500))
        except (TypeError, ValueError):
            return 300

    @staticmethod
    def _part_text(part: Any) -> str:
        value = getattr(part, "text", "")
        return value if isinstance(value, str) else ""

    def _captions_from_text(self, text: str) -> list[str]:
        if not isinstance(text, str):
            return []
        return [
            m.strip()
            for m in self.CAPTION_RE.findall(text)
            if isinstance(m, str) and m.strip()
        ]

    def _existing_caption(self, req: ProviderRequest) -> str:
        captions: list[str] = []
        captions.extend(self._captions_from_text(getattr(req, "prompt", "") or ""))

        for part in getattr(req, "extra_user_content_parts", []) or []:
            captions.extend(self._captions_from_text(self._part_text(part)))

        return "\n".join(dict.fromkeys(captions)).strip()

    def _inject_silent_fallback_hint(self, req: ProviderRequest) -> None:
        """Keep the final model from exposing internal image-processing failure.

        The hint intentionally does not say that image recognition failed. It only
        constrains the final answer to available text and hides modality/internal
        processing status from the user. On AstrBot >= 4.24 it is marked temporary
        so it will not be persisted into conversation history.
        """
        if not self.config.get("suppress_failure_reply", True):
            return

        hint = (
            "<runtime_hint>"
            "本轮只依据当前可用的文本信息自然回复。"
            "不要讨论图片是否可见、识图状态或任何内部处理过程，"
            "也不要编造未提供的视觉细节。"
            "</runtime_hint>"
        )

        for part in getattr(req, "extra_user_content_parts", []) or []:
            if "<runtime_hint>" in self._part_text(part):
                return

        try:
            from astrbot.core.agent.message import TextPart

            part = TextPart(text=hint)
            mark_temp = getattr(part, "mark_as_temp", None)
            if callable(mark_temp):
                marked = mark_temp()
                if marked is not None:
                    part = marked
            req.extra_user_content_parts.append(part)
            return
        except Exception as exc:
            if self.config.get("debug_log", False):
                logger.debug(
                    "[ForceImageCaption] failed to add temporary fallback hint: %s",
                    exc,
                )

        # Compatibility fallback for older AstrBot versions.
        prompt = getattr(req, "prompt", "")
        prompt = prompt if isinstance(prompt, str) else ""
        if "<runtime_hint>" not in prompt:
            req.prompt = f"{prompt.rstrip()}\n\n{hint}".strip()

    def _strip_failure_markers(self, req: ProviderRequest) -> None:
        """Remove AstrBot's visible image-caption failure marker from this request."""
        prompt = getattr(req, "prompt", "")
        if isinstance(prompt, str) and prompt:
            req.prompt = self.FAILURE_RE.sub("", prompt).strip()

        parts = getattr(req, "extra_user_content_parts", None)
        if not isinstance(parts, list):
            return

        cleaned = []
        for part in parts:
            text = self._part_text(part)
            if text and self.FAILURE_RE.search(text):
                # Do not let the text-only main model see a visible failure marker.
                # If this part contains only the marker, drop it completely.
                remaining = self.FAILURE_RE.sub("", text).strip()
                if not remaining:
                    continue
                try:
                    part.text = remaining
                except Exception:
                    continue
            cleaned.append(part)

        req.extra_user_content_parts = cleaned

    def _remove_caption_parts(self, req: ProviderRequest) -> None:
        parts = getattr(req, "extra_user_content_parts", None)
        if not isinstance(parts, list):
            self._strip_failure_markers(req)
            return

        cleaned = []
        for part in parts:
            text = self._part_text(part)
            if "<image_caption>" in text.lower():
                continue
            if self.FAILURE_RE.search(text):
                remaining = self.FAILURE_RE.sub("", text).strip()
                if not remaining:
                    continue
                try:
                    part.text = remaining
                except Exception:
                    continue
            cleaned.append(part)

        req.extra_user_content_parts = cleaned
        self._strip_failure_markers(req)

    @staticmethod
    def _inject_caption_into_prompt(req: ProviderRequest, caption: str) -> None:
        prompt = getattr(req, "prompt", "")
        prompt = prompt if isinstance(prompt, str) else ""

        if "<image_caption>" in prompt.lower():
            return

        block = f"<image_caption>\n{caption.strip()}\n</image_caption>"
        req.prompt = f"{prompt.rstrip()}\n\n{block}".strip()

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        return list(dict.fromkeys(v for v in values if isinstance(v, str) and v))

    @staticmethod
    def _session_key(event: AstrMessageEvent) -> str:
        umo = getattr(event, "unified_msg_origin", "")
        if isinstance(umo, str) and umo.strip():
            return umo.strip()

        obj = getattr(event, "message_obj", None)
        session_id = getattr(obj, "session_id", "") if obj is not None else ""
        if session_id:
            return str(session_id)

        try:
            group_id = event.get_group_id()
        except Exception:
            group_id = ""
        try:
            sender_id = event.get_sender_id()
        except Exception:
            sender_id = ""
        return f"fallback:{group_id}:{sender_id}"

    @staticmethod
    def _message_id(event: AstrMessageEvent) -> str:
        obj = getattr(event, "message_obj", None)
        value = getattr(obj, "message_id", "") if obj is not None else ""
        return str(value or "")

    @staticmethod
    def _sender_id(event: AstrMessageEvent) -> str:
        try:
            value = event.get_sender_id()
            if value is not None and str(value).strip():
                return str(value).strip()
        except Exception:
            pass
        obj = getattr(event, "message_obj", None)
        sender = getattr(obj, "sender", None) if obj is not None else None
        for attr in ("user_id", "id", "sender_id"):
            value = getattr(sender, attr, None) if sender is not None else None
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    @staticmethod
    def _is_group_chat(event: AstrMessageEvent) -> bool:
        try:
            group_id = event.get_group_id()
            if group_id is not None and str(group_id).strip():
                return True
        except Exception:
            pass

        # Adapter compatibility fallback.  aiocqhttp and several other adapters
        # encode the message type in UMO/session identifiers.
        candidates = [
            getattr(event, "unified_msg_origin", ""),
            getattr(getattr(event, "message_obj", None), "session_id", ""),
        ]
        return any(
            isinstance(value, str) and "groupmessage" in value.lower()
            for value in candidates
        )

    def _inject_group_reply_scope_hint(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
        *,
        image_source: str,
    ) -> None:
        """Prevent a group image turn from becoming a backlog/catch-up reply.

        This does *not* delete chat history.  Older group messages remain available
        to the model as background context.  We only add a temporary current-turn
        instruction so the model does not interpret every historical line as an
        unanswered request that now needs a response.
        """
        if not self._group_reply_scope_guard_enabled() or not self._is_group_chat(event):
            return

        source_text = "当前消息中的图片" if image_source == "current" else "当前追问所指向的最近图片"
        hint = (
            "<group_reply_scope_hint>"
            "这是群聊中的当前轮回复。历史群消息仅用于理解语境，不代表现在需要补回复。"
            "请只直接回应当前触发这次 LLM 请求的发言者和当前这条消息，"
            f"并结合{source_text}。"
            "不要因为上下文里存在此前未回复的群聊文本，就逐条补答、补回应或主动回到旧话题。"
            "只有当当前发言明确引用、追问或要求回应此前内容时，才回应对应的历史内容。"
            "如果当前消息只有图片，就只围绕本轮图片自然回应。"
            "</group_reply_scope_hint>"
        )

        for part in getattr(req, "extra_user_content_parts", []) or []:
            if "<group_reply_scope_hint>" in self._part_text(part):
                return

        try:
            from astrbot.core.agent.message import TextPart

            part = TextPart(text=hint)
            mark_temp = getattr(part, "mark_as_temp", None)
            if callable(mark_temp):
                marked = mark_temp()
                if marked is not None:
                    part = marked
            req.extra_user_content_parts.append(part)
            if self.config.get("debug_log", False):
                logger.info(
                    "[ForceImageCaption] group reply scope guard injected source=%s",
                    image_source,
                )
            return
        except Exception as exc:
            if self.config.get("debug_log", False):
                logger.debug(
                    "[ForceImageCaption] failed to add temporary group reply scope hint: %s",
                    exc,
                )

        # Older AstrBot fallback. ProviderRequest.prompt is request-local here; the
        # XML-like tag also makes duplicate insertion easy to detect.
        prompt = getattr(req, "prompt", "")
        prompt = prompt if isinstance(prompt, str) else ""
        if "<group_reply_scope_hint>" not in prompt:
            req.prompt = f"{prompt.rstrip()}\n\n{hint}".strip()

    @staticmethod
    def _cacheable_image_ref(ref: str) -> bool:
        if not isinstance(ref, str) or not ref.strip():
            return False
        # data: URLs can be several MB each. Keeping them for every session would
        # turn a tiny follow-up cache into an unbounded memory sink.
        return not ref.lstrip().lower().startswith("data:")

    def _current_user_text(self, event: AstrMessageEvent) -> str:
        candidates = [
            getattr(event, "message_str", ""),
            getattr(getattr(event, "message_obj", None), "message_str", ""),
        ]
        text = next((v for v in candidates if isinstance(v, str) and v.strip()), "")
        if not text:
            return ""

        # Some adapters render non-text components as placeholders. They are not
        # useful context for the vision model.
        text = re.sub(r"\[\s*(?:图片|image)\s*\]", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        limit = self._followup_context_max_chars()
        if limit and len(text) > limit:
            text = text[:limit].rstrip() + "…"
        return text

    @staticmethod
    def _looks_like_image_followup(text: str) -> bool:
        """Conservative heuristic for follow-ups such as “第二个是谁 / 这张呢”."""
        if not isinstance(text, str):
            return False
        value = re.sub(r"\s+", "", text).lower()
        if not value:
            return False

        # Natural image references are often statements rather than questions,
        # e.g. “这个超级好吃”“这个也太可爱了”“那玩意我也想买”.
        # Keeping this short avoids turning every long message containing “这个”
        # into an image follow-up.
        natural_reference = re.match(
            r"^(?:这个|那个|这张|那张|这玩意|那玩意|这东西|那东西|它|他|她|ta)(?:也|真|太|还|好|超|挺|有点|简直|居然|看着|感觉|怎么|为什么|是|不)?",
            value,
        )
        if natural_reference and len(value) <= 36:
            return True

        explicit = (
            "图片", "照片", "截图", "表情包", "表情", "图里", "图中", "画面",
            "这张", "那张", "上一张", "刚才那张", "刚刚那张", "这幅", "那幅",
            "左边", "右边", "中间", "上面", "下面", "前面", "后面",
        )
        if any(token in value for token in explicit):
            return True

        ordinal = re.search(r"第(?:[一二三四五六七八九十百两\d]+)(?:个|位|只|张|排|行|列)?", value)
        interrogative = any(token in value for token in ("谁", "哪", "什么", "啥", "怎么", "是不是", "叫啥", "叫什"))
        if ordinal and interrogative:
            return True

        demonstrative = re.search(r"(?:这个|那个|这人|那人|这位|那位|它|他|她|ta).{0,12}(?:是谁|谁|什么|啥|哪|怎么|干嘛|意思)", value)
        if demonstrative:
            return True

        # Very short conversational follow-ups are common immediately after an
        # image. Keep this list narrow to avoid attaching stale images to normal chat.
        short_followups = {
            "谁啊", "谁呀", "是谁", "这谁", "那谁", "这是谁", "那是谁",
            "什么意思", "啥意思", "怎么回事", "这是啥", "这是什么", "那是什么",
            "这个呢", "那个呢", "然后呢", "还有呢",
        }
        return len(value) <= 18 and value in short_followups

    @staticmethod
    def _needs_fresh_visual_followup(text: str) -> bool:
        """Whether a follow-up really needs another look at the original image.

        Natural reactions such as “笑死 / 这个好可爱 / 确实” can safely reuse the
        previous caption. Requests for OCR, position, identity, counting, comparison,
        or fine visual details should re-query the vision model.
        """
        if not isinstance(text, str):
            return False
        value = re.sub(r"\s+", "", text).lower()
        if not value:
            return False

        detail_tokens = (
            "写了什么", "写的什么", "写了啥", "写的啥", "什么字", "文字", "字幕",
            "第一个", "第二个", "第三个", "第四个", "第五个", "第几",
            "左边", "右边", "中间", "上面", "下面", "前面", "后面", "角落",
            "是谁", "谁啊", "谁呀", "哪一个", "哪个", "哪位", "叫什么", "名字",
            "拿着", "穿着", "戴着", "颜色", "几个人", "几个", "多少", "数量",
            "区别", "不同", "对比", "比较", "哪张", "哪边", "哪里", "位置",
            "细节", "放大", "看清", "认一下", "识别", "ocr",
        )
        if any(token in value for token in detail_tokens):
            return True

        # Ordinal + question is almost always a request for a more specific look.
        if re.search(r"第(?:[一二三四五六七八九十百两\d]+)(?:个|位|只|张|排|行|列)?", value):
            if any(token in value for token in ("谁", "什么", "啥", "哪", "怎么", "干嘛", "在做")):
                return True

        return False

    async def _remember_images(self, event: AstrMessageEvent, images: list[str]) -> None:
        if not self._recent_memory_enabled():
            return

        cacheable = [v for v in self._dedupe(images) if self._cacheable_image_ref(v)]
        if not cacheable:
            if self.config.get("debug_log", False) and images:
                logger.debug("[ForceImageCaption] recent-image cache skipped non-cacheable refs")
            return

        cacheable = cacheable[: self._recent_image_max_count()]
        key = self._session_key(event)
        now = time.time()
        async with self._recent_images_lock:
            previous = self._recent_images.get(key) or {}
            same_images = self._dedupe(list(previous.get("images", []) or [])) == cacheable
            self._recent_images[key] = {
                "ts": now,
                "images": cacheable,
                "message_id": self._message_id(event),
                "sender_id": self._sender_id(event),
                # Preserve a successful caption when the same image is observed again
                # by another AstrBot stage. This prevents duplicate vision calls.
                "caption": str(previous.get("caption", "") or "") if same_images else "",
                "caption_ts": float(previous.get("caption_ts", 0.0) or 0.0) if same_images else 0.0,
                "caption_provider_id": str(previous.get("caption_provider_id", "") or "") if same_images else "",
            }
            ttl = self._recent_image_ttl()
            if len(self._recent_images) > 256:
                expired = [
                    k for k, item in self._recent_images.items()
                    if now - float(item.get("ts", 0.0) or 0.0) > ttl
                ]
                for k in expired:
                    self._recent_images.pop(k, None)

        if self.config.get("debug_log", False):
            logger.info(
                "[ForceImageCaption] remembered recent image(s) session=%s count=%d preserved_caption=%s",
                key,
                len(cacheable),
                bool(same_images and previous.get("caption")),
            )

    async def _get_recent_context(
        self, event: AstrMessageEvent
    ) -> tuple[list[str], float, str, str]:
        if not self._recent_memory_enabled():
            return [], 0.0, "", ""

        key = self._session_key(event)
        now = time.time()
        ttl = self._recent_image_ttl()
        async with self._recent_images_lock:
            item = self._recent_images.get(key)
            if not item:
                return [], 0.0, "", ""
            age = max(0.0, now - float(item.get("ts", 0.0) or 0.0))
            if age > ttl:
                self._recent_images.pop(key, None)
                return [], age, "", ""
            images = list(item.get("images", []) or [])
            sender_id = str(item.get("sender_id", "") or "")
            caption = str(item.get("caption", "") or "").strip()
        return self._dedupe(images), age, sender_id, caption

    async def _get_recent_images(self, event: AstrMessageEvent) -> tuple[list[str], float, str]:
        images, age, sender_id, _caption = await self._get_recent_context(event)
        return images, age, sender_id

    async def _store_recent_caption(
        self,
        event: AstrMessageEvent,
        images: list[str],
        caption: str,
        provider_id: str = "",
    ) -> None:
        caption = str(caption or "").strip()
        if not caption or not self._recent_memory_enabled():
            return

        key = self._session_key(event)
        normalized = [v for v in self._dedupe(images) if self._cacheable_image_ref(v)]
        async with self._recent_images_lock:
            item = self._recent_images.get(key)
            if not item:
                return
            remembered = self._dedupe(list(item.get("images", []) or []))
            if normalized and remembered != normalized[: self._recent_image_max_count()]:
                return
            item["caption"] = caption
            item["caption_ts"] = time.time()
            item["caption_provider_id"] = str(provider_id or "")

    async def _forget_recent_images(self, event: AstrMessageEvent) -> bool:
        key = self._session_key(event)
        async with self._recent_images_lock:
            return self._recent_images.pop(key, None) is not None

    def _caption_prompt_for_request(
        self,
        event: AstrMessageEvent,
        *,
        image_source: str,
    ) -> str:
        base = self._caption_prompt(event)
        user_text = self._current_user_text(event)
        if not user_text:
            return base

        source_hint = (
            "这是用户刚才发过、现在继续追问的图片。"
            if image_source == "recent"
            else "这是用户当前消息携带或引用的图片。"
        )
        return (
            f"{base}\n\n"
            f"{source_hint}\n"
            f"当前用户消息：{user_text}\n"
            "请优先提取回答这句话所必需的视觉事实；如果问题涉及人物顺序、位置、"
            "文字、动作或表情，要把相关信息说清楚。只输出供主聊天模型使用的视觉事实，"
            "不要称呼用户，不要解释你的识图过程。"
        )

    async def _image_ref(self, image: Image) -> str:
        url = getattr(image, "url", None)
        if isinstance(url, str) and url.strip():
            return url.strip()

        try:
            path = await image.convert_to_file_path()
            if isinstance(path, str) and path.strip():
                return path.strip()
        except Exception:
            pass

        for attr in ("file", "path"):
            value = getattr(image, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()

        return ""

    async def _walk_components(
        self,
        components: Any,
        *,
        depth: int = 0,
        seen: set[int] | None = None,
    ) -> list[str]:
        if depth > 3 or not isinstance(components, (list, tuple)):
            return []

        if seen is None:
            seen = set()

        refs: list[str] = []

        for comp in components:
            obj_id = id(comp)
            if obj_id in seen:
                continue
            seen.add(obj_id)

            if isinstance(comp, Image):
                ref = await self._image_ref(comp)
                if ref:
                    refs.append(ref)
                continue

            if isinstance(comp, Reply):
                chain = getattr(comp, "chain", None)
                refs.extend(
                    await self._walk_components(
                        chain,
                        depth=depth + 1,
                        seen=seen,
                    )
                )

            for attr in ("chain", "message", "content", "nodes"):
                nested = getattr(comp, attr, None)
                if isinstance(nested, (list, tuple)):
                    refs.extend(
                        await self._walk_components(
                            nested,
                            depth=depth + 1,
                            seen=seen,
                        )
                    )

        return self._dedupe(refs)

    async def _event_images(self, event: AstrMessageEvent) -> list[str]:
        try:
            chain = event.message_obj.message
        except Exception:
            return []
        return await self._walk_components(chain)

    async def _resolve_images(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ) -> list[str]:
        req_images = [
            str(v).strip()
            for v in (getattr(req, "image_urls", None) or [])
            if isinstance(v, str) and v.strip()
        ]
        event_images = await self._event_images(event)

        # A framework temp path may already have disappeared while the Image
        # component has been rewritten to our persistent copy. Replace stale refs
        # by basename before sending them to the vision provider.
        event_by_name: dict[str, str] = {}
        for ref in event_images:
            path = self._ref_to_local_path(ref)
            if path is not None and path.exists():
                event_by_name[path.name] = ref

        normalized_req: list[str] = []
        for ref in req_images:
            local = self._ref_to_local_path(ref)
            if local is None:
                normalized_req.append(ref)
                continue
            if local.exists():
                normalized_req.append(ref)
                continue
            replacement = event_by_name.get(local.name)
            if replacement:
                normalized_req.append(replacement)
                if self.config.get("debug_log", False):
                    logger.info(
                        "[ForceImageCaption] replaced stale request image ref by persistent copy basename=%s",
                        local.name,
                    )
            elif self.config.get("debug_log", False):
                logger.debug(
                    "[ForceImageCaption] dropped stale local request image ref: %s",
                    ref,
                )

        # Event refs first: if the same image exists as both a stale framework
        # reference and a stable plugin_data copy, the stable one wins.
        return self._dedupe(event_images + normalized_req)

    @staticmethod
    def _error_text(exc: Exception) -> str:
        return str(exc).lower()

    @classmethod
    def _is_hard_stop_error(cls, exc: Exception) -> bool:
        """Errors where immediately sending more requests is counterproductive."""
        text = cls._error_text(exc)
        hard_stop = (
            "429", "rpm exhausted", "tpm", "rate_limit", "rate limit",
            "quota_exceeded", "quota exceeded", "429001",
            "account_billing_suspended", "billing status", "billing suspended",
            "account is suspended", "401", "invalid api key", "authentication",
        )
        return any(token in text for token in hard_stop)

    @classmethod
    def _is_retryable(cls, exc: Exception) -> bool:
        text = cls._error_text(exc)

        if cls._is_hard_stop_error(exc):
            return False

        # These usually cannot be fixed by sending the exact same request again.
        non_retryable = (
            "400", "403", "404", "413", "422",
            "invalid_request", "sensitive", "content policy",
        )
        if any(token in text for token in non_retryable):
            return False

        retryable = (
            "timeout", "timed out", "connection", "temporarily",
            "reset by peer", "eof", "500", "502", "503", "504",
        )
        return any(token in text for token in retryable)

    @classmethod
    def _should_split_multi_image_error(cls, exc: Exception) -> bool:
        """Split only when per-image retry could plausibly fix a batch/format issue."""
        text = cls._error_text(exc)
        if cls._is_hard_stop_error(exc) or cls._is_retryable(exc):
            return False

        # Payload/image-count/client-format failures can often be recovered by
        # sending images one by one. Do not do this for RPM/TPM/server outages.
        split_hints = (
            "multiple image", "multiple images", "multi-image", "multi image",
            "too many image", "too many images", "image count", "image_num",
            "unsupported image", "unsupported images", "payload",
            "400", "413", "422", "invalid_request", "invalid request",
        )
        return any(token in text for token in split_hints)

    @staticmethod
    def _normalize_cache_text(text: str) -> str:
        value = re.sub(r"\s+", " ", str(text or "")).strip().lower()
        return value[:500]

    def _image_signature(self, images: list[str]) -> str:
        parts: list[str] = []
        for ref in self._dedupe(images):
            local = self._ref_to_local_path(ref)
            if local is not None and local.exists():
                try:
                    stat = local.stat()
                    parts.append(f"file:{local}:{stat.st_size}:{stat.st_mtime_ns}")
                    continue
                except OSError:
                    pass
            parts.append(str(ref))
        raw = "\n".join(parts).encode("utf-8", errors="ignore")
        return hashlib.sha1(raw).hexdigest()

    def _caption_cache_key(
        self, provider_id: str, images: list[str], *, user_text: str = "", kind: str = "query"
    ) -> str:
        image_sig = self._image_signature(images)
        text_sig = hashlib.sha1(
            self._normalize_cache_text(user_text).encode("utf-8", errors="ignore")
        ).hexdigest()[:16]
        return f"{provider_id}|{image_sig}|{kind}|{text_sig}"

    async def _get_caption_cache(self, key: str) -> str:
        now = time.time()
        ttl = self._caption_cache_ttl()
        async with self._caption_cache_lock:
            item = self._caption_cache.get(key)
            if not item:
                return ""
            if now - float(item.get("ts", 0.0) or 0.0) > ttl:
                self._caption_cache.pop(key, None)
                return ""
            return str(item.get("caption", "") or "").strip()

    async def _put_caption_cache(self, key: str, caption: str) -> None:
        caption = str(caption or "").strip()
        if not caption:
            return
        now = time.time()
        async with self._caption_cache_lock:
            self._caption_cache[key] = {"ts": now, "caption": caption}
            if len(self._caption_cache) > 512:
                oldest = sorted(
                    self._caption_cache.items(),
                    key=lambda kv: float(kv[1].get("ts", 0.0) or 0.0),
                )[:128]
                for cache_key, _item in oldest:
                    self._caption_cache.pop(cache_key, None)

    async def _provider_cooldown_remaining(self, provider_id: str) -> float:
        now = time.time()
        async with self._provider_cooldown_lock:
            until = float(self._provider_cooldown_until.get(provider_id, 0.0) or 0.0)
            if until <= now:
                self._provider_cooldown_until.pop(provider_id, None)
                return 0.0
            return until - now

    async def _trip_provider_cooldown(self, provider_id: str, exc: Exception) -> None:
        if not self._is_hard_stop_error(exc):
            return
        seconds = self._rate_limit_cooldown()
        until = time.time() + seconds
        async with self._provider_cooldown_lock:
            self._provider_cooldown_until[provider_id] = max(
                float(self._provider_cooldown_until.get(provider_id, 0.0) or 0.0),
                until,
            )
        logger.warning(
            "[ForceImageCaption] provider进入冷却 %ss，避免继续触发RPM/TPM或计费拒绝 provider=%s error=%s",
            seconds,
            provider_id,
            exc,
        )

    async def _generate_caption_dedup(
        self,
        event: AstrMessageEvent,
        provider_id: str,
        images: list[str],
        *,
        image_source: str,
        cache_key: str,
    ) -> str:
        cached = await self._get_caption_cache(cache_key)
        if cached:
            if self.config.get("debug_log", False):
                logger.info("[ForceImageCaption] caption cache hit key=%s", cache_key[-40:])
            return cached

        async with self._caption_inflight_lock:
            task = self._caption_inflight.get(cache_key)
            if task is None:
                task = asyncio.create_task(
                    self._generate_caption(
                        event,
                        provider_id,
                        images,
                        image_source=image_source,
                    )
                )
                self._caption_inflight[cache_key] = task
                created = True
            else:
                created = False

        try:
            caption = await asyncio.shield(task)
        finally:
            if created:
                async with self._caption_inflight_lock:
                    if self._caption_inflight.get(cache_key) is task:
                        self._caption_inflight.pop(cache_key, None)

        if caption:
            await self._put_caption_cache(cache_key, caption)
        return caption

    async def _caption_once(
        self,
        provider: Any,
        prompt: str,
        images: list[str],
    ) -> str:
        async def _do_request():
            return await asyncio.wait_for(
                provider.text_chat(
                    prompt=prompt,
                    image_urls=images,
                ),
                timeout=self._caption_timeout(),
            )

        if self._serialize_caption_requests():
            async with self._caption_request_lock:
                response = await _do_request()
        else:
            response = await _do_request()
        return str(getattr(response, "completion_text", "") or "").strip()

    async def _caption_with_retries(
        self,
        provider: Any,
        prompt: str,
        images: list[str],
        *,
        label: str,
    ) -> str:
        """Caption one image set, retrying transient failures and optional empty replies."""
        max_retries = self._max_retries()
        retry_empty = bool(self.config.get("retry_on_empty_caption", True))
        delay = self._retry_delay()
        backoff = self._retry_backoff()
        last_exc: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                text = await self._caption_once(provider, prompt, images)
                if text:
                    if self.config.get("debug_log", False) and attempt:
                        logger.info(
                            "[ForceImageCaption] %s succeeded after retry attempt=%d/%d",
                            label,
                            attempt,
                            max_retries,
                        )
                    return text

                if not retry_empty:
                    return ""
                last_exc = RuntimeError("图片转述模型返回空内容")
            except Exception as exc:
                last_exc = exc
                if not self._is_retryable(exc):
                    raise

            if attempt >= max_retries:
                break

            wait_seconds = delay * (backoff ** attempt)
            if self.config.get("debug_log", False):
                logger.warning(
                    "[ForceImageCaption] %s failed, retry %d/%d in %.2fs: %s",
                    label,
                    attempt + 1,
                    max_retries,
                    wait_seconds,
                    last_exc,
                )
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)

        if last_exc is not None:
            raise last_exc
        return ""

    async def _generate_caption(
        self,
        event: AstrMessageEvent,
        provider_id: str,
        images: list[str],
        *,
        image_source: str = "current",
    ) -> str:
        provider = self.context.get_provider_by_id(provider_id=provider_id)
        if provider is None:
            raise ValueError(f"找不到图片转述模型 Provider：{provider_id}")

        prompt = self._caption_prompt_for_request(
            event,
            image_source=image_source,
        )

        batch_exc: Exception | None = None
        try:
            return await self._caption_with_retries(
                provider,
                prompt,
                images,
                label="batch",
            )
        except Exception as exc:
            batch_exc = exc

        # If multiple images fail as one request, rescue them one by one only
        # when the failure looks like a batch/payload/image-count problem. A 429,
        # billing suspension, timeout or server outage must never fan out into N
        # additional requests.
        should_split = bool(
            len(images) > 1
            and self.config.get("split_multi_image_on_failure", True)
            and batch_exc is not None
            and self._should_split_multi_image_error(batch_exc)
        )
        if should_split:
            captions: list[str] = []
            for index, image in enumerate(images, 1):
                try:
                    text = await self._caption_with_retries(
                        provider,
                        prompt,
                        [image],
                        label=f"image-{index}",
                    )
                except Exception as exc:
                    # A hard-stop error during split recovery aborts the whole rescue
                    # immediately, rather than continuing with the remaining images.
                    if self._is_hard_stop_error(exc):
                        raise
                    if self.config.get("debug_log", False):
                        logger.warning(
                            "[ForceImageCaption] image-%d failed after retries: %s",
                            index,
                            exc,
                        )
                    continue
                if text:
                    captions.append(f"图片{index}：{text}")

            if captions:
                return "\n".join(captions)
        elif (
            len(images) > 1
            and self.config.get("split_multi_image_on_failure", True)
            and batch_exc is not None
            and self.config.get("debug_log", False)
        ):
            logger.info(
                "[ForceImageCaption] skipped multi-image split because error is not safely recoverable: %s",
                batch_exc,
            )

        if batch_exc is not None:
            raise batch_exc
        return ""

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def remember_image_message(self, event: AstrMessageEvent):
        """Remember images even when an image-only message does not trigger the LLM."""
        if not self.config.get("enabled", True) or not self._recent_memory_enabled():
            return
        try:
            persisted = await self._persist_event_images(event)
            images = persisted or await self._event_images(event)
            if images:
                await self._remember_images(event, images)
        except Exception as exc:
            if self.config.get("debug_log", False):
                logger.debug("[ForceImageCaption] failed to remember event image: %s", exc)

    @filter.on_llm_request(priority=-999999)
    async def force_image_caption(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ):
        if not self.config.get("enabled", True):
            return

        if self.config.get("silent_failure", True):
            self._strip_failure_markers(req)

        await self._persist_event_images(event)

        current_images = await self._resolve_images(event, req)
        if current_images:
            await self._remember_images(event, current_images)

        existing = self._existing_caption(req)
        if existing:
            if current_images:
                self._inject_group_reply_scope_hint(
                    event, req, image_source="current"
                )
                await self._store_recent_caption(event, current_images, existing)
            self._inject_caption_into_prompt(req, existing)
            self._remove_caption_parts(req)

            if self.config.get("remove_images_from_main_model", True):
                req.image_urls = []

            if self.config.get("debug_log", False):
                logger.info(
                    "[ForceImageCaption] reused existing caption, length=%d",
                    len(existing),
                )
            return

        user_text = self._current_user_text(event)
        images = current_images
        image_source = "current"
        recent_caption = ""
        explicit_followup = False
        natural_followup = False
        fresh_visual_followup = False

        if not images and self._recent_memory_enabled():
            followup_only = bool(self.config.get("recent_image_followup_only", True))
            recent_images, age, image_sender_id, recent_caption = await self._get_recent_context(event)
            if recent_images:
                explicit_followup = self._looks_like_image_followup(user_text)
                fresh_visual_followup = self._needs_fresh_visual_followup(user_text)
                if self._natural_followup_enabled() and age <= self._natural_followup_window():
                    current_sender_id = self._sender_id(event)
                    same_sender = bool(
                        current_sender_id
                        and image_sender_id
                        and current_sender_id == image_sender_id
                    )
                    natural_followup = (
                        not self._natural_followup_same_sender_only()
                        or same_sender
                    )

                should_reuse = (
                    (not followup_only)
                    or explicit_followup
                    or natural_followup
                )
                if should_reuse:
                    images = recent_images
                    image_source = "recent"
                    if self.config.get("debug_log", False):
                        reason = (
                            "explicit" if explicit_followup
                            else "natural-window" if natural_followup
                            else "always"
                        )
                        logger.info(
                            "[ForceImageCaption] reused recent image context count=%d age=%.1fs reason=%s fresh_visual=%s cached_caption=%s text=%r",
                            len(images),
                            age,
                            reason,
                            fresh_visual_followup,
                            bool(recent_caption),
                            user_text[:80],
                        )

        if images:
            self._inject_group_reply_scope_hint(
                event, req, image_source=image_source
            )

        if not images:
            if self._looks_like_image_followup(user_text):
                self._inject_silent_fallback_hint(req)
            if self.config.get("debug_log", False):
                logger.info("[ForceImageCaption] no usable image for this LLM request.")
            return

        # v1.2.5 keeps the v1.2.4 optimization: ordinary short-window continuation does not
        # re-run the vision model. It reuses the most recent successful caption.
        # Fine-grained visual questions still trigger a fresh vision request.
        if (
            image_source == "recent"
            and recent_caption
            and self._reuse_caption_for_natural_followup()
            and not fresh_visual_followup
        ):
            self._inject_caption_into_prompt(req, recent_caption)
            self._remove_caption_parts(req)
            if self.config.get("remove_images_from_main_model", True):
                req.image_urls = []
            if self.config.get("debug_log", False):
                logger.info(
                    "[ForceImageCaption] reused cached recent caption without vision API call length=%d",
                    len(recent_caption),
                )
            return

        provider_id = self._caption_provider_id(event)
        if not provider_id:
            logger.warning(
                "[ForceImageCaption] 检测到图片，但未配置图片转述模型。"
                "请在 AstrBot 中选择‘默认图片转述模型’，"
                "或在插件配置中填写 Provider ID。"
            )
            if self.config.get("silent_failure", True):
                self._strip_failure_markers(req)
            self._inject_silent_fallback_hint(req)
            if self.config.get("remove_images_on_failure", True):
                req.image_urls = []
            return

        query_cache_key = self._caption_cache_key(
            provider_id,
            images,
            user_text=user_text,
            kind="query",
        )
        caption = await self._get_caption_cache(query_cache_key)
        if caption:
            await self._store_recent_caption(event, images, caption, provider_id)
            self._inject_caption_into_prompt(req, caption)
            self._remove_caption_parts(req)
            if self.config.get("remove_images_from_main_model", True):
                req.image_urls = []
            if self.config.get("debug_log", False):
                logger.info("[ForceImageCaption] exact caption cache hit; skipped vision API")
            return

        cooldown_remaining = await self._provider_cooldown_remaining(provider_id)
        if cooldown_remaining > 0:
            # If a general recent caption exists, use it as a safe fallback even for
            # a detail question rather than repeatedly striking a known-limited API.
            if recent_caption:
                self._inject_caption_into_prompt(req, recent_caption)
                self._remove_caption_parts(req)
                if self.config.get("remove_images_from_main_model", True):
                    req.image_urls = []
                logger.info(
                    "[ForceImageCaption] provider冷却中，复用已有图片理解，跳过视觉请求 remaining=%.1fs provider=%s",
                    cooldown_remaining,
                    provider_id,
                )
                return

            logger.warning(
                "[ForceImageCaption] provider冷却中，跳过新的视觉请求 remaining=%.1fs provider=%s",
                cooldown_remaining,
                provider_id,
            )
            self._inject_silent_fallback_hint(req)
            if self.config.get("remove_images_on_failure", True):
                req.image_urls = []
            return

        try:
            caption = await self._generate_caption_dedup(
                event,
                provider_id,
                images,
                image_source=image_source,
                cache_key=query_cache_key,
            )
        except Exception as exc:
            await self._trip_provider_cooldown(provider_id, exc)
            logger.error(
                "[ForceImageCaption] 图片转述失败 provider=%s images=%d plugin_retries=%d error=%s",
                provider_id,
                len(images),
                self._max_retries(),
                exc,
            )

            if self.config.get("silent_failure", True):
                self._strip_failure_markers(req)
            self._inject_silent_fallback_hint(req)

            if self.config.get("remove_images_on_failure", True):
                req.image_urls = []
            return

        if not caption:
            logger.warning("[ForceImageCaption] 图片转述模型最终仍返回空内容。")
            if self.config.get("silent_failure", True):
                self._strip_failure_markers(req)
            self._inject_silent_fallback_hint(req)
            if self.config.get("remove_images_on_failure", True):
                req.image_urls = []
            return

        await self._store_recent_caption(event, images, caption, provider_id)

        self._inject_caption_into_prompt(req, caption)
        self._remove_caption_parts(req)

        if self.config.get("remove_images_from_main_model", True):
            req.image_urls = []

        if self.config.get("debug_log", False):
            logger.info(
                "[ForceImageCaption] caption ready provider=%s source=%s images=%d length=%d",
                provider_id,
                image_source,
                len(images),
                len(caption),
            )

    @filter.command("force_caption_status")
    async def force_caption_status(self, event: AstrMessageEvent):
        provider_id = self._caption_provider_id(event) or "（未配置）"
        recent, age, _image_sender_id, recent_caption = await self._get_recent_context(event)
        followup_only = bool(self.config.get("recent_image_followup_only", True))
        cooldown = await self._provider_cooldown_remaining(provider_id) if provider_id != "（未配置）" else 0.0
        yield event.plain_result(
            "Force Image Caption v1.2.6\n"
            f"状态：{'启用' if self.config.get('enabled', True) else '关闭'}\n"
            f"图片转述模型：{provider_id}\n"
            f"插件额外失败重试：{self._max_retries()} 次（AstrBot 4.28+ 建议 0）\n"
            f"自然续聊复用转述：{'开启' if self._reuse_caption_for_natural_followup() else '关闭'}\n"
            f"群聊当前轮聚焦：{'开启' if self._group_reply_scope_guard_enabled() else '关闭'}\n"
            f"视觉请求串行保护：{'开启' if self._serialize_caption_requests() else '关闭'}\n"
            f"限流冷却：{self._rate_limit_cooldown()} 秒"
            + (f"（当前剩余约 {cooldown:.0f} 秒）\n" if cooldown > 0 else "\n")
            + f"静默失败：{'开启' if self.config.get('silent_failure', True) else '关闭'}\n"
            f"最近图片记忆：{'开启' if self._recent_memory_enabled() else '关闭'}\n"
            f"临时图持久化：{'开启' if self._persist_recent_images_enabled() else '关闭'}\n"
            f"追问复用：{'图片追问 + 短时自然续聊' if followup_only else 'TTL 内所有 LLM 请求'}\n"
            f"自然续聊窗口：{'开启' if self._natural_followup_enabled() else '关闭'}"
            + (f"（{self._natural_followup_window()} 秒，{'仅同一发送者' if self._natural_followup_same_sender_only() else '群内任意发送者'}）\n" if self._natural_followup_enabled() else "\n")
            + f"记忆有效期：{self._recent_image_ttl()} 秒\n"
            f"本会话缓存：{len(recent)} 张"
            + (f"（约 {age:.0f} 秒前，{'已有可复用转述' if recent_caption else '尚无转述'}）" if recent else "")
        )

    @filter.command("force_caption_forget")
    async def force_caption_forget(self, event: AstrMessageEvent):
        removed = await self._forget_recent_images(event)
        yield event.plain_result(
            "已清除本会话最近图片记忆。" if removed else "本会话当前没有最近图片记忆。"
        )

    async def terminate(self):
        async with self._recent_images_lock:
            self._recent_images.clear()
        async with self._caption_cache_lock:
            self._caption_cache.clear()
        async with self._provider_cooldown_lock:
            self._provider_cooldown_until.clear()
        async with self._caption_inflight_lock:
            tasks = list(self._caption_inflight.values())
            self._caption_inflight.clear()
        for task in tasks:
            if not task.done():
                task.cancel()
        try:
            await self._cleanup_persistent_cache(force=True)
        except Exception:
            pass
