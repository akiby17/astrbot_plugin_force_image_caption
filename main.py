from __future__ import annotations

import asyncio
import re
import time
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

    def _max_retries(self) -> int:
        try:
            return max(0, min(int(self.config.get("max_retries", 2)), 10))
        except (TypeError, ValueError):
            return 2

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
            self._recent_images[key] = {
                "ts": now,
                "images": cacheable,
                "message_id": self._message_id(event),
            }
            # Opportunistic cleanup prevents long-running bots from accumulating
            # one cache entry for every group/private chat ever seen.
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
                "[ForceImageCaption] remembered recent image(s) session=%s count=%d",
                key,
                len(cacheable),
            )

    async def _get_recent_images(self, event: AstrMessageEvent) -> tuple[list[str], float]:
        if not self._recent_memory_enabled():
            return [], 0.0

        key = self._session_key(event)
        now = time.time()
        ttl = self._recent_image_ttl()
        async with self._recent_images_lock:
            item = self._recent_images.get(key)
            if not item:
                return [], 0.0
            age = max(0.0, now - float(item.get("ts", 0.0) or 0.0))
            if age > ttl:
                self._recent_images.pop(key, None)
                return [], age
            images = list(item.get("images", []) or [])
        return self._dedupe(images), age

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
        return self._dedupe(req_images + event_images)

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        text = str(exc).lower()

        # These usually cannot be fixed by sending the exact same request again.
        non_retryable = (
            "400",
            "401",
            "403",
            "404",
            "413",
            "422",
            "429",
            "invalid_request",
            "rate_limit",
            "sensitive",
            "content policy",
        )
        if any(token in text for token in non_retryable):
            return False

        retryable = (
            "timeout",
            "timed out",
            "connection",
            "temporarily",
            "reset by peer",
            "eof",
            "500",
            "502",
            "503",
            "504",
        )
        return any(token in text for token in retryable)

    async def _caption_once(
        self,
        provider: Any,
        prompt: str,
        images: list[str],
    ) -> str:
        response = await asyncio.wait_for(
            provider.text_chat(
                prompt=prompt,
                image_urls=images,
            ),
            timeout=self._caption_timeout(),
        )
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

        # If multiple images fail as one request, rescue them one by one.
        if len(images) > 1 and self.config.get("split_multi_image_on_failure", True):
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

        if batch_exc is not None:
            raise batch_exc
        return ""

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def remember_image_message(self, event: AstrMessageEvent):
        """Remember images even when an image-only message does not trigger the LLM."""
        if not self.config.get("enabled", True) or not self._recent_memory_enabled():
            return
        try:
            images = await self._event_images(event)
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

        # AstrBot's built-in caption path may have failed before this late hook.
        # Never expose its visible failure marker to the final text model.
        if self.config.get("silent_failure", True):
            self._strip_failure_markers(req)

        # Resolve and remember the current image before any early return. This is
        # important when AstrBot has already generated a caption for the current turn:
        # later turns can still refer back to the same image.
        current_images = await self._resolve_images(event, req)
        if current_images:
            await self._remember_images(event, current_images)

        existing = self._existing_caption(req)
        if existing:
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

        images = current_images
        image_source = "current"

        if not images and self._recent_memory_enabled():
            user_text = self._current_user_text(event)
            followup_only = bool(self.config.get("recent_image_followup_only", True))
            should_reuse = (not followup_only) or self._looks_like_image_followup(user_text)
            if should_reuse:
                recent_images, age = await self._get_recent_images(event)
                if recent_images:
                    images = recent_images
                    image_source = "recent"
                    if self.config.get("debug_log", False):
                        logger.info(
                            "[ForceImageCaption] reused recent image(s) for follow-up count=%d age=%.1fs text=%r",
                            len(images),
                            age,
                            user_text[:80],
                        )

        if not images:
            user_text = self._current_user_text(event)
            if self._looks_like_image_followup(user_text):
                # A likely visual follow-up reached us without a usable current/recent
                # image (for example, cache expired or the adapter did not expose the
                # original image). Keep the main model from narrating internal
                # modality failures; it can ask a natural clarification instead.
                self._inject_silent_fallback_hint(req)
            if self.config.get("debug_log", False):
                logger.info("[ForceImageCaption] no usable image for this LLM request.")
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

        try:
            caption = await self._generate_caption(
                event,
                provider_id,
                images,
                image_source=image_source,
            )
        except Exception as exc:
            logger.error(
                "[ForceImageCaption] 图片转述失败，已结束重试 provider=%s images=%d retries=%d error=%s",
                provider_id,
                len(images),
                self._max_retries(),
                exc,
            )

            # Important: do not leave AstrBot's "[Image Captioning Failed]"
            # in the final request, otherwise the main model tends to reply
            # with "没看到图片/无法识别图片".
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
        recent, age = await self._get_recent_images(event)
        followup_only = bool(self.config.get("recent_image_followup_only", True))
        yield event.plain_result(
            "Force Image Caption\n"
            f"状态：{'启用' if self.config.get('enabled', True) else '关闭'}\n"
            f"图片转述模型：{provider_id}\n"
            f"失败重试：{self._max_retries()} 次\n"
            f"静默失败：{'开启' if self.config.get('silent_failure', True) else '关闭'}\n"
            f"最近图片记忆：{'开启' if self._recent_memory_enabled() else '关闭'}\n"
            f"追问复用：{'仅疑似图片追问' if followup_only else 'TTL 内所有 LLM 请求'}\n"
            f"记忆有效期：{self._recent_image_ttl()} 秒\n"
            f"本会话缓存：{len(recent)} 张"
            + (f"（约 {age:.0f} 秒前）" if recent else "")
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
