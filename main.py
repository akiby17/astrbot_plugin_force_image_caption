from __future__ import annotations

import asyncio
import re
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
    ) -> str:
        provider = self.context.get_provider_by_id(provider_id=provider_id)
        if provider is None:
            raise ValueError(f"找不到图片转述模型 Provider：{provider_id}")

        prompt = self._caption_prompt(event)

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

        images = await self._resolve_images(event, req)
        if not images:
            if self.config.get("debug_log", False):
                logger.info("[ForceImageCaption] no image found in this LLM request.")
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
                "[ForceImageCaption] caption ready provider=%s images=%d length=%d",
                provider_id,
                len(images),
                len(caption),
            )

    @filter.command("force_caption_status")
    async def force_caption_status(self, event: AstrMessageEvent):
        provider_id = self._caption_provider_id(event) or "（未配置）"
        yield event.plain_result(
            "Force Image Caption\n"
            f"状态：{'启用' if self.config.get('enabled', True) else '关闭'}\n"
            f"图片转述模型：{provider_id}\n"
            f"失败重试：{self._max_retries()} 次\n"
            f"静默失败：{'开启' if self.config.get('silent_failure', True) else '关闭'}"
        )
