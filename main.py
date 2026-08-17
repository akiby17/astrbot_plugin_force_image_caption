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
            "请识别并简洁描述用户发送的图片，供另一个无法直接看图的聊天模型理解。"
            "说明主要人物、物体、场景、动作和重要文字；"
            "如果是表情包、梗图或聊天截图，也说明其主要含义和情绪。"
            "不要回答用户，只输出图片描述。"
        )

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

    def _remove_caption_parts(self, req: ProviderRequest) -> None:
        parts = getattr(req, "extra_user_content_parts", None)
        if not isinstance(parts, list):
            return

        cleaned = []
        for part in parts:
            text = self._part_text(part)
            if "<image_caption>" in text.lower():
                continue
            if "[Image Captioning Failed]" in text:
                continue
            cleaned.append(part)

        req.extra_user_content_parts = cleaned

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
        )
        if any(token in text for token in non_retryable):
            return False

        retryable = (
            "timeout",
            "timed out",
            "connection",
            "temporarily",
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
            timeout=60,
        )
        return str(getattr(response, "completion_text", "") or "").strip()

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

        try:
            return await self._caption_once(provider, prompt, images)
        except Exception as exc:
            if self._is_retryable(exc):
                await asyncio.sleep(0.8)
                try:
                    return await self._caption_once(provider, prompt, images)
                except Exception as retry_exc:
                    exc = retry_exc

            if len(images) > 1:
                captions: list[str] = []
                for index, image in enumerate(images, 1):
                    try:
                        text = await self._caption_once(provider, prompt, [image])
                    except Exception:
                        continue
                    if text:
                        captions.append(f"图片{index}：{text}")

                if captions:
                    return "\n".join(captions)

            raise exc

    @filter.on_llm_request(priority=-999999)
    async def force_image_caption(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ):
        if not self.config.get("enabled", True):
            return

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
                "请在 AstrBot 中选择“默认图片转述模型”，"
                "或在插件配置中填写 Provider ID。"
            )
            return

        try:
            caption = await self._generate_caption(
                event,
                provider_id,
                images,
            )
        except Exception as exc:
            logger.error(
                "[ForceImageCaption] 图片转述失败 provider=%s images=%d error=%s",
                provider_id,
                len(images),
                exc,
            )

            if self.config.get("remove_images_on_failure", False):
                req.image_urls = []
            return

        if not caption:
            logger.warning("[ForceImageCaption] 图片转述模型返回空内容。")
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
            f"图片转述模型：{provider_id}"
        )
