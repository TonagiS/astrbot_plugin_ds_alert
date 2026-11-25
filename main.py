"""当群里出现 ds / DS娘 / 肥鱼 相关的关键词或图片时，主动接话。

思路：平时群聊只有 1% 的主动回复概率（provider_ltm_settings.active_reply）。
本插件在群消息上做一层旁路检测，命中关键词（或图片经视觉模型判定相关）时，
把这条消息标成"等同被 @ 唤醒"，于是走 AstrBot 原有的 LLM 链路正常回复，
人格、群上下文、历史注入全都照旧，不另开一条回复通路。

成本控制：文本关键词零成本；图片判定才会调一次视觉模型，
所以限制成"每条消息最多看一眼图"，并且同一个群有冷却时间和每小时次数上限。
"""

from __future__ import annotations

import random
import time

from pathlib import Path

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image, Plain
from astrbot.api.star import Context, Star

PLUGIN_DIR = Path(__file__).resolve().parent
LOG_PATH = PLUGIN_DIR / "data" / "ds_alert.log"


def _dbg(msg: str) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if LOG_PATH.exists() and LOG_PATH.stat().st_size > 1024 * 1024:
            LOG_PATH.write_text("", encoding="utf-8")
        stamp = time.strftime("%m-%d %H:%M:%S")
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(f"[{stamp}] {msg}\n")
    except Exception:  # noqa: BLE001
        pass

VISION_PROMPT = (
    "判断这张图的内容。如果和 DeepSeek（简称 DS / 深度求索 / D 娘）、"
    "鲸鱼 / 鲸鱼娘 / 大肥鱼、或者二次元女仆角色有关，就回答「是」，"
    "否则回答「否」。然后在同一行用不超过 15 个字描述画面。"
)


class DsAlert(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None) -> None:
        super().__init__(context)
        self.config = config or {}
        self._cooldown: dict[str, float] = {}
        self._quota: dict[str, tuple[int, int]] = {}

    # ------------------------------------------------------------------ config

    def _cfg(self, key: str, default=None):
        try:
            value = self.config.get(key, default)
        except Exception:  # noqa: BLE001
            value = default
        return default if value is None else value

    def _keywords(self) -> list[str]:
        raw = self._cfg("keywords", [])
        if isinstance(raw, str):
            raw = [x.strip() for x in raw.replace("，", ",").split(",")]
        out = []
        for item in raw or []:
            item = str(item).strip()
            if item:
                out.append(item)
        return out

    def _groups(self) -> list[str]:
        raw = self._cfg("groups", [])
        if isinstance(raw, str):
            raw = [x.strip() for x in raw.replace("，", ",").split(",")]
        return [str(x).strip() for x in (raw or []) if str(x).strip()]

    def _in_group(self, event: AstrMessageEvent) -> bool:
        groups = self._groups()
        if not groups:
            return True
        gid = str(event.get_group_id() or "")
        umo = event.unified_msg_origin
        return gid in groups or umo in groups

    # ------------------------------------------------------------------ image

    async def _judge_image(self, url: str) -> str:
        provider_id = str(self._cfg("image_provider_id", "deepseek/deepseek-flash") or "")
        try:
            if provider_id:
                provider = self.context.get_provider_by_id(provider_id)
            else:
                provider = await self.context.get_using_provider_async()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"ds_alert: 取视觉模型失败: {exc}")
            return ""
        if not provider:
            return ""
        try:
            resp = await provider.text_chat(
                prompt=VISION_PROMPT,
                session_id=f"ds-alert-{int(time.time())}",
                image_urls=[url],
                persist=False,
            )
            return str(getattr(resp, "completion_text", "") or "")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"ds_alert: 图片判定失败: {exc}")
            return ""

    async def _image_hit(self, event: AstrMessageEvent) -> str:
        if not self._cfg("image_check", True):
            return ""
        keywords = [k.lower() for k in self._keywords()]
        for comp in event.get_messages():
            if not isinstance(comp, Image):
                continue
            url = comp.url or comp.file or ""
            if not url:
                continue
            caption = await self._judge_image(url)
            if not caption:
                continue
            low = caption.lower()
            if low.startswith("是") or "相关" in low:
                return f"image:{caption.strip()[:30]}"
            for kw in keywords:
                if kw and kw in low:
                    return f"image:{kw}"
        return ""

    # ------------------------------------------------------------------ limits

    def _allow(self, umo: str) -> bool:
        now = time.time()
        cooldown = float(self._cfg("cooldown_seconds", 240))
        last = self._cooldown.get(umo, 0.0)
        if now - last < cooldown:
            return False
        hour = int(now // 3600)
        mark, count = self._quota.get(umo, (hour, 0))
        if mark != hour:
            mark, count = hour, 0
        cap = int(self._cfg("max_per_hour", 5))
        if count >= cap:
            return False
        if random.random() > float(self._cfg("probability", 0.8)):
            return False
        self._cooldown[umo] = now
        self._quota[umo] = (mark, count + 1)
        return True

    # ------------------------------------------------------------------ handler

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        if not self._cfg("enable", True):
            return
        if not self._in_group(event):
            return
        umo = event.unified_msg_origin
        text = (event.message_str or "").strip()
        _dbg(f"收到 ({umo}): {text[:40]}")
        if event.is_at_or_wake_command:
            return  # 已经 @ 了，本来就会回，不许重复
        if text.startswith("/") or text.startswith("#"):
            return

        hit = ""
        low = text.lower()
        for kw in self._keywords():
            if kw.lower() in low:
                hit = f"keyword:{kw}"
                break
        if not hit:
            try:
                has_image = any(isinstance(c, Image) for c in event.get_messages())
            except Exception:  # noqa: BLE001
                has_image = False
            # 看图规则：带文字的消息一律看；纯图按概率抽样
            if has_image:
                has_text = len(text) >= int(self._cfg("image_min_chars", 2))
                if has_text:
                    _dbg(f"文字没命中，带文字直接看图 ({umo})")
                    hit = await self._image_hit(event)
                elif random.random() < float(self._cfg("image_probability", 0.3)):
                    _dbg(f"文字没命中，纯图抽样命中，看图 ({umo})")
                    hit = await self._image_hit(event)
                else:
                    _dbg(f"跳过看图：纯图且抽样未命中 ({umo})")
        if not hit:
            return

        if not self._allow(umo):
            _dbg(f"命中 {hit} 但被冷却/概率/配额挡下 ({umo})")
            return

        _dbg(f"命中 {hit}，主动接话 ({umo})")
        # 标成"等同被 @ 唤醒"，交给 AstrBot 原有 LLM 链路正常回复
        event.is_wake = True
        event.is_at_or_wake_command = True
