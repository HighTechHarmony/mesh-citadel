import asyncio
import json
import logging
import unicodedata

import requests

from citadel.auth.permissions import PermissionLevel
from citadel.commands.base import BaseCommand, CommandCategory
from citadel.commands.registry import register_command
from citadel.room.room import Room, SystemRoomIDs
from citadel.transport.packets import ToUser
from citadel.user.user import User

log = logging.getLogger(__name__)
WEATHER_ACKNOWLEDGEMENT = "Request received; please wait while the forecast is generated."
ASCII_PUNCTUATION = {
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2026": "...",
}

WEATHER_SECTIONS = {
    "SYN": "synopsis",
    "NT": "near_term",
    "ST": "short_term",
    "LT": "long_term",
}


class WeatherServiceError(Exception):
    """The weather service returned an unavailable or invalid response."""


def _sanitize_sentence(sentence):
    normalized = unicodedata.normalize("NFKD", sentence)
    printable = []
    for char in normalized:
        if ord(char) < 128:
            printable.append(char if 0x20 <= ord(char) <= 0x7E else " ")
        elif not unicodedata.combining(char):
            printable.append(ASCII_PUNCTUATION.get(char, " "))

    sentence = " ".join("".join(printable).split())
    if not sentence:
        return None, False

    if sentence[-1] not in ".!?":
        sentence += "."

    truncated = len(sentence) > 133
    if truncated:
        prefix = sentence[:130]
        boundary = prefix.rfind(" ")
        if boundary >= 65:
            prefix = prefix[:boundary]
        sentence = f"{prefix.rstrip()}..."
    return sentence, truncated


def validate_summary(payload, expected_section):
    if not isinstance(payload, dict) or payload.get("section") != expected_section:
        raise WeatherServiceError("Weather service returned an invalid section")

    sentences = payload.get("sentences")
    if not isinstance(sentences, list) or not sentences:
        raise WeatherServiceError("Weather service returned invalid sentences")

    sanitized = []
    invalid_count = 0
    truncated_count = 0
    for sentence in sentences:
        if not isinstance(sentence, str):
            invalid_count += 1
            continue
        clean_sentence, truncated = _sanitize_sentence(sentence)
        if clean_sentence is None:
            invalid_count += 1
            continue
        sanitized.append(clean_sentence)
        truncated_count += truncated

    if invalid_count or truncated_count or len(sanitized) > 4:
        log.warning(
            "Adjusted weather summary for %s: discarded=%s truncated=%s "
            "excess_sentences=%s",
            expected_section, invalid_count, truncated_count,
            max(0, len(sanitized) - 4))

    sanitized = sanitized[:4]
    if not sanitized:
        raise WeatherServiceError("Weather service returned no usable sentences")
    return sanitized


def number_sentences(sentences, max_packet_size):
    total = len(sentences)
    numbered = []
    for index, sentence in enumerate(sentences, start=1):
        suffix = f" ({index}/{total})"
        content_limit = max_packet_size - len(suffix)
        if content_limit < 3:
            raise WeatherServiceError(
                "MeshCore packet limit is too small for forecast numbering")
        if len(sentence) > content_limit:
            sentence = f"{sentence[:content_limit - 3].rstrip()}..."
        numbered.append(f"{sentence}{suffix}")
    return numbered


async def fetch_summary(api_url, section, timeout_seconds):
    url = f"{api_url.rstrip('/')}/api/v1/forecasts/{section}/message_summary"

    def request_summary():
        with requests.get(url, timeout=timeout_seconds,
                          allow_redirects=False) as response:
            if not 200 <= response.status_code < 300:
                try:
                    error_payload = response.json()
                except ValueError:
                    error_payload = response.text
                if isinstance(error_payload, dict):
                    diagnostic = error_payload.get("detail", error_payload)
                else:
                    diagnostic = error_payload
                if not isinstance(diagnostic, str):
                    diagnostic = json.dumps(diagnostic, ensure_ascii=True,
                                            default=str)
                diagnostic = " ".join(diagnostic.split())[:5000]
                error = f"Weather service returned HTTP {response.status_code}"
                if diagnostic:
                    error = f"{error}: {diagnostic}"
                raise WeatherServiceError(
                    error)
            return response.json()

    try:
        payload = await asyncio.to_thread(request_summary)
    except asyncio.CancelledError:
        raise
    except WeatherServiceError:
        raise
    except requests.Timeout as error:
        raise WeatherServiceError(
            f"Weather service request timed out after {timeout_seconds} seconds") from error
    except requests.RequestException as error:
        raise WeatherServiceError(
            f"Weather service connection failed ({type(error).__name__})") from error
    except ValueError as error:
        raise WeatherServiceError("Weather service returned invalid JSON") from error

    return validate_summary(payload, section)


class WeatherCommand(BaseCommand):
    name = "weather"
    category = CommandCategory.UNCOMMON
    permission_level = PermissionLevel.USER
    args = ""
    section = None
    short_text = "Get a weather forecast"
    help_text = "Request a current NOAA forecast summary in the wx room."

    async def run(self, context):
        state = context.session_mgr.get_session_state(context.session_id)
        if not state or state.current_room != SystemRoomIDs.WX_ID:
            return ToUser(
                session_id=context.session_id,
                text="Weather commands are available in wx. Use C wx.",
                is_error=True,
                error_code="wrong_room",
            )
        if self.args.strip():
            return ToUser(
                session_id=context.session_id,
                text="Weather commands do not take arguments.",
                is_error=True,
                error_code="invalid_arguments",
            )

        user = User(context.db, state.username)
        await user.load()
        room = Room(context.db, context.config, state.current_room)
        await room.load()
        if not room.can_user_read(user):
            return ToUser(
                session_id=context.session_id,
                text="You do not have permission to read the wx room.",
                is_error=True,
                error_code="permission_denied",
            )

        meshcore_config = context.config.transport.get("meshcore", {})
        max_packet_size = meshcore_config.get("max_packet_size", 140)
        if (not isinstance(max_packet_size, int)
                or isinstance(max_packet_size, bool) or max_packet_size < 1):
            log.error("Invalid MeshCore max_packet_size for weather delivery")
            return ToUser(
                session_id=context.session_id,
                text="Weather service is unavailable. Try again later.",
                is_error=True,
                error_code="weather_unavailable",
            )

        await context.session_mgr.send_msg(
            context.session_id,
            ToUser(session_id=context.session_id,
                   text=WEATHER_ACKNOWLEDGEMENT),
        )
        try:
            sentences = await fetch_summary(
                context.config.weather["api_url"], self.section,
                context.config.weather["timeout_seconds"])
            sentences = number_sentences(sentences, max_packet_size)
        except WeatherServiceError as error:
            log.warning("Weather request for %s failed: %s", self.section, error)
            return ToUser(
                session_id=context.session_id,
                text="Weather service is unavailable. Try again later.",
                is_error=True,
                error_code="weather_unavailable",
            )

        return [
            ToUser(
                session_id=context.session_id,
                text=sentence,
                hints={"verbatim": True},
            )
            for sentence in sentences
        ]


@register_command
class SynopsisWeatherCommand(WeatherCommand):
    code = "SYN"
    section = "synopsis"
    short_text = "Weather synopsis"
    help_text = "Get the NOAA weather synopsis. Available in the wx room."


@register_command
class NearTermWeatherCommand(WeatherCommand):
    code = "NT"
    section = "near_term"
    short_text = "Near-term weather"
    help_text = "Get the NOAA near-term forecast. Available in the wx room."


@register_command
class ShortTermWeatherCommand(WeatherCommand):
    code = "ST"
    section = "short_term"
    short_text = "Short-term weather"
    help_text = "Get the NOAA short-term forecast. Available in the wx room."


@register_command
class LongTermWeatherCommand(WeatherCommand):
    code = "LT"
    section = "long_term"
    short_text = "Long-term weather"
    help_text = "Get the NOAA long-term forecast. Available in the wx room."