import asyncio
import logging

import requests

from citadel.auth.permissions import PermissionLevel
from citadel.commands.base import BaseCommand, CommandCategory
from citadel.commands.registry import register_command
from citadel.room.room import Room, SystemRoomIDs
from citadel.transport.packets import ToUser
from citadel.user.user import User

log = logging.getLogger(__name__)
WEATHER_ACKNOWLEDGEMENT = "Request received; please wait while the forecast is generated."

WEATHER_SECTIONS = {
    "SYN": "synopsis",
    "NT": "near_term",
    "ST": "short_term",
    "LT": "long_term",
}


class WeatherServiceError(Exception):
    """The weather service returned an unavailable or invalid response."""


def validate_summary(payload, expected_section):
    if not isinstance(payload, dict) or payload.get("section") != expected_section:
        raise WeatherServiceError("Weather service returned an invalid section")

    sentences = payload.get("sentences")
    if not isinstance(sentences, list) or not 1 <= len(sentences) <= 4:
        raise WeatherServiceError("Weather service returned invalid sentences")

    total_length = 0
    for sentence in sentences:
        if not isinstance(sentence, str) or not sentence.strip():
            raise WeatherServiceError("Weather service returned invalid text")
        if len(sentence) > 133 or any(not 0x20 <= ord(char) <= 0x7E for char in sentence):
            raise WeatherServiceError("Weather sentence cannot be sent as one packet")
        total_length += len(sentence)

    if total_length > 532:
        raise WeatherServiceError("Weather service returned too much text")
    return sentences


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
                raise WeatherServiceError(
                    f"Weather service returned HTTP {response.status_code}")
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