import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from citadel.commands import weather
from citadel.commands.builtins import ChangeRoomCommand, room_entry_text
from citadel.commands.registry import registry
from citadel.auth.permissions import PermissionLevel, is_allowed
from citadel.transport.engines.meshcore.message_router import MessageRouter
from citadel.transport.engines.meshcore.protocol_handler import ProtocolHandler
from citadel.transport.parser import TextParser
from citadel.transport.packets import ToUser


@pytest.mark.parametrize("alias,section", [
    ("SYN", "synopsis"),
    ("NT", "near_term"),
    ("ST", "short_term"),
    ("LT", "long_term"),
])
def test_weather_aliases_are_registered(alias, section):
    command = registry.get(alias)(username="")
    assert command.section == section
    assert command.name == "weather"


@pytest.mark.parametrize("alias,section", [
    ("syn", "synopsis"),
    ("SYN", "synopsis"),
    ("SyN", "synopsis"),
    ("nt", "near_term"),
    ("NT", "near_term"),
    ("Nt", "near_term"),
    ("st", "short_term"),
    ("ST", "short_term"),
    ("St", "short_term"),
    ("lt", "long_term"),
    ("LT", "long_term"),
    ("Lt", "long_term"),
])
def test_weather_aliases_parse_as_commands(alias, section):
    command = TextParser().parse_command(alias)
    assert command.section == section


@pytest.mark.parametrize("payload", [
    None,
    {"section": "short_term", "sentences": ["Rain."]},
    {"section": "near_term", "sentences": []},
    {"section": "near_term", "sentences": ["Rain."] * 5},
    {"section": "near_term", "sentences": [""]},
    {"section": "near_term", "sentences": ["é"]},
    {"section": "near_term", "sentences": ["line\nbreak"]},
    {"section": "near_term", "sentences": ["x" * 134]},
    {"section": "near_term", "sentences": ["x" * 133] * 5},
])
def test_invalid_summaries_are_rejected(payload):
    with pytest.raises(weather.WeatherServiceError):
        weather.validate_summary(payload, "near_term")


def test_summary_sentences_are_preserved():
    sentences = ["Rain is likely.", "  Winds increase tonight."]
    assert weather.validate_summary(
        {"section": "near_term", "sentences": sentences,
         "summary": "Ignored joined value"},
        "near_term") == sentences


def test_numbered_sentences_include_total_and_fit_packet_limit():
    sentences = weather.number_sentences(["Rain is likely.", "Winds increase."], 20)

    assert sentences == ["Rain is lik... (1/2)", "Winds incre... (2/2)"]
    assert all(len(sentence.encode("ascii")) <= 20 for sentence in sentences)


def test_numbered_sentences_truncate_with_ellipsis_to_fit():
    numbered = weather.number_sentences(["x" * 133, "Clear."], 16)

    assert numbered[0] == "xxxxxxx... (1/2)"
    assert len(numbered[0]) == 16
    assert numbered[1] == "Clear. (2/2)"


def test_numbered_sentences_reject_packet_limit_too_small_for_suffix():
    with pytest.raises(weather.WeatherServiceError):
        weather.number_sentences(["Rain."], 8)


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def json(self):
        return self.payload


@pytest.mark.asyncio
async def test_fetch_summary_uses_configured_endpoint_without_redirects(monkeypatch):
    response = FakeResponse(
        200, {"section": "near_term", "sentences": ["Rain is likely."]})
    get = Mock()

    def fake_get(url, timeout, allow_redirects):
        get(url, timeout=timeout, allow_redirects=allow_redirects)
        return response

    monkeypatch.setattr(weather.requests, "get", fake_get)

    sentences = await weather.fetch_summary(
        "http://noaa-ct:8765/", "near_term", 12)

    assert sentences == ["Rain is likely."]
    get.assert_called_once_with(
        "http://noaa-ct:8765/api/v1/forecasts/near_term/message_summary",
        timeout=12, allow_redirects=False)


@pytest.mark.asyncio
async def test_fetch_summary_handles_http_error(monkeypatch):
    monkeypatch.setattr(
        weather.requests, "get",
        lambda *args, **kwargs: FakeResponse(503, {"detail": "internal info"}))

    with pytest.raises(weather.WeatherServiceError):
        await weather.fetch_summary("http://weather", "synopsis", 10)


@pytest.mark.asyncio
async def test_fetch_summary_handles_timeout(monkeypatch):
    def timeout(*args, **kwargs):
        raise weather.requests.Timeout

    monkeypatch.setattr(weather.requests, "get", timeout)

    with pytest.raises(weather.WeatherServiceError, match="timed out after 10 seconds"):
        await weather.fetch_summary("http://weather", "synopsis", 10)


@pytest.mark.asyncio
async def test_weather_command_only_requests_in_wx(monkeypatch):
    fetch = AsyncMock(return_value=["Rain is likely.", "Winds increase tonight."])
    monkeypatch.setattr(weather, "fetch_summary", fetch)

    class FakeSessionManager:
        def get_session_state(self, session_id):
            return SimpleNamespace(current_room=1, username="user")

    context = SimpleNamespace(
        session_id="session",
        session_mgr=FakeSessionManager(),
        config=SimpleNamespace(weather={"api_url": "http://weather",
                                        "timeout_seconds": 10},
                              transport={"meshcore": {"max_packet_size": 150}}),
        db=object(),
    )
    response = await registry.get("NT")(username="").run(context)

    assert response.is_error
    assert response.error_code == "wrong_room"
    fetch.assert_not_awaited()


@pytest.mark.asyncio
async def test_weather_command_returns_ordered_verbatim_sentences(monkeypatch):
    events = []

    async def fetch(*args):
        events.append("request_started")
        return ["Rain is likely.", "Winds  increase tonight."]

    async def send_msg(session_id, message):
        events.append("acknowledgement_queued")

    fetch_mock = AsyncMock(side_effect=fetch)
    monkeypatch.setattr(weather, "fetch_summary", fetch_mock)

    class FakeUser:
        async def load(self):
            pass

    class FakeRoom:
        room_id = 7

        def __init__(self, db, config, room_id):
            pass

        async def load(self):
            pass

        def can_user_read(self, user):
            return True

    monkeypatch.setattr(weather, "User", lambda db, username: FakeUser())
    monkeypatch.setattr(weather, "Room", FakeRoom)
    context = SimpleNamespace(
        session_id="session",
        session_mgr=SimpleNamespace(
            get_session_state=lambda session_id: SimpleNamespace(
                current_room=7, username="user"),
            send_msg=send_msg),
        config=SimpleNamespace(weather={"api_url": "http://weather/",
                                        "timeout_seconds": 10},
                              transport={"meshcore": {"max_packet_size": 150}}),
        db=object(),
    )

    response = await registry.get("NT")(username="").run(context)

    assert [item.text for item in response] == [
        "Rain is likely. (1/2)", "Winds  increase tonight. (2/2)"]
    assert all(item.hints == {"verbatim": True} for item in response)
    assert events == ["acknowledgement_queued", "request_started"]
    fetch_mock.assert_awaited_once_with("http://weather/", "near_term", 10)


def test_wx_room_entry_includes_weather_hint_only_for_wx():
    wx_room = SimpleNamespace(room_id=7)
    lobby = SimpleNamespace(room_id=1)
    assert "syn (synopsis)" in room_entry_text(wx_room, "Entered wx")
    assert room_entry_text(lobby, "Entered Lobby") == "Entered Lobby"


@pytest.mark.asyncio
async def test_change_room_to_wx_shows_weather_hint(monkeypatch):
    class FakeUser:
        async def load(self):
            pass

    class FakeRoom:
        def __init__(self, db, config, room_id):
            self.room_id = room_id
            self.name = "Lobby" if room_id == 1 else "wx"

        async def load(self):
            pass

        async def go_to_room(self, room_name):
            assert room_name == "wx"
            return FakeRoom(None, None, 7)

    monkeypatch.setattr("citadel.commands.builtins.User", lambda db, username: FakeUser())
    monkeypatch.setattr("citadel.commands.builtins.Room", FakeRoom)
    state = SimpleNamespace(current_room=1, username="user")
    session_mgr = SimpleNamespace(
        get_session_state=lambda session_id: state,
        set_current_room=lambda session_id, room_id: setattr(
            state, "current_room", room_id),
    )
    context = SimpleNamespace(
        db=object(), config=object(), session_mgr=session_mgr,
        session_id="session")

    response = await ChangeRoomCommand(username="", args="wx").run(context)

    assert state.current_room == 7
    assert "syn (synopsis)" in response.text
    assert "lt (long term)" in response.text


def test_weather_permission_is_limited_to_readable_wx_room():
    class User:
        permission_level = PermissionLevel.USER

    class Room:
        def __init__(self, room_id, can_read=True):
            self.room_id = room_id
            self.name = "wx" if room_id == 7 else "Lobby"
            self.can_read = can_read

        def can_user_read(self, user):
            return self.can_read

    user = User()
    assert is_allowed("weather", user, Room(7))
    assert not is_allowed("weather", user, Room(1))
    assert not is_allowed("weather", user, Room(7, can_read=False))


@pytest.mark.asyncio
async def test_router_queues_verbatim_sentences_without_prompt_or_header():
    sentences = [
        ToUser(session_id="session", text="First.", hints={"verbatim": True}),
        ToUser(session_id="session", text="Second.", hints={"verbatim": True}),
    ]
    queued = []

    class SessionManager:
        def get_session_by_node_id(self, node_id):
            return "session"

        def get_workflow(self, session_id):
            return None

        async def touch_password_cache(self, username, node_id):
            pass

        async def set_cache_username(self, username, node_id):
            pass

        async def mark_logged_in(self, session_id, logged_in):
            pass

        def mark_username(self, session_id, username):
            pass

        async def send_msg(self, session_id, message):
            queued.append(message)

    router = MessageRouter.__new__(MessageRouter)
    router.dedupe = SimpleNamespace(is_duplicate=AsyncMock(return_value=False))
    router.session_mgr = SessionManager()
    router.node_auth = SimpleNamespace(
        node_has_password_cache=AsyncMock(return_value="user"),
        touch_password_cache=AsyncMock(),
        set_cache_username=AsyncMock(),
    )
    router.text_parser = SimpleNamespace(parse_command=lambda text: object())
    router.command_processor = SimpleNamespace(process=AsyncMock(return_value=sentences))
    router.insert_prompt = AsyncMock(side_effect=AssertionError("unexpected prompt"))
    router.send_msg_header = AsyncMock(side_effect=AssertionError("unexpected header"))

    await router._process_mc_message_safe(SimpleNamespace(payload={
        "pubkey_prefix": "node", "text": "nt", "sender_timestamp": 1}))

    assert queued == sentences
    router.insert_prompt.assert_not_awaited()
    router.send_msg_header.assert_not_awaited()


@pytest.mark.asyncio
async def test_protocol_sends_verbatim_text_as_one_packet(monkeypatch):
    config = SimpleNamespace(transport={"meshcore": {
        "max_packet_size": 150, "inter_packet_delay": 0}})
    meshcore = SimpleNamespace(commands=SimpleNamespace(
        send_msg_with_retry=AsyncMock()))
    handler = ProtocolHandler(config, object(), meshcore)
    handler._send_packet = AsyncMock(return_value=True)
    monkeypatch.setattr(weather.asyncio, "sleep", AsyncMock())
    sentence = ToUser(session_id="session", text="Rain  is likely.",
                      hints={"verbatim": True})

    result = await handler.send_to_node("node", "user", sentence)

    assert result is True
    handler._send_packet.assert_awaited_once_with("user", "node", "Rain  is likely.")


@pytest.mark.asyncio
async def test_protocol_rejects_oversized_verbatim_text():
    config = SimpleNamespace(transport={"meshcore": {
        "max_packet_size": 5, "inter_packet_delay": 0}})
    meshcore = SimpleNamespace(commands=SimpleNamespace(
        send_msg_with_retry=AsyncMock()))
    handler = ProtocolHandler(config, object(), meshcore)
    handler._send_packet = AsyncMock(return_value=True)

    result = await handler.send_to_node(
        "node", "user", ToUser(session_id="session", text="Too long",
                                  hints={"verbatim": True}))

    assert result is False
    handler._send_packet.assert_not_awaited()
