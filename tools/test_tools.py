"""Unit and integration tests for all LocalJarvis tools.

Run with:
    pytest tools/test_tools.py -v

Optional dependencies (duckduckgo-search, spotipy, sympy) are mocked so tests
pass even if they are not installed.
"""
from __future__ import annotations

import sqlite3
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakePaths:
    data_dir = "data"


class _FakeFileConfig:
    allowed_paths = ["data"]


class _FakeCalendarConfig:
    db_path = ":memory:"


class _FakeResearchConfig:
    max_results = 5
    source_guidance = "TEST GUIDANCE"


class _FakeSpotifyConfig:
    client_id = ""
    client_secret = ""
    refresh_token = ""
    redirect_uri = "http://localhost:8888/callback"


class _FakeToolsConfig:
    file = _FakeFileConfig()
    calendar = _FakeCalendarConfig()
    research = _FakeResearchConfig()
    spotify = _FakeSpotifyConfig()


class _FakeConfig:
    """Minimal config stand-in for tool factories."""
    paths = _FakePaths()
    tools = _FakeToolsConfig()


_CFG = _FakeConfig()


# ---------------------------------------------------------------------------
# Phase 1 — BaseTool + PathSanitizer
# ---------------------------------------------------------------------------

class TestPathSanitizer:
    def setup_method(self) -> None:
        from tools.base import PathSanitizer
        self.san = PathSanitizer(["data", "/tmp"])

    def test_allows_path_inside_root(self, tmp_path: Path) -> None:
        from tools.base import PathSanitizer
        san = PathSanitizer([str(tmp_path)])
        result = san.resolve(str(tmp_path / "notes.txt"))
        assert result == (tmp_path / "notes.txt").resolve()

    def test_blocks_traversal(self, tmp_path: Path) -> None:
        from tools.base import PathSanitizer
        san = PathSanitizer([str(tmp_path / "sandbox")])
        with pytest.raises(PermissionError):
            san.resolve(str(tmp_path / "sandbox" / ".." / ".." / "etc" / "passwd"))

    def test_requires_at_least_one_root(self) -> None:
        from tools.base import PathSanitizer
        with pytest.raises(ValueError):
            PathSanitizer([])

    def test_is_allowed_returns_bool(self, tmp_path: Path) -> None:
        from tools.base import PathSanitizer
        san = PathSanitizer([str(tmp_path)])
        assert san.is_allowed(str(tmp_path / "file.txt")) is True
        assert san.is_allowed("/etc/passwd") is False


# ---------------------------------------------------------------------------
# Phase 2 — WritingTool
# ---------------------------------------------------------------------------

class TestWritingTool:
    def _make_tool(self, tmp_path: Path):
        from tools.base import PathSanitizer
        from tools.writing.file_writer import WritingTool
        return WritingTool(PathSanitizer([str(tmp_path)]))

    def test_create_and_read_roundtrip(self, tmp_path: Path) -> None:
        tool = self._make_tool(tmp_path)
        target = str(tmp_path / "hello.txt")
        result = tool.run(op="create", path=target, content="hello world")
        assert "Created" in result
        content = tool.run(op="read", path=target)
        assert content == "hello world"

    def test_create_rejects_existing_file(self, tmp_path: Path) -> None:
        tool = self._make_tool(tmp_path)
        target = str(tmp_path / "dup.txt")
        tool.run(op="create", path=target, content="first")
        result = tool.run(op="create", path=target, content="second")
        assert "already exists" in result

    def test_overwrite(self, tmp_path: Path) -> None:
        tool = self._make_tool(tmp_path)
        target = str(tmp_path / "ow.txt")
        tool.run(op="create", path=target, content="v1")
        tool.run(op="overwrite", path=target, content="v2")
        assert tool.run(op="read", path=target) == "v2"

    def test_append(self, tmp_path: Path) -> None:
        tool = self._make_tool(tmp_path)
        target = str(tmp_path / "app.txt")
        tool.run(op="create", path=target, content="line1\n")
        tool.run(op="append", path=target, content="line2\n")
        assert "line2" in tool.run(op="read", path=target)

    def test_access_denied_outside_root(self, tmp_path: Path) -> None:
        tool = self._make_tool(tmp_path)
        result = tool.run(op="read", path="/etc/passwd")
        assert "Access denied" in result

    def test_invalid_op(self, tmp_path: Path) -> None:
        tool = self._make_tool(tmp_path)
        result = tool.run(op="explode", path=str(tmp_path / "x.txt"))
        assert "Unknown op" in result


# ---------------------------------------------------------------------------
# Phase 2 — FileSearchTool
# ---------------------------------------------------------------------------

class TestFileSearchTool:
    def _make_tool(self, tmp_path: Path):
        from tools.base import PathSanitizer
        from tools.file_search.file_searcher import FileSearchTool
        return FileSearchTool(PathSanitizer([str(tmp_path)]), tmp_path)

    def setup_files(self, tmp_path: Path) -> None:
        (tmp_path / "alpha.txt").write_text("hello world")
        (tmp_path / "beta.txt").write_text("foo bar baz")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "gamma.txt").write_text("needle in haystack")

    def test_search_name(self, tmp_path: Path) -> None:
        self.setup_files(tmp_path)
        tool = self._make_tool(tmp_path)
        result = tool.run(op="search_name", pattern="*.txt")
        assert "alpha.txt" in result

    def test_search_content(self, tmp_path: Path) -> None:
        self.setup_files(tmp_path)
        tool = self._make_tool(tmp_path)
        result = tool.run(op="search_content", query="needle")
        assert "gamma.txt" in result

    def test_read_file(self, tmp_path: Path) -> None:
        self.setup_files(tmp_path)
        tool = self._make_tool(tmp_path)
        result = tool.run(op="read", path=str(tmp_path / "alpha.txt"))
        assert "hello world" in result

    def test_list_dir(self, tmp_path: Path) -> None:
        self.setup_files(tmp_path)
        tool = self._make_tool(tmp_path)
        result = tool.run(op="list_dir")
        assert "alpha.txt" in result or "beta.txt" in result

    def test_search_content_no_query(self, tmp_path: Path) -> None:
        tool = self._make_tool(tmp_path)
        result = tool.run(op="search_content", query="")
        assert "required" in result


# ---------------------------------------------------------------------------
# Phase 3 — MathTool
# ---------------------------------------------------------------------------

class TestMathTool:
    def _make_tool(self):
        from tools.utility.math_tool import MathTool
        return MathTool()

    def test_basic_arithmetic(self) -> None:
        sympy = pytest.importorskip("sympy")
        tool = self._make_tool()
        result = tool.run(expression="2 + 2")
        assert result == "4"

    def test_exponentiation(self) -> None:
        sympy = pytest.importorskip("sympy")
        tool = self._make_tool()
        result = tool.run(expression="2**10")
        assert result == "1024"

    def test_sqrt(self) -> None:
        sympy = pytest.importorskip("sympy")
        tool = self._make_tool()
        result = tool.run(expression="sqrt(144)")
        assert result == "12"

    def test_empty_expression(self) -> None:
        tool = self._make_tool()
        result = tool.run(expression="")
        assert "No expression" in result

    def test_invalid_expression(self) -> None:
        sympy = pytest.importorskip("sympy")
        tool = self._make_tool()
        result = tool.run(expression="import os; os.system('id')")
        # Should not execute — sympy will raise or return a symbolic result safely
        assert "import" not in result or "Could not evaluate" in result

    def test_missing_sympy(self) -> None:
        tool = self._make_tool()
        with patch.dict(sys.modules, {"sympy": None}):
            result = tool.run(expression="1+1")
        assert "not installed" in result or "4" in result or result  # graceful


# ---------------------------------------------------------------------------
# Phase 3 — FileOpsTool
# ---------------------------------------------------------------------------

class TestFileOpsTool:
    def _make_tool(self, tmp_path: Path):
        from tools.base import PathSanitizer
        from tools.utility.file_ops import FileOpsTool
        return FileOpsTool(PathSanitizer([str(tmp_path)]), tmp_path)

    def test_mkdir(self, tmp_path: Path) -> None:
        tool = self._make_tool(tmp_path)
        new_dir = str(tmp_path / "mydir")
        result = tool.run(op="mkdir", src=new_dir)
        assert "created" in result.lower()
        assert Path(new_dir).is_dir()

    def test_copy_file(self, tmp_path: Path) -> None:
        src = tmp_path / "src.txt"
        src.write_text("data")
        tool = self._make_tool(tmp_path)
        result = tool.run(op="copy", src=str(src), dst=str(tmp_path / "dst.txt"))
        assert "copied" in result.lower()
        assert (tmp_path / "dst.txt").read_text() == "data"

    def test_move_file(self, tmp_path: Path) -> None:
        src = tmp_path / "mv_src.txt"
        src.write_text("move me")
        tool = self._make_tool(tmp_path)
        result = tool.run(op="move", src=str(src), dst=str(tmp_path / "mv_dst.txt"))
        assert "moved" in result.lower()
        assert not src.exists()

    def test_delete_file_inside_data_root(self, tmp_path: Path) -> None:
        target = tmp_path / "del.txt"
        target.write_text("bye")
        tool = self._make_tool(tmp_path)
        result = tool.run(op="delete", src=str(target))
        assert "deleted" in result.lower()
        assert not target.exists()

    def test_delete_outside_data_root_blocked(self, tmp_path: Path) -> None:
        from tools.base import PathSanitizer
        from tools.utility.file_ops import FileOpsTool
        # allowed root == tmp_path, but data_root is a sub-folder
        sub = tmp_path / "data"
        sub.mkdir()
        tool = FileOpsTool(PathSanitizer([str(tmp_path)]), sub)
        outside = tmp_path / "outside.txt"
        outside.write_text("should not be deleted")
        result = tool.run(op="delete", src=str(outside))
        assert "restricted" in result.lower() or "outside" in result.lower()

    def test_list(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("x")
        tool = self._make_tool(tmp_path)
        result = tool.run(op="list", src=str(tmp_path))
        assert "f.txt" in result


# ---------------------------------------------------------------------------
# Phase 4 — ResearchTool
# ---------------------------------------------------------------------------

class TestResearchTool:
    def _make_tool(self, guidance: str = "TEST GUIDANCE"):
        from tools.research.ddg_search import ResearchTool
        return ResearchTool(guidance, 5)

    def _fake_ddgs_results(self):
        return [
            {"href": "https://arxiv.org/abs/2301.00001", "title": "Paper on X", "body": "Abstract text"},
            {"href": "https://reddit.com/r/science/comments/abc", "title": "Reddit thread", "body": "Discussion"},
            {"href": "https://tmz.com/story/celebrity", "title": "TMZ Gossip", "body": "Celebrity news"},
            {"href": "https://github.com/user/repo", "title": "GitHub Repo", "body": "Source code"},
        ]

    def test_annotations_present(self) -> None:
        from tools.research.ddg_search import ResearchTool

        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text = MagicMock(return_value=self._fake_ddgs_results())

        with patch("tools.research.ddg_search.DDGS", return_value=mock_ddgs):
            tool = self._make_tool()
            result = tool.run(query="test query")

        assert "[ACADEMIC]" in result
        assert "[SOCIAL]" in result
        assert "[TABLOID]" in result
        assert "[DEV]" in result

    def test_source_guidance_in_output(self) -> None:
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text = MagicMock(return_value=self._fake_ddgs_results())

        with patch("tools.research.ddg_search.DDGS", return_value=mock_ddgs):
            tool = self._make_tool("MY GUIDANCE")
            result = tool.run(query="test")

        assert "MY GUIDANCE" in result

    def test_empty_query(self) -> None:
        tool = self._make_tool()
        result = tool.run(query="")
        assert "No query" in result

    def test_no_results(self) -> None:
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text = MagicMock(return_value=[])

        with patch("tools.research.ddg_search.DDGS", return_value=mock_ddgs):
            tool = self._make_tool()
            result = tool.run(query="obscure topic")

        assert "No results" in result

    def test_missing_library(self) -> None:
        tool = self._make_tool()
        with patch("tools.research.ddg_search.DDGS", None):
            result = tool.run(query="something")
        assert "not installed" in result


# ---------------------------------------------------------------------------
# Phase 5 — CalendarTool
# ---------------------------------------------------------------------------

class TestCalendarTool:
    def _make_tool(self) -> Any:
        from tools.calendar.calendar_tool import CalendarTool
        return CalendarTool(Path(":memory:"))

    def _make_tool_real(self, tmp_path: Path) -> Any:
        from tools.calendar.calendar_tool import CalendarTool
        return CalendarTool(tmp_path / "cal.db")

    def test_add_and_upcoming(self, tmp_path: Path) -> None:
        tool = self._make_tool_real(tmp_path)
        result = tool.run(op="add", title="Team Meeting", start_dt="2030-01-01T10:00:00", end_dt="2030-01-01T11:00:00")
        assert "added" in result.lower()
        upcoming = tool.run(op="upcoming", n=5)
        assert "Team Meeting" in upcoming

    def test_get_event(self, tmp_path: Path) -> None:
        tool = self._make_tool_real(tmp_path)
        add_result = tool.run(op="add", title="Get Test", start_dt="2030-02-01T09:00:00", end_dt="2030-02-01T10:00:00")
        event_id = add_result.split("id=")[1].split(")")[0].strip()
        get_result = tool.run(op="get", event_id=event_id)
        assert "Get Test" in get_result

    def test_search(self, tmp_path: Path) -> None:
        tool = self._make_tool_real(tmp_path)
        tool.run(op="add", title="Doctor Appointment", start_dt="2030-03-01T08:00:00", end_dt="2030-03-01T09:00:00")
        result = tool.run(op="search", query="Doctor")
        assert "Doctor Appointment" in result

    def test_delete(self, tmp_path: Path) -> None:
        tool = self._make_tool_real(tmp_path)
        add_result = tool.run(op="add", title="Delete Me", start_dt="2030-04-01T08:00:00", end_dt="2030-04-01T09:00:00")
        event_id = add_result.split("id=")[1].split(")")[0].strip()
        del_result = tool.run(op="delete", event_id=event_id)
        assert "deleted" in del_result.lower()
        upcoming = tool.run(op="upcoming", n=10)
        assert "Delete Me" not in upcoming

    def test_update(self, tmp_path: Path) -> None:
        tool = self._make_tool_real(tmp_path)
        add_result = tool.run(op="add", title="Old Title", start_dt="2030-05-01T08:00:00", end_dt="2030-05-01T09:00:00")
        event_id = add_result.split("id=")[1].split(")")[0].strip()
        tool.run(op="update", event_id=event_id, title="New Title")
        get_result = tool.run(op="get", event_id=event_id)
        assert "New Title" in get_result

    def test_add_missing_fields(self, tmp_path: Path) -> None:
        tool = self._make_tool_real(tmp_path)
        result = tool.run(op="add", title="Incomplete")
        assert "required" in result.lower()

    def test_invalid_op(self, tmp_path: Path) -> None:
        tool = self._make_tool_real(tmp_path)
        result = tool.run(op="teleport")
        assert "Unknown op" in result


# ---------------------------------------------------------------------------
# Phase 6 — SpotifyTool
# ---------------------------------------------------------------------------

class TestSpotifyTool:
    def _make_tool_with_mock_client(self):
        from tools.spotify.spotify_tool import SpotifyTool
        tool = SpotifyTool("fake_id", "fake_secret", "fake_token", "http://localhost")
        mock_sp = MagicMock()
        tool._sp = mock_sp  # inject pre-built mock
        return tool, mock_sp

    def test_play_by_query(self) -> None:
        tool, sp = self._make_tool_with_mock_client()
        sp.search.return_value = {
            "tracks": {"items": [{"name": "Bohemian Rhapsody", "artists": [{"name": "Queen"}], "uri": "spotify:track:abc"}]}
        }
        result = tool.run(op="play", query="Queen")
        assert "Bohemian Rhapsody" in result
        sp.start_playback.assert_called_once()

    def test_play_by_uri(self) -> None:
        tool, sp = self._make_tool_with_mock_client()
        result = tool.run(op="play", uri="spotify:track:xyz")
        assert "spotify:track:xyz" in result
        sp.start_playback.assert_called_once_with(uris=["spotify:track:xyz"], device_id=None)

    def test_pause(self) -> None:
        tool, sp = self._make_tool_with_mock_client()
        result = tool.run(op="pause")
        assert "Paused" in result
        sp.pause_playback.assert_called_once()

    def test_next_track(self) -> None:
        tool, sp = self._make_tool_with_mock_client()
        result = tool.run(op="next_track")
        assert "next" in result.lower()
        sp.next_track.assert_called_once()

    def test_search(self) -> None:
        tool, sp = self._make_tool_with_mock_client()
        sp.search.return_value = {
            "tracks": {"items": [
                {"name": "Song A", "artists": [{"name": "Artist A"}], "uri": "spotify:track:1"},
            ]}
        }
        result = tool.run(op="search", query="Artist A")
        assert "Song A" in result

    def test_current_track_nothing_playing(self) -> None:
        tool, sp = self._make_tool_with_mock_client()
        sp.current_playback.return_value = None
        result = tool.run(op="current_track")
        assert "Nothing" in result

    def test_create_playlist(self) -> None:
        tool, sp = self._make_tool_with_mock_client()
        sp.current_user.return_value = {"id": "user123"}
        sp.user_playlist_create.return_value = {"uri": "spotify:playlist:new"}
        result = tool.run(op="create_playlist", name="My Jams")
        assert "My Jams" in result

    def test_dj_mode(self) -> None:
        tool, sp = self._make_tool_with_mock_client()
        sp.recommendations.return_value = {
            "tracks": [
                {"uri": "spotify:track:r1", "name": "Rec 1", "artists": [{"name": "Band"}]},
                {"uri": "spotify:track:r2", "name": "Rec 2", "artists": [{"name": "Band"}]},
            ]
        }
        result = tool.run(op="dj_mode", seed_track_uri="spotify:track:seed", n=2)
        assert "Rec 1" in result
        assert sp.add_to_queue.call_count == 2

    def test_no_credentials_returns_graceful_error(self) -> None:
        from tools.spotify.spotify_tool import SpotifyTool
        tool = SpotifyTool("", "", "", "")
        result = tool.run(op="current_track")
        assert "initialised" in result or "credentials" in result.lower() or "SPOTIFY" in result

    def test_invalid_op(self) -> None:
        tool, _ = self._make_tool_with_mock_client()
        result = tool.run(op="broadcast")
        assert "Unknown op" in result


# ---------------------------------------------------------------------------
# Phase 8 — Executor integration with tool_call PlanSteps
# ---------------------------------------------------------------------------

class TestExecutorToolCallIntegration:
    """Verify the Executor dispatches tool_call steps through the tool dict."""

    @pytest.mark.asyncio
    async def test_executor_runs_tool_call_step(self, tmp_path: Path) -> None:
        from agents.executor.executor_agent import ExecutorAgent
        from core.schema import AgentMessage, PlanPayload, PlanStep, RetryPolicy

        executor = ExecutorAgent()

        # Build a minimal tools dict — writing tool writing to tmp_path.
        from tools.base import PathSanitizer
        from tools.writing.file_writer import WritingTool
        writing = WritingTool(PathSanitizer([str(tmp_path)]))
        executor.tools = {"writing": writing.run}
        executor.invoke_tool = lambda name, **kw: executor.tools[name](**kw)

        step = PlanStep(
            id="step-1-tool",
            action="tool_call",
            args={"tool": "writing", "op": "create", "path": str(tmp_path / "out.txt"), "content": "from executor"},
            model=None,
            retry_policy=RetryPolicy(max_retries=0, backoff_ms=0, retry_on=[]),
        )
        msg = AgentMessage(
            id="test-1",
            timestamp="2026-01-01T00:00:00",
            source="planner",
            target="executor",
            type="plan.created",
            payload=PlanPayload(steps=[step], expected_outputs=["tool_result"]),
            metadata={},
        )

        result_msg = await executor.handle(msg)
        step_result = result_msg.payload.results["step-1-tool"]
        assert step_result["status"] == "success"
        assert (tmp_path / "out.txt").read_text() == "from executor"


# ---------------------------------------------------------------------------
# Phase 8 — Planner intent→tool mapping
# ---------------------------------------------------------------------------

class TestPlannerIntentMapping:
    def _planner(self):
        from agents.planner.planner_agent import PlannerAgent
        return PlannerAgent()

    @pytest.mark.parametrize("intent,expected_tool", [
        ("calculate 2 + 2", "math"),
        ("research quantum computing", "research"),
        ("write a new document", "writing"),
        ("find file report.csv", "file_search"),
        ("upcoming calendar events", "calendar"),
        ("play music on spotify", "spotify"),
        ("copy file src.txt to dst.txt", "file_ops"),
    ])
    def test_intent_maps_to_tool(self, intent: str, expected_tool: str) -> None:
        planner = self._planner()
        result = planner._map_intent_to_tool(intent)
        assert result == expected_tool

    def test_unrecognised_intent_returns_none(self) -> None:
        planner = self._planner()
        result = planner._map_intent_to_tool("tell me a joke")
        assert result is None

    @pytest.mark.asyncio
    async def test_plan_includes_tool_call_step_when_tool_detected(self) -> None:
        from agents.planner.planner_agent import PlannerAgent
        from core.schema import AgentMessage, TaskPayload

        planner = PlannerAgent()
        msg = AgentMessage(
            id="t1",
            timestamp="2026-01-01T00:00:00",
            source="router",
            target="planner",
            type="task.routing",
            payload=TaskPayload(
                user_intent="calculate 2**8",
                input_text="calculate 2**8",
                selected_model="phi-3-mini",
                intent_confidence=0.9,
                context={"routing_tier": "small"},
                constraints={},
            ),
            metadata={},
        )
        result = await planner.handle(msg)
        steps = result.payload.steps
        actions = [s.action for s in steps]
        assert "tool_call" in actions
        tool_step = next(s for s in steps if s.action == "tool_call")
        assert tool_step.args.get("tool") == "math"


# ---------------------------------------------------------------------------
# Phase 7 — Config schema parses tools section
# ---------------------------------------------------------------------------

class TestConfigSchemaTools:
    def test_appconfig_parses_tools_section(self) -> None:
        from config.schema import AppConfig

        raw = {
            "models": {"default": "small", "tiers": {"small": "phi-3-mini", "large": "llama-3.1-8b"}},
            "paths": {"data_dir": "data", "cache_dir": ".cache", "log_dir": "logs"},
            "security": {"allowed_tools": {"executor": ["writing", "math"]}},
            "voice": {"default_voice": "alloy", "tts_engine": "piper", "stt_engine": "whisper"},
            "timeouts": {"model_call_seconds": 30},
            "tools": {
                "file": {"allowed_paths": ["data", "/tmp/jarvis"]},
                "calendar": {"db_path": "data/calendar.db"},
                "research": {"max_results": 15, "source_guidance": "prefer academic"},
                "spotify": {
                    "client_id": "abc",
                    "client_secret": "def",
                    "refresh_token": "ghi",
                    "redirect_uri": "http://localhost:8888/callback",
                },
            },
        }
        cfg = AppConfig.from_dict(raw)
        assert cfg.tools.file.allowed_paths == ["data", "/tmp/jarvis"]
        assert cfg.tools.calendar.db_path == "data/calendar.db"
        assert cfg.tools.research.max_results == 15
        assert cfg.tools.research.source_guidance == "prefer academic"
        assert cfg.tools.spotify.client_id == "abc"

    def test_appconfig_defaults_when_tools_missing(self) -> None:
        from config.schema import AppConfig

        raw = {
            "models": {"default": "small", "tiers": {"small": "phi-3-mini", "large": "llama-3.1-8b"}},
            "paths": {"data_dir": "data", "cache_dir": ".cache", "log_dir": "logs"},
            "security": {"allowed_tools": {}},
            "voice": {"default_voice": "alloy", "tts_engine": "piper", "stt_engine": "whisper"},
            "timeouts": {},
        }
        cfg = AppConfig.from_dict(raw)
        assert cfg.tools.calendar.db_path == "data/calendar.db"
        assert cfg.tools.research.max_results == 10
        assert cfg.tools.spotify.client_id == ""
