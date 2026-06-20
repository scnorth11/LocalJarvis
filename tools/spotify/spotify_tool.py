"""SpotifyTool — Spotify DJ skill via the Spotify Web API (spotipy).

Authentication
--------------
Credentials are read from environment variables:

    SPOTIFY_CLIENT_ID      — your Spotify app's client ID
    SPOTIFY_CLIENT_SECRET  — your Spotify app's client secret
    SPOTIFY_REFRESH_TOKEN  — a long-lived refresh token obtained via the
                             Spotify OAuth flow (once-only browser step)

The tool exchanges the refresh token for an access token on first use.
No browser is needed at runtime.

Operations
----------
play            Play by search query or direct Spotify URI.
pause           Pause playback.
resume          Resume playback.
next_track      Skip to next track.
prev_track      Go back to previous track.
search          Search for tracks, artists, albums, or playlists.
queue           Add a track URI to the playback queue.
create_playlist Create a new playlist on the user's account.
add_to_playlist Add track URIs to an existing playlist.
current_track   Return info about the currently playing track.
dj_mode         Auto-queue N recommendations seeded from a track URI.
"""
from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

from tools.base import BaseTool

logger = logging.getLogger(__name__)

_VALID_OPS = {
    "play",
    "pause",
    "resume",
    "next_track",
    "prev_track",
    "search",
    "queue",
    "create_playlist",
    "add_to_playlist",
    "current_track",
    "dj_mode",
}

_SCOPES = " ".join([
    "user-read-playback-state",
    "user-modify-playback-state",
    "user-read-currently-playing",
    "playlist-modify-public",
    "playlist-modify-private",
])


class SpotifyTool(BaseTool):
    """Control Spotify playback and manage playlists.

    Parameters (passed as ``**kwargs`` from the Executor):

    op : str
        One of the operations listed in the module docstring.

    play
        query (str, optional), uri (str, optional), device_id (str, optional)
    pause / resume / next_track / prev_track
        device_id (str, optional)
    search
        query (str), type (str, default "track"), limit (int, default 10)
    queue
        uri (str), device_id (str, optional)
    create_playlist
        name (str), description (str, optional), public (bool, default False)
    add_to_playlist
        playlist_id (str), uris (list[str])
    current_track
        (no additional params)
    dj_mode
        seed_track_uri (str), n (int, default 5), device_id (str, optional)
    """

    name = "spotify"
    description = "Control Spotify playback and manage playlists like a DJ."

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
        redirect_uri: str,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._refresh_token = refresh_token
        self._redirect_uri = redirect_uri
        self._sp: Optional[Any] = None  # lazy spotipy.Spotify instance

    # ------------------------------------------------------------------
    # BaseTool implementation
    # ------------------------------------------------------------------

    def run(self, *, op: str, **kwargs: Any) -> str:
        op = op.lower().strip()
        if op not in _VALID_OPS:
            return f"[spotify] Unknown op '{op}'. Valid ops: {sorted(_VALID_OPS)}"

        client = self._get_client()
        if client is None:
            return (
                "[spotify] Spotify client could not be initialised. "
                "Ensure SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, and "
                "SPOTIFY_REFRESH_TOKEN are set."
            )

        try:
            if op == "play":
                return self._play(client, **kwargs)
            if op == "pause":
                return self._pause(client, **kwargs)
            if op == "resume":
                return self._resume(client, **kwargs)
            if op == "next_track":
                return self._next(client, **kwargs)
            if op == "prev_track":
                return self._prev(client, **kwargs)
            if op == "search":
                return self._search(client, **kwargs)
            if op == "queue":
                return self._queue(client, **kwargs)
            if op == "create_playlist":
                return self._create_playlist(client, **kwargs)
            if op == "add_to_playlist":
                return self._add_to_playlist(client, **kwargs)
            if op == "current_track":
                return self._current_track(client)
            if op == "dj_mode":
                return self._dj_mode(client, **kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SpotifyTool op=%s failed: %s", op, exc)
            return f"[spotify] Error during '{op}': {exc}"

        return "[spotify] Unexpected state"

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def _play(
        self,
        sp: Any,
        *,
        query: str = "",
        uri: str = "",
        device_id: Optional[str] = None,
        **_: Any,
    ) -> str:
        if uri:
            sp.start_playback(uris=[uri], device_id=device_id)
            return f"[spotify] Playing URI: {uri}"
        if query:
            results = sp.search(q=query, type="track", limit=1)
            items = results.get("tracks", {}).get("items", [])
            if not items:
                return f"[spotify] No tracks found for: {query}"
            track = items[0]
            sp.start_playback(uris=[track["uri"]], device_id=device_id)
            return (
                f"[spotify] Now playing: {track['name']} — "
                f"{', '.join(a['name'] for a in track['artists'])}"
            )
        # No query or URI — resume/start contextual playback.
        sp.start_playback(device_id=device_id)
        return "[spotify] Playback started."

    @staticmethod
    def _pause(sp: Any, *, device_id: Optional[str] = None, **_: Any) -> str:
        sp.pause_playback(device_id=device_id)
        return "[spotify] Paused."

    @staticmethod
    def _resume(sp: Any, *, device_id: Optional[str] = None, **_: Any) -> str:
        sp.start_playback(device_id=device_id)
        return "[spotify] Resumed."

    @staticmethod
    def _next(sp: Any, *, device_id: Optional[str] = None, **_: Any) -> str:
        sp.next_track(device_id=device_id)
        return "[spotify] Skipped to next track."

    @staticmethod
    def _prev(sp: Any, *, device_id: Optional[str] = None, **_: Any) -> str:
        sp.previous_track(device_id=device_id)
        return "[spotify] Went back to previous track."

    @staticmethod
    def _search(
        sp: Any,
        *,
        query: str,
        type: str = "track",  # noqa: A002
        limit: int = 10,
        **_: Any,
    ) -> str:
        if not query:
            return "[spotify] 'query' is required for search."
        results = sp.search(q=query, type=type, limit=int(limit))
        lines: List[str] = [f"[spotify] Search results for '{query}' ({type}):"]
        key = type + "s"
        items = results.get(key, {}).get("items", [])
        if not items:
            return f"[spotify] No {type} results for: {query}"
        for item in items:
            if type == "track":
                artists = ", ".join(a["name"] for a in item.get("artists", []))
                lines.append(f"  • {item['name']} — {artists}  [{item['uri']}]")
            elif type == "artist":
                lines.append(f"  • {item['name']}  [{item['uri']}]")
            elif type == "album":
                artist = item.get("artists", [{}])[0].get("name", "")
                lines.append(f"  • {item['name']} — {artist}  [{item['uri']}]")
            elif type == "playlist":
                lines.append(f"  • {item['name']} by {item.get('owner', {}).get('display_name', '?')}  [{item['uri']}]")
            else:
                lines.append(f"  • {item.get('name', str(item))}")
        return "\n".join(lines)

    @staticmethod
    def _queue(sp: Any, *, uri: str, device_id: Optional[str] = None, **_: Any) -> str:
        if not uri:
            return "[spotify] 'uri' is required for queue."
        sp.add_to_queue(uri=uri, device_id=device_id)
        return f"[spotify] Added to queue: {uri}"

    def _create_playlist(
        self,
        sp: Any,
        *,
        name: str,
        description: str = "",
        public: bool = False,
        **_: Any,
    ) -> str:
        if not name:
            return "[spotify] 'name' is required for create_playlist."
        user_id = sp.current_user()["id"]
        playlist = sp.user_playlist_create(
            user=user_id,
            name=name,
            public=bool(public),
            description=description,
        )
        return f"[spotify] Playlist created: '{name}'  [{playlist['uri']}]"

    @staticmethod
    def _add_to_playlist(
        sp: Any,
        *,
        playlist_id: str,
        uris: List[str],
        **_: Any,
    ) -> str:
        if not playlist_id or not uris:
            return "[spotify] 'playlist_id' and 'uris' are required for add_to_playlist."
        sp.playlist_add_items(playlist_id=playlist_id, items=uris)
        return f"[spotify] Added {len(uris)} track(s) to playlist {playlist_id}."

    @staticmethod
    def _current_track(sp: Any) -> str:
        playback = sp.current_playback()
        if not playback or not playback.get("item"):
            return "[spotify] Nothing is currently playing."
        item = playback["item"]
        artists = ", ".join(a["name"] for a in item.get("artists", []))
        progress_ms = playback.get("progress_ms", 0)
        duration_ms = item.get("duration_ms", 0)
        progress = f"{progress_ms // 60000}:{(progress_ms % 60000) // 1000:02d}"
        duration = f"{duration_ms // 60000}:{(duration_ms % 60000) // 1000:02d}"
        is_playing = "▶" if playback.get("is_playing") else "⏸"
        return (
            f"[spotify] {is_playing} {item['name']} — {artists}\n"
            f"          Album: {item.get('album', {}).get('name', '?')}\n"
            f"          Progress: {progress} / {duration}  URI: {item['uri']}"
        )

    @staticmethod
    def _dj_mode(
        sp: Any,
        *,
        seed_track_uri: str,
        n: int = 5,
        device_id: Optional[str] = None,
        **_: Any,
    ) -> str:
        if not seed_track_uri:
            return "[spotify] 'seed_track_uri' is required for dj_mode."
        # Extract track ID from URI (spotify:track:ID).
        track_id = seed_track_uri.split(":")[-1]
        recs = sp.recommendations(seed_tracks=[track_id], limit=int(n))
        tracks = recs.get("tracks", [])
        if not tracks:
            return "[spotify] No recommendations found."
        uris = [t["uri"] for t in tracks]
        for uri in uris:
            sp.add_to_queue(uri=uri, device_id=device_id)
        names = [f"{t['name']} — {', '.join(a['name'] for a in t['artists'])}" for t in tracks]
        return "[spotify] DJ mode: queued recommendations:\n" + "\n".join(
            f"  {i+1}. {n}" for i, n in enumerate(names)
        )

    # ------------------------------------------------------------------
    # Client initialisation
    # ------------------------------------------------------------------

    def _get_client(self) -> Optional[Any]:
        if self._sp is not None:
            return self._sp
        try:
            import spotipy  # noqa: PLC0415
            from spotipy.oauth2 import SpotifyOAuth  # noqa: PLC0415
        except ImportError:
            logger.warning("SpotifyTool: spotipy not installed. Run: pip install spotipy")
            return None

        if not all([self._client_id, self._client_secret, self._refresh_token]):
            logger.warning(
                "SpotifyTool: missing credentials "
                "(SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET / SPOTIFY_REFRESH_TOKEN)"
            )
            return None

        try:
            auth_manager = SpotifyOAuth(
                client_id=self._client_id,
                client_secret=self._client_secret,
                redirect_uri=self._redirect_uri,
                scope=_SCOPES,
            )
            # Inject the pre-supplied refresh token so no browser is needed.
            token_info = auth_manager.refresh_access_token(self._refresh_token)
            self._sp = spotipy.Spotify(auth=token_info["access_token"])
            logger.debug("SpotifyTool: authenticated successfully")
            return self._sp
        except Exception as exc:  # noqa: BLE001
            logger.warning("SpotifyTool: authentication failed: %s", exc)
            return None


# ------------------------------------------------------------------
# Factory consumed by ToolLoader
# ------------------------------------------------------------------

def create_tool(config: Any) -> SpotifyTool:
    # Prefer env vars over config values for credentials.
    client_id = (
        os.environ.get("SPOTIFY_CLIENT_ID")
        or _safe_attr(config, "tools.spotify.client_id")
        or ""
    )
    client_secret = (
        os.environ.get("SPOTIFY_CLIENT_SECRET")
        or _safe_attr(config, "tools.spotify.client_secret")
        or ""
    )
    refresh_token = (
        os.environ.get("SPOTIFY_REFRESH_TOKEN")
        or _safe_attr(config, "tools.spotify.refresh_token")
        or ""
    )
    redirect_uri = (
        os.environ.get("SPOTIFY_REDIRECT_URI")
        or _safe_attr(config, "tools.spotify.redirect_uri")
        or "http://localhost:8888/callback"
    )
    return SpotifyTool(client_id, client_secret, refresh_token, redirect_uri)


def _safe_attr(obj: Any, dotpath: str) -> Any:
    """Safely traverse a dotted attribute path; return None on AttributeError."""
    try:
        for part in dotpath.split("."):
            obj = getattr(obj, part)
        return obj
    except AttributeError:
        return None
