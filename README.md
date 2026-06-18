# LocalJarvis
Building skills with LLMs and local AI through creating a personalized Jarvis assistant that can call various tools, skills, and even sub agents to help with everyday tasks.

DB: SQLite

Memory: SQLite tables + embeddings

Embeddings: local (Ollama or SentenceTransformers)

LLMs: tiered (small local, medium local, optional big‑rig)

Voice: Whisper STT → Router; Piper TTS; Vosk wake‑word

External APIs: DuckDuckGo‑style fetch tool + Spotify

Agents: Router, Planner, Executor, Memory, Persona

Tools: writing, file search, Spotify DJ, research, calendar, utility