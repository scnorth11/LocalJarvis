## Plan: LocalJarvis multi-agent implementation

TL;DR: Use the current scaffold to build the full Jarvis voice assistant in phases. Start with reusable agent interfaces, then implement Router, Planner, Executor, Memory, and Persona agents. Add the voice pipeline and workflow wiring last, keeping each component integration-friendly and testable.

**Phases**
1. Foundations and agent contracts
   - Create the basic Python modules for each agent folder.
   - Define clear method contracts and return types.
   - Keep agent responsibilities narrow: Router routes, Planner plans, Executor executes, Memory stores/retrieves, Persona adds voice/context style.

2. Router and tiered model selection
   - Implement `agents/router/router_agent.py`.
   - Add local Ollama CLI support and a simple `models/selection_logic.py` helper.
   - Build intent classification and tier selection for `phi-3-mini` vs `llama-3.1-8b`.
   - Return a structured task payload for Planner.

3. Planner + Executor + Persona skeletons
   - Implement `agents/planner/planner_agent.py` with `plan(task: dict) -> dict`.
   - Implement `agents/executor/executor_agent.py` with `execute(plan: dict) -> dict`.
   - Implement `agents/persona/persona_agent.py` to enrich output for voice style and persona context.
   - Keep these modules independent and wireable through `main.py`.

4. Memory agent and local embeddings
   - Implement `agents/memory_agent/memory_agent.py` with CRUD methods.
   - Implement `memory/sqlite_store.py` for SQLite persistence.
   - Implement `memory/embeddings.py` for a local MiniLM embedding path.
   - Expose a simple memory API for `read`, `write`, and `search`.

5. Voice pipeline integration
   - Implement `voice/whisper_stt.py` for Whisper audio-to-text entry.
   - Implement `voice/piper_tts.py` for local voice synthesis output.
   - Implement `voice/vosk_wakeword.py` for wake-word detection.
   - Wire these with the Router in a lightweight `main.py` proof-of-concept.

6. Workflow engine and tools
   - Use `workflows/` for predefined flows like coding, research, and task management.
   - Define a workflow engine in `main.py` or a new module if needed.
   - Map tool folders to agent calls; keep actual tool implementations scoped for later expansion.

7. Wiring and validation
   - Build the end-to-end path in `main.py`: STT → Router → Planner → Executor → Persona → TTS.
   - Add config entries in `config/settings.yaml` for model names, timeouts, and tool mappings.
   - Add smoke tests and CLI helpers for each agent.

**Relevant files**
- `agents/router/router_agent.py`
- `agents/planner/planner_agent.py`
- `agents/executor/executor_agent.py`
- `agents/persona/persona_agent.py`
- `agents/memory_agent/memory_agent.py`
- `models/selection_logic.py`
- `memory/sqlite_store.py`
- `memory/embeddings.py`
- `voice/whisper_stt.py`
- `voice/piper_tts.py`
- `voice/vosk_wakeword.py`
- `main.py`
- `config/settings.yaml`
- `workflows/coding_workflow.py`
- `workflows/research_workflow.py`
- `workflows/task_management.py`
- `workflows/voice_command.py`

**Verification**
1. Start with a CLI driver in `main.py` that routes typed text through Router → Planner → Executor.
2. Validate Router returns `selected_model` and `planner_payload` for both simple and complex inputs.
3. Validate memory can insert and retrieve notes; confirm embedding generation path works.
4. Validate Whisper and Piper modules can process test audio files if available.
5. Confirm `config/settings.yaml` controls model tier names and timeouts.

**Decisions**
- Use Ollama CLI subprocess calls for model execution.
- Use SQLite for memory persistence and local MiniLM for embeddings.
- Keep each agent stateless and focused on one stage of the pipeline.
- Build the voice loop around a simple Router-first architecture.

**Next concrete step**
- Create the first skeleton files and the Router agent with model selection support.
- Once done, implement Planner and Executor to consume Router output.
