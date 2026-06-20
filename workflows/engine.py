"""Workflow engine — state machine runner with SQLite persistence.

Usage::

    from workflows.engine import WorkflowEngine, WorkflowRun
    from workflows.daily_briefing import DailyBriefingWorkflow

    async def run_pipeline(text: str) -> str: ...   # passed from main.py

    engine = WorkflowEngine(pipeline=run_pipeline)
    engine.initialize()

    run = await engine.run(DailyBriefingWorkflow(), {"query": "..."})
    print(run.state, run.context.get("output"))
"""

import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS workflow_runs (
    run_id        TEXT PRIMARY KEY,
    workflow_name TEXT NOT NULL,
    state         TEXT NOT NULL,
    current_step  INTEGER NOT NULL DEFAULT 0,
    context_json  TEXT NOT NULL DEFAULT '{}',
    started_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    error         TEXT
);
CREATE INDEX IF NOT EXISTS idx_workflow_runs_name ON workflow_runs (workflow_name);
CREATE INDEX IF NOT EXISTS idx_workflow_runs_state ON workflow_runs (state);
"""


class WorkflowState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WorkflowStep:
    """A single named step in a workflow."""

    name: str
    description: str
    action: Callable[[Dict[str, Any]], Awaitable[str]]
    required: bool = True


@dataclass
class WorkflowRun:
    """Live record for a single workflow execution."""

    run_id: str
    workflow_name: str
    state: WorkflowState
    current_step: int
    context: Dict[str, Any]
    started_at: str
    updated_at: str
    error: Optional[str] = None


class BaseWorkflow(ABC):
    """Abstract base class for all workflow implementations."""

    name: str

    @abstractmethod
    def build_steps(self, context: Dict[str, Any]) -> List[WorkflowStep]:
        """Return the ordered list of steps for this workflow.

        *context* is the initial context dict passed to
        :meth:`WorkflowEngine.run`.  Steps may inspect it to conditionally
        include or exclude steps.
        """


class WorkflowEngine:
    """Runs :class:`BaseWorkflow` instances with SQLite-persisted state.

    Parameters
    ----------
    pipeline:
        Async callable ``(text: str) -> str`` used by workflow steps that
        need to invoke the agent pipeline (Router → Planner → Executor →
        Persona).  Injected at construction so workflows stay decoupled from
        ``main.py`` and are straightforward to test.
    db_path:
        Path to the SQLite database file for run persistence.
    """

    def __init__(
        self,
        pipeline: Callable[[str], Awaitable[str]],
        db_path: str = "data/workflows.db",
    ) -> None:
        self._pipeline = pipeline
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create the database and schema if they do not already exist."""
        import os

        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_DB_SCHEMA)
        self._conn.commit()
        logger.info("WorkflowEngine: database ready at %s", self._db_path)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        workflow: BaseWorkflow,
        initial_context: Optional[Dict[str, Any]] = None,
    ) -> WorkflowRun:
        """Execute *workflow* from the first step and return the final run record."""
        context = dict(initial_context or {})
        context["_pipeline"] = self._pipeline

        now = datetime.now(timezone.utc).isoformat()
        run = WorkflowRun(
            run_id=str(uuid4()),
            workflow_name=workflow.name,
            state=WorkflowState.IDLE,
            current_step=0,
            context=context,
            started_at=now,
            updated_at=now,
        )
        self._persist(run)
        return await self._execute(run, workflow)

    async def resume(self, run_id: str, workflow: BaseWorkflow) -> WorkflowRun:
        """Resume a PAUSED or FAILED run from its saved step."""
        run = self._load(run_id)
        if run is None:
            raise KeyError(f"WorkflowRun not found: {run_id}")
        if run.state in (WorkflowState.COMPLETED, WorkflowState.RUNNING):
            logger.warning(
                "WorkflowEngine: cannot resume run %s in state %s", run_id, run.state
            )
            return run
        run.context["_pipeline"] = self._pipeline
        return await self._execute(run, workflow)

    def get_run(self, run_id: str) -> Optional[WorkflowRun]:
        """Return a :class:`WorkflowRun` by ID, or *None* if not found."""
        return self._load(run_id)

    def list_runs(
        self, workflow_name: Optional[str] = None
    ) -> List[WorkflowRun]:
        """Return all runs, optionally filtered by *workflow_name*."""
        self._require_conn()
        if workflow_name:
            rows = self._conn.execute(  # type: ignore[union-attr]
                "SELECT run_id,workflow_name,state,current_step,context_json,started_at,updated_at,error "
                "FROM workflow_runs WHERE workflow_name=? ORDER BY started_at DESC",
                (workflow_name,),
            ).fetchall()
        else:
            rows = self._conn.execute(  # type: ignore[union-attr]
                "SELECT run_id,workflow_name,state,current_step,context_json,started_at,updated_at,error "
                "FROM workflow_runs ORDER BY started_at DESC"
            ).fetchall()
        return [self._row_to_run(r) for r in rows]

    # ------------------------------------------------------------------
    # Internal execution loop
    # ------------------------------------------------------------------

    async def _execute(self, run: WorkflowRun, workflow: BaseWorkflow) -> WorkflowRun:
        steps = workflow.build_steps(run.context)
        run.state = WorkflowState.RUNNING
        self._persist(run)

        for idx, step in enumerate(steps):
            if idx < run.current_step:
                continue  # skip already-completed steps on resume

            logger.info(
                "WorkflowEngine [%s] step %d/%d: %s",
                run.workflow_name,
                idx + 1,
                len(steps),
                step.name,
            )
            try:
                result = await step.action(run.context)
                if result is not None:
                    run.context[step.name] = result
            except Exception as exc:
                logger.error(
                    "WorkflowEngine [%s] step %s failed: %s",
                    run.workflow_name,
                    step.name,
                    exc,
                )
                if step.required:
                    run.state = WorkflowState.FAILED
                    run.error = f"Step '{step.name}' failed: {exc}"
                    run.current_step = idx
                    run.updated_at = datetime.now(timezone.utc).isoformat()
                    self._persist(run)
                    return run
                # Optional step — log and continue
                run.context[step.name] = ""

            run.current_step = idx + 1
            run.updated_at = datetime.now(timezone.utc).isoformat()
            self._persist(run)

        run.state = WorkflowState.COMPLETED
        run.updated_at = datetime.now(timezone.utc).isoformat()
        self._persist(run)
        logger.info("WorkflowEngine [%s] completed run %s", run.workflow_name, run.run_id)
        return run

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _persist(self, run: WorkflowRun) -> None:
        self._require_conn()
        # Exclude the non-serialisable _pipeline callable before encoding.
        safe_ctx = {k: v for k, v in run.context.items() if k != "_pipeline"}
        self._conn.execute(  # type: ignore[union-attr]
            "INSERT OR REPLACE INTO workflow_runs "
            "(run_id,workflow_name,state,current_step,context_json,started_at,updated_at,error) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                run.run_id,
                run.workflow_name,
                run.state.value,
                run.current_step,
                json.dumps(safe_ctx),
                run.started_at,
                run.updated_at,
                run.error,
            ),
        )
        self._conn.commit()  # type: ignore[union-attr]

    def _load(self, run_id: str) -> Optional[WorkflowRun]:
        self._require_conn()
        row = self._conn.execute(  # type: ignore[union-attr]
            "SELECT run_id,workflow_name,state,current_step,context_json,started_at,updated_at,error "
            "FROM workflow_runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        return self._row_to_run(row) if row else None

    @staticmethod
    def _row_to_run(row: tuple) -> WorkflowRun:
        run_id, workflow_name, state, current_step, ctx_json, started_at, updated_at, error = row
        return WorkflowRun(
            run_id=run_id,
            workflow_name=workflow_name,
            state=WorkflowState(state),
            current_step=current_step,
            context=json.loads(ctx_json),
            started_at=started_at,
            updated_at=updated_at,
            error=error,
        )

    def _require_conn(self) -> None:
        if self._conn is None:
            raise RuntimeError("WorkflowEngine not initialized. Call initialize() first.")
