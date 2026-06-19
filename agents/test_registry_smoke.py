import asyncio

from agents.agent_registry import AgentRegistry
from security.enforcement import SecurityEnforcer


class FakeAgent:
    def __init__(self):
        self.tools = {
            "allowed_tool": lambda x: f"echo:{x}",
            "forbidden_tool": lambda: "secret",
        }

    async def handle(self, message):
        return self.invoke_tool("allowed_tool", message or "hello")


async def main():
    registry = AgentRegistry()

    # Make enforcement deterministic for the test by replacing the enforcer
    registry.enforcer = SecurityEnforcer({"fake_agent": {"allowed_tool"}})

    agent = FakeAgent()
    await registry.register_agent("fake_agent", agent)

    proxy = registry.get("fake_agent")
    out = await proxy.handle("world")
    print("allowed->", out)

    try:
        registry._agents["fake_agent"].tool_invoker("fake_agent", "forbidden_tool")
        print("forbidden-> allowed (unexpected)")
    except Exception as e:
        print("forbidden->", type(e).__name__, str(e))


if __name__ == "__main__":
    asyncio.run(main())
