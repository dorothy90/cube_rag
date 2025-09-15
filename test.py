from agent_orchestrator import AgentOrchestrator
res = AgentOrchestrator().run("컨테이너 오케스트레이션이 뭔가요?")
print(res["answer"]); print(res["sources"])