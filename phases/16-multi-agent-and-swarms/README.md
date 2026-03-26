# Phase 16: Multi-Agent & Swarms
Moving from single agents to teams of specialized agents that collaborate.

## Roadmap
| Lesson | Description | Status |
|--------|-------------|--------|
| 01. CrewAI Fundamentals | Leaders, Workers, and Tasks. | ✅ |
| 02. AutoGen | Conversational agents. | ⬚ |
| 03. Agent Consensus | Resolving disagreements between agents. | ⬚ |

## Code Example: CrewAI
```python
from crewai import Agent, Task, Crew

# Define Agents
researcher = Agent(role='Researcher', goal='Find latest AI news', backstory='Expert analyst', allow_delegation=False)
writer = Agent(role='Writer', goal='Summarize news', backstory='Tech journalist', allow_delegation=False)

# Define Tasks
task1 = Task(description='Search for AI news today.', expected_output='A bullet list of news', agent=researcher)
task2 = Task(description='Write a blog post from the list.', expected_output='A blog post', agent=writer)

# Create Crew
crew = Crew(agents=[researcher, writer], tasks=[task1, task2], verbose=True)
result = crew.kickoff()
```
