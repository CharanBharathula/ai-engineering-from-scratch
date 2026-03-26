# Lesson 01: CrewAI Fundamentals (Leader-Worker Architecture)

## 1. Moving from Single Agents to Teams
A single ReAct agent (like we built in Phase 14) is great for simple tasks. However, if you give one agent 20 different tools and a massive goal (like "Research the market, write a 10-page report, and code a website"), it will fail. 

The prompt becomes too large, the LLM loses focus, and tool selection accuracy plummets. 

The solution is **Multi-Agent Systems**. Just like human companies, we break the work down and assign it to specialized agents. 

## 2. What is CrewAI?
CrewAI is the leading framework for building hierarchical, role-playing agent teams. It is built on top of LangChain.

The core philosophy of CrewAI is: **Role-Playing**. You define agents with distinct personalities, backgrounds, and specific tools, preventing them from stepping on each other's toes.

## 3. The Core Components of CrewAI

1. **Agents:** The individual "employees". Defined by a `role`, a `goal`, and a `backstory`.
2. **Tasks:** The specific jobs that need to be done. Defined by a `description` and an `expected_output`. Tasks are assigned to Agents.
3. **Tools:** The functions agents can use (e.g., Google Search, File Reader).
4. **Crew:** The "company" that groups the Agents and Tasks together.
5. **Process:** How the work is managed (Sequential vs. Hierarchical).

## 4. The Architectures

### A. Sequential Process (The Conveyor Belt)
In a sequential process, Task 1 is completed by Agent 1. The output of Task 1 is then passed directly as the input to Task 2, which is handled by Agent 2. 
*Best for: Linear pipelines (Research -> Write -> Publish).*

### B. Hierarchical Process (The Manager)
In a hierarchical process, a "Manager Agent" (usually powered by a smarter model like GPT-4) receives the overarching goal. The Manager figures out what needs to be done and delegates sub-tasks to the worker agents. 
*Best for: Complex, ambiguous goals.*

## 5. Building a "Research & Writing" Team

Here is a complete, runnable example of a Sequential CrewAI team.

```python
# pip install crewai langchain-openai duckduckgo-search
import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun

# 1. Set up the LLM (The "Brain" for our agents)
os.environ["OPENAI_API_KEY"] = "your-api-key"
llm = ChatOpenAI(model="gpt-4o-mini")

# 2. Define Tools
search_tool = DuckDuckGoSearchRun()

# 3. Create the Agents
researcher = Agent(
    role='Senior Research Analyst',
    goal='Uncover cutting-edge developments in AI engineering',
    backstory="""You work at a leading tech think tank.
    Your expertise lies in identifying emerging trends.
    You have a knack for dissecting complex data and presenting actionable insights.""",
    verbose=True,
    allow_delegation=False, # We don't want the researcher assigning work
    tools=[search_tool],
    llm=llm
)

writer = Agent(
    role='Tech Content Strategist',
    goal='Craft compelling content on tech advancements',
    backstory="""You are a renowned Content Strategist, known for your insightful
    and engaging articles. You transform complex concepts into compelling narratives.""",
    verbose=True,
    allow_delegation=False,
    llm=llm # Notice the writer does not have the search tool. They just write.
)

# 4. Create the Tasks
task1 = Task(
    description="""Conduct a comprehensive analysis of the latest advancements in Multi-Agent AI systems in 2024.
    Identify key frameworks (like CrewAI, AutoGen) and their use cases.""",
    expected_output="A full analysis report with at least 3 bullet points on key frameworks.",
    agent=researcher
)

task2 = Task(
    description="""Using the insights provided, develop an engaging blog post that highlights the most significant 
    Multi-Agent AI advancements. Your post should be informative yet accessible.""",
    expected_output="A 4-paragraph blog post formatted in markdown.",
    agent=writer
)

# 5. Form the Crew and Start
# Process.sequential means Task 1 will run, and its output will feed into Task 2
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=2,
    process=Process.sequential 
)

# Start the execution!
result = crew.kickoff()

print("========================================")
print("FINAL RESULT:")
print(result)
```

## 6. Best Practices for Multi-Agent Systems
- **Narrow Roles:** Do not make a "General Developer" agent. Make a "Python Backend Engineer" and a "React Frontend Developer".
- **Clear Expected Outputs:** Always define exactly what the task should return in the `expected_output` field (e.g., "A valid JSON object" or "A 3-paragraph summary").
- **Prevent Endless Delegation:** If workers are allowed to delegate tasks to each other, they can get caught in an infinite loop passing the buck. Use `allow_delegation=False` for bottom-tier workers.