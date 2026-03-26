# Lesson 01: The ReAct Loop (Reason + Act)

## 1. What is an Agent?
In traditional software engineering, code executes in a deterministic, linear path (e.g., `if A then B`). 
In LLM Engineering, an **Agent** is an LLM that is given agency to control a loop. It decides *what* to do next, *which* tools to use, and *when* the task is complete. 

## 2. The Core Mechanism: ReAct
The most foundational architecture for AI agents is the **ReAct** pattern, which stands for **Reasoning and Acting**. Introduced in a 2022 paper by researchers at Princeton and Google, ReAct forces the LLM to think out loud before taking an action.

### The Loop:
1. **Thought:** The LLM analyzes the user's request and the current state.
2. **Action:** The LLM decides to call a specific tool (e.g., `Search("Weather in Tokyo")`).
3. **Observation:** The system executes the tool and feeds the result back to the LLM.
4. **Repeat:** The LLM loops back to step 1 until it resolves the query.

## 3. Why is "Thinking" Important?
If an LLM just outputs an action without thinking, it often hallucinates or chooses the wrong tool. By forcing a `Thought:` step, we allocate "compute tokens" to the LLM to plan its next move. This drastically increases the success rate of complex tasks.

## 4. Building a ReAct Loop from Scratch (No Frameworks)
Before using LangChain or CrewAI, you must understand how to build this in raw Python. 

```python
import re

# Simulated LLM function
def call_llm(prompt):
    # In reality, this calls OpenAI or Anthropic API
    pass

# Simulated Tools
def calculate(expression):
    return eval(expression)

def wikipedia_search(query):
    return "The capital of France is Tokyo... wait, no, it's Paris."

tools = {
    "Calculator": calculate,
    "Wikipedia": wikipedia_search
}

# The System Prompt that forces the ReAct format
SYSTEM_PROMPT = """
You are a smart assistant. You work in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer.

Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:
calculate:
e.g. Action: calculate: 4 * 7 / 3
Returns the math result.

wikipedia:
e.g. Action: wikipedia: Django
Returns a summary from Wikipedia.

Example session:
Question: What is the capital of France?
Thought: I should look up France on Wikipedia
Action: wikipedia: France
PAUSE

You will be called again with this:
Observation: France is a country. The capital is Paris.
Thought: I know the answer now.
Answer: The capital of France is Paris.
"""

def agent_loop(user_query, max_iterations=5):
    prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {user_query}"
    
    for i in range(max_iterations):
        result = call_llm(prompt)
        print(result)
        
        # If the LLM output the final answer, break the loop
        if "Answer:" in result:
            return result.split("Answer:")[1].strip()
            
        # Parse the Action
        action_match = re.search(r"Action: ([a-z_]+): (.+)", result)
        if action_match:
            tool_name = action_match.group(1)
            tool_input = action_match.group(2)
            
            # Execute the tool
            if tool_name in tools:
                observation = tools[tool_name](tool_input)
                print(f"Observation: {observation}")
                
                # Append the observation to the prompt for the next loop
                prompt += f"\n{result}\nObservation: {observation}\n"
            else:
                prompt += f"\n{result}\nObservation: Tool {tool_name} not found.\n"
                
    return "Failed to complete task within max iterations."
```

## 5. Challenges in ReAct
1. **Context Window Exhaustion:** Every iteration adds to the prompt. If an agent loops 10 times, the prompt becomes massive, costing more money and confusing the LLM.
2. **Infinite Loops:** An agent might get stuck trying the exact same failed tool 5 times in a row.
3. **Tool Hallucination:** The LLM might invent a tool that doesn't exist (e.g., `Action: hack_mainframe: target`).

*In the next lesson, we will learn how to solve these issues using Memory Management and Context Compression.*