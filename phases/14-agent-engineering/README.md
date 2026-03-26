# Phase 14: Agent Engineering
Building loops that allow LLMs to reason, plan, and act autonomously.

## Roadmap
| Lesson | Description | Status |
|--------|-------------|--------|
| 01. The ReAct Loop | Reason, Act, Observe. | ✅ |
| 02. Memory Management | Short-term vs Long-term memory. | ⬚ |
| 03. Planning | Goal decomposition. | ⬚ |

## Code Example: ReAct Loop Concept
```python
def react_agent(user_query, max_steps=5):
    prompt = f"Goal: {user_query}. You can use tools: [Search, Calculator]. Output format: Thought, Action, Action Input."
    
    for step in range(max_steps):
        # 1. Reason & Act
        response = llm(prompt)
        print(response)
        
        # 2. Parse Action
        if "Action: Search" in response:
            observation = search_tool("...")
        elif "Action: None" in response:
            return "Task Complete!"
            
        # 3. Observe
        prompt += f"\nObservation: {observation}"
```
