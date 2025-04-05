
from typing import Dict, List, Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
import operator

class AgentState(TypedDict):
    """The state of the agent in the graph."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

def create_agent(llm: ChatOpenAI, tools: List[BaseTool], system_prompt: str) -> AgentExecutor:
    """Create an agent executor with the given LLM and tools."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_openai_functions_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)

def create_graph(agents: Dict[str, AgentExecutor], tools: List[BaseTool]) -> StateGraph:
    """Create a graph with the given agents and tools."""
    
    # Create a graph
    workflow = StateGraph(AgentState)
    
    # Create tool executor
    tool_executor = ToolExecutor(tools)
    
    # Define the nodes
    for agent_name, agent in agents.items():
        def agent_node(state: AgentState, agent=agent):
            result = agent.invoke(state)
            return {"messages": [HumanMessage(content=result["output"])], "next": ""}
        
        workflow.add_node(agent_name, agent_node)
    
    # Add tool executor node
    def tool_node(state: AgentState):
        # Get the last message
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return {"messages": [], "next": ""}
        
        # Parse the function call
        function_call = last_message.additional_kwargs.get("function_call")
        if not function_call:
            return {"messages": [], "next": ""}
        
        # Execute the function
        result = tool_executor.invoke(function_call)
        return {"messages": [HumanMessage(content=str(result))], "next": ""}
    
    workflow.add_node("tool_executor", tool_node)
    
    # Define the edges
    for agent_name in agents:
        # Connect agent to tool executor
        workflow.add_edge(agent_name, "tool_executor")
        # Connect tool executor back to agent
        workflow.add_edge("tool_executor", agent_name)
        
        # Add conditional edges
        def should_continue(state: AgentState) -> bool:
            return state["next"] != ""
        
        def should_end(state: AgentState) -> bool:
            return state["next"] == ""
        
        workflow.add_conditional_edges(
            agent_name,
            # If there's a next agent, continue
            {
                agent_name: should_continue,
                END: should_end
            }
        )
    
    # Set the entry point
    workflow.set_entry_point("primary")
    
    return workflow

def run_graph(graph: StateGraph, query: str) -> List[BaseMessage]:
    """Run the graph with the given query."""
    # Initialize the state
    state = {"messages": [HumanMessage(content=query)], "next": ""}
    
    # Run the graph
    result = graph.invoke(state)
    
    return result["messages"]