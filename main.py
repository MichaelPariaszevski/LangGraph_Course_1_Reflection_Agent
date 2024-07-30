from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

from typing import List, Sequence

from langchain_core.messages import BaseMessage, HumanMessage

from langgraph.graph import END, MessageGraph

from chains_and_prompts import generation_chain, reflection_chain

REFLECT = "reflect"  # All uppercase letters indicate that a variable is meant to stay constant (however, python does not enforce this or have a rule for this)
GENERATE = "generate"  # All uppercase letters indicate that a variable is meant to stay constant (however, python does not enforce this or have a rule for this)

# LLM: generate
# Human (really, the reflection node): critique
# LLM: generate revised response
# Human (really, the reflection node): critique
# LLM: generate revised response
# Human (really, the reflection node): critique
# LLM: generate revised response


def generation_node(
    state: Sequence[BaseMessage],
):  # state is a list of all of our messages
    return generation_chain.invoke({"messages": state})


def reflection_node(state: Sequence[BaseMessage]):
    response = reflection_chain.invoke({"messages": state})
    return [
        HumanMessage(content=response.content)
    ]  # Here, we are trying to trick the LLM into thinking that it is having a conversation with a human


builder = MessageGraph()

builder.add_node(node=GENERATE, action=generation_node)
builder.add_node(node=REFLECT, action=reflection_node)
builder.set_entry_point(
    key=GENERATE
)  # Where to start the LangGraph graph (starting node)


def should_continue(
    state: List[BaseMessage],
):  # if the length of state is greater than 6, end the LangGraph graph, else go to the REFLECT node
    if len(state) > 6:
        return END
    return REFLECT


builder.add_conditional_edges(source=GENERATE, path=should_continue)

builder.add_edge(start_key=REFLECT, end_key=GENERATE)

graph=builder.compile() 

print(graph.get_graph().draw_mermaid()) # To see the graph as an illustration, paste the output of this graph.get_graph().draw_mermaid() line into mermaid.live

if __name__ == "__main__":
    print("Hello LangGraph") 
    
    inputs=HumanMessage(content="""Make this tweet better: 
                        @LangChainAI 
                        -newly Tool Calling feature is seriously underrated. 
                        After a long wait, it's here- making the implementation of agents across different models with function calling- super easy. 
                        Made a video covering their newest blog post.""")
    
    response=graph.invoke(input=inputs)
    
    print("-"*100) 
    print(response)
    print("-"*100)
    print(response[-1].content)
