from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from pydantic import BaseModel
from prompts import *
from states import *
from tools import *
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent



# from langchain_core.globals import set_debug
# import logging
# set_debug(True)
# logging.basicConfig(level=logging.DEBUG)



# llm = ChatGroq(model="openai/gpt-oss-120b")

# llm  = ChatOpenAI(model='gpt-4')

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')




def planner_agent(state: dict) -> dict:
    user_prompt = state["user_prompt"]
    resp = llm.with_structured_output(Plan).invoke(planner_prompt(user_prompt))
    if resp is None:
        return ValueError("Architect agent returned None")
    return {"plan":resp}


def architect_agent(state: dict) -> dict:
    plan : Plan = state["plan"]
    resp = llm.with_structured_output(TaskPlan).invoke(architect_prompt(plan))
    if resp is None:
        return ValueError("Architect agent returned None")
    
    resp.plan = plan

    return {"task_plan":resp}

def coader_agent(state: dict) -> dict:

    coder_state = state.get("coder_state")
    if coder_state is None:
        coder_state = CoderState(task_plan=state["task_plan"], current_step_idx=0)

    

    steps = coder_state.task_plan.implementation_steps
    
    if coder_state.current_step_idx >= len(steps):
        return {"coder_state": coder_state, "status": "Done"}
    
    current_task = steps[coder_state.current_step_idx]

    existing_content = read_file.run(current_task.filepath)


    user_prompt = (
        f"Task : {current_task.task_description}\n"
        f"File: {current_task.filepath}\n"
        f"Existing content \n{existing_content}\n"
        "Use write_file(path, content) to save your changes."
    )
    system_prompt = coder_system_prompt()


    coder_tools = [write_file, read_file, list_files, get_current_directory]
    react_agent = create_react_agent(llm, coder_tools)

    react_agent.invoke(
        {"messages": [
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_prompt}
        ]}
    )
    coder_state.current_step_idx += 1
    return {"coder_state": coder_state}
    


graph = StateGraph(dict)
graph.add_node("planner",planner_agent)
graph.add_node("architect",architect_agent)
graph.add_node("coder", coader_agent)

graph.add_edge("planner","architect")
graph.add_edge("architect","coder")
graph.add_conditional_edges(
    "coder",
    lambda s: "END" if s.get("status") == "Done" else "coder",
    {"END": END, "coder":"coder"}
)
graph.set_entry_point("planner")


agent = graph.compile()

user_prompt = "create TODO list web application for me"


result = agent.invoke({"user_prompt":user_prompt, "recursion_limit":100})  
print(result)