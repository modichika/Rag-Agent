from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from retrieve import get_rag_agent

app = FastAPI()

class ChatRequest(BaseModel):
    query: str


@app.get("/query/")
async def query_rag_system(query):
    try:
        







 # -1 for the last one the newest thought, Why we use it: As the agent thinks, the messages list grows like this:
# [User Question]
# [User Question, AI Thought/Tool Call]
# [User Question, AI Thought, Tool Result]
# [User Question, AI Thought, Tool Result, Final AI Answer]


# The first AI message is the tool call that we need the AI to perform, it calls the particular tool for a particular task associated with it. First this happens without the tool call AI/LLM won't know what task to perform for the clarity.


# THE OUTPUT:
# ================================== Ai Message ==================================
# Tool Calls:
#   retrieve_context (call_CR2eWWMJnlJCnGqcHMd9PBcR)
#  Call ID: call_CR2eWWMJnlJCnGqcHMd9PBcR
#   Args:
#     query: standard method for Task Decomposition
# ================================= Tool Message =================================
# Name: retrieve_context

# Source: {'pk': '6a0e31e7-1fa5-4e1b-8d8b-f7730e38da63', 'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/', 'start_index': 2578}
# Content: Task decomposition can be done (1) by LLM with simple prompting like "Steps for XYZ.\n1.", "What are the subgoals for achieving XYZ?", (2) by using task-specific instructions; e.g. "Write a story outline." for writing a novel, or (3) with human inputs.
# Another quite distinct approach, LLM+P (Liu et al. 2023), involves relying on an external classical planner to do long-horizon planning. This approach utilizes the Planning Domain Definition Language (PDDL) as an intermediate interface to describe the planning problem. In this process, LLM (1) translates the problem into “Problem PDDL”, then (2) requests a classical planner to generate a PDDL plan based on an existing “Domain PDDL”, and finally (3) translates the PDDL plan back into natural language. Essentially, the planning step is outsourced to an external tool, assuming the availability of domain-specific PDDL and a suitable planner which is common in certain robotic setups but not in many other domains.
# Self-Reflection#

# Source: {'pk': '3b294245-2370-4741-aa98-a55ecb8b68e9', 'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/', 'start_index': 1638}
# Content: Component One: Planning#
# A complicated task usually involves many steps. An agent needs to know what they are and plan ahead.
# Task Decomposition#
# Chain of thought (CoT; Wei et al. 2022) has become a standard prompting technique for enhancing model performance on complex tasks. The model is instructed to “think step by step” to utilize more test-time computation to decompose hard tasks into smaller and simpler steps. CoT transforms big tasks into multiple manageable tasks and shed lights into an interpretation of the model’s thinking process.
# Tree of Thoughts (Yao et al. 2023) extends CoT by exploring multiple reasoning possibilities at each step. It first decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS (breadth-first search) or DFS (depth-first search) with each state evaluated by a classifier (via a prompt) or majority vote.
# ================================== Ai Message ==================================

# - Standard method for Task Decomposition:
#   - Chain of Thought (CoT) prompting. The model is guided to think step-by-step, breaking a complex task into smaller, more manageable steps and showing its reasoning process.

# - Common extensions of CoT:
#   - Tree of Thoughts (ToT): Extends CoT by exploring multiple reasoning paths per step. It builds a tree of intermediate thoughts and uses search (e.g., BFS or DFS) with evaluation to prune less promising branches, allowing the model to compare alternative plans.
#   - Self-Reflection: The model reflects on its own previous outputs and reasoning to revise or improve subsequent steps, aiming to correct errors or refine plans.
#   - LLM+P / external planning: Delegates long-horizon planning to an external classical planner using PDDL. The workflow: (1) translate the problem to PDDL, (2) have a planner generate a PDDL plan, (3) translate the plan back into natural language. This external tool-based approach offloads planning from the LLM.

# If you’d like, I can pull brief examples or references for each extension.


# From where does the AI when calls the tool gets the original docs the url from?
# The ai in the Tool Message is getting the blog post from the doc.metadata that included the uuid, and source url the original content through which we need the answers

# It's not any memory if the script has ended it will again start from scratch to answer and not use it's previous knowledges.