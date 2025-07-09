"""GraphAgent

The GraphAgent is a hierarchical planning agent that works as follows.

* A Goal is created in an initialized state.

* The Goal enters the ACTIVE state and begins the execution loop:
    * An agent is prompted with a view of the graph execution state and a set of tool-call options:
        * Create sub-goal: Create an agent to execute the next sub-goal.
        * LLM call: Prompt an LLM with a given prompt.
        * User-provided tools: Invoke one of the user's provided tools.
        * Finalize the goal: Set the goal's state to a terminal state.
    * The agent is prompted with the same view of the graph execution state including the most recent
      tool-call outcome and is asked to update the Goal's current status description

* The agent's view of the graph execution state includes
    * The current Goal's representation.
    * The history of all tool calls made for this goal so far.
    * The direct ancestry tree of goals up to the root Goal


The GraphAgent class manages all of this complex state in a graph data structure (DAG).

Each Node in the graph represents a Goal. Each Goal stores its tool call history and current status.
Edges can represent (parent -> child) relationships or (previous -> next) relationships.

The GraphAgent manages the execution loop, building the graph structure and constructing prompts.

The agent should create sub-Goals until the current goal can be executed with a single function call.
"""

import json
from enum import Enum
from textwrap import dedent
from typing import List, Literal
from uuid import uuid4

import llama_index.core.instrumentation as instrument
import networkx as nx
from llama_index.core.llms import LLM, ChatMessage, MessageRole
from llama_index.core.output_parsers.pydantic import PydanticOutputParser
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.tools import BaseTool, FunctionTool
from pydantic import BaseModel, Field

# Rich imports for formatted output
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text

dispatcher = instrument.get_dispatcher()


def new_uuid():
    """Generates a new unique identifier."""
    return uuid4().hex


class GoalState(Enum):
    """Enumeration for the possible states of a Goal."""

    INITIALIZED: str = "INITIALIZED"
    ACTIVE: str = "ACTIVE"
    PENDING_SUBGOAL: str = "PENDING_SUBGOAL"
    COMPLETED: str = "COMPLETED"
    FAILED: str = "FAILED"


class Goal(BaseModel):
    """A specific, achievable objective managed by an agent.

    Goals are the planning unit for the hierarchical graph planning agent.

    Each Goal has an agent responsible for completing it. The agent can invoke Tools and create
    sub-Goals in order to achieve the goal, until the agent moves the Goal into a terminal state.
    """

    name: str = Field(description="Short name / title for this goal")
    description: str = Field(
        description="Short description. Can include rationale, context, and additional details."
    )
    acceptance_criteria: str = Field(
        description="What is required for this goal to be met?"
    )
    status: str = Field(
        description="Detailed description of the state of this goal's execution?",
        default="This goal has been initialized.",
    )
    output: str = Field(
        description="The final output for this goal, created when the goal is finalized.",
        default="The goal is not finalized yet - no output.",
    )
    id: str = Field(default_factory=new_uuid)
    state: GoalState = Field(default=GoalState.INITIALIZED)
    chat_history: List[ChatMessage] = Field(default_factory=list)

    def __str__(self):
        return (
            f"Goal: {self.name}\n"
            f"Description: {self.description}\n"
            f"Acceptance Criteria: {self.acceptance_criteria}\n"
            f"Status: {self.status}\n"
            f"State: {self.state.value}"
            f"Output: {self.output}"
        )


def complete_goal(
    state: Literal[GoalState.COMPLETED.value, GoalState.FAILED.value],
) -> GoalState:
    """A tool to mark a goal as completed or failed."""
    return GoalState(state)


complete_goal_tool = FunctionTool.from_defaults(complete_goal)


goal_output_parser = PydanticOutputParser(
    Goal, excluded_schema_keys_from_format=["id", "_tool_calls", "state"]
)


STATUS_UPDATE_PROMPT = RichPromptTemplate(
    dedent(
        """
        The current goal status is:
        -----
        {{status_str}}
        -----

        Given the latest messages and tool call results, write an updated status.
        * Return only the status field, not the state, name, description, or details.
        * Compress the existing status field information while adding the latest information.
        * Evaluate progress towards the goal and the relevance of recent tool call outputs.
        * Mention what remains to be done, if anything, for accomplishing the goal.
        """
    )
)

EXECUTE_GOAL_PROMPT = RichPromptTemplate(
    dedent(
        """
        You are an agent responsible for achieving a Goal.

        # Execution State
        {{execution_state}}

        ## Instructions
        Your task is to take the next step to achieve the current goal.
        Review the current goal, the history of previous steps, and the overall plan (ancestor goals).
        You have the following tool call options:
        1.  **Call an LLM**: If the goal can be achieved by a direct question to an LLM, use the `call_llm` tool. The LLM will have the
            same context as you for completing its task.
        2.  **Use a user-provided tool**: If a user tool can directly accomplish the goal, invoke it. Consider this option strongly.
            Be thoughtful about using these tools. For example, perhaps if one tool is not working as expected, another can be used.
            For example a code interpreter can perform many tasks.
        3.  **Create a sub-goal**: If the current goal cannot be achieved through a single tool call, select this option.
            The sub-goal should be the next step in the solution process. You may end up recursively generating sub-goals to get more
            and more specific until you have the first actionable step.
        4.  **Finalize the goal**: If the goal has been achieved, call `finalize_current_goal` with the appropriate state ('COMPLETED' or 'FAILED').
            Only finalize the goal if it is completely and entirely finished - if there are ANY remaining finalization steps to take,
            create a sub-goal for them or invoke a tool to complete them.
            Mark the goal as FAILED if there are no possible steps which can be taken with the tools available to you to achieve the current
            goal.

        Choose the single best action to take now.
        """
    )
)


class GraphAgent:
    def __init__(
        self,
        llm: LLM,
        tools: List[BaseTool],
        goal: Goal,
        max_depth: int = 5,
        max_breadth: int = 5,
        quiet: bool = False,
        interactive: bool = False,
    ):
        self.llm = llm
        self.graph = nx.DiGraph()
        self.graph.add_node(goal.id, data=goal)
        self.current_goal = goal
        self._user_tools = tools
        self.max_depth = max_depth
        self.max_breadth = max_breadth
        self.quiet = quiet
        self.interactive = interactive
        if not self.quiet:
            self.console = Console(width=120)

    def _print_history_delta(self, goal: Goal, start_index: int):
        """Prints new additions to the chat history using rich."""
        if self.quiet:
            return

        role_colors = {
            "SYSTEM": "blue",
            "USER": "green",
            "ASSISTANT": "magenta",
            "TOOL": "yellow",
        }

        for i in range(start_index, len(goal.chat_history)):
            message = goal.chat_history[i]
            role_str = str(getattr(message.role, "value", message.role)).upper()

            if role_str == "SYSTEM":
                continue

            color = role_colors.get(role_str, "white")
            title = f"[bold {color}]{role_str}[/bold {color}]"

            if role_str == "ASSISTANT":
                # FIXED: Print assistant text and tool calls as separate panels.
                # First, print the text content of the assistant's message, if any.
                if message.content:
                    self.console.print(
                        Panel(
                            Text(message.content),
                            title=title,
                            border_style=color,
                            expand=False,
                        )
                    )

                # Then, iterate through tool calls and print each in its own panel.
                if tool_calls := message.additional_kwargs.get("tool_calls"):
                    for tool_call in tool_calls:
                        tool_data = {}
                        if hasattr(tool_call, "function"):
                            tool_data = {
                                "id": tool_call.id,
                                "name": tool_call.function.name,
                                "input": json.loads(tool_call.function.arguments),
                            }
                        elif hasattr(tool_call, "tool_name"):
                            tool_data = {
                                "id": tool_call.id,
                                "name": tool_call.tool_name,
                                "input": tool_call.tool_kwargs,
                            }
                        else:
                            continue

                        json_str = json.dumps(tool_data, indent=2)
                        syntax = Syntax(
                            json_str,
                            "json",
                            theme="monokai",
                            line_numbers=True,
                            word_wrap=True,
                        )
                        # Print each tool call panel directly to the console.
                        self.console.print(
                            Panel(
                                syntax,
                                title=f"[bold yellow]TOOL CALL: {tool_data['name']}[/bold yellow]",
                                border_style="yellow",
                                expand=False,
                            )
                        )

            elif role_str == "TOOL":
                tool_name = message.additional_kwargs.get("tool_name", "unknown_tool")
                panel_title = f"{title} | Output from [b]{tool_name}[/b]"
                self.console.print(
                    Panel(
                        Text(str(message.content), overflow="fold"),
                        title=panel_title,
                        border_style=color,
                        expand=False,
                    )
                )

    def ask_the_user(self, query) -> str:
        """Ask the user a question

        This can be used for getting further clarification from the user, or advice on how to proceed with the current goal.
        """
        response = input(f"GraphAgent asks: {query}\n > ")
        return response

    def call_llm(self, query) -> str:
        """Call an LLM

        If the goal can be achieved by a direct question to an LLM, use the `call_llm` tool.
            Be sure to provide all required context in the LLM prompt, as this session context will not be included automatically.
            Include full source document texts, full tool outputs, citation URLs, etc in the prompt. If you do not include required context,
            the LLM may hallucinate context instead, leading to invalid results.

        The LLM is merely a text-in, text-out interface without access to the Internet or any tools whatsoever. It only has access
        to the context you provide.
        """
        chat_history = self.current_goal.chat_history[:-1] + [
            ChatMessage(role=MessageRole.USER, content=query)
        ]
        response = self.llm.chat(chat_history)
        if response.message.content is None:
            breakpoint()
        return response.message.content

    def finalize_current_goal(
        self,
        state: Literal[GoalState.COMPLETED.value, GoalState.FAILED.value],
        output: str,
    ) -> str:
        """Finalize the current goal.

        If the goal has been achieved, call `finalize_current_goal` with the appropriate state ('COMPLETED' or 'FAILED') and output.
        Only finalize the goal if it is completely and entirely finished - if there are ANY remaining finalization steps to take,
        create a sub-goal for them or invoke a tool to complete them.

        The output should contain the entirety of the goal's output. For example, if the goal is to write a research report, the
        output should contain the entire text of the research report. If the goal failed with a traceback, the entire traceback
        should be set in the output.
        """
        state = GoalState(state)
        self.current_goal.state = state
        self.current_goal.output = output
        return self.current_goal.output

    def create_and_execute_sub_goal(
        self, description: str, acceptance_criteria: str, name: str
    ) -> str:
        """Create the next sub-goal for the current goal and execute it.

            description: str = Field(
                description="Short description. Can include rationale, context, and additional details."
            )
            acceptance_criteria: str = Field(
                description="What is required for this goal to be met?"
            )
            name: str = Field(description="Short name / title for this goal")

        Note that the sub-goal execution process will not have access to any tools which are not available to you already.
        If the sub-goal cannot be executed using the tools available to you, an alternative approach will be required.
        """
        parent_goal = self.current_goal
        path = nx.shortest_path(self.graph, self.root_goal.id, parent_goal.id)
        if len(path) >= self.max_depth:
            raise RuntimeError(
                "Max subgoal depth reached. Use another tool or fail the current goal."
            )
        parent_goal.state = GoalState.PENDING_SUBGOAL
        goal = Goal(
            name=name,
            description=description,
            acceptance_criteria=acceptance_criteria,
            state=GoalState.ACTIVE,
        )
        self.graph.add_node(goal.id, data=goal)
        self.graph.add_edge(parent_goal.id, goal.id)
        self.current_goal = goal
        try:
            self.execute_goal(self.current_goal)
        finally:
            self.current_goal = parent_goal
            self.current_goal.state = GoalState.ACTIVE

        return str(goal)

    @property
    def tools_by_name(self) -> dict[str, BaseTool]:
        tools_by_name = {t.metadata.name: t for t in (self._user_tools or [])}
        tools_by_name["call_llm"] = FunctionTool.from_defaults(self.call_llm)
        tools_by_name["finalize_current_goal"] = FunctionTool.from_defaults(
            self.finalize_current_goal
        )
        tools_by_name["create_and_execute_sub_goal"] = FunctionTool.from_defaults(
            self.create_and_execute_sub_goal
        )
        if self.interactive:
            tools_by_name["ask_the_user"] = FunctionTool.from_defaults(
                self.ask_the_user
            )
        return tools_by_name

    @property
    def root_goal(self) -> Goal:
        goal_id: str = next(
            node for node in self.graph if self.graph.in_degree(node) == 0
        )
        goal: Goal = self.graph.nodes[goal_id]["data"]
        return goal

    def _get_graph_execution_state_view(self, goal: Goal) -> str:
        """Constructs the string representation of the agent's view of the graph state."""
        path = nx.shortest_path(self.graph, self.root_goal.id, goal.id)
        ancestry_view = "## Goal Hierarchy (Path from Root to Current)\n"
        for i, node_id in enumerate(path):
            indent = "  " * i
            node_goal = self.graph.nodes[node_id]["data"]
            ancestry_view += f"{indent}- Goal: {node_goal.name}\n"
            ancestry_view += f"{indent}  Description: {node_goal.description}\n"
            ancestry_view += f"{indent}  Status: {node_goal.status}\n"
            if node_id == goal.id:
                ancestry_view += f"{indent}  (This is the CURRENT GOAL)\n"

        # Find and display completed sibling goals
        completed_siblings_view = "## Recently Completed Sibling Goals\n"
        parent_id = next(self.graph.predecessors(goal.id), None)
        if parent_id:
            sibling_ids = [
                sid for sid in self.graph.successors(parent_id) if sid != goal.id
            ]
            if sibling_ids:
                for i, sibling_id in enumerate(sibling_ids):
                    sibling_goal = self.graph.nodes[sibling_id]["data"]
                    completed_siblings_view += f"- Goal: {sibling_goal.name}\n"
                    completed_siblings_view += (
                        f"  Description: {sibling_goal.description}\n"
                    )
                    completed_siblings_view += f"  Output: {sibling_goal.output}\n\n"
            else:
                completed_siblings_view += "No sibling goals have been completed yet.\n"
        else:
            completed_siblings_view += (
                "This is the root goal, so there are no sibling goals.\n"
            )

        completed_children_view = (
            "## Recently Completed Sub-Goals (for the Current Goal)\n"
        )
        child_ids = list(self.graph.successors(goal.id))
        completed_child_ids = [
            cid
            for cid in child_ids
            if self.graph.nodes[cid]["data"].state == GoalState.COMPLETED
        ]

        if completed_child_ids:
            for child_id in completed_child_ids:
                child_goal = self.graph.nodes[child_id]["data"]
                completed_children_view += f"- Goal: {child_goal.name}\n"
                completed_children_view += f"  Output: {child_goal.output}\n\n"
        else:
            completed_children_view += (
                "No sub-goals have been completed for the current goal yet.\n"
            )

        current_goal_view = f"## Current Goal Details\n{str(goal)}"

        # Combine all the views into a single string
        return f"{ancestry_view}\n{completed_siblings_view}\n{completed_children_view}\n{current_goal_view}"

    @dispatcher.span
    def run_until_complete(self) -> Goal:
        """Executes the entire goal hierarchy until the root goal is complete."""
        self.current_goal = self.root_goal
        final_root_goal = self.execute_goal(self.current_goal)
        return final_root_goal

    @dispatcher.span
    def execute_goal(self, goal: Goal) -> Goal:
        """Execute current goal until it reaches a terminal state."""
        while goal.state not in (GoalState.COMPLETED, GoalState.FAILED):
            try:
                self._execute_step(goal)
            except Exception as exc:
                goal.state = GoalState.FAILED
                goal.status = (goal.status or "") + f"\n\nGoal FAILED: {exc}"
                if not self.quiet:
                    self.console.print_exception()
        return goal

    @dispatcher.span
    def _execute_step(self, goal: Goal):
        """Executes a single step of the goal-oriented reasoning loop."""
        if not self.quiet:
            self.console.print(
                Rule(
                    f"[bold] Executing Step for Goal: {goal.name} [/bold]", style="cyan"
                )
            )

        execution_state_view = self._get_graph_execution_state_view(goal)
        system_prompt = EXECUTE_GOAL_PROMPT.format(execution_state=execution_state_view)
        goal.chat_history.append(
            ChatMessage(role=MessageRole.USER, content=system_prompt)
        )

        history_len_before_chat = len(goal.chat_history)
        response = self.llm.chat_with_tools(
            list(self.tools_by_name.values()), chat_history=goal.chat_history
        )
        assistant_message = response.message
        goal.chat_history.append(assistant_message)
        self._print_history_delta(goal, history_len_before_chat)

        tool_calls = assistant_message.additional_kwargs.get("tool_calls", [])

        if not tool_calls:
            goal.status += "\nAgent did not select a tool. A tool must be selected. Repeating step."
            return

        for tool_call in tool_calls:
            tool_name, tool_kwargs, tool_call_id = None, None, None
            if hasattr(tool_call, "function"):
                tool_name = tool_call.function.name
                tool_kwargs = json.loads(tool_call.function.arguments)
                tool_call_id = tool_call.id
            elif hasattr(tool_call, "tool_name"):
                tool_name = tool_call.tool_name
                tool_kwargs = tool_call.tool_kwargs
                tool_call_id = tool_call.id
            elif isinstance(tool_call, dict):
                tool_name = tool_call["name"]
                tool_kwargs = tool_call["input"]
                tool_call_id = tool_call["id"]
            else:
                raise ValueError(
                    f"Don't know how to extract tool call info from call: {tool_call=}"
                )

            if tool_name not in self.tools_by_name:
                raise ValueError(f"Invalid tool_name specified: {tool_name}")

            history_len_before_tool_exec = len(goal.chat_history)
            try:
                tool_output = self.tools_by_name[tool_name](**tool_kwargs)
            except Exception as exc:
                goal.chat_history.append(
                    ChatMessage(
                        role="tool",
                        content=str(exc),
                        additional_kwargs={
                            "tool_call_id": tool_call_id,
                            "tool_name": tool_name,
                        },
                    )
                )
            else:
                goal.chat_history.append(
                    ChatMessage(
                        role="tool",
                        content=str(tool_output),
                        additional_kwargs={
                            "tool_call_id": tool_call_id,
                            "tool_name": tool_name,
                        },
                    )
                )
            self._print_history_delta(goal, history_len_before_tool_exec)

        status_update_history = goal.chat_history + [
            ChatMessage(
                role="system",
                content=STATUS_UPDATE_PROMPT.format(status_str=goal.status),
            )
        ]
        response = self.llm.chat(status_update_history)
        if response.message.content is None:
            breakpoint()
        goal.status = response.message.content

        if not self.quiet:
            self.console.print(
                Panel(
                    Text(goal.status, justify="left"),
                    title="[bold blue]STATUS UPDATE[/bold blue]",
                    border_style="blue",
                    expand=False,
                )
            )
            self.console.print(Rule(style="cyan"))


if __name__ == "__main__":
    import argparse

    from llama_index.tools.arxiv import ArxivToolSpec
    from llama_index.tools.code_interpreter import CodeInterpreterToolSpec
    from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
    from llama_index.tools.wikipedia import WikipediaToolSpec
    from llama_index.tools.yahoo_finance import YahooFinanceToolSpec

    from syftr.instrumentation.arize import instrument_arize
    from syftr.llm import LLMs

    instrument_arize()

    gpt = LLMs["gpt-4o"]
    ddg = DuckDuckGoSearchToolSpec().to_tool_list()
    arxiv = ArxivToolSpec().to_tool_list()
    code = CodeInterpreterToolSpec().to_tool_list()[0]
    code.metadata.description += "Use print statements to view code execution results"
    code = [code]
    wiki = WikipediaToolSpec().to_tool_list()
    yahoo = YahooFinanceToolSpec().to_tool_list()

    parser = argparse.ArgumentParser()
    parser.add_argument("goal", type=str)
    args = parser.parse_args()

    goal = Goal(
        # name="Write a research report about the latest agent techniques for LLMs",
        # description="Use the arxiv search tool only. Synthesize a report based on recent papers",
        # acceptance_criteria="At least 6 recent reports on agentic LLM techniques have been collected. A research report has been written.",
        name=args.goal,
        description="",
        acceptance_criteria="",
    )

    agent = GraphAgent(gpt, wiki + arxiv + code + yahoo, goal, interactive=True)

    agent.run_until_complete()

    print(goal.output)
