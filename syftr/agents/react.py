import asyncio
from typing import Any

from llama_index.core.agent.react import ReActChatFormatter, ReActOutputParser
from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    ObservationReasoningStep,
)
from llama_index.core.llms import ChatMessage, ChatResponse
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import ToolOutput, ToolSelection
from llama_index.core.tools.types import BaseTool
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)


class PrepEvent(Event):
    pass


class InputEvent(Event):
    input: list[ChatMessage]


class StreamEvent(Event):
    delta: str


class ToolCallEvent(Event):
    tool_calls: list[ToolSelection]


class FunctionOutputEvent(Event):
    output: ToolOutput


class ReActAgent(Workflow):
    """Implement of RAG using a ReAcT Agent"""

    def __init__(
        self,
        *args: Any,
        llm: LLM,
        tools: list[BaseTool] | None = None,
        extra_context: str | None = None,
        max_iterations: int = 10,
        default_tool_choice="required",
        react_chat_formatter: ReActChatFormatter | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tools = tools or []
        self.llm = llm
        self.formatter = react_chat_formatter or ReActChatFormatter.from_defaults(
            context=extra_context or ""
        )
        self.output_parser = ReActOutputParser()
        self.default_tool_choice = default_tool_choice
        self.max_iterations = max_iterations

    @step
    async def new_user_msg(self, ctx: Context, ev: StartEvent) -> PrepEvent:
        # clear sources
        await ctx.store.set("sources", [])

        # init memory if needed
        memory = await ctx.store.get("memory", default=None)
        if not memory:
            memory = ChatMemoryBuffer.from_defaults(llm=self.llm)

        # get user input
        user_input = ev.input
        user_msg = ChatMessage(role="user", content=user_input)
        memory.put(user_msg)

        # clear current reasoning
        await ctx.store.set("current_reasoning", [])

        # set memory
        await ctx.store.set("memory", memory)
        await ctx.store.set("iteration_counter", 0)

        return PrepEvent()

    @step
    async def prepare_chat_history(self, ctx: Context, ev: PrepEvent) -> InputEvent:
        # get chat history
        memory = await ctx.store.get("memory")
        chat_history = memory.get()
        current_reasoning = await ctx.store.get("current_reasoning", default=[])

        # format the prompt with react instructions
        llm_input = self.formatter.format(
            self.tools, chat_history, current_reasoning=current_reasoning
        )
        return InputEvent(input=llm_input)

    @step
    async def handle_llm_input(
        self, ctx: Context, ev: InputEvent
    ) -> ToolCallEvent | StopEvent:
        chat_history = ev.input
        current_reasoning = await ctx.store.get("current_reasoning", default=[])
        memory = await ctx.store.get("memory")
        iteration = await ctx.store.get("iteration_counter")
        print(iteration)
        if iteration == self.max_iterations:
            sources = await ctx.store.get("sources", default=[])
            memory = await ctx.store.get("memory")
            response = memory.get()[-1].content
            return StopEvent(
                result={
                    "response": response,
                    "sources": [sources],
                    "reasoning": current_reasoning,
                }
            )

        response = self.llm.chat(
            chat_history,
            tools=[x.metadata.to_openai_tool() for x in self.tools],
            # use default for the first iteration, then default to auto
            # that allows it to stop
            tool_choice=self.default_tool_choice if iteration == 0 else "auto",
        )
        try:
            tool_calls = response.raw.choices[0].message.tool_calls
        except AttributeError:  # Ocassionaly get Response which has different schema
            tool_calls = []
        if tool_calls:  # Convert tool calls into proposed text action
            tool = tool_calls[0].function
            message_content = (
                "Thought: I need to use a tool to help me answer the question.\n"
                + f"Action: {tool.name}\nAction Input: {tool.arguments}"
            )
        else:
            message_content = response.message.content or ""
        ctx.write_event_to_stream(StreamEvent(delta=message_content))
        try:
            reasoning_step = self.output_parser.parse(message_content)
            current_reasoning.append(reasoning_step)
            if reasoning_step.is_done:
                memory.put(
                    ChatMessage(role="assistant", content=reasoning_step.response)
                )
                await ctx.store.set("memory", memory)
                await ctx.store.set("current_reasoning", current_reasoning)

                sources = await ctx.store.get("sources", default=[])

                return StopEvent(
                    result={
                        "response": reasoning_step.response,
                        "sources": [sources],
                        "reasoning": current_reasoning,
                    }
                )
            elif isinstance(reasoning_step, ActionReasoningStep):
                tool_name = reasoning_step.action
                tool_args = reasoning_step.action_input
                return ToolCallEvent(
                    tool_calls=[
                        ToolSelection(
                            tool_id="fake",
                            tool_name=tool_name,
                            tool_kwargs=tool_args,
                        )
                    ]
                )
        except Exception as e:
            current_reasoning.append(
                ObservationReasoningStep(
                    observation=f"There was an error in parsing my reasoning: {e}"
                )
            )
            await ctx.store.set("current_reasoning", current_reasoning)

        iteration += 1
        await ctx.store.set("iteration_counter", iteration)
        # if no tool calls or final response, iterate again
        return PrepEvent()

    @step
    async def handle_tool_calls(self, ctx: Context, ev: ToolCallEvent) -> PrepEvent:
        tool_calls = ev.tool_calls
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}
        current_reasoning = await ctx.store.get("current_reasoning", default=[])
        sources = await ctx.store.get("sources", default=[])

        # call tools -- safely!
        for tool_call in tool_calls:
            tool = tools_by_name.get(tool_call.tool_name)
            if not tool:
                current_reasoning.append(
                    ObservationReasoningStep(
                        observation=f"Tool {tool_call.tool_name} does not exist"
                    )
                )
                continue

            try:
                tool_output = tool(**tool_call.tool_kwargs)
                sources.append(tool_output)
                current_reasoning.append(
                    ObservationReasoningStep(observation=tool_output.content)
                )
            except Exception as e:
                current_reasoning.append(
                    ObservationReasoningStep(
                        observation=f"Error calling tool {tool.metadata.get_name()}: {e}"
                    )
                )

        # save new state in context
        await ctx.store.set("sources", sources)
        await ctx.store.set("current_reasoning", current_reasoning)

        # iterate the counter
        iteration = await ctx.store.get("iteration_counter")
        iteration += 1
        await ctx.store.set("iteration_counter", iteration)
        # if no tool calls or final response, iterate again

        # prep the next iteraiton
        return PrepEvent()

    def chat(self, query, invocation_id=None) -> ChatResponse:
        # Backwards compatibility which used chat
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError()
        else:

            async def wrap_run():
                result = await self.run(input=query)
                result = result["response"]
                print(result)
                result = ChatResponse(
                    message=ChatMessage.from_str(result), additional_kwargs={}
                )

                return result

            return loop.run_until_complete(wrap_run())


class IntrospectiveAgentWorker(Workflow):
    pass


class ToolInteractiveReflectionAgentWorker(Workflow):
    pass


class LATSAgentWorker(Workflow):
    pass


class CoAAgentPack(Workflow):
    pass
