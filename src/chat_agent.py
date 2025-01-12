import asyncio
import os
import logging
# from dotenv import load_dotenv
from livekit.plugins import deepgram, elevenlabs, openai, silero

from src.system_prompt import get_system_prompt
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.agents.llm import ChatContext, ChatMessage
from src.tool_calls import Actions

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# These are needed for the *chat* agent to be created, not required for the *chat* agent to work
os.environ.setdefault("DEEPGRAM_API_KEY", "")
os.environ.setdefault("ELEVENLABS_API_KEY", "")

logger = logging.getLogger("chat-agent")

class ChatAgent:
    def __init__(self):
        self.agent : VoicePipelineAgent | None = None


    def initialize_agent(self):
        
        system_prompt = get_system_prompt()
        logger.info(f"System prompt: {system_prompt}")
        
        initial_ctx = ChatContext()
        initial_ctx.append(role="system", text=system_prompt)

        call_actions = Actions()

        self.agent = VoicePipelineAgent(
            vad=silero.VAD.load(),
            stt=deepgram.STT(model="nova-2-phonecall"),
            llm=openai.LLM(model="gpt-4o-mini"),  # Using latest model with higher rate limits
            tts=elevenlabs.TTS(api_key=os.environ.get("ELEVENLABS_API_KEY")),
            chat_ctx=initial_ctx,
            fnc_ctx=call_actions
        )

    async def chat_response(self, message: str) -> str:
        if self.agent is None:
            raise ValueError("Agent is not initialized")

        self.agent.chat_ctx.messages.append(ChatMessage.create(text=message, role="user"))  
        while True:
            # Get the stream and process chunks
            stream = self.agent.llm.chat(chat_ctx=self.agent.chat_ctx, fnc_ctx=self.agent.fnc_ctx)

            # Process response stream
            response_buffer = []
            tool_calls = []
            async for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                
                if choice.delta.content:
                    response_buffer.append(choice.delta.content)
                if choice.delta.tool_calls:
                    has_function_calls = True
                    for fnc in choice.delta.tool_calls:
                        tool_calls.append(fnc)

            complete_response = "".join(response_buffer)
            if complete_response:
                self.agent.chat_ctx.messages.append(ChatMessage.create(text=complete_response, role="assistant"))
                return complete_response

            called_fncs = []
            for fnc in tool_calls:
                logger.debug(f"Function call: {fnc}")

                called_fnc = fnc.execute()
                called_fncs.append(called_fnc)
                logger.debug(f"Executing ai function: {fnc.function_info.name}")
                try:
                    # Add 10 second timeout for function execution
                    await asyncio.wait_for(called_fnc.task, timeout=2.0)
                except asyncio.TimeoutError:
                    logger.error(f"Function {fnc.function_info.name} timed out after 2 seconds")
                except Exception as e:
                    logger.error(f"Error executing ai function: {fnc.function_info.name} - {str(e)}")

            
            tool_calls_info = []
            tool_calls_results = []

            for called_fnc in called_fncs:
                # ignore the function calls that returns None
                if called_fnc.result is None and called_fnc.exception is None:
                    continue

                logger.debug(f"Called result function: {called_fnc}")
                tool_calls_info.append(called_fnc.call_info)
                tool_calls_results.append(
                    ChatMessage.create_tool_from_called_function(called_fnc)
                )

            if tool_calls_info:
                # create a nested speech handle
                extra_tools_messages = [
                    ChatMessage.create_tool_calls(tool_calls_info)
                ]
                extra_tools_messages.extend(tool_calls_results)

                # synthesize the tool speech with the chat ctx from llm_stream
                self.agent.chat_ctx.messages.extend(extra_tools_messages)
                # agent.chat_ctx.messages.extend(call_ctx.extra_chat_messages)
                logger.debug(self.agent.chat_ctx.messages)

                continue
            else:
                break

        logger.error("Agent unable to respond")
        return ""
    
    def reset(self):
        if self.agent is None:
            return
        self.agent.chat_ctx.messages = []
        self.agent = None

    def get_agent(self) -> VoicePipelineAgent:
        if self.agent is None:
            raise ValueError("Agent is not initialized")
        return self.agent
    
    def get_chat_ctx(self) -> ChatContext:
        if self.agent is None:
            raise ValueError("Agent is not initialized")
        return self.agent.chat_ctx
    
    def get_chat_messages(self) -> list[ChatMessage]:
        if self.agent is None:
            raise ValueError("Agent is not initialized")
        return self.agent.chat_ctx.messages
