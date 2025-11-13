from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    cartesia,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

import asyncio
import os
import httpx

# --------------------------------------------------------
# Load environment variables
# --------------------------------------------------------
load_dotenv()


# ========================================================
# Custom LLM class for OpenRouter (free-tier models)
# ========================================================
class OpenRouterLLM:
    def __init__(self, model_name="meta-llama/llama-3.1-8b-instruct"):
        self.model_name = model_name
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"

    async def complete(self, messages):
        """Send the user messages to the LLM asynchronously."""
        user_message = messages[-1]["content"]
        print("\n🧠 OpenRouter Input:", user_message)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": user_message}],
        }

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(self.endpoint, json=body, headers=headers)
            response.raise_for_status()
            data = response.json()
            reply = data["choices"][0]["message"]["content"].strip()

        print("💬 OpenRouter Output:", reply)
        return reply


# ========================================================
# The Voice Assistant logic
# ========================================================
class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful and friendly voice AI assistant.")


# ========================================================
# Main entrypoint
# ========================================================
async def entrypoint(ctx: agents.JobContext):
    # Initialize OpenRouter LLM (free API key required)
    llm = OpenRouterLLM(model_name="meta-llama/llama-3.1-8b-instruct")

    # Initialize the session
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=llm,
        tts=cartesia.TTS(
            model="sonic-2",
            voice="f786b574-daa5-4673-aa0c-cbe3e8534c02",
        ),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

    # Greeting message
    await session.generate_reply(
        instructions="Greet the user and introduce yourself as a voice assistant."
    )


# ========================================================
# Run the agent
# ========================================================
if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
