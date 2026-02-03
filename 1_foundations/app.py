from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr

# ─────────────────────────────────────────────
# Environment setup
# ─────────────────────────────────────────────

load_dotenv(override=True)

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

UNKNOWN_PATTERNS: list[str] = [
    "i'm not sure",
    "i am not sure",
    "i don't have information",
    "i do not have information",
    "i'm not aware",
    "could you clarify",
    "provide more details",
    "i don't have details",
]

# ─────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────

def push(text: str) -> None:
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        },
        timeout=10,
    )


def record_user_details(
    email: str,
    name: str = "Shawn",
    notes: str = "He is just the best guy out there",
) -> dict:
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}


def record_unknown_question(question: str) -> dict:
    push(f"Recording unknown question: {question}")
    return {"recorded": "ok"}


def should_push(reply: str) -> bool:
    reply = reply.lower()
    reply = reply.replace("’", "'").replace("‘", "'")
    return any(p in reply for p in UNKNOWN_PATTERNS)

# ─────────────────────────────────────────────
# Tool schemas
# ─────────────────────────────────────────────

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string"},
            "name": {"type": "string"},
            "notes": {"type": "string"},
        },
        "required": ["email"],
        "additionalProperties": False,
    },
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
        },
        "required": ["question"],
        "additionalProperties": False,
    },
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json},
]

# ─────────────────────────────────────────────
# Core class
# ─────────────────────────────────────────────

class Me:
    def __init__(self) -> None:
        self.name = "Shawn"

        reader = PdfReader("me/linkedin.pdf")
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text

        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()

    def system_prompt(self) -> str:
        return (
            f"You are acting as {self.name}. You are answering questions on {self.name}'s website, "
            f"particularly questions related to career, background, skills and experience. "
            f"Be professional and engaging. "
            f"If you don't know the answer, use record_unknown_question. "
            f"If appropriate, encourage the user to share their email and record it.\n\n"
            f"## Summary:\n{self.summary}\n\n"
            f"## LinkedIn Profile:\n{self.linkedin}\n"
        )

    def handle_tool_calls(self, tool_calls) -> list[dict]:
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            tool = globals().get(tool_name)

            if callable(tool):
                result = tool(**arguments)
            else:
                result = {}

            results.append(
                {
                    "role": "tool",
                    "content": json.dumps(result),
                    "tool_call_id": tool_call.id,
                }
            )
        return results

# ─────────────────────────────────────────────
# Chat handler
# ─────────────────────────────────────────────

def chat(message, history):
    me = chat.me

    messages = (
        [{"role": "system", "content": me.system_prompt()}]
        + history
        + [{"role": "user", "content": message}]
    )

    while True:
        response = client.chat.completions.create(
            model="arcee-ai/trinity-large-preview:free",
            messages=messages,
            tools=tools,
        )

        choice = response.choices[0]

        if choice.finish_reason == "tool_calls":
            msg = choice.message
            messages.append(msg)
            messages.extend(me.handle_tool_calls(msg.tool_calls))
        else:
            final_reply = choice.message.content
            break

    if should_push(final_reply):
        record_unknown_question(message)

    return final_reply

# attach state
chat.me = Me()

if __name__ == "__main__":
    gr.ChatInterface(chat).launch()

