"""
AI Chatbot using the Anthropic Claude API.
Uses claude-opus-4-6 for conversational AI.
"""

import anthropic

MODEL = "claude-opus-4-6"
SYSTEM_PROMPT = "You are a helpful, friendly AI assistant. Keep responses concise and conversational."


def chat():
    client = anthropic.Anthropic()
    messages = []

    print("=" * 50)
    print("  AI Chatbot (Claude Opus 4.6)")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 50)

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        with client.messages.stream(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages,
        ) as stream:
            print("Bot: ", end="", flush=True)
            response_text = ""
            for text in stream.text_stream:
                print(text, end="", flush=True)
                response_text += text
            print()

        messages.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    chat()
