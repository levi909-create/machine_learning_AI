"""
AI Chatbot using HuggingFace Transformers
Uses microsoft/DialoGPT-medium for conversational AI (runs locally, no API key needed).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "microsoft/DialoGPT-medium"


def load_model(model_name: str = MODEL_NAME):
    print(f"Loading model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    print("Model loaded.\n")
    return tokenizer, model


def chat(tokenizer, model, history_ids=None, max_history_turns: int = 5):
    """
    Run an interactive chat session.
    Maintains conversation history for context-aware responses.
    """
    print("=" * 50)
    print("  AI Chatbot (HuggingFace Transformers)")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    past_key_values = None
    chat_history_ids = None

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        # Encode user input and EOS token
        new_input_ids = tokenizer.encode(
            user_input + tokenizer.eos_token, return_tensors="pt"
        ).to(device)

        # Append to chat history
        if chat_history_ids is not None:
            bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
        else:
            bot_input_ids = new_input_ids

        # Trim history to avoid hitting max token limit
        max_len = 1000
        if bot_input_ids.shape[-1] > max_len:
            bot_input_ids = bot_input_ids[:, -max_len:]

        # Generate response
        with torch.no_grad():
            chat_history_ids = model.generate(
                bot_input_ids,
                max_new_tokens=150,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.75,
                repetition_penalty=1.3,
            )

        # Decode only the new tokens (bot's response)
        response_ids = chat_history_ids[:, bot_input_ids.shape[-1]:]
        response = tokenizer.decode(response_ids[0], skip_special_tokens=True)

        print(f"Bot: {response}")


def main():
    tokenizer, model = load_model()
    chat(tokenizer, model)


if __name__ == "__main__":
    main()
