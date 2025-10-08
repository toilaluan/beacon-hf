MESSAGES = [
    {
        "content": "What is the capital of France?",
        "role": "user"
    },
    {"content":"", "role": "assistant"}
]

if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    conversation = tokenizer.apply_chat_template(MESSAGES, tokenize=True, add_generation_prompt=False, continue_final_message=True)
    print(tokenizer.decode(conversation))