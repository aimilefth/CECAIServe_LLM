from llama_cpp import Llama

def main():
    llm = Llama.from_pretrained(
        repo_id="Qwen/Qwen2-0.5B-Instruct-GGUF",
        filename="*q8_0.gguf",
        verbose=True,
    )
    output = llm(
        "Q: Name the planets in the solar system? A: ", # Prompt
        max_tokens=None, # Generate up to 32 tokens, set to None to generate up to the end of the context window
        stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
        echo=True # Echo the prompt back in the output
    ) # Generate a completion, can also call create_completion
    print(output)


if __name__ == "__main__":
    main()