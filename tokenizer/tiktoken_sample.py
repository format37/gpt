import tiktoken
import yaml

def main():
    tokenizer = tiktoken.get_encoding("gpt2")
    text = "Jack embraced beauty through art and life."
    integers = tokenizer.encode(text)
    print("integers:", integers)
    print([tokenizer.decode([i]) for i in integers])
    decoded = tokenizer.decode(integers)
    print("decoded:", decoded)

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)["tokenizer"]

    with open(cfg["sample_file"], encoding="utf-8") as f:
        text = f.read(10_000_000)
    ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(f"{len(text) / len(ids):.2f} chars per token")

if __name__=="__main__":
    main()
    