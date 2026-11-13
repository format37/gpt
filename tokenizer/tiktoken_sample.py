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

    # Special tokens: the same string can be encoded in two different ways.
    text = "end of book<|endoftext|>next book"

    # 1. Default: tiktoken refuses to encode text that contains a special token.
    try:
        tokenizer.encode(text)
    except ValueError as e:
        print("default: ValueError:", str(e).splitlines()[0])

    # 2. Allowed: "<|endoftext|>" becomes ONE special token, id 50256.
    ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print("allowed:", [tokenizer.decode([i]) for i in ids], ids)

    # 3. Treated as plain text: the same characters become ordinary pieces.
    ids = tokenizer.encode(text, disallowed_special=())
    print("as text:", [tokenizer.decode([i]) for i in ids], ids)

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)["tokenizer"]

    with open(cfg["sample_file"], encoding="utf-8") as f:
        text = f.read(10_000_000)
    ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(f"{len(text) / len(ids):.2f} chars per token")

if __name__=="__main__":
    main()
    