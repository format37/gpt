from tqdm import tqdm
import os
import yaml
import pandas as pd
from cleanup import strip_headers

def file_list(config):
    target_dir = config["target_dir"]
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Select only english texts
    df = pd.read_csv(config["metadata_folder"])
    print('Initial size:', len(df))
    df = df[(df.language=="['en']") & (df.type=='Text')]
    print('English texts:', len(df))

    # Only existing files
    files = os.listdir(config["data_dir"])
    downloaded = [file.split("_")[0] for file in files]
    df = df[df.id.isin(downloaded)]
    print('Existing:', len(df))

    # Compose file path
    pathes = []
    for name in df.id:
        path = os.path.join(config['data_dir'], f"{name}_raw.txt")
        pathes.append(path)
    return pathes

def collapse_blank_lines(text):
    result = []
    for line in text.split("\n"):
        if line.strip():
            # Has text in a line
            result.append(line)
        elif result and result[-1] != "":
            # First empty string after the text
            result.append("")
        # Skip other empty strings
    return "\n".join(result)

def combine_files(files, config):
    max_size_mb = 500
    separator = "<|endoftext|>"
    target_dir = config['target_dir']

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    current_content = []
    current_size = 0
    file_counter = 1

    for file_path in tqdm(files):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
        except UnicodeDecodeError:
            # Not UTF-8 (UTF-16, cp1252): skip the book
            tqdm.write(f"Skipping {file_path}: not UTF-8")
            continue

        # Remove extra \ufeff garbage
        content = content.replace("\ufeff", "")
        # Remove license
        content = strip_headers(content)
        # Collapse blank lines
        content = collapse_blank_lines(content)

        # Write each 500 mb
        estimated_size = len(content.encode("utf-8"))
        if current_size + estimated_size > max_size_mb * 1024 * 1024:
            target_file_path = os.path.join(
                target_dir, 
                f"combined_{file_counter}.txt"
                )
            with open(target_file_path, "w", encoding="utf-8") as target_file:
                target_file.write(separator.join(current_content))
            file_counter += 1
            current_content = [content]
            current_size = estimated_size
        else:
            current_content.append(content)
            current_size += estimated_size

    # Write tail file
    if current_content:
        target_file_path = os.path.join(target_dir, f"combined_{file_counter}.txt")
        with open(target_file_path, "w", encoding="utf-8") as target_file:
            target_file.write(separator.join(current_content))

    print(f"{file_counter} file(s) saved in {target_dir}")

def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)["prepare_dataset"]
    files = file_list(config)
    combine_files(files, config)

if __name__=='__main__':
    main()
