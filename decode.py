import json
import os
import argparse

def decode_unicode_escaped_string(s):
    if isinstance(s, str):
        try:
            return json.loads(f'"{s}"')
        except json.JSONDecodeError:
            return s
    return s

def main():
    parser = argparse.ArgumentParser(description="Decode unicode-escaped strings in JSON.")
    parser.add_argument("--input_path", type=str, help="Path to the input JSON file")
    args = parser.parse_args()

    input_path = args.input_path
    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' not found.")
        return

    output_path = input_path.replace(".json", "_decoded.json")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        for item in data:
            for key in item:
                item[key] = decode_unicode_escaped_string(item[key])
    else:
        for key in data:
            data[key] = decode_unicode_escaped_string(data[key])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Decoded JSON saved at: {output_path}")

if __name__ == "__main__":
    main()