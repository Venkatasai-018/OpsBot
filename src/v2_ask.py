import json
from datetime import datetime
from pathlib import Path

import boto3


ROOT_DIR = Path(__file__).resolve().parent.parent
PROMPT_FILE = ROOT_DIR / "prompts" / "v2" / "v2.txt"
RUNBOOK_FILE = ROOT_DIR / "data" / "runbooks" / "issue.txt"
OUTPUT_FILE = ROOT_DIR / "notes" / "v2" / "responses.txt"
System_file = ROOT_DIR / "prompts" / "v2" / "system_prompt.txt"



def load_client():
    return boto3.client(
        "bedrock-runtime",
        region_name="us-east-1"
    )


def load_prompt(file_path):
    prompt_path = Path(file_path)
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Prompt file is empty: {prompt_path}")
    return prompt


def load_runbook(file_path):
    runbook_path = Path(file_path)
    if not runbook_path.is_file():
        raise FileNotFoundError(f"Runbook file not found: {runbook_path}")
    return runbook_path.name, runbook_path.read_text(encoding="utf-8").strip()

system_prompt=load_prompt(System_file)
def build_system_prompt(runbooks):
    
    return (system_prompt
        + ".\n\n"
        f"RUNBOOK\n{runbooks}"
    )


def ask_model(client, prompt, runbook):
    response = client.invoke_model(
        modelId="amazon.nova-micro-v1:0",
        body=json.dumps({
            "system": [{"text": build_system_prompt(runbook)}],
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt}]
                }
            ],
            "inferenceConfig": {"maxTokens": 512, "temperature": 1.0},
            
        })
    )
    return json.loads(response["body"].read())


def append_output(output, usage, runbook_name):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with OUTPUT_FILE.open("a", encoding="utf-8") as file:
        file.write(
            f"[{timestamp}]\n"
            f"Runbook: {runbook_name}\n"
            f"{output}\n"
            f"Input tokens: {usage['inputTokens']}\n"
            f"Output tokens: {usage['outputTokens']}\n"
            f"Total tokens: {usage['totalTokens']}\n"
            f"{'-' * 60}\n"
        )


def main():
    prompt = load_prompt(PROMPT_FILE)
    runbook_name, runbook = load_runbook(RUNBOOK_FILE)
    result = ask_model(load_client(), prompt, runbook)

    output = result["output"]["message"]["content"][0]["text"]
    usage = result["usage"]
    append_output(output, usage, runbook_name)

    print(output)
    print(f"Input tokens: {usage['inputTokens']}")
    print(f"Output tokens: {usage['outputTokens']}")
    print(f"Total tokens: {usage['totalTokens']}")
    print(f"Appended to {OUTPUT_FILE.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()


