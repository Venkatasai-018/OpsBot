import csv
import json
from pathlib import Path

import boto3


ROOT_DIR = Path(__file__).resolve().parent.parent
ALERTS_FILE = ROOT_DIR / "data" / "alerts.csv"
MODEL_ID = "amazon.nova-micro-v1:0"
REGION_NAME = "us-east-1"
MAX_OUTPUT_TOKENS = 512


def load_examples(file_path=ALERTS_FILE):
	with file_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
		rows = list(csv.DictReader(csv_file))[:3]

	required_fields = {
		"alert_id",
		"alert_name",
		"priority",
		"category",
		"example_condition",
		"evaluation_window",
		"runbook_file",
	}
	if not rows:
		raise ValueError(f"No examples found in {file_path}")
	if not required_fields.issubset(rows[0]):
		missing_fields = ", ".join(sorted(required_fields - set(rows[0])))
		raise ValueError(f"Missing columns in {file_path}: {missing_fields}")

	return [
		{
			"input": row["example_condition"],
			"expected_output": {
				"content": row["example_condition"],
				"priority": row["priority"],
			},
		}
		for row in rows
	]


def build_system_prompt(examples):
	examples_json = json.dumps(examples, indent=2)
	return (
		"You classify DevOps alert descriptions using the labeled examples below. "
		"Match the closest example. Return only a JSON object with exactly two keys: "
		"content and priority. Use P1, P2, or P3 for priority."

		f"LABELED EXAMPLES\n{examples_json}"
	)


def ask_model(prompt, examples):
	client = boto3.client("bedrock-runtime", region_name=REGION_NAME)
	response = client.invoke_model(
		modelId=MODEL_ID,
		body=json.dumps({
			"system": [{"text": build_system_prompt(examples)}],
			"messages": [
				{
					"role": "user",
					"content": [{"text": prompt}],
				}
			],
			"inferenceConfig": {
				"maxTokens": MAX_OUTPUT_TOKENS,
				"temperature": 0.0,
			},
		}),
	)
	return json.loads(response["body"].read())


def main():
	prompt = input("Describe the alert to classify: ").strip()
	if not prompt:
		raise ValueError("The alert description cannot be empty.")

	examples = load_examples()
	result = ask_model(prompt, examples)
	answer_text = result["output"]["message"]["content"][0]["text"]
	try:
		answer = json.loads(answer_text)
	except json.JSONDecodeError as error:
		raise ValueError(f"Model did not return valid JSON: {answer_text}") from error

	print(f"Content: {answer.get('content')}")
	print(f"Priority: {answer.get('priority')}")


if __name__ == "__main__":
	main()
