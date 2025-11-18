import sys
import os
import json
from pathlib import Path
from preprocess import generate_questions
from ollama import generate


def read_results_json(run_path: str):
    folder_path = Path(run_path)
    folder_name = folder_path.name

    run_date = folder_name.split("_", 1)[1]
    results_json = list(folder_path.glob(run_date+"_results.json"))[0]
    with open(results_json, "r") as f:
        results_json = json.load(f)
    return results_json


def _remove_substrings(text, remove_list):
    for s in remove_list:
        text = text.replace(s, "")
    text = text.strip().replace("  ", " ")
    return text.strip()


def prepare_questions_and_responses(response: dict, questions:dict, remove_list: list, style: str = "question_response") -> str | None:
    raw_response_scraps = response.get("responses_scraps")
    survey_question_variables = response.get("logic_flow")

    responses_scraps = [s for s in raw_response_scraps if isinstance(s, str)]
    indices_to_remove = [i for i, s in enumerate(raw_response_scraps) if not isinstance(s, str)]

    filtered_survey_variables = [item for i, item in enumerate(survey_question_variables) if i not in indices_to_remove]
    cleaned_responses_scraps = [_remove_substrings(s, remove_list) for s in responses_scraps]

    formatted_responses = []
    i = 1
    for response in cleaned_responses_scraps:
        if (response != "") and (response != None):
            formatted_responses.append(
                f"{i}. {response}"
            )
            i += 1

    final_dialog = ""

    if len(formatted_responses) == len(filtered_survey_variables):
        for i, question_response in enumerate(zip(filtered_survey_variables, cleaned_responses_scraps)):
            survey_variable, survey_response = question_response
            survey_question = questions.get(survey_variable).get("question")

            if style == "question_response":
                dialog = f"Question {i+1}: {survey_question}\nResponse: {survey_response}"
            elif style == "response_only":
                dialog = f"Response {i+1}: {survey_response}"

            final_dialog += dialog + "\n"

        return final_dialog

    else:
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py path/to/config/ path/to/result_folder")
        sys.exit(1)

        CONFIG_FOLDER = sys.argv[1]
        RESULT_FOLDER = sys.argv[2]

        run_results = read_results_json(RESULT_FOLDER)
        questions = generate_questions(CONFIG_FOLDER)

        remove_list = [
            "TOOL: ",
            "TEXT:",
            "request",
            "discreteNumericResponse",
            "singleAnswerResponse",
            "multipleAnswerResponse",
            "tool",
            "TEXT",
            "{",
            "}",
            "[",
            "]",
            ":",
            '"',
            "\n",
        ]

        for result in run_results:
            agent_response = prepare_questions_and_responses(result, questions, remove_list, style="response_only")