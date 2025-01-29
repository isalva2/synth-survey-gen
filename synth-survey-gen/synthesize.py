import ollama
import json
from typing import List, Dict
from pathlib import Path

from ollama import chat, ChatResponse


def read_config(config_folder:Path=Path("config/")):
    """
    Prepares system configuration and survey questions for LLM.
    """
    with open(config_folder / "config.json", "r") as file:
        model_config = json.load(file)

    with open( config_folder / "survey.json", "r") as file:
        survey_config = json.load(file)

    return model_config, survey_config

def generate_population_sample(n: int) -> List[List[str]]:
    """
    Returns a list of string attributes for `n` many agents. Uses iterative proportional fitting.
    """
    pass


def prepare_system_prompts(population_sample: List[List[str]], prompt_template: str):
    """
    Prepare a system prompt for each
    """
    system_prompts = []
    for individiual_attributes in population_sample:
        system_prompt = prompt_template.format(*individiual_attributes)
        system_prompts.append(system_prompt)


def build_survey(population_sample: List[List[str]], model_config:Dict, survey_config:Dict):
    """
    Prepare agent ID, individual unique system prompt, and list to store agent responses
    """

    # prepare system prompts and response
    survey = []
    for agent_id, individual_attributes in enumerate(population_sample):
        system_prompt = survey_config["prompt_template"].format(*individual_attributes)

        agent = {
            "agent_id": agent_id,
            "system_prompt": system_prompt,
            "survey_responses": []
        }

        survey.append(agent)

    return agents


def begin_survey(agents: Dict, model_config: Dict, survey_config: Dict):

    model_name = model_config["model_name"]
    model_parameters = model_config["model_params"]

    survey_questions = survey_config["survey_questions"]
    response_types = survey_config["response_types"]

    # run through agents and survey questions:
    for agent in agents:
        system_prompt = agent["system_prompt"]
        messages = [{"role": "system", "content": system_prompt}]

        responses = []

        for question, response_type in zip(survey_questions, survey_config["response_type_id"]):

            # append survey question to chat history
            chat_question = {"role": "user", "content": f"{question} {response_types[str(response_type)]}"}
            messages.append(chat_question)
            print(messages)

            # llm time
            response: ChatResponse = ollama.chat(
                model = model_name,
                options = model_parameters,
                messages = messages,
                stream=False
            )

            # append response to chat history:
            assistant_response = {"role": "assistant", "content": response["message"]["content"]}
            print(assistant_response)
            agent["survey_responses"].append(assistant_response["content"])
            messages.append(assistant_response)

            print(messages)




def main():
    pass


if __name__ == "__main__":
    main()
