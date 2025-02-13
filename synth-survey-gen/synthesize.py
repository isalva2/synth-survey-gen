import ollama
import json
from typing import List, Dict
from pathlib import Path

from ollama import chat, ChatResponse


def read_config(config_folder:str="config/"):
    """
    Prepares system configuration and survey questions for LLM.
    """
    config_folder = Path(config_folder)
    with open(config_folder / "config.json", "r") as file:
        model_config = json.load(file)

    with open( config_folder / "questions_condensed.json", "r") as file:
        questions = json.load(file)

    return model_config, questions


class survey_agent:
    def __init__(self, agent_id, prompt_template: str, agent_attrs:List[str] = None):
        self.id = agent_id
        self.system_prompt = prompt_template.format(*agent_attrs)

        self.messages = []
        self.survey_responses = []
        self.person_response = None
        self.is_compiled = False

        initial_prompt = {"role": "system", "content": self.system_prompt}
        self.messages.append(initial_prompt)

    def record_question(self, user_response:str):
        user_message = {"role": "user", "content": user_response}
        self.messages.append(user_message)

    def record_response(self, assistant_response:str):
        self.survey_responses.append(assistant_response)
        assistant_message = {"role": "assistant", "content": assistant_response}
        self.messages.append(assistant_message)


def run_survey(agents:List[survey_agent], model_config: Dict, questions: Dict):
    model_name = model_config["model_name"]
    model_parameters = model_config["model_params"]

    for agent in agents:
        for item in questions.items:
            pass


def build_survey(population_sample: List[List[str]], model_config:Dict, survey_config:Dict):
    """
    Prepare agent ID, individual unique system prompt, and list to store agent responses
    """

    # prepare system prompts and response
    agents = []
    for agent_id, individual_attributes in enumerate(population_sample):
        system_prompt = survey_config["prompt_template"].format(*individual_attributes)

        agent = {
            "agent_id": agent_id,
            "system_prompt": system_prompt,
            "survey_responses": [],
            "chat_history": None
        }

        agents.append(agent)

    return agents


def begin_survey(agents: List[Dict], model_config: Dict, survey_config: Dict):

    # get model params and survey questions
    model_name = model_config["model_name"]
    model_parameters = model_config["model_params"]
    survey_questions = survey_config["survey_questions"]
    response_types = survey_config["response_types"]

    # run through agents and survey questions:
    for agent in agents:
        system_prompt = agent["system_prompt"]
        messages = [{"role": "system", "content": system_prompt}]

        for i, question, response_type in enumerate(zip(survey_questions, survey_config["response_type_id"])):

            # append survey question to chat history
            chat_question = {
                "role": "user",
                "content": f"{question} {response_types[str(response_type)]}"
                }
            messages.append(chat_question)
            print(messages)

            # call model using ollama API
            response: ChatResponse = ollama.chat(
                model = model_name,
                options = model_parameters,
                messages = messages,
                stream=False
            )

            # extract assistant response from ChatResponse
            assistant_content = response["message"]["content"]
            agent["survey_responses"].append(assistant_content)

            # append response to
            assistant_response = {"role": "assistant", "content": assistant_content}
            messages.append(assistant_response)

            if i == len(survey_questions)-1:
                agent["chat_history"] = messages


def main():
    pass


if __name__ == "__main__":
    main()
