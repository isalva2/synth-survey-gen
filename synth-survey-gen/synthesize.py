import ollama
import json
from typing import List, Dict
import copy
from tqdm import tqdm
from pathlib import Path
from ollama import chat, ChatResponse


def read_config(config_folder:str="config/"):
    """
    Prepares system configuration and survey questions for LLM.
    """
    config_folder = Path(config_folder)
    with open(config_folder / "config.json", "r") as file:
        model_config: dict = json.load(file)

    with open( config_folder / "questions_condensed.json", "r") as file:
        questions: dict = json.load(file)

    return model_config, questions


class survey_agent:
    def __init__(self, agent_id, prompt_template: str, agent_attrs:List[str] = None):
        self.id = agent_id
        self.system_prompt = prompt_template.format(*agent_attrs)

        self.messages = []
        self.survey_variables = []
        self.survey_responses = []
        self.person_response = None

        initial_prompt = {"role": "system", "content": self.system_prompt}
        self.messages.append(initial_prompt)

    def record_question(self, user_response:str)->None:
        user_message = {"role": "user", "content": user_response}
        self.messages.append(user_message)

    def record_response(self, assistant_response:str)->None:
        self.survey_responses.append(assistant_response)
        assistant_message = {"role": "assistant", "content": assistant_response}
        self.messages.append(assistant_message)

    def write_person_response(self)->None:
        for variable, response in zip(self.survey_variables, self.survey_responses):
            pass
            

def run_survey(agents:List[survey_agent], model_config: Dict, questions: Dict, max_q: int = None, truncate_memory: bool=True):
    """
    Run's the survey sequentially on a list of survey agents.
    Uses a truncated short-term memory system.
    """
    model_name = model_config["model_name"]
    model_parameters = model_config["model_params"]

    # Loop through list of agent
    for agent in agents:
        # Loop through every survey qeustion/response variable in question dictionary
        for key in tqdm(list(questions.keys())[:max_q], desc="Asking questions"):
            
            # get question and possible repsonses corresponding to survey variable, this is in order
            survey_question = questions[key]["question"]
            possible_responses = "; ".join(f"{k}: {v}" for k, v in questions[key]["response"].items())
            formatted_question: str =  f"{survey_question} Please respond in the format 'number: option'. Possible choices are {possible_responses}"

            # build chat question with possible responses and add to chat message input
            chat_question = {
                "role": "user",
                "content": formatted_question
                }
            
            # choose to truncate memory
            if truncate_memory:
                chat_message = copy.deepcopy(agent.messages)
                chat_message.append(chat_question)

                # record survey question
                agent.record_question(survey_question)
            else:
                agent.record_question(formatted_question)
                chat_message = agent.messages

            # actual ollama API call
            response: ChatResponse = ollama.chat(
                model = model_name,
                options = model_parameters,
                messages = chat_message,
                stream=False
            )

            # add survey variable to agent, extract context - need to add COT processing here I believe
            agent.survey_variables.append(key)
            assistant_content = response["message"]["content"]
            agent.record_response(assistant_content)


class dummy_chat():
    def __init__(self, model_config: Dict):
        self.model_name = model_config["model_name"]
        self.model_parameters = model_config["model_params"]

        self.system_prompt: str = None
        self.messages = []
        self.ChatResponses = []

    def initialize(self, system_prompt: str):
        if self.system_prompt is None:
            self.system_prompt = system_prompt
            self.messages.append(self._prepare_message(0, system_prompt))

    def chat(self, chat_msg:str, verbose=True, ):
        self.messages.append(self._prepare_message(1, chat_msg))

        response: ChatResponse = ollama.chat(
            model = self.model_name,
            options = self.model_parameters,
            messages = self.messages,
            stream = False
        )

        assistant_content = response["message"]["content"]
        self.ChatResponses.append(response)
        self.messages.append(self._prepare_message(2, assistant_content))

        if verbose:
            return assistant_content
        
    def chat_history(self):
        return [message["content"] for message in self.messages]

    def clear(self):
        self.system_prompt = None
        self.messages = []
        self.ChatResponses = []


    def _prepare_message(self, role:int, message:str)-> Dict[str,str]:
        role_ids = ["system", "user", "assistant"]
        return {"role":role_ids[role], "content":message}


def main():
    pass


if __name__ == "__main__":
    main()
