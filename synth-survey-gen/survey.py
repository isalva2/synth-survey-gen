from preprocess import *
from langroid.language_models import LLMMessage
from synthesize import SurveyAgent
from typing import Dict, List
from datetime import datetime

import re
import json


def _response_from_tool_message(survey_response: LLMMessage) -> str | Dict[str, str] | None:
    """ Returns either a formatted Tool Message response as a Dict, None in
    case of error, or text only survey response from survey agent.

    Args:
        survey_response (LLMMessage): langroid LLMMessage implementation

    Returns:
        _type_: str | Dict[str, str] | None
    """
    message_content = survey_response.content
    match = re.search(r'{\s*(.*)\s*}', message_content, re.DOTALL)
    if match:
        json_string = match.group(1)

        try:
            tool_content = json.loads(f"{{{json_string}}}")
            return tool_content  # Return the parsed dictionary
        except json.JSONDecodeError:
            return None
    else:
        return message_content


class ResponsePackage:
    def __init__(self, id):
        self.id = id
        pass
        self.survey_logic = []
        self.raw_llm_contents = []
        self.question_dtypes = []
        self.timestamps = []
        self.encoded_responses = []
        self.parse_errors = []
        self.dtype_matches = []

    def record_instance(
        self,
        survey_variable: str, # Survey question variable
        raw_content: str, # LLMMessage content
        question_dtype: str, # NUMERIC or TEXT survey question type (primarily TEXT)
        timestamp: datetime, # langroid runtime
        encoded_response: int | List[int] | str | None, # possible response type based on available agent tools
        parse_error: bool, # flag if parsed correctly
        dtype_match: bool): # flag if mismatch on question type and response type
        self.survey_logic.append(survey_variable)
        self.raw_llm_contents.append(raw_content)
        self.question_dtypes.append(question_dtype)
        self.timestamps.append(timestamp)
        self.encoded_responses.append(encoded_response)
        self.parse_errors.append(parse_error)
        self.dtype_matches.append(dtype_match)


class SurveyEngine:
    def __init__(self, survey_conf: Dict, survey_questions: Dict, agents: List[SurveyAgent]):
        self.survey_conf = survey_conf
        self.questions = survey_questions
        self.agents = agents
        self.respondent_summaries = []

    def run(self):
        # survey logic and tool mapping
        survey_logic = self.survey_conf["logic"]
        type_tool_map = self.survey_conf["dtype_tools"]

        for agent in self.agents:
            id = agent.agent_id
            queued_variable = self.survey_conf["start"]               # queue up first question variable
            print(queued_variable)
            queued_question_package = self.questions[queued_variable] # get corresponding question package
            queued_question_dtype = queued_question_package["dtype"]

            # logging response
            agent_response_package = ResponsePackage(id=id)
            logic_log = []
            n_iter = 0

            while queued_variable != None:
                agent.queue_question(queued_variable, queued_question_package)
                agent.ask_question()

                last_message = agent.message_history[-1]
                content = last_message.content
                timestamp = last_message.timestamp
                parsed_response = _response_from_tool_message(last_message)

                # this match case needs to return this at minimum
                tool_type: str
                encoded_response: int | List[int] | str | None
                parse_error: bool = False
                dtype_match: bool

                match parsed_response: # extensible
                    case str():
                        tool_type = "textResponse"
                        encoded_response = parsed_response
                    case dict():
                        try:
                            tool_type = parsed_response["request"]
                            if tool_type == "singleAnswerResponse":
                                encoded_response = parsed_response["answer_key"]
                            elif tool_type == "multipleAnswerResponse":
                                encoded_response = parsed_response["answer_keys"]
                            elif tool_type == "discreteNumericResponse":
                                encoded_response = parsed_response["discrete_response"]
                        except:
                            tool_type = "BadResponse"
                            encoded_response = None
                            parse_error=True
                    case None:
                        tool_type = "BadResponse"
                        encoded_response = None
                        parse_error=True

                    case _:
                        tool_type = "BadResponse"
                        encoded_response = None
                        parse_error=True

                # dtype match
                dtype_match = tool_type in type_tool_map[queued_question_dtype]

                # log results and append
                agent_response_package.record_instance(
                    survey_variable=queued_variable,
                    raw_content=content,
                    question_dtype=queued_question_dtype,
                    timestamp=timestamp,
                    encoded_response=encoded_response,
                    parse_error=parse_error,
                    dtype_match=dtype_match)
                self.respondent_summaries.append(agent_response_package)

                logic_log.append(queued_variable)

                # survey logic control
                # queue up next variable






                if isinstance(survey_logic[queued_variable], dict):
                    try:
                        print(isinstance(encoded_response, int))
                        if isinstance(encoded_response, int):
                            queued_variable = survey_logic[queued_variable][str(encoded_response)]
                            print(survey_logic[queued_variable])
                        elif isinstance(encoded_response, list) or encoded_response is None:
                            queued_variable = survey_logic[queued_variable]["ELSE"]
                    except:
                        print("bingo bitch")
                        queued_variable = survey_logic[queued_variable]
                elif isinstance(survey_logic[queued_variable], str):
                    queued_variable = survey_logic[queued_variable]

                n_iter += 1