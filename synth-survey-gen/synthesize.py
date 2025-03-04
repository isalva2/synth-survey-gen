import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely import Point
import json
from typing import List, Dict, Tuple
import copy
from tqdm import tqdm
from pathlib import Path
import ollama
from ollama import ChatResponse



def read_config(config_folder:str):
    """
    Prepares config folder
    """

    # read config, question, and mappers
    config_path = Path(config_folder)
    with open(config_path / "config.json", "r") as file:
        model_config: dict = json.load(file)

    with open(config_path / "questions.json", "r") as file:
        questions: dict = json.load(file)


def synthesize_population(config_folder:str, n_sample:int, source:str="pums", random_state=0):

    data_folder = Path(config_folder) / "data"

    if source == "pums":
        """
        Data derived from 2019 Public Use Microdata Sample (PUMS) dataset.
        PUMS: https://www.census.gov/programs-surveys/acs/microdata/access.2019.html#list-tab-735824205
        PUMA: https://catalog.data.gov/dataset/tiger-line-shapefile-2019-series-information-for-the-2010-census-public-use-microdata-area-puma
        """

        crs:str = "EPSG:4326"

        # load PUMS and PUMA data
        person_df = pd.read_csv(data_folder/"person.csv", low_memory=False)
        location_df = pd.read_csv(data_folder/"location.csv")
        pums_person_df = pd.read_csv(data_folder/"psam_p17.csv")

        puma_gdf = gpd.read_file(data_folder/"tl_2019_17_puma10.shp")
        puma_gdf = puma_gdf.to_crs(crs=crs)
        puma_gdf["PUMA"] = puma_gdf["PUMACE10"].astype(int)

        # get the size of each household
        household_sizes = person_df[["sampno", "perno"]].groupby(by="sampno")["perno"].max()
        household_locations = location_df[location_df.loctype==1] \
            .drop_duplicates(subset="sampno") \
            .set_index("sampno")[["longitude", "latitude"]] # always x, y for spatial operations apparently
        household_locations = household_locations.join(household_sizes)

        # get household locations as gdf
        geometry = household_locations.apply(
            lambda row: Point(row.longitude, row.latitude), axis=1
        )
        household_gdf = GeoDataFrame(household_locations, geometry=geometry, crs=crs)

        # get count of households in each PUMA zone
        household_join = household_gdf.sjoin(puma_gdf, predicate="within")
        household_counts = household_join.groupby(by=["PUMA"])["perno"].sum()
        puma_household_counts = puma_gdf \
            .set_index("PUMA") \
            .join(household_counts, how="inner")[["GEOID10", "NAMELSAD10", "perno", "geometry"]]
        puma_household_counts.reset_index(inplace=True)

        # sample by PUMA area
        samples = []
        puma_household_counts["weight"] = puma_household_counts.perno / puma_household_counts.perno.sum()
        for _, row in puma_household_counts.iterrows():
            puma = row.PUMA
            weight = row.weight
            n = int(weight*n_sample)

            sample = pums_person_df[pums_person_df.PUMA==puma].sample(n, random_state=random_state)
            samples.append(sample)

        population_sample = pd.concat(samples)

        return population_sample

    pass


class survey_agent:
    def __init__(self, agent_id, prompt_template: str, agent_attrs: List[str]|None = None):
        self.id = agent_id
        self.system_prompt = prompt_template.format(*agent_attrs)

        self.messages = []
        self.survey_variables = []
        self.survey_responses = []
        self.person_response = None

        initial_prompt = {"role": "system", "content": self.system_prompt}
        self.messages.append(initial_prompt)

    def record_question(self, user_response: str|None) -> None:
        user_message = {"role": "user", "content": user_response}
        self.messages.append(user_message)

    def record_response(self, assistant_response: str|None) -> None:
        self.survey_responses.append(assistant_response)
        assistant_message = {"role": "assistant", "content": assistant_response}
        self.messages.append(assistant_message)

    def write_person_response(self)->None:
        for variable, response in zip(self.survey_variables, self.survey_responses):
            pass


def run_survey(agents:List[survey_agent], model_config: Dict, questions: Dict, max_q: int|None = None, truncate_memory: bool=True):
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
            # survey_question = questions[key]["question"]
            # responses =

            survey_question, responses = questions[key].values()
            # possible_responses = "; ".join(f"{k}: {v}" for k, v in responses.items())
            # print(possible_responses)
            # formatted_question: str =  f"{survey_question} Please respond in the format 'number: option'. Possible choices are {possible_responses}"

            formatted_question, question_format = question_formatter(survey_question, responses)

            # build chat question with possible responses and add to chat message input
            chat_question = {
                "role": "user",
                "content": formatted_question
                }

            # choose to truncate memory
            if truncate_memory:
                chat_message = copy.deepcopy(agent.messages)
                chat_message.append(chat_question)
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

            # There should be a post process here, e.g. deepseek remove COT think process.

            agent.record_response(assistant_content)


def question_formatter(survey_question: str, responses: Dict[str, str])->Tuple[str,str]:
    """
    Takes survey question and string representation of all possible answers and formats the instructions for survey response.
    Currently supports the following question type that is linked to specific text in the MyDailyTravel Survey:
        1. Single variable response. Requests a single item from the list of possible responses. Returns a key-value pair. This is the default question format.
        2. Select all that apply (SAA): Selects multiple single variable responses.
        2. Numeric (NUM): Request a numeric value to be returned. No further elaboration is required.
        3. Text (TXT): Request a text response of a few words.
    """

    substring_mapper = {
        "Select all that apply.": "SAA",
        "Please describe.": "TXT",
        "please describe.": "TXT",
        "please provide the degree": "TXT",
        "please provide details": "TXT",
        "in a few words": "TXT",
        "How many hours do you work in": "NUM",
        "On average, how much do you pay": "NUM",
        "How many days do you telecommute per week": "NUM",
        "How many jobs do you": "NUM",
        "How old are you?": "NUM"
    }
    question_format = "DEF"
    for substring in substring_mapper.keys():
        if substring in survey_question:
            question_format = substring_mapper[substring]

    possible_responses = "; ".join(f"{k}: {v}" for k, v in responses.items())

    if question_format == "DEF":
        formatted_question = f"{survey_question} Please respond with only one of the following choices. Possible choices are: {possible_responses}"
    elif question_format == "TXT":
        formatted_question = f"{survey_question} Please respond in a few words or if appropriate, select only one of the possible alternative choices: {possible_responses}"
    elif question_format == "NUM":
        formatted_question = f"{survey_question} Please response with only a nunmeric value or if appropriate, select one on the possible alternative choices: {possible_responses}"
    elif question_format == "SAA":
        formatted_question = f"{survey_question} Please place multiple selections on a new line. Possible choices are {possible_responses}"
    else:
        formatted_question = "Please skip this and proceed to the next question."
        question_format = "BAD"

    return formatted_question, question_format

class dummy_chat():
    def __init__(self, model_config: Dict, model: str|None=None):
        self.model_name = model_config["model_name"] if model is None else model
        self.model_parameters = model_config["model_params"]

        self.system_prompt: str | None = None
        self.messages: List = []
        self.ChatResponses: List = []

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


    def _prepare_message(self, role:int, message:str)->Dict[str,str]:
        role_ids = ["system", "user", "assistant"]
        return {"role":role_ids[role], "content":message}


def main():
    pass


if __name__ == "__main__":
    main()
