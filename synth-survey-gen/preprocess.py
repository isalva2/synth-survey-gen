from typing import Dict, List
import pandas as pd
import geopandas as gpd
import random
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
import json
import re
import locale

"""
Preprocessing steps for census and travel survey data.
These functions prepare json files for a specific survey
configuration, however there are a couple steps (~10%)
that needs to be done manually.
"""

# set for US currency atm
locale.setlocale(locale.LC_ALL, "en_US.UTF-8") # this will be a problem in the future

def process_MyDailyTravelData(config_folder: str):
    def value_to_int(x):
        try:
            return int(x)
        except:
            return int(x.split("-")[0])

    def clean_and_replace(text: str, replacements: dict) -> str:

        text = " ".join(text.split())  # Remove excess whitespace and new lines

        def replace_match(match):
            key = match.group(1)
            return replacements.get(key, match.group(0))  # Replace if found, else keep original

        return re.sub(r"\[\$(.*?)\]", replace_match, text)

    replacement_dict = {
        "AGE_COMPUTED": "",
        "ARE_YOU": "are you",
        "ARE_YOU_CAP": "Are you",
        "CURRENTDATE": "today",
        "DAYCARE": "",
        "DO_YOU": "do you",
        "DO_YOU_CAP": "Do you",
        "HAVE_YOU": "have you",
        "I_DO": "I do",
        "JOBTEXT": "",
        "NONWORKER_TEXT": "",
        "ON_DAY": "today",
        "PRIMARY": " primary",
        "WERE_ACTIVITIES": "Were activities",
        "WORK_PRE": "",
        "WORKER_TEXT": "",
        "YOUR": "your",
        "YOUR1": "your",
        "YOUR_EMPLOYER": "your employer",
        "YOUR_THEIR": "your",
        "YOU": "you",
        "YOU1": "you",
        "YOU_DO": "you do",
        "YOU_HAVE": "you",
        "YOU_TELECOMMUTE": "you telecommute",
        "YOU_THEIR": "your",
        "YOU_WORK": "you work"
    }

    # get variables and table from data dictionary
    data_path = Path(config_folder) / "data"
    file_path = data_path / "data_dictionary.xlsx"
    data_dictionary = pd.read_excel(file_path, sheet_name=None)
    variables_df = data_dictionary["Variables"]
    variables_df = variables_df[variables_df["QUESTION TEXT"].notna()]

    lookup_df = data_dictionary["Value Lookup"]
    lookup_df["VALUE_INT"] = lookup_df["VALUE"].apply(
        lambda x: value_to_int(x)
        )

    # get response dictionary from person.csv
    file_path = data_path / "person.csv"
    person_cols = pd.read_csv(file_path, nrows=0).columns.to_list()

    # query dictionary
    query_dictionary = {}
    for col in person_cols:
        if col.upper() in variables_df["NAME"].to_list():
            try:
                lookup_table = lookup_df[lookup_df["NAME"]==col.upper()]
                query_dictionary[col.upper()] = {
                    "question":(variables_df[variables_df["NAME"] == col.upper()]["QUESTION TEXT"].values)[0],
                    "dtype":(variables_df[variables_df["NAME"] == col.upper()]["DATA TYPE"].values)[0],
                    "response": lookup_table.set_index("VALUE_INT")["LABEL"].to_dict()
                }
            except:
                query_dictionary[col.upper()] = "This didnt work"

    for item in query_dictionary.items():
        survey_variable, question_response = item
        if "question" in question_response.keys():
            question_text = query_dictionary[survey_variable]["question"]
            query_dictionary[survey_variable]["question"] = clean_and_replace(question_text, replacement_dict)

    # add introduction
    introduction = {"INTRO":{
        "question": "Please tell us your name and a little about yourself.",
        "dtype": "TEXT",
        "response": {
            "-8": "I don't know",
            "-7": "I prefer not to answer",}}}
    query_dictionary = {**introduction, **query_dictionary}

    return query_dictionary



def process_pums_data(config_folder: str, person: bool = True, write: str | None = None) -> Dict[str, str] | None:
    """
    Reads US Census Public Use Microdata Sample (PUMS) Uses 1-year ACS data.
    Househould and person data: https://www2.census.gov/programs-surveys/acs/data/pums/2019/1-Year/
    """
    data_path = Path(config_folder) / "data"
    dd_path = data_path.glob("PUMS_Data_Dictionary*.csv")
    dd = pd.read_csv(*dd_path, header=None, names=list("abcdefg"))
    variable_desc = dd[dd.a == "NAME"].set_index("b")["e"].to_dict()

    na_str = "MISSING"

    if person:
        df_path = data_path.glob("psam_p*.csv")
    else:
        df_path = data_path.glob("psam_h*.csv")
    df = pd.read_csv(*df_path, nrows=1)
    pums_variables = df.columns.values
    pums_variable_dict = {k:v for k,v in variable_desc.items() if k in pums_variables}
    mapper = {}
    for variable,description in pums_variable_dict.items():
        description_row = dd[(dd.a!="NAME") & (dd.b==variable)]
        dtype = description_row.c.iloc[0]
        answers = description_row[["f","g"]] \
            .drop_duplicates() \
            .fillna(na_str) \
            .set_index("f")["g"] \
            .to_dict()
        mapper[variable] = {
            "description": description,
            "dtype": dtype,
            "answers": answers
        }

    if write != None:
        with open(write, "w") as file:
            json.dump(mapper, file, indent=4)
    else:
        return mapper


def puma_locations(config_folder: str) -> Dict[str, List[str]]:
    """
    returns a dictionary of PUMA codes keys and list of cities within PUMA.
    derived from:https://catalog.data.gov/dataset/tiger-line-shapefile-2019-state-illinois-current-place-state-based
    """
    epsg = 26971 # NAD83 StatePlane Illinois East FIPS 1201
    data_path = Path(config_folder) / "data"
    puma = gpd.read_file(data_path / "tl_2019_17_puma10.shp").to_crs(epsg=epsg)
    place = gpd.read_file(data_path / "tl_2019_17_place.shp").to_crs(epsg=epsg)

    place["geometry"] = place.geometry.centroid
    joined_place = gpd.sjoin_nearest(place, puma, how="left")

    puma_locations = joined_place[["NAME", "PUMACE10"]].groupby("PUMACE10")["NAME"].apply(list).to_dict()
    return puma_locations


def attribute_decoder_dict(encoded_attributes: Dict[str, str], decoder_dict: Dict[any, any]) -> Dict[str, str]:
    """
    prepares a dictionary of encoded individual attributes and their
    descriptions (from the person.json and house.json files) for system prompt templating
    """
    individual_attributes = {}
    keyCatch = ["SERIALNO", "PUMA"]
    for key, val in encoded_attributes.items():
        try:
                if (decoder_dict[key]["dtype"] == "N") or (key in keyCatch) :
                    individual_attributes[key] = val
                else:
                    individual_attributes[key] = decoder_dict[key]["answers"][val]
        except:
            individual_attributes[key] = "MISSING"

    return individual_attributes


def get_attribute_descriptions(decoder_dict: Dict[any, any]) -> Dict[str, str]:
    return {key+"_desc": decoder_dict[key]["description"] for key in decoder_dict.keys()}


def _decapitalize(sentence: str)->str:
    """
    Returns sentence without capitalization
    """
    return sentence[0].lower()+sentence[1:]


def _indefinite(noun_phrase: str) -> str:
    """
    Adds an indefinite article to noun phrase
    """
    vowels = "aeiou"
    indefinite_article = "an" if noun_phrase[0].lower() in vowels else "a"
    return f"{indefinite_article} {noun_phrase}"


def _wspace(word: str) -> str:
    "Adds singe space to start of phrase"
    return f" {word}"


def _random_select(key: str, mapper: Dict[str, List[str]]):
    """Takes a string phrase as a key and returns a random selection from mapper dictionary

    Args:
        key (str): jinja template variable
        mapper (Dict[str, List[str]]): A dictionary with items corresponding to jinja environment variables and there possible values

    Returns:
        _type_: _description_
    """
    try:
        return random.choice(mapper.get(key))
    except:
        return "RANDOM_SELECT_MISSING"


def _decontext(phrase: str, sep: str = " (") -> str:
    """
    Removes anything in parentheses (default behavior)
    """
    if sep in phrase: return phrase.split(sep, maxsplit=1)[0]
    else: return phrase


def _to_currency(val: str | float, symbol: bool = True, grouping: bool = True):
    return locale.currency(float(val), symbol=True, grouping=grouping)[:-3]


def _random_includes(seq: List[str]) -> List[str]:
    shuffled = list(seq)
    random.shuffle(shuffled)
    return shuffled


def _listify(items: List[str]) -> str:
    if not items:
        return ''
    if len(items) == 1:
        return items[0]
    elif len(items) == 2:
        return f'{items[0]} and {items[1]}'
    else:
        return f"{', '.join(items[:-1])}, and {items[-1]}"

def write_individual_bio(attributes: Dict[str, str], descriptions: Dict[str, str], config_folder: str, **kwargs) -> str:

    # globals -> to be derived from a config file eventually, ahahahah
    year = "2019"
    month = "March"
    day = "25"

    env_path = Path(config_folder) / "templates"
    env = Environment(
        loader=FileSystemLoader(env_path),
    )
    env.trim_blocks = True
    env.lstrip_blocks=True
    env.filters["desentence"] = _decapitalize
    env.filters["indefinite"] = _indefinite
    env.filters["wspace"] = _wspace
    env.filters["random_s"] = _random_select
    env.filters["decontext"] = _decontext

    bio_template = env.get_template("bio.j2")
    bio = bio_template.render(**attributes, **descriptions, **kwargs, YEAR = year)
    return bio

class SystemMessageGenerator:
    def __init__(self, config_folder: str, template: str, verbose_debug:bool = False, shuffle:bool = False, wrap: int|None = None):
        """
        WRAP IS DEFAULT BEHAVIOR ON SYS MSG
        """
        # load environment
        self.env = Environment(
            loader=FileSystemLoader(
                Path(config_folder) / "templates"
            ),
            extensions=["jinja2.ext.do"]
        )
        self.template = template
        self.verbose_debug = verbose_debug
        self.shuffle = shuffle
        self.wrap = wrap

        # initialize environment preferences and filters
        self.env.trim_blocks=True
        self.env.lstrip_blocks=True
        self.env.filters["desentence"]         = _decapitalize
        self.env.filters["indefinite"]         = _indefinite
        self.env.filters["wspace"]             = _wspace
        self.env.filters["random_s"]           = _random_select
        self.env.filters["decontext"]          = _decontext
        self.env.filters["to_currency"]        = _to_currency
        self.env.filters["randomize_includes"] = _random_includes
        self.env.filters["listify"]            = _listify

        # get system message template
        self.system_message_template = self.env.get_template(self.template)

    def _wrap_text(self, text, n):
        """
        Utility to break text on char width
        """
        words = text.split()
        lines = []
        current_line = []

        for word in words:
            if sum(len(w) for w in current_line) + len(current_line) + len(word) <= n:
                current_line.append(word)
            else:
                # Join current line and start a new one
                lines.append(' '.join(current_line))
                current_line = [word]

        # Add last line if not empty
        if current_line:
            lines.append(' '.join(current_line))

        return '\n'.join(lines)

    def write_system_message(self, **kwargs) -> str:
        """Writes agent system messages

        kwargs are specific to population synthesis dataset

        Returns:
            str: _description_
        """
        rendered_msg = self.system_message_template.render(**kwargs, _all_args = kwargs, verbose_debug = self.verbose_debug, shuffle = self.shuffle)
        if self.wrap is not None:
            return self._wrap_text(rendered_msg, self.wrap)
        else:
            return rendered_msg[1:] # there is an extra space on intro to make shuffling work


if __name__ == "__main__":
    pass