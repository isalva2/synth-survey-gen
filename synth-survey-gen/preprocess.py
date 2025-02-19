from typing import Dict
import pandas as pd
from pathlib import Path
import json

def process_MyDailyTravelData(source:Path):

    # get variables and table from data dictionary
    file_path = source / "data_dictionary.xlsx"
    data_dictionary = pd.read_excel(file_path, sheet_name=None)
    variables_df = data_dictionary["Variables"]
    variables_df = variables_df[variables_df["QUESTION TEXT"].notna()]
    lookup_df = data_dictionary["Value Lookup"]

    # get response dictionary from person.csv
    file_path = source / "person.csv"
    person_cols = pd.read_csv(file_path, nrows=0).columns.to_list()
    person_response = {col:None for col in person_cols}

    # query dictionary
    query_dictionary = {}
    for col in person_cols:
        if col.upper() in variables_df["NAME"].to_list():
            try:
                lookup_table = lookup_df[lookup_df["NAME"]==col.upper()]
                query_dictionary[col.upper()] = {
                    "question":(variables_df[variables_df["NAME"] == col.upper()]["QUESTION TEXT"].values)[0],
                    "response": lookup_table.set_index("VALUE")["LABEL"].to_dict()
                }
            except:
                query_dictionary[col.upper()] = "This didnt work"


    return variables_df, query_dictionary, person_response

def process_pums_data(source: str, write: str | None = None) -> Dict[str, str] | None:
    data_path = Path(source)
    df_path = data_path.glob("psam_p*.csv")
    dd_path = data_path.glob("PUMS_Data_Dictionary*.csv")
    df = pd.read_csv(*df_path, nrows=1)
    dd = pd.read_csv(*dd_path, header=None, names=list("abcdefg"))

    person_variables = df.columns.values
    variable_desc = dd[dd.a == "NAME"].set_index("b")["e"].to_dict()

    person_variable_dict = {k:v for k,v in variable_desc.items() if k in person_variables}

    person_answer_mapper = {}
    for variable,description in person_variable_dict.items():
        answers = dd[(dd.a!="NAME") & (dd.b==variable)][["f","g"]].drop_duplicates().set_index("f")["g"].to_dict()
        person_answer_mapper[variable] = {
            "description": description,
            "answers": answers
        }

    if write != None:
        with open(write, "w") as file:
            json.dump(person_answer_mapper, file)
    else:
        return person_answer_mapper


if __name__ == "__main__":
    person_answer_mapper = process_pums_data("data/pums", "configs/pums_answer_mapper.json")