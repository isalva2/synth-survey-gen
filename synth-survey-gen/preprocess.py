"""
Preprocessing form https://cmap.illinois.gov/data/transportation/travel-survey/ 2019 Survey.
"""


import pandas as pd
from pathlib import Path

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
                    "question":variables_df[variables_df["NAME"] == col.upper()]["QUESTION TEXT"].values[0],
                    "response": lookup_table.set_index("VALUE")["LABEL"].to_dict()
                }
            except:
                query_dictionary[col.upper()] = "This didnt work"


    return variables_df, query_dictionary, person_response