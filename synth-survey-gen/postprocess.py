from survey import AgentReponsePackage
from pathlib import Path
from dataclasses import asdict
import pandas as pd
import json

class PostProcessMyDailyTravelResponse:
    def __init__(self, config_folder):
        self.data_path = Path(config_folder) / "data"
        self.ground_truth_df = pd.read_csv(self.data_path / "person.csv", low_memory=False)

        self._prepare_dataset()

    def _prepare_dataset(self):
        ground_truth_cols = self.ground_truth_df.columns
        synthetic_columns = ["agent_id", "bio", "intro"]
        synthetic_columns.extend(ground_truth_cols)
        self.synthetic_dataset = pd.DataFrame(columns=synthetic_columns)
        self.synthetic_asdict = []

        self.multiple_choice_cols = ["NOGOWHY2", "TRAVELDATAMODE", "DTYPE"]

    def serialize_response(self, agent_response: AgentReponsePackage):
        response_dict = asdict(agent_response)
        response_cols = [col.lower() for col in response_dict["logic_flow"]]
        new_row = {}

        # Loop through the logic flow and encoded responses to build the new row
        for col, val in zip(response_cols, response_dict["encoded_responses"]):
            # Handle multiple choice columns which might be lists of ints
            if col in self.multiple_choice_cols:
                if isinstance(val, list):
                    # Convert each element in the list to an integer, if possible
                    new_row[col] = [self._coerce_to_int(x) for x in val]
                else:
                    # If it's not a list, try to convert it to an int and wrap it in a list
                    new_row[col] = [self._coerce_to_int(val)]
            else:
                # For other types of values, handle coercion based on type
                if isinstance(val, str):
                    val = val.replace('\n', '').replace('\r', '')
                    new_row[col] = val  # Ensure it's a string
                else:
                    new_row[col] = self._coerce_to_int(val)  # Coerce to integer if possible

        # serialize result to dataset
        self.synthetic_dataset = pd.concat([self.synthetic_dataset, pd.DataFrame([new_row])], ignore_index=True)
        self.synthetic_asdict.append(response_dict)

    def _coerce_to_int(self, value):
        """
        Helper function to attempt to coerce a value into an integer.
        Returns the value if coercion is not possible.
        """
        try:
            # If the value is already an integer, return it directly
            if isinstance(value, int):
                return value
            # If it's a string that can be converted to an integer, try to do that
            return int(value)
        except (ValueError, TypeError):
            # Return the value as is if it cannot be coerced into an integer
            return value

    def write_results(self, RUN_FOLDER, date_str) -> bool | str:
        # write csv and json this may fail
        write_success = True
        try:
            self.synthetic_dataset.to_csv(Path(RUN_FOLDER / "_".join((date_str, "results.csv"))))
        except Exception as e:
            write_success = e

        try:
            with open(Path(RUN_FOLDER) / "_".join((date_str, "results.json")), "w") as f:
                json.dump(self.synthetic_asdict, f, indent=4)
        except Exception as e:
            write_success = e

        return write_success

def main():
    pass

if __name__ == "__main__":
    main()