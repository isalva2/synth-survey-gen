from survey import AgentResponsePackage
from pathlib import Path
from dataclasses import asdict
import copy
import pandas as pd
import json

class ProcessSurveyResponse:
    def __init__(self, config_folder: str, batch_size: int, RUN_FOLDER: str, source: str, date_str: str):
        self.data_path = Path(config_folder) / "data"
        self.batch_size = batch_size
        self.n_batches = 0
        self.batches_written = 1
        self.RUN_FOLDER = RUN_FOLDER
        self.source = source
        self.date_str = date_str
        self._prepare_dataset()

    def _prepare_dataset(self):
        if self.source == "US":
            self.multiple_choice_cols = ["NOGOWHY2", "TRAVELDATAMODE", "DTYPE"]
            self.ground_truth_df = pd.read_csv(self.data_path / "person.csv", low_memory=False)
        else:
            self.multiple_choice_cols = None
            questions_df = pd.read_csv(self.data_path / "../questions.csv").T
            self.ground_truth_df = questions_df

        ground_truth_cols = self.ground_truth_df.columns
        self.synthetic_columns = ["agent_id", "serial_number", "agent_bio", "intro"]
        self.synthetic_columns.extend(ground_truth_cols)
        self.synthetic_dataset = pd.DataFrame(columns=self.synthetic_columns)
        self.batch_dataset = copy.deepcopy(self.synthetic_dataset)
        self.synthetic_asdict = []
        self.batch_asdict = []

    def serialize_response(self, agent_response: AgentResponsePackage):
        response_dict = asdict(agent_response)
        response_cols = [col.lower() for col in response_dict["logic_flow"]]
        new_row = {}

        # get agent id and system_message
        new_row["agent_id"]       = agent_response.agent_id
        new_row["serial_number"]  = agent_response.serial_number
        new_row["agent_bio"]      = agent_response.agent_bio

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

        # batch dataset
        self.batch_dataset = pd.concat([self.batch_dataset, pd.DataFrame([new_row])], ignore_index=True)
        self.batch_asdict.append(response_dict)

        self._batch_write_results()

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
            self.synthetic_dataset.to_csv(Path(RUN_FOLDER) / "_".join((date_str, "results.csv")), index=False)
        except Exception as e:
            write_success = e

        try:
            with open(Path(RUN_FOLDER) / "_".join((date_str, "results.json")), "w") as f:
                json.dump(self.synthetic_asdict, f, indent=4)
        except Exception as e:
            write_success = e

        return write_success

    def _batch_write_results(self) -> None:
        # write csv and json this may fail
        if self.n_batches % self.batch_size == 0:
            try:
                self.batch_dataset.to_csv(Path(self.RUN_FOLDER) / f"batch_{self.batches_written}_{self.date_str}_results.csv", index=False)
            except Exception as e:
                print(e)

            try:
                with open(Path(self.RUN_FOLDER) / f"batch_{self.batches_written}_{self.date_str}_results.json", "w") as f:
                    json.dump(self.batch_asdict, f, indent=4)
            except Exception as e:
                print(e)

            # reset batch datasets
            self.batches_written += 1
            self.batch_dataset = self.batch_dataset[0:0]
            self.batch_asdict = []

        # iterate n_batches written
        self.n_batches += 1

def main():
    pass

if __name__ == "__main__":
    main()