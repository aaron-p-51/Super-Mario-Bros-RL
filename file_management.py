import os
import time
import datetime


class File_Management:
    def __init__(
        self,
        best_rewards_filepath="models\\best_rewards",
        savepoints_filepath="models\\savepoints",
        checkpoints_filepath="models\\checkpoints",
    ):
        self.best_rewards_filepath = best_rewards_filepath
        self.savepoints_filepath = savepoints_filepath
        self.checkpoints_filepath = checkpoints_filepath

    def get_savepoint_files(self):
        files = [
            f
            for f in os.listdir(self.savepoints_filepath)
            if os.path.isfile(os.path.join(self.savepoints_filepath, f))
        ]
        return files

    def get_first_file(self, path: str):
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        if not files:
            return None
        files.sort(key=lambda f: os.path.getmtime(os.path.join(path, f)))
        return os.path.join(path, files[0])

    def get_current_date_time_string(self):
        return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

    def get_savepoint_path(self):
        savepoint_file_name = "savepoint_" + "_" + self.get_current_date_time_string()
        return os.path.join(self.savepoints_filepath, savepoint_file_name)

    def get_checkpoint_path(self):
        checkpoint_file_name = "checkpoint_" + self.get_current_date_time_string()
        return os.path.joing(self.checkpoints_filepath, checkpoint_file_name)
