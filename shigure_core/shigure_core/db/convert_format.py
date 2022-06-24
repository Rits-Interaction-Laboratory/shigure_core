import rosidl_runtime_py
import json
import os


class ConvertMsg:
    @staticmethod
    def message_to_json(msg_data):
        dict_data = rosidl_runtime_py.message_to_ordereddict(msg_data)
        json_data = json.dumps(dict_data)
        return json_data

    @staticmethod
    def write_msg_as_json(save_dir_path: str, save_file_name: str, msg_data):
        dict_data = rosidl_runtime_py.message_to_ordereddict(msg_data)
        with open(os.path.join(save_dir_path, save_file_name) + '.json', 'w') as f:
            json.dump(dict_data, f, indent=2)
