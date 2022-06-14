import json
import yaml


class ConvertJson:
    @staticmethod
    def msg_to_json(msg):
        # Convert a ROS message to JSON format
        y = yaml.load(str(msg), Loader=yaml.SafeLoader)
        return json.dumps(y, indent=11)