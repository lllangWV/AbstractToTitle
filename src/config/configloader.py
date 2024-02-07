import yaml

class ConfigLoader:
    def __init__(self, config_file):
        self.config_file = config_file
        self.load_config()

    def load_config(self):
        with open(self.config_file, 'r') as file:
            config_data = yaml.safe_load(file)
            for key, value in config_data.items():
                setattr(self, key, value['value'])

    def __str__(self):
        return f"ConfigLoader with properties: {', '.join(f'{key}={value}' for key, value in self.__dict__.items() if key != 'config_file')}"


# # Usage
# config_loader = ConfigLoader('configuration.yaml')
# print(config_loader)