import yaml


with open("./configs/models.yaml", "r") as stream:
    try:
        documents = yaml.safe_load(stream)
        for model in documents:
            print(f"sdmodel,{model}")
    except yaml.YAMLError as exc:
        print(exc)