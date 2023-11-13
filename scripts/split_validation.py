import torch
import torch.utils.data as data
import json


def save_split(train, validation, name):
    with open(f"train_{name}.jsonl", "w") as t:
        t.writelines(train)
    with open(f"validation_{name}.jsonl", "w") as t:
        t.writelines(validation)


def load_datasets(ds):
    d = []
    for dataset in ds:
        with open(dataset, "r") as f:
            if "jsonl" in dataset:
                lines = f.readlines()
                d += lines
            else:
                data = f.read()
                data = json.loads(data)
                d += [json.dumps(js) + "\n" for js in data]

    return d


seed = 42
ds = ["your-dataset.jsonl"]
dataset = load_datasets(ds)
generator = torch.Generator().manual_seed(seed)
train, validation = data.random_split(dataset, [0.8, 0.2], generator)  # this shuffles

save_split(train, validation, "dataset")
