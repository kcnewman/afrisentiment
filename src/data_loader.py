import os
from dataset import load_data, Dataset, DatasetDict
from typing import Union


def fetch_dataset(
    name: str,
    split: Union[STR, None] = "train",
    save_format: str = "csv",
    save_dir: str = "../data/raw",
):
    """
    Load dataset from Hugging Face.

    Args:
        name (str): Dataset name
        split (str or None): train or test split. If None, returns full DatasetDict.
        save_format (str): dfault = 'csv'
        save_dir (str): Save location.

    Returns:
        Dataset
    """
    try:
        if split is None:
            dataset = load_data(name)
        else:
            dataset = load_data(name, split=split)
        os.makedirs(save_dir, exist_ok=True)

        if isinstance(dataset, Dataset):
            save_path = os.path.join(save_dir, f"{name}_{split}.{save_format}")
            if save_format == "csv":
                dataset.to_csv(save_path)
            elif save_format == "json":
                dataset.to_json(save_path)
            else:
                raise ValueError("Unsupported save format. Use 'json' or 'csv'.")
        elif isinstance(dataset, DatasetDict):
            for k in dataset:
                subset = dataset[k]
                save_path = os.path.join(save_dir, f"{name}_{k}.{save_format}")
                if save_format == "csv":
                    dataset.to_csv(save_path)
                elif save_format == "json":
                    dataset.to_json(save_path)
        return dataset
    except Exception as e:
        raise RuntimeError(
            f"Failed to load or save dataset '{name} with split '{split}': {e}"
        )
