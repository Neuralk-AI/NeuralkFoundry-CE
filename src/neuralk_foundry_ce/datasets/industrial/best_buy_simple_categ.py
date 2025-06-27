from dataclasses import dataclass
import pandas as pd

from ..base import DownloadDataConfig


@dataclass
class DataConfig(DownloadDataConfig):
    """
    Configuration for the Best Buy product categorization dataset.
    
    This dataset is used for product categorization tasks where the goal is to
    classify products into predefined categories based on their features.
    """
    name: str = "best_buy_simple_categ"
    task: str = "classification"
    target: str = "type"
    file_name: str = 'data.parquet'

    def download_data(self, dataset_dir):
        """
        Download and preprocess the Best Buy dataset.
        
        Downloads the JSON file from Best Buy's open dataset repository,
        filters for specific product categories (HardGood, Game, Software),
        and saves the processed data as a Parquet file.
        
        Parameters
        ----------
        dataset_dir : Path
            Directory path where the processed dataset should be saved.
        """
        ds_url = 'https://raw.githubusercontent.com/BestBuyAPIs/open-data-set/refs/heads/master/products.json'
        df = pd.read_json(ds_url)[['name', 'type', 'price', 'manufacturer']]
        df = df[df.type.isin(['HardGood', 'Game', 'Software'])]
        df = df.reset_index(drop=True)
        df.to_parquet(dataset_dir / self.file_name)
