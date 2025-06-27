from dataclasses import dataclass
import pandas as pd

from ..base import DownloadDataConfig


@dataclass
class DataConfig(DownloadDataConfig):
    """
    Configuration for the Abt-Buy entity linkage dataset.
    
    This dataset is used for entity resolution tasks where the goal is to
    identify matching products between two different e-commerce catalogs.
    """
    name: str = "abt_buy"
    task: str = "linkage"
    target: str = "group_id"
    file_name: str = 'data.parquet'

    def download_data(self, target_path):
        """
        Download and preprocess the Abt-Buy dataset.
        
        Downloads the raw CSV files from the RLTK experimentation repository,
        performs basic preprocessing including manufacturer extraction from
        product names, and creates entity linkage labels.
        
        Parameters
        ----------
        target_path : Path
            Directory path where the processed dataset should be saved.
        """
        ds_url = 'https://raw.githubusercontent.com/usc-isi-i2/rltk-experimentation/refs/heads/master/datasets'
        abt = pd.read_csv(ds_url + '/Abt-Buy/Abt.csv', encoding='unicode_escape', index_col='id')
        buy = pd.read_csv(ds_url + '/Abt-Buy/Buy.csv', encoding='unicode_escape', index_col='id')
        y = pd.read_csv(ds_url + '/Abt-Buy/abt_buy_perfectMapping.csv', encoding='unicode_escape')
        abt['manufacturer'] = abt['name'].str.split(' ').str[0]
        dataset = pd.concat([abt, buy])
        dataset['group_id'] = np.arange(dataset.shape[0])
        dataset.loc[y['idBuy'], 'group_id'] = dataset.loc[y['idAbt'], 'group_id'].values
        dataset.reset_index(drop=True, inplace=True)
        dataset.to_parquet(target_path)
