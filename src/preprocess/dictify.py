"""This script reads a directory into a dictionary.""" 
import os
from pathlib import Path
from preprocess.utils import load_hpcp
from tqdm import tqdm
from loguru import logger



class Dictify:
    def __init__(self, directory: str|Path) -> None:
        """Dictify the directory
        
        Args:
            directory (str|Path): The directory to dictify

        Example:
            dictify = Dictify("data_path")
            data = dictify.get_data()
        """
        self.directory = directory
        self.files: list = []
        self.data: dict = {}
        self.max_length: int = 0

        self.get_file_list()
        logger.info(f"Found {len(self.files)} files.")

        self.dictify()
    
    def get_file_list(self):
        """Get list with all .h5 files' paths"""

        self.files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(self.directory) for f in filenames if
                      os.path.splitext(f)[1] == '.h5']

    def dictify(self) -> None:
        """Turns a directory into a dictionary."""
        for file in tqdm(self.files):
            _file = os.path.basename(file)
            _dir = os.path.basename(os.path.dirname(file))

            _hpcp = load_hpcp(file)
            self.max_length = max(self.max_length, len(_hpcp)) # check for max length

            if _dir in self.data:
                self.data[_dir][_file] = _hpcp
            else:   
                self.data[_dir] = {_file: _hpcp}

        return
    
    def get_data(self) -> dict:
        return self.data
    
    def get_max_length(self) -> int:
        return self.max_length
    
    def get_file_count(self) -> int:
        return len(self.files)

