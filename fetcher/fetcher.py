import kagglehub as khub
import shutil
import logging
import os

# Define log file path
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "fetch_data.log")

# Ensure the log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging to write to a file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),  # Log to file
        logging.StreamHandler()
    ]
)

logging.info("Logging initialized. Writing logs to fetch_data.log")

# Constants
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
KAGGLE_SOURCE: str = "awsaf49/cbis-ddsm-breast-cancer-image-dataset"
DIR_NAME: str = KAGGLE_SOURCE.split("/")[-1]

class FetchData:
    """Handles downloading, moving, and cleaning up a Kaggle dataset."""

    def __init__(self, source: str = KAGGLE_SOURCE) -> None:
        """Initialize the FetchData class."""
        self.source_path: str = source
        self.dir_name: str = self.source_path.split("/")[-1]
        self.dest_path = os.path.join(BASE_DIR.split("/fetcher")[0], "kaggle","data", self.dir_name)
        print(self.dest_path)

    def download(self) -> str:
        """
        Download the dataset from Kaggle.

        Returns:
            str: The path where the dataset is stored.
        """
        try:
            logging.info(f"Downloading dataset from Kaggle: {self.source_path}")
            path = khub.dataset_download(self.source_path)
            logging.info(f"Dataset downloaded successfully: {path}")
            return path
        except Exception as e:
            logging.error(f"Error downloading dataset: {e}")
            raise

    def move(self, cache_path: str, dest_path: str) -> bool:
        """
        Move the downloaded dataset to the destination path.

        Args:
            cache_path (str): Path of the downloaded dataset.
            dest_path (str): Destination directory.

        Returns:
            bool: True if moved successfully, False otherwise.
        """
        try:
            logging.info(f"Moving dataset from {cache_path} to {dest_path}")
            shutil.move(cache_path, dest_path)
            logging.info("Dataset moved successfully.")
            return True
        except Exception as e:
            logging.error(f"Error moving dataset: {e}")
            cleaning: bool = self.cleanup(cache_path)
            return False

    def cleanup(self, cache_path: str) -> bool:
        """
        Remove the cached dataset files.

        Args:
            cache_path (str): Path to the downloaded dataset.

        Returns:
            bool: True if cleanup was successful, False otherwise.
        """
        try:
            logging.info(f"Cleaning up dataset cache: {cache_path}")
            shutil.rmtree(cache_path)
            logging.info("Cleanup successful.")
            return True
        except Exception as e:
            logging.error(f"Error cleaning up dataset: {e}")
            return False


    def fetch_dataset(self) -> tuple[bool, bool]:
        """
        Fetch, move, and clean up the Kaggle dataset.

        Returns:
            tuple[bool, bool]: Status of move and cleanup operations.
        """
        cache_path = self.download()
        move_status = fetcher.move(cache_path, self.dest_path)
        return move_status

if __name__ == "__main__":
    dataset: str = input("Enter the Kaggle dataset source: ")
    # Initialize the FetchData class
    fetcher = FetchData(source=dataset)

    # Fetch, move, and clean up the dataset
    move_status = fetcher.fetch_dataset()

    # Print the results
    print(f"Move status: {move_status}")
    logging.info(f"Move status: {move_status}")