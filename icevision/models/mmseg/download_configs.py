__all__ = ["download_mmseg_configs"]

from icevision.imports import *
from icevision.utils import *

# VERSION = "v2.10.0"  # TODO: Update
VERSION = "v0.17.0"
BASE_URL = "https://codeload.github.com/Orbis-International/mmsegmentation_configs/zip/refs/tags"


def download_mmseg_configs() -> Path:

    save_dir = get_root_dir() / f"mmsegmentation_configs"

    mmseg_config_path = save_dir / f"mmsegmentation_configs-{VERSION[1:]}/configs"
    download_path = save_dir / f"{VERSION}.zip"

    if mmseg_config_path.exists():
        logger.info(
            f"The mmseg config folder already exists. No need to downloaded it. Path : {mmseg_config_path}"
        )
    elif download_path.exists():
        # The zip file was downloaded by not extracted yet
        # Extract zip file
        logger.info(f"Extracting the {VERSION}.zip file.")
        save_dir = Path(download_path).parent
        shutil.unpack_archive(filename=str(download_path), extract_dir=str(save_dir))
    else:
        save_dir.mkdir(parents=True, exist_ok=True)

        download_path = save_dir / f"{VERSION}.zip"
        if not download_path.exists():
            logger.info("Downloading mmseg configs")
            download_and_extract(f"{BASE_URL}/{VERSION}", download_path)

    return mmseg_config_path
