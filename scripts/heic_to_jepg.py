from pathlib import Path
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

source_dir = Path("data/21_08_borad")
output_dir = Path("data/21_08_borad_jpg")
output_dir.mkdir(exist_ok=True)

for path in source_dir.glob("*.heic"):
    image = Image.open(path).convert("RGB")
    image.save(output_dir / f"{path.stem}.jpg", quality=95)