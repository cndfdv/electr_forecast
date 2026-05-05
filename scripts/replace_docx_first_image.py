from __future__ import annotations

import argparse
import re
import shutil
import struct
import tempfile
import zipfile
from pathlib import Path


EMU_PER_INCH = 914400


def png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG file")
    return struct.unpack(">II", data[16:24])


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace the first embedded image in a DOCX file.")
    parser.add_argument("--docx", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--width-in", type=float, default=3.28)
    args = parser.parse_args()

    docx_path = args.docx
    image_path = args.image
    temp_dir = Path(tempfile.mkdtemp(prefix="docx_first_image_"))

    try:
        with zipfile.ZipFile(docx_path) as archive:
            archive.extractall(temp_dir)

        xml_path = temp_dir / "word" / "document.xml"
        rels_path = temp_dir / "word" / "_rels" / "document.xml.rels"
        xml = xml_path.read_text(encoding="utf-8")
        rels = rels_path.read_text(encoding="utf-8")

        drawing_match = re.search(r"<w:drawing>.*?r:embed=\"([^\"]+)\".*?</w:drawing>", xml, flags=re.S)
        if drawing_match is None:
            raise RuntimeError("No embedded drawing was found in document.xml")

        relation_id = drawing_match.group(1)
        relation_match = re.search(
            rf"<Relationship[^>]+Id=\"{re.escape(relation_id)}\"[^>]+Target=\"([^\"]+)\"",
            rels,
        )
        if relation_match is None:
            raise RuntimeError(f"No relationship target found for {relation_id}")

        target = relation_match.group(1)
        media_path = (temp_dir / "word" / target).resolve()
        word_dir = (temp_dir / "word").resolve()
        if not str(media_path).startswith(str(word_dir)):
            raise RuntimeError(f"Unexpected image target outside word/: {target}")

        media_path.write_bytes(image_path.read_bytes())

        image_width, image_height = png_size(image_path)
        cx = int(args.width_in * EMU_PER_INCH)
        cy = int(cx * image_height / image_width)

        first_drawing = xml[drawing_match.start() : drawing_match.end()]
        first_drawing = re.sub(
            r"<wp:extent cx=\"\d+\" cy=\"\d+\"/>",
            f'<wp:extent cx="{cx}" cy="{cy}"/>',
            first_drawing,
            count=1,
        )
        first_drawing = re.sub(
            r"<a:ext cx=\"\d+\" cy=\"\d+\"/>",
            f'<a:ext cx="{cx}" cy="{cy}"/>',
            first_drawing,
            count=1,
        )
        xml = xml[: drawing_match.start()] + first_drawing + xml[drawing_match.end() :]
        xml_path.write_text(xml, encoding="utf-8")

        backup_path = docx_path.with_suffix(".docx.tmpbak")
        shutil.copyfile(docx_path, backup_path)
        with zipfile.ZipFile(docx_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in temp_dir.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(temp_dir).as_posix())
        backup_path.unlink()

        print(
            f"replaced {target} via {relation_id}; "
            f"image={image_width}x{image_height}; extent={cx}x{cy}"
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
