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
    parser = argparse.ArgumentParser(description="Replace embedded DOCX images by drawing order.")
    parser.add_argument("--docx", type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", required=True)
    args = parser.parse_args()

    docx_path = args.docx
    images = args.image
    temp_dir = Path(tempfile.mkdtemp(prefix="docx_images_"))

    try:
        with zipfile.ZipFile(docx_path) as archive:
            archive.extractall(temp_dir)

        xml_path = temp_dir / "word" / "document.xml"
        rels_path = temp_dir / "word" / "_rels" / "document.xml.rels"
        xml = xml_path.read_text(encoding="utf-8")
        rels = rels_path.read_text(encoding="utf-8")

        drawings = list(re.finditer(r"<w:drawing>.*?r:embed=\"([^\"]+)\".*?</w:drawing>", xml, flags=re.S))
        if len(images) > len(drawings):
            raise RuntimeError(f"DOCX has only {len(drawings)} drawings, got {len(images)} images")

        replacements: list[tuple[int, int, str]] = []
        word_dir = (temp_dir / "word").resolve()
        new_xml = xml
        offset = 0

        for drawing, image_path in zip(drawings, images):
            relation_id = drawing.group(1)
            relation_match = re.search(
                rf"<Relationship[^>]+Id=\"{re.escape(relation_id)}\"[^>]+Target=\"([^\"]+)\"",
                rels,
            )
            if relation_match is None:
                raise RuntimeError(f"No relationship target found for {relation_id}")

            target = relation_match.group(1)
            media_path = (temp_dir / "word" / target).resolve()
            if not str(media_path).startswith(str(word_dir)):
                raise RuntimeError(f"Unexpected image target outside word/: {target}")

            media_path.write_bytes(image_path.read_bytes())

            image_width, image_height = png_size(image_path)
            drawing_xml = new_xml[drawing.start() + offset : drawing.end() + offset]
            extent_match = re.search(r"<wp:extent cx=\"(\d+)\" cy=\"(\d+)\"/>", drawing_xml)
            if extent_match is not None:
                cx = int(extent_match.group(1))
                cy = int(cx * image_height / image_width)
                drawing_xml = re.sub(
                    r"<wp:extent cx=\"\d+\" cy=\"\d+\"/>",
                    f'<wp:extent cx="{cx}" cy="{cy}"/>',
                    drawing_xml,
                    count=1,
                )
                drawing_xml = re.sub(
                    r"<a:ext cx=\"\d+\" cy=\"\d+\"/>",
                    f'<a:ext cx="{cx}" cy="{cy}"/>',
                    drawing_xml,
                    count=1,
                )
            else:
                cx = cy = 0

            start = drawing.start() + offset
            end = drawing.end() + offset
            new_xml = new_xml[:start] + drawing_xml + new_xml[end:]
            offset += len(drawing_xml) - (drawing.end() - drawing.start())
            replacements.append((cx, cy, target))

        xml_path.write_text(new_xml, encoding="utf-8")

        backup_path = docx_path.with_suffix(".docx.tmpbak")
        shutil.copyfile(docx_path, backup_path)
        with zipfile.ZipFile(docx_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in temp_dir.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(temp_dir).as_posix())
        backup_path.unlink()

        for index, (cx, cy, target) in enumerate(replacements, start=1):
            print(f"{index}: {target}; extent={cx}x{cy}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
