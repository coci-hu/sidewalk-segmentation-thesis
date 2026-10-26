"""Read-only text extraction for this legacy Word .doc; no Office automation.

Follows MS-DOC Retrieving Text and FibRgFcLcb97 / PlcPcd definitions.
Extracts main story only; does not claim to reproduce document layout.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path


def extract(path):
    raw = path.read_bytes()
    if raw[:8] != bytes.fromhex("D0CF11E0A1B11AE1"):
        raise ValueError("Not an OLE compound document")
    def u16(data, offset):
        return struct.unpack_from("<H", data, offset)[0]
    def u32(data, offset):
        return struct.unpack_from("<I", data, offset)[0]
    sector_size = 1 << u16(raw, 30)
    mini_size = 1 << u16(raw, 32)
    def sector(index):
        offset = (index + 1) * sector_size
        value = raw[offset:offset + sector_size]
        # This file's final Table sector omits unused padding; stream length
        # checks below still require every declared meaningful byte.
        if not value:
            raise ValueError("Sector outside file")
        return value
    fat_ids = list(struct.unpack_from("<109I", raw, 76))
    next_difat = u32(raw, 68)
    for _ in range(u32(raw, 72)):
        values = struct.unpack("<" + "I" * (sector_size // 4), sector(next_difat))
        fat_ids.extend(values[:-1])
        next_difat = values[-1]
    fat_ids = [i for i in fat_ids if i < 0xFFFFFFFA][:u32(raw, 44)]
    fat = []
    for index in fat_ids:
        fat.extend(struct.unpack("<" + "I" * (sector_size // 4), sector(index)))
    def chain(start, allocation, reader, size=None):
        result, visited = [], set()
        while start < 0xFFFFFFFA:
            if start in visited or start >= len(allocation):
                raise ValueError("Invalid allocation chain")
            visited.add(start)
            result.append(reader(start))
            start = allocation[start]
        joined = b"".join(result)
        if size is not None and len(joined) < size:
            raise ValueError("Stream truncated")
        return joined if size is None else joined[:size]
    directory = chain(u32(raw, 48), fat, sector)
    entries = {}
    for offset in range(0, len(directory), 128):
        entry = directory[offset:offset + 128]
        name_bytes = u16(entry, 64)
        kind = entry[66]
        if not kind:
            continue
        name = entry[:name_bytes - 2].decode("utf-16le")
        entries[name] = {"kind": kind, "start": u32(entry, 116), "size": struct.unpack_from("<Q", entry, 120)[0]}
    root = next(e for e in entries.values() if e["kind"] == 5)
    mini_stream = chain(root["start"], fat, sector, root["size"])
    mini_fat_bytes = chain(u32(raw, 60), fat, sector) if u32(raw, 64) else b""
    mini_fat = list(struct.unpack("<" + "I" * (len(mini_fat_bytes) // 4), mini_fat_bytes))
    def stream(name):
        entry = entries[name]
        if entry["size"] < u32(raw, 56):
            return chain(entry["start"], mini_fat,
                         lambda i: mini_stream[i * mini_size:(i + 1) * mini_size], entry["size"])
        return chain(entry["start"], fat, sector, entry["size"])
    word = stream("WordDocument")
    flags = u16(word, 10)
    if flags & 0x0100:
        raise ValueError("Encrypted documents are unsupported")
    table_name = "1Table" if flags & 0x0200 else "0Table"
    table = stream(table_name)
    words_count = u16(word, 32)
    long_count_offset = 34 + words_count * 2
    long_count = u16(word, long_count_offset)
    longs_offset = long_count_offset + 2
    main_count = u32(word, longs_offset + 3 * 4)
    fc_count_offset = longs_offset + long_count * 4
    fc_array_offset = fc_count_offset + 2
    if u16(word, fc_count_offset) < 34:
        raise ValueError("Missing CLX entry")
    clx_offset, clx_size = struct.unpack_from("<II", word, fc_array_offset + 33 * 8)
    clx = table[clx_offset:clx_offset + clx_size]
    position = 0
    while clx[position] == 1:
        position += 3 + u16(clx, position + 1)
    if clx[position] != 2:
        raise ValueError("Missing piece table")
    plc_size = u32(clx, position + 1)
    plc = clx[position + 5:position + 5 + plc_size]
    if (plc_size - 4) % 12:
        raise ValueError("Invalid piece table size")
    pieces = (plc_size - 4) // 12
    cps = struct.unpack_from("<" + "I" * (pieces + 1), plc, 0)
    if cps[0] != 0 or cps[-1] < main_count:
        raise ValueError("Main-story range missing")
    parts, units = [], 0
    for index in range(pieces):
        begin, end = cps[index], min(cps[index + 1], main_count)
        if begin >= main_count:
            break
        if end <= begin:
            raise ValueError("Invalid character range")
        encoded = u32(plc, 4 * (pieces + 1) + 8 * index + 2)
        compressed = bool(encoded & 0x40000000)
        offset = encoded & 0x3FFFFFFF
        if compressed:
            data = word[offset // 2:offset // 2 + end - begin]
            text = data.decode("cp1252", errors="replace")
        else:
            data = word[offset:offset + 2 * (end - begin)]
            text = data.decode("utf-16le")
        parts.append(text)
        units += end - begin
    if units != main_count:
        raise ValueError("Character coverage mismatch")
    return "".join(parts), {"file_sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
        "main_story_utf16_units": main_count, "piece_count": pieces, "table_stream": table_name,
        "main_story_complete": True, "layout_or_embedded_objects_extracted": False,
        "format_source": "https://learn.microsoft.com/en-us/openspecs/office_file_formats/ms-doc/01d5d8c4-cf9c-4ef9-80fd-439e763cfe01"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    root = Path(__file__).resolve().parent
    if not output.is_relative_to(root) or output.exists():
        raise ValueError("Use new experiment output")
    text, metadata = extract(args.input)
    output.mkdir(parents=True)
    with (output / "thought_document_extracted.txt").open("x", encoding="utf-8") as f:
        f.write(text.replace("\r", "\n").replace("\x07", "\t"))
    with (output / "document_read_metadata.json").open("x", encoding="utf-8") as f:
        json.dump({"source_file": str(args.input), **metadata, "read_only": True}, f, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
