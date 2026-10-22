#!/usr/bin/env python3
"""Download the public G1 archive with resumable, checked HTTP byte ranges."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile

URL = ("https://drive.usercontent.google.com/download?"
       "id=1g6OzxXGJczZ4klVqKCpTzy-orKkllCr4&export=download&confirm=t")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("InterAct/downloads/OMOMO_retarget_aug_v4.zip"))
    parser.add_argument("--prefix", type=Path, help="Optional contiguous partial download to reuse")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.is_file():
        with zipfile.ZipFile(args.output) as archive:
            print("Archive already downloaded: {} ({} members)".format(args.output, len(archive.infolist())))
        return
    probe = urllib.request.Request(URL, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(probe, timeout=60) as response:
        if response.status != 206:
            raise RuntimeError("Server does not support byte ranges")
        total = int(response.headers["Content-Range"].rsplit("/", 1)[1])
    prefix_size = args.prefix.stat().st_size if args.prefix else 0
    if prefix_size > total:
        raise ValueError("Partial download exceeds archive size")
    chunk_size = 64 * 1024 * 1024
    directory = args.output.with_suffix(".chunks")
    directory.mkdir(parents=True, exist_ok=True)
    ranges = [(start, min(start + chunk_size, total) - 1)
              for start in range(prefix_size, total, chunk_size)]
    print("{} bytes total; reusing {} bytes; {} chunks".format(total, prefix_size, len(ranges)), flush=True)

    def download(bounds):
        start, end = bounds
        path = directory / ("{}-{}.bin".format(start, end))
        expected = end - start + 1
        if path.exists() and path.stat().st_size == expected:
            return path
        temporary = path.with_suffix(".partial")
        for attempt in range(5):
            try:
                request = urllib.request.Request(URL, headers={"Range": "bytes={}-{}".format(start, end)})
                with urllib.request.urlopen(request, timeout=60) as response:
                    content_range = "bytes {}-{}/{}".format(start, end, total)
                    if response.status != 206 or response.headers.get("Content-Range") != content_range:
                        raise RuntimeError("Unexpected Content-Range: {}".format(response.headers.get("Content-Range")))
                    with temporary.open("wb") as stream:
                        shutil.copyfileobj(response, stream, length=1024 * 1024)
                if temporary.stat().st_size != expected:
                    raise RuntimeError("Incomplete chunk {}".format(start))
                temporary.replace(path)
                print("Downloaded bytes {}-{}".format(start, end), flush=True)
                return path
            except Exception as error:
                print("Chunk {} attempt {}: {}".format(start, attempt + 1, error), flush=True)
                if attempt == 4:
                    raise
                time.sleep(2 * (attempt + 1))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        chunks = list(pool.map(download, ranges))
    assembled = args.output.with_suffix(".assembling")
    with assembled.open("wb") as output:
        if args.prefix:
            with args.prefix.open("rb") as source:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
        for chunk in chunks:
            with chunk.open("rb") as source:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
    if assembled.stat().st_size != total:
        raise RuntimeError("Assembled archive size does not match server")
    with zipfile.ZipFile(assembled) as archive:
        print("ZIP central directory: {} members".format(len(archive.infolist())), flush=True)
    assembled.replace(args.output)
    print("Downloaded {}".format(args.output), flush=True)


if __name__ == "__main__":
    main()
