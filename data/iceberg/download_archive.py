from pathlib import Path
import pandas as pd
import requests
import time


ICEBERG_DIR = Path(__file__).parent

MANIFEST_FILE = ICEBERG_DIR / "iceberg_download_manifest.csv"
OUTPUT_DIR = ICEBERG_DIR / "raw" / "historical"


def main():
    print("=== USNIC Antarctic Iceberg Downloader ===")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_FILE)

    print(f"Files in manifest: {len(manifest)}")
    print(f"Download directory: {OUTPUT_DIR}")
    print()

    session = requests.Session()

    successful = 0
    skipped = 0
    failed = 0

    for index, row in manifest.iterrows():

        filename = row["filename"]
        url = row["download_url"]

        output_file = OUTPUT_DIR / filename

        if output_file.exists() and output_file.stat().st_size > 0:
            print(
                f"[{index + 1}/{len(manifest)}] "
                f"SKIP {filename} (already exists)"
            )
            skipped += 1
            continue

        print(
            f"[{index + 1}/{len(manifest)}] "
            f"Downloading {filename}..."
        )

        try:
            response = session.get(
                url,
                timeout=60
            )

            response.raise_for_status()

            if not response.content:
                raise RuntimeError("Empty response")

            output_file.write_bytes(response.content)

            print(
                f"    OK - "
                f"{len(response.content) / 1024:.1f} KB"
            )

            successful += 1

            time.sleep(0.3)

        except Exception as error:
            print(f"    FAILED: {error}")
            failed += 1

    print()
    print("=== Download Summary ===")
    print(f"Successful: {successful}")
    print(f"Skipped:    {skipped}")
    print(f"Failed:     {failed}")
    print(f"Total:      {len(manifest)}")
    print()
    print("Files are in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()