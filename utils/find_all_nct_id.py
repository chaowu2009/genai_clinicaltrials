
## Parallel + Resume-Safe Python Script


import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
OUTPUT_FILE = "nct_ids.txt"

START_ID = 1

# Parallelism / throttling
WORKERS = 20
BATCH_SIZE = 2000
REQUEST_TIMEOUT = 10
BATCH_SLEEP_SECONDS = 0.1

# Stop condition
MAX_CONSECUTIVE_MISSES = 5000


def nct_id(num: int) -> str:
    return f"NCT{num:08d}"


def load_existing_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def study_exists(session: requests.Session, nct: str) -> bool:
    r = session.get(f"{BASE_URL}/{nct}", timeout=REQUEST_TIMEOUT)
    return r.status_code == 200


def append_ids(path: str, ids: list[str], lock: threading.Lock) -> None:
    if not ids:
        return
    ids.sort()
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            for x in ids:
                f.write(x + "\n")


def main():
    # ---- Load previously saved IDs ----
    saved_ids = load_existing_ids(OUTPUT_FILE)
    print(f"Loaded {len(saved_ids)} existing NCT IDs from {OUTPUT_FILE}")

    file_lock = threading.Lock()

    current = START_ID
    misses_at_top = 0
    highest_found = None

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=WORKERS,
        pool_maxsize=WORKERS
    )
    session.mount("https://", adapter)

    while misses_at_top < MAX_CONSECUTIVE_MISSES:
        batch_start = current
        batch_end = current + BATCH_SIZE - 1

        # ---- Skip IDs already in file ----
        batch = []
        for i in range(batch_start, batch_end + 1):
            nct = nct_id(i)
            if nct not in saved_ids:
                batch.append((i, nct))

        print(
            f"\nBatch {nct_id(batch_start)} → {nct_id(batch_end)} | "
            f"Skipping {BATCH_SIZE - len(batch)} already saved"
        )

        found = []
        found_numbers = set()

        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {
                executor.submit(study_exists, session, nct): (i, nct)
                for i, nct in batch
            }

            for future in as_completed(futures):
                i, nct = futures[future]
                try:
                    exists = future.result()
                except Exception as e:
                    print(f"ERROR {nct}: {e}")
                    continue

                if exists:
                    found.append(nct)
                    found_numbers.add(i)

        # ---- Persist results ----
        if found:
            append_ids(OUTPUT_FILE, found, file_lock)
            saved_ids.update(found)
            highest_found = max(highest_found or batch_start, max(found_numbers))
            print(f"Found {len(found)} | Highest so far: {nct_id(highest_found)}")
        else:
            print("Found 0")

        # ---- Update stop condition (top-end only) ----
        tail_misses = 0
        for i in range(batch_end, batch_start - 1, -1):
            if i in found_numbers:
                break
            if nct_id(i) not in saved_ids:
                tail_misses += 1

        if tail_misses == BATCH_SIZE:
            misses_at_top += BATCH_SIZE
        else:
            misses_at_top = tail_misses

        print(f"Misses at top: {misses_at_top}/{MAX_CONSECUTIVE_MISSES}")

        current += BATCH_SIZE
        time.sleep(BATCH_SLEEP_SECONDS)

    print("\nFinished scanning.")
    print(f"Last checked: {nct_id(current - 1)}")
    if highest_found:
        print(f"Highest found: {nct_id(highest_found)}")
    print(f"Total IDs saved: {len(saved_ids)}")


if __name__ == "__main__":
    main()
