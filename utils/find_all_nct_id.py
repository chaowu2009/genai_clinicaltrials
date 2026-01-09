import requests
import time

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
OUTPUT_FILE = "nct_ids.txt"

START_ID = 1
MAX_CONSECUTIVE_MISSES = 2000   # stop after 2000 missing IDs in a row
SLEEP_SECONDS = 0.2             # be polite to the API

def nct_id(num):
    return f"NCT{num:08d}"

def study_exists(nct):
    url = f"{BASE_URL}/{nct}"
    r = requests.get(url, timeout=10)
    return r.status_code == 200

def main():
    found = []
    misses = 0
    current = START_ID

    while misses < MAX_CONSECUTIVE_MISSES:
        nct = nct_id(current)

        try:
            if study_exists(nct):
                print(f"FOUND: {nct}")
                found.append(nct)
                misses = 0
            else:
                misses += 1
        except Exception as e:
            print(f"Error on {nct}: {e}")

        current += 1
        time.sleep(SLEEP_SECONDS)

    print(f"\nStopping after {misses} consecutive missing IDs.")
    print(f"Highest checked: {nct_id(current-1)}")
    print(f"Total found: {len(found)}")

    with open(OUTPUT_FILE, "w") as f:
        for nct in found:
            f.write(nct + "\n")

    print(f"NCT IDs saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
