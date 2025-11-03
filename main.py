import requests
import time
import csv
import urllib.parse

APP_ID = "2807960"  # Battlefield 6
OUT_CSV = f"steam_reviews_{APP_ID}_ptbr.csv"
TARGET = 1000        # metas de reviews em português
BATCH = 100          # num por página
SLEEP_BETWEEN = 0.8  # segundos

def fetch_batch(app_id, cursor="*", num_per_page=100, language="brazilian", filter_type="all", purchase_type="all"):
    url = f"https://store.steampowered.com/appreviews/{app_id}"
    params = {
        "json": 1,
        "num_per_page": num_per_page,
        "cursor": cursor,
        "language": language,
        "filter": filter_type,
        "purchase_type": purchase_type
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def main():
    collected = 0
    cursor = "*"
    fieldnames = ["recommendationid","author_steamid","language","review","timestamp_created","voted_up","votes_up"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        while collected < TARGET:
            data = fetch_batch(APP_ID, cursor=urllib.parse.quote(cursor, safe=''), num_per_page=BATCH)
            reviews = data.get("reviews", [])
            if not reviews:
                print("Sem mais reviews ou idioma não retornado.")
                break
            for r in reviews:
                if r.get("language", "").startswith("braz"):
                    row = {
                        "recommendationid": r.get("recommendationid"),
                        "author_steamid": r.get("author", {}).get("steamid"),
                        "language": r.get("language"),
                        "review": r.get("review"),
                        "timestamp_created": r.get("timestamp_created"),
                        "voted_up": r.get("voted_up"),
                        "votes_up": r.get("votes_up"),
                    }
                    writer.writerow(row)
                    collected += 1
                    if collected >= TARGET:
                        break
            cursor = data.get("cursor", "")
            print(f"Coletados: {collected} – próximo cursor: {cursor[:60]}")
            time.sleep(SLEEP_BETWEEN)

    print("Finalizado — arquivo:", OUT_CSV)

if __name__ == "__main__":
    main()
