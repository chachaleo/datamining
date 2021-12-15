import ujson as json
import pandas as pd
path = "apple.ndjson"

records = map(json.loads, open((path),  encoding="utf-8", newline="\n" ))
df = pd.DataFrame.from_records(records, nrows=10)
