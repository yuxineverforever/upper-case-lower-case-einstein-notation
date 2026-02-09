import sys

from timeit import default_timer as timer
import psycopg2 as psy

iterations = 1

if len(sys.argv) != 3:
    print(f"Usage: {__file__} [INSERT SQL filepath] [SELECT SQL filepath]")
    exit(-1)

with open(sys.argv[1], "r") as f:
    insert = f.read()

with open(sys.argv[2], "r") as f:
    select = f.read()

print(f"len(query): {len(insert)}")

conn = psy.connect(
    user="postgres", password="newpassword", database="postgres", host="localhost"
)
conn.set_isolation_level(psy.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
cur = conn.cursor()

tic = timer()
cur.execute(insert)
toc = timer()
print(f"insert finished in {(toc - tic) / iterations:.3f} s")

for i in range(iterations):
    print(f"{i + 1}/{iterations}")
    cur.execute(select)

tic_total = timer()
for i in range(iterations):
    print(f"{i + 1}/{iterations}")
    tic = timer()
    cur.execute(select)
    toc = timer()
    print(f"select finished in {(toc - tic):.3f} s")
toc_total = timer()
print(f"average: {(toc_total - tic_total) / iterations:.3f} s")

counter = 0
for record in cur:
    print(record)
    counter += 1
    if counter == 30:
        break
print(f"{counter} rows in result")

cur.close()
conn.close()
