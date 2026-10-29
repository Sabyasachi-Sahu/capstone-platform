# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 4: load the capstone lake as Bronze, Silver and Gold tables
# MAGIC Synthetic data only. The model never runs here: Databricks Free cannot
# MAGIC reach Ollama on the laptop, so only tables move.
# MAGIC 1. Run cell 1. It creates the schemas and a volume.
# MAGIC 2. Upload `data/lake/lake_upload.zip` into the volume (Catalog > workspace >
# MAGIC    capstone_landing > files > Upload to this volume).
# MAGIC 3. Run the remaining cells.
 
# COMMAND ----------
 
import json
import zipfile
 
CATALOG = "workspace"
LANDING = f"/Volumes/{CATALOG}/capstone_landing/files"
for schema in (
    "capstone_landing",
    "capstone_bronze",
    "capstone_silver",
    "capstone_gold",
):
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{schema}")  # noqa: F821
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.capstone_landing.files")  # noqa: F821
print("Now upload lake_upload.zip to", LANDING)
 
# COMMAND ----------
 
with zipfile.ZipFile(f"{LANDING}/lake_upload.zip") as zf:
    zf.extractall(f"{LANDING}/lake")
with open(f"{LANDING}/lake/manifest.json") as handle:
    manifest = json.load(handle)
print("Data end:", manifest["data_end"], "| built:", manifest["created"])
 
# COMMAND ----------
 
results = []
for layer, tables in manifest["layers"].items():
    for table, expected in tables.items():
        frame = spark.read.parquet(f"{LANDING}/lake/{layer}/{table}.parquet")  # noqa: F821
        name = f"{CATALOG}.capstone_{layer}.{table}"
        frame.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
            name
        )
        loaded = spark.table(name).count()  # noqa: F821
        results.append(
            (layer, table, expected, loaded, "OK" if loaded == expected else "MISMATCH")
        )
for row in results:
    print(
        f"{row[0]:7} {row[1]:20} expected {row[2]:>9,}  loaded {row[3]:>9,}  {row[4]}"
    )
bad = [r for r in results if r[4] != "OK"]
print(f"\n{len(results) - len(bad)} of {len(results)} tables reconcile")
if bad:
    raise RuntimeError("Row counts do not match the manifest")
 
# COMMAND ----------
 
# MAGIC %sql
# MAGIC SELECT d.label_drug, COUNT(DISTINCT r.person_id) AS people_readmitted
# MAGIC FROM workspace.capstone_silver.prescription r
# MAGIC JOIN workspace.capstone_silver.drug d ON d.drug_id = r.drug_id
# MAGIC JOIN workspace.capstone_silver.admission a ON a.person_id = r.person_id
# MAGIC WHERE d.label_drug IS NOT NULL AND a.readmitted_30d
# MAGIC GROUP BY d.label_drug ORDER BY people_readmitted DESC
