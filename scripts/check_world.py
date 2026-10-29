import duckdb

con = duckdb.connect()
W = "data/world"

print("=== 1. INTEGRITY CHECKS (all must be 0) ===")
print(
    con.sql(f"""
SELECT
  (SELECT COUNT(*) FROM '{W}/person.parquet')                               AS persons,
  (SELECT COUNT(*) - COUNT(DISTINCT person_id) FROM '{W}/person.parquet')   AS dup_person_ids,
  (SELECT COUNT(*) - COUNT(DISTINCT synthea_id) FROM '{W}/person.parquet')  AS dup_synthea_ids,
  (SELECT COUNT(*) FROM '{W}/patient.parquet' p
     LEFT JOIN '{W}/person.parquet' x USING(person_id)
     WHERE x.person_id IS NULL)                                             AS orphan_patients,
  (SELECT COUNT(*) FROM '{W}/policyholder.parquet' p
     LEFT JOIN '{W}/person.parquet' x USING(person_id)
     WHERE x.person_id IS NULL)                                             AS orphan_policyholders,
  (SELECT COUNT(*) FROM '{W}/bank_customer.parquet' b
     LEFT JOIN '{W}/person.parquet' x USING(person_id)
     WHERE x.person_id IS NULL)                                             AS orphan_customers
""")
)

print("=== 2. CROSS-DOMAIN COVERAGE ===")
print(
    con.sql(f"""
SELECT COUNT(*) AS persons_in_all_three_roles
FROM '{W}/patient.parquet'       pa
JOIN '{W}/policyholder.parquet'  po USING(person_id)
JOIN '{W}/bank_customer.parquet' bc USING(person_id)
""")
)

print("=== 3. SAMPLE: ONE PERSON, THREE ROLES ===")
print(
    con.sql(f"""
SELECT x.person_id, x.first_name, x.last_name,
       pa.patient_id, po.policyholder_id, bc.customer_id
FROM '{W}/person.parquet' x
JOIN '{W}/patient.parquet'       pa USING(person_id)
JOIN '{W}/policyholder.parquet'  po USING(person_id)
JOIN '{W}/bank_customer.parquet' bc USING(person_id)
ORDER BY x.person_id
LIMIT 3
""")
)
