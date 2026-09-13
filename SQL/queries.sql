PRAGMA foreign_keys = ON;

-- ============================================
-- Create normalized tables
-- ============================================

-- machines: static/descriptive info for each machine
CREATE TABLE IF NOT EXISTS machines (
    UDI INTEGER PRIMARY KEY,
    product_id TEXT UNIQUE NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('H', 'L', 'M'))
);

-- sensor_readings: operational sensor data per machine
CREATE TABLE IF NOT EXISTS sensor_readings (
    reading_id INTEGER PRIMARY KEY AUTOINCREMENT,
    udi INTEGER NOT NULL,
    air_temperature_k REAL NOT NULL,
    process_temperature_k REAL NOT NULL,
    rotational_speed_rpm INTEGER NOT NULL,
    torque_nm REAL NOT NULL,
    tool_wear_min INTEGER NOT NULL,
    FOREIGN KEY (udi) REFERENCES machines(UDI)
);

-- failures: failure outcome and cause flags per machine
CREATE TABLE IF NOT EXISTS failures (
    failure_id INTEGER PRIMARY KEY AUTOINCREMENT,
    udi INTEGER NOT NULL,
    machine_failure INTEGER NOT NULL CHECK (machine_failure IN (0, 1)),
    twf INTEGER NOT NULL CHECK (twf IN (0, 1)),
    hdf INTEGER NOT NULL CHECK (hdf IN (0, 1)),
    pwf INTEGER NOT NULL CHECK (pwf IN (0, 1)),
    osf INTEGER NOT NULL CHECK (osf IN (0, 1)),
    rnf INTEGER NOT NULL CHECK (rnf IN (0, 1)),
    FOREIGN KEY (udi) REFERENCES machines(UDI)
);

-- ============================================
-- Migrate data from staging_ai4i
-- ============================================

-- Insert machine info (deduplicated by UDI)
INSERT INTO machines (UDI, product_id, type)
SELECT DISTINCT UDI, "Product ID", "Type"
FROM staging_ai4i;

-- Insert sensor readings (one row per original record)
INSERT INTO sensor_readings (udi, air_temperature_k, process_temperature_k, rotational_speed_rpm, torque_nm, tool_wear_min)
SELECT UDI, "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
FROM staging_ai4i;

-- Insert failure flags (one row per original record)
INSERT INTO failures (udi, machine_failure, twf, hdf, pwf, osf, rnf)
SELECT UDI, "Machine failure", TWF, HDF, PWF, OSF, RNF
FROM staging_ai4i;

-- Verify row counts (expect 10000 in each)
SELECT 'machines' AS table_name, COUNT(*) AS row_count FROM machines
UNION ALL
SELECT 'sensor_readings', COUNT(*) FROM sensor_readings
UNION ALL
SELECT 'failures', COUNT(*) FROM failures;

-- ============================================
-- Analytical Queries
-- ============================================

-- 1) Number of machines per type
SELECT type, COUNT(*) AS machine_count
FROM machines
GROUP BY type;

-- 2) Total machine failures per machine type (JOIN)
SELECT m.type, COUNT(*) AS failure_count
FROM failures f
JOIN machines m ON f.udi = m.UDI
WHERE f.machine_failure = 1
GROUP BY m.type;

-- 3) Failure rate (%) per machine type (JOIN)
SELECT m.type,
       COUNT(*) AS total_machines,
       SUM(f.machine_failure) AS total_failures,
       ROUND(100.0 * SUM(f.machine_failure) / COUNT(*), 2) AS failure_rate_pct
FROM machines m
JOIN failures f ON f.udi = m.UDI
GROUP BY m.type;

-- 4) Average sensor readings for failed vs non-failed records (JOIN)
SELECT f.machine_failure,
       ROUND(AVG(s.air_temperature_k), 2) AS avg_air_temp,
       ROUND(AVG(s.torque_nm), 2) AS avg_torque,
       ROUND(AVG(s.tool_wear_min), 2) AS avg_tool_wear
FROM sensor_readings s
JOIN failures f ON s.udi = f.udi
GROUP BY f.machine_failure;

-- 5) Top 5 machines with the highest torque reading (JOIN)
SELECT m.UDI, m.product_id, m.type, s.torque_nm
FROM sensor_readings s
JOIN machines m ON s.udi = m.UDI
ORDER BY s.torque_nm DESC
LIMIT 5;

-- 6) Breakdown of failure cause counts across all failed machines
SELECT
    SUM(twf) AS tool_wear_failures,
    SUM(hdf) AS heat_dissipation_failures,
    SUM(pwf) AS power_failures,
    SUM(osf) AS overstrain_failures,
    SUM(rnf) AS random_failures
FROM failures
WHERE machine_failure = 1;

-- 7) CTE: average torque per machine type, then list machines above their type's average
WITH type_avg_torque AS (
    SELECT m.type, AVG(s.torque_nm) AS avg_torque
    FROM machines m
    JOIN sensor_readings s ON s.udi = m.UDI
    GROUP BY m.type
)
SELECT m.UDI, m.product_id, m.type, s.torque_nm, t.avg_torque
FROM machines m
JOIN sensor_readings s ON s.udi = m.UDI
JOIN type_avg_torque t ON t.type = m.type
WHERE s.torque_nm > t.avg_torque
ORDER BY m.type, s.torque_nm DESC;

-- 8) CTE: machines with more than one failure-cause flag triggered
WITH failure_flag_count AS (
    SELECT udi,
           (twf + hdf + pwf + osf + rnf) AS flags_triggered
    FROM failures
)
SELECT m.UDI, m.product_id, m.type, ffc.flags_triggered
FROM failure_flag_count ffc
JOIN machines m ON m.UDI = ffc.udi
WHERE ffc.flags_triggered > 1;

-- 9) Window function: rank machines by torque within each machine type
SELECT m.UDI, m.product_id, m.type, s.torque_nm,
       RANK() OVER (PARTITION BY m.type ORDER BY s.torque_nm DESC) AS torque_rank_in_type
FROM machines m
JOIN sensor_readings s ON s.udi = m.UDI;

-- 10) Window function: running average of process temperature ordered by UDI
SELECT m.UDI, s.process_temperature_k,
       ROUND(AVG(s.process_temperature_k) OVER (
           ORDER BY m.UDI
           ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
       ), 2) AS rolling_avg_process_temp
FROM machines m
JOIN sensor_readings s ON s.udi = m.UDI
ORDER BY m.UDI;
