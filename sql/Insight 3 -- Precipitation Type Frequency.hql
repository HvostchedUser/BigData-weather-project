SELECT precip_type, COUNT(*) AS count
FROM weather_records
WHERE precip_type IS NOT NULL
GROUP BY precip_type;

