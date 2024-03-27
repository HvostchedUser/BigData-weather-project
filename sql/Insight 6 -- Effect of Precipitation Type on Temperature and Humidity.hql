SELECT precip_type, AVG(temperature) AS avg_temperature, AVG(humidity) AS avg_humidity
FROM weather_records
WHERE precip_type IS NOT NULL
GROUP BY precip_type;

