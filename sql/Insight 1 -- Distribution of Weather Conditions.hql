SELECT summary, COUNT(*) AS count
FROM weather_records
GROUP BY summary
ORDER BY count DESC;

