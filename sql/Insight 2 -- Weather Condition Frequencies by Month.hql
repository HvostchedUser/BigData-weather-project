SELECT MONTH(from_unixtime(CAST(formatted_date AS BIGINT))) AS month, summary, COUNT(*) AS count
FROM weather_records
GROUP BY MONTH(from_unixtime(CAST(formatted_date AS BIGINT))), summary
ORDER BY month, count DESC;

