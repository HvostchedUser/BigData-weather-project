SELECT wind_speed, COUNT(*) AS count
FROM weather_records
GROUP BY wind_speed
ORDER BY wind_speed;

