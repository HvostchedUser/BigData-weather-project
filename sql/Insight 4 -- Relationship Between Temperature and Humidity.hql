INSERT OVERWRITE DIRECTORY '/user/hive/warehouse/temperature_humidity_output'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
SELECT temperature, humidity FROM weather_records;

