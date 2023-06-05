# Windows_Anomaly_Detection

## Data Collection

1. Windows Event Logs generated by Sysmon
2. ElasticAgent winlogbeat forwards all Windows Event Logs to SIEM (ELK)
3. On Kibana, goto Home -> Analytics -> Discover
4. Create a tabular dashboard to filter Windows Event ID 4699 (Process Creation)
5. Columns: @timestamp, host.name, process.name, process.parent.name, process.pid, process.command_line, user.name, user.id, user.domain.
6. Download .csv file

## Pre-processing

1. Convert CSV to python pandas dataframe (pre_processing/src/csv_to_df.py)
2. Extract columns and create a list of words to minic a sentence
3. Input the sentences to word embedding module (word2vec, etc.) (pre_processing/src/vectorize.py)
4. next

```
bazel run //:main
```
