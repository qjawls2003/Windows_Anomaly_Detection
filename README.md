# Windows Anomaly Detection for Processes

## Data Collection

1. Windows Event Logs generated by Sysmon and others
2. ElasticAgent forwards all Windows Event Logs to SIEM (ELK)
3. On Kibana, goto Home -> Analytics -> Discover
4. Create a tabular dashboard to filter Windows Event ID 4688 (Process Creation)
5. Columns: @timestamp, host.name, process.name, process.parent.name, process.pid, process.command_line, user.name, user.id, user.domain.
6. Download .csv file (pre_processing/data/WinEvent4688.csv)

## Pre-processing

1. Convert CSV to python pandas dataframe (pre_processing/src/csv_to_df.py)
2. Extract columns and create a list of words to minic a sentence
3. Input the sentences to word embedding module (word2vec, etc.) (pre_processing/src/vectorize.py)
4. next

```
bazel run //:main
```

# TODO

#### Modeling
- Get PyTorch CUDA using NVIDIA GPU working
- Complete layers

#### Embedding
- ~~Figure out which word embedding to use~~ Using WordPiece
- ~~Figure out what the multi-dimensional vectors really mean~~
- How to detect anomaly using the vectors?
- ~~What if the new commands have words not in the embedded model? How do we deal with new words not in the dictionary?~~ Masking Language Model

#### Clustering
- PCA for dimension reductions?
- Multiple clusters based time (month, day, hour), process name, parent process name, user, domain: assign an cluster id for each log entry and the use word embedding on the command lines

#### More pre-processing
- ~~How to formulate a sentence so that all features are added (time, users, parent processes, and ultimately command line words)~~
- ~~Generalize @timestamp to "month, day of the week, and day/night"~~
