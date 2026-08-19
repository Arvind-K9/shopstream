#!/bin/bash
LOCAL_DATA_DIR="$HOME/project/data/raw"
HDFS_RAW_DIR="/group_project/raw"

if [ ! -d "$LOCAL_DATA_DIR" ]; then
  echo "Local data directory $LOCAL_DATA_DIR not found."
  exit 1
fi

hdfs dfs -put -f $LOCAL_DATA_DIR/*.csv $HDFS_RAW_DIR/
hdfs dfs -ls $HDFS_RAW_DIR
