#!/bin/bash

hdfs dfs -mkdir -p /group/project/raw
hdfs dfs -mkdir -p /group_project/cleaned
hdfs dfs -mkdir -p /group_project/features
hdfs dfs -mkdir -p /group_project/lookup
hdfs dfs -chmod -R 755 /group_project
hdfs dfs -ls -R /group_project

mkdir -p ~/project/data/cleaned
mkdir -p ~/project/data/features
