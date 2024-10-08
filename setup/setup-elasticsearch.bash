#!/bin/bash

####################################################################################
# elasticsearch 환경 세팅 스크립트
# 실행 방법 
# nohup /usr/share/elasticsearch/bin/elasticsearch > /var/log/elasticsearch.log 2>&1 &
# 종료 방법
# ps aux | grep elasticsearch 
# kill -9 [pid]
####################################################################################

# Install required packages
sudo apt-get update
sudo apt-get install -y wget apt-transport-https openjdk-11-jdk

# Import Elasticsearch PGP Key
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo apt-key add -

# Add Elasticsearch repository
sudo sh -c 'echo "deb https://artifacts.elastic.co/packages/7.x/apt stable main" > /etc/apt/sources.list.d/elastic-7.x.list'

# Install Elasticsearch
sudo apt-get update
sudo apt-get install -y elasticsearch
