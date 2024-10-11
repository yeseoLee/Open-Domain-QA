#!/bin/bash

####################################################################################
# elasticsearch 환경 세팅 스크립트
# 실행 방법 (로그 남기기/남기지 않기 선택)
# sudo -u elasticsearch nohup /usr/share/elasticsearch/bin/elasticsearch > /dev/null 2>&1 &
# sudo -u elasticsearch nohup /usr/share/elasticsearch/bin/elasticsearch > /var/log/elasticsearch.log 2>&1 &
# 실행 상태 확인
# curl -X GET "localhost:9200"
# Nori 플러그인 설치 확인
# curl -X GET "localhost:9200/_analyze" -H 'Content-Type: application/json' -d '{ "tokenizer": "nori_tokenizer", "text": "동해물과 백두산이" }'
# 종료 방법
# ps aux | grep elasticsearch | grep -v grep
# kill -9 [pid]
####################################################################################

# Update and install prerequisites
sudo apt update
sudo apt-get install -y wget apt-transport-https openjdk-11-jdk curl gnupg

# Import the Elasticsearch public GPG key
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo apt-key add -

# Add the Elasticsearch APT repository
sudo sh -c 'echo "deb https://artifacts.elastic.co/packages/7.x/apt stable main" > /etc/apt/sources.list.d/elastic-7.x.list'

# Install Elasticsearch
sudo apt-get update
sudo apt-get install -y elasticsearch

# Install Nori Plugin 
sudo /usr/share/elasticsearch/bin/elasticsearch-plugin install analysis-nori

# Change Ownership of elasticsearch Directory
if id "elasticsearch" &>/dev/null; then
    echo "User 'elasticsearch' already exists."
else
    sudo adduser --system --no-create-home --group elasticsearch
fi

sudo chown -R elasticsearch:elasticsearch /etc/elasticsearch
sudo chown -R elasticsearch:elasticsearch /usr/share/elasticsearch

# Option: setting log file
sudo touch /var/log/elasticsearch.log
sudo chown elasticsearch:elasticsearch /var/log/elasticsearch.log
