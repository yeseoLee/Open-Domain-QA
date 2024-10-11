#!/bin/bash

####################################################################################
# elasticsearch 환경 세팅 스크립트
# 실행 방법 
# sudo -u elasticsearch nohup /usr/share/elasticsearch/bin/elasticsearch > /var/log/elasticsearch/elasticsearch.log 2>&1 &
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
sudo apt-get install -y wget apt-transport-https openjdk-11-jdk curl

# Import the Elasticsearch public GPG key
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo apt-key add -

# Add the Elasticsearch APT repository
sudo sh -c 'echo "deb https://artifacts.elastic.co/packages/7.x/apt stable main" > /etc/apt/sources.list.d/elastic-7.x.list'

# Install Elasticsearch
sudo apt-get update
sudo apt-get install -y elasticsearch

# ES 사용자 생성 및 ES 디렉토리의 소유권 변경
if id "elasticsearch" &>/dev/null; then
    echo "User 'elasticsearch' already exists."
else
    sudo adduser --system --no-create-home --group elasticsearch
fi

sudo chown -R elasticsearch:elasticsearch /etc/elasticsearch
sudo chown -R elasticsearch:elasticsearch /usr/share/elasticsearch
sudo chown elasticsearch:elasticsearch /var/log/elasticsearch.log

# Nori Plugin 설치
sudo /usr/share/elasticsearch/bin/elasticsearch-plugin install analysis-nori