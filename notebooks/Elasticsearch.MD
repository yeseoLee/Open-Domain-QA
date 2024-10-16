# Elasticsearch
- Apache Lucene(아파치 루씬) 기반의 Java 오픈소스 분산형 RESTful 검색 엔진
- 대용량 데이터를 거의 실시간(Near Real Time)으로 검색, 저장, 분석할 수 있음
-  로그 분석, 웹사이트 검색, 빅데이터 처리 등 실시간 데이터 분석에 많이 사용됨
- Elasticsearch는 검색을 위해 단독으로 사용되기도 하며, ELK(Elasticsearch/Logstatsh/Kibana)스택으로 사용되기도 함

## Architecture
Elasticsearch는 대규모 데이터를 효과적으로 처리하고 빠르고 확장 가능한 검색을 제공하기 위한 분산형 구조로 설계되어 있다. 자세한 내용은 아래 키워드를 참고하여 검색해보면 되겠다.  
- 클러스터
- Node
- Index, Type, Field
- Document
- Shard

### 참고
- https://victorydntmd.tistory.com/308  

## Inverted index
- 역색인(inverted index)은 검색을 빠르게 수행하기 위해 사용되는 핵심 데이터 구조
- RDBMS의 인덱싱과는 다르게, 주로 **전체 텍스트 검색**에서 많이 사용

### 개념
사전의 끝에 있는 색인은 단어가 등장하는 페이지 번호를 나열하는데,역색인이 사전의 색인처럼 동작한다. 역색인도 **토큰**을 키로 하고, 그 토큰이 포함된 **문서들**을 값으로 저장한다.

예를 들어, 다음과 같은 두 문서가 있을 떄,  
- 문서 1: "Elasticsearch is a search engine"
- 문서 2: "A search engine is powerful"

역색인은 다음과 같이 만들어진다.  

- "a" → [문서 1, 문서 2]
- "search" → [문서 1, 문서 2]
- "engine" → [문서 1, 문서 2]
- "elasticsearch" → [문서 1]
- "is" → [문서 1, 문서 2]
- "powerful" → [문서 2]

이렇게 역색인이 생성되면, 특정 단어에 대해 그 단어가 포함된 문서들을 빠르게 찾을 수 있다.  

### 과정
역색인은 데이터를 토큰화하고 전처리하는 과정을 거쳐 생성
1. **토큰화**: 텍스트를 단어 단위로 쪼갬  
  e.g.) "Elasticsearch is powerful" -> ["elasticsearch", "is", "powerful"]
2. **정규화**: 모든 텍스트를 소문자로 변환하고, 불필요한 문장 부호 등을 제거
3. **불용어 제거**: "is", "a", "the"와 같은 자주 등장하지만 의미가 없는 단어들을 제거
4. **역색인 생성**: 각 단어에 대해 등장하는 문서 ID를 기록하여 역색인을 생성

### 장점
- 특정 단어나 구문이 포함된 문서를 빠르게 찾아낼 수 있음
- 단어의 위치 정보도 포함할 수 있어, 문서 내에서 단어가 등장하는 위치를 기반으로 정렬된 결과를 제공할 수 있음
- 중복된 단어를 한 번만 저장하면서 여러 문서에서 해당 단어가 사용된 위치만을 기록하므로 공간 효율성이 높음
### 참고
- https://esbook.kimjmin.net/06-text-analysis/6.1-indexing-data
- https://needjarvis.tistory.com/345

# Elasticsearch for Retrieval

## 한글 형태소 분석기 nori
https://esbook.kimjmin.net/06-text-analysis/6.7-stemming/6.7.2-nori

## 내가 정리하는 것보다 좋은 자료가 있었네요
https://github.com/thejungwon/search-engine-tutorial

## 더 알아보기
- [Elastic 가이드 북](https://esbook.kimjmin.net)
- [검색엔진(Search Engine)의 현황 및 구동원리](https://needjarvis.tistory.com/167)
