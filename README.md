# Capstone Design 2023-2
# 이웃(IoT) : IoT 기기 디지털 포렌식 도구
* 김시환(컴퓨터공학 2019102158), 정태규(컴퓨터공학 2020105661)

## Overview
* 디지털 포렌식은 현대 사회에서 중요한 역할을 수행하고 있는 분야 중 하나로, 디지털 기기 및 시스템에서 증거를 수집하고 분석하여 범죄나 인위적인 활동을 조사하는 핵심 기술이다.
* 제조사에 따라 다른 파일시스템으로 인해 다지털 포렌식에 많은 어려움이 있는데 다양한 파일시스템 내에서 동작이 가능한 효율적인 방식의 IoT 기기 디지털 포렌식 도구를 제시한다.
* 제안하는 방법은 IoT 기기에서 추출한 로그 데이터로 학습시킨 BERT를 활용하는 방법으로, 딥러닝 알고리즘 기반의 포렌식 도구를 개발하여 기존 포렌식의 복잡성을 해결하고자 한다.

## Results
BERT 모델 학습
* [capstonDesign.ipynb](capstonDesign.ipynb)
* [model_compute.py](model_compute.py)

## Conclusion
학습시킨 BERT 모델을 포렌식 도구에 적용하고, 해당 포렌식 도구를 실제 IoT 기기에서 작동시키며 활용해볼 수 있다.

#### 향후 연구
* 정기적으로 실행되는 자동화된 작업을 지정하는 데 사용되는 시스템 스케줄러인 cron job을 활용하여 주기적으로 로그들을 수집하여 모델을 거쳐 침해활동이 있는지 확인하도록 할 계획이다.
* 그 후 이를 웹서버로 전송하여 실시간으로 많은 IoT 기기들을 동시에 확인할 수 있는 도구로 발전할 수 있도록 연구할 계획이다.
* 추가적으로 로그 데이터뿐만 아니라 침해 사례를 인지할 수 있는 데이터는 파일시스템 내에 다양하게 존재하는데, 추후에는 파일 시스템을 분석하여 인지할 수 있도록 확장하는 연구를 진행할 예정이다.

## References
[1] 정익래, 홍도원, 정교일, “디지털 포렌식 기술 및 동향", 전자통신동향분석 제 22권 제 1호  
[2] 정규식, 김정길, 곽후근, 장훈, “유무선공유기를 이용한 임베디드 리눅스 시스템 구축 및 응용”  
[3] 이진오, 손태식, “IoT 플랫폼에 탑재되는 안드로이드 및 리눅스 기반 파일시스템 포렌식  
[4] 박현진, 인공지능(AI) 언어모델 ‘ 버트 는 무엇인가”, 인공지능 신문

## Reports
* [기초조사서(docx)](reports/CD_이웃(기초조사서).docx) 
* [중간보고서(pdf)](reports/CD_이웃(중간보고서).pdf)
* [최종보고서(pdf)](reports/CD_이웃(최종보고서).pdf)
