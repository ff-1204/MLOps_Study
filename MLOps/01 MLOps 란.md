# MLOps란?

ML(머신러닝) 모델을 만드는 것에서 끝나지 않고, **실제 서비스로 배포하고 안정적으로 운영**하기 위한 기술과 방법을 다루는 분야.

**ML**(Machine Learning) + **Dev**(Development) + **Ops**(Operations) 의 합성어다.

---

## 기존 ML 프로젝트의 한계

실제 ML 프로젝트를 진행하다 보면 이런 문제들이 생긴다.

- **실험 기록 부재**: 수십 번 모델을 바꿔가며 실험했는데, 가장 성능이 좋았던 모델의 파라미터(설정값)나 환경을 기록하지 않아 다시 만들 수 없는 상황이 생긴다.
- **배포 환경 불일치**: 내 컴퓨터에서는 잘 되는데 서버에 올리면 성능이 떨어지거나 오류가 난다. 학습 환경과 배포 환경이 다르기 때문이다.
- **협업 어려움**: 팀으로 개발할 때 각자 Python 버전, 라이브러리 버전이 달라 같은 코드가 다르게 동작한다. 담당자가 퇴사하면 그 사람만 알던 정보가 사라지기도 한다.
- **서비스화 실패**: 위 문제들이 쌓이면 AI 프로젝트가 실험에서 끝나고 실제 서비스로 이어지지 못한다.

> 실제로 AI 프로젝트의 **85%가 서비스화에 실패**한다는 조사 결과가 있다.
> - [Why 85% of AI projects fail — TechRepublic](https://www.techrepublic.com/article/why-85-of-ai-projects-fail/)
> - [Pactera White Paper Reveals 85 percent of AI Projects Ultimately Fail — Pactera Edge](https://www.pacteraedge.com/pactera-white-paper-reveals-85-percent-ai-projects-ultimately-fail-0)

> **왜 ML 코드만 잘 짜면 안 될까?**  
> 실제 ML 시스템에서 학습·예측 코드는 전체의 극히 일부에 불과하다.  
> 데이터 수집, 데이터 검증, 설정 관리, 모니터링, 서빙 인프라 등 훨씬 많은 부분이 코드 주변을 둘러싸고 있으며,  
> 이 부분을 제대로 관리하지 않으면 나중에 고치기 어려운 문제들이 쌓인다.

---

## MLOps가 해결하는 것

| 문제 | MLOps 해결책 |
|---|---|
| 실험 재현 불가 | 실험 추적 도구로 파라미터·결과를 자동 기록 (예: MLflow) |
| 배포 환경 불일치 | 컨테이너(Docker)로 환경을 통째로 포장해서 어디서나 동일하게 실행 |
| 협업 어려움 | 중앙 모델 저장소에 모든 팀원이 모델을 등록·공유 |
| 수동 배포·모니터링 | 자동화 파이프라인으로 배포·재학습을 사람 없이 처리 |

---

## MLOps 주요 구성 요소

MLOps는 **데이터 · 모델 · 서빙** 세 영역으로 구성된다.  
각 영역의 세부 내용은 `03 MLOps의 구성 요소.md` 참고.

---

## 다음으로

MLOps의 뿌리는 **DevOps**다.  
DevOps가 무엇인지, 그리고 일반 SW 개발과 ML 개발이 어떻게 다른지 이해하면 MLOps가 왜 이런 구조를 가졌는지 자연스럽게 납득된다.

→ `02 DevOps와 MLOps의 차이.md` 에서 이어서 학습

---

## 참고 문헌

- Sculley et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NeurIPS.
- Google Cloud. MLOps: Continuous delivery and automation pipelines in machine learning
