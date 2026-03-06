# AI 신용평가 모형 프로젝트 가이드

## 1. 프로젝트 개요

### 무엇을 만드는가?
**AI를 활용한 개인 신용평가(Credit Scoring) 모형**을 만든다.
- 개인의 금융 데이터(소득, 연체 이력, 부채비율 등)를 입력하면
- AI 모형이 **"이 사람이 대출금을 갚지 못할 확률"**을 예측한다.
- 추가로 LLM(대규모 언어모형)을 활용하여 **예측 결과를 자연어로 해석**한다.

### 왜 신용평가인가?
- 은행, 카드사, 핀테크 기업의 **핵심 업무**
- 데이터가 정형(tabular)이라 입문자가 다루기 쉬움
- AI/ML의 효과가 명확하게 드러남 (전통 통계 vs ML 비교 가능)
- 실무와 직결되어 포트폴리오 가치가 높음

### 수업 연계
| 수업 주제 | 프로젝트 적용 |
|-----------|-------------|
| Python 기초 | 데이터 전처리, pandas 활용 |
| LLM 소개 | Claude/GPT로 모형 결과 해석 |
| RAG 활용 | 금융 규제/용어 문서 검색하여 해석에 활용 (심화) |
| AI 에이전트 | 자동 신용평가 파이프라인 구축 (심화) |

---

## 2. 문헌 리뷰 (Literature Review)

AI 신용평가는 전통 통계 모형에서 시작하여, 머신러닝, 그리고 최근 LLM까지 빠르게 진화하고 있다.
아래는 이 프로젝트와 관련된 주요 연구 흐름을 정리한 것이다.

### 2.1 전통 신용평가 모형 (1960s~2010s)

| 시기 | 모형 | 특징 |
|------|------|------|
| 1960s | Discriminant Analysis | 최초의 정량적 신용평가 (Altman Z-Score) |
| 1980s | Logistic Regression | 연체 확률을 직접 추정. 현재도 금융권 표준 |
| 2000s | Credit Scorecard | 로지스틱 회귀 기반. WoE/IV로 변수 변환. 규제 친화적 |

- **Altman (1968)**: Z-Score 모형으로 기업 부도 예측의 기초를 확립
- **Thomas et al. (2002)**: *Credit Scoring and Its Applications* - 신용평가 교과서의 표준
- **Siddiqi (2012)**: *Credit Risk Scorecards* - 실무 스코어카드 구축 방법론

> 전통 모형의 강점은 **해석 가능성**과 **규제 준수**. 은행 감독 당국이 "왜 이 대출이 거절됐는가"를 설명할 수 있어야 하므로, 블랙박스 모형은 오랫동안 기피되었다.

### 2.2 머신러닝 기반 신용평가 (2015~현재)

Gradient Boosting 계열 모형이 정형 데이터에서 로지스틱 회귀를 압도하면서 패러다임이 전환되었다.

**핵심 모형 비교**:

| 모형 | 장점 | 단점 | 대표 논문/연도 |
|------|------|------|-------------|
| Random Forest | 과적합에 강함, 병렬 처리 | 부스팅 대비 성능 열세 | Breiman (2001) |
| XGBoost | 높은 정확도, 정규화 | 메모리 사용량 큼 | Chen & Guestrin (2016) |
| **LightGBM** | 빠른 학습, 대규모 데이터 | 소규모 데이터 과적합 주의 | Ke et al. (2017) |
| CatBoost | 범주형 변수 자동 처리 | 학습 느림 | Prokhorenkova et al. (2018) |
| Neural Network | 비선형 패턴 포착 | 해석 어려움, 데이터 많이 필요 | - |

**최근 연구 동향 (2024~2025)**:

- **ML 기반 신용평가 체계적 문헌 리뷰 (2025)**: 2018~2024년 330편의 논문을 분석한 대규모 서베이. XGBoost와 LightGBM이 가장 널리 사용되며, 앙상블 기법이 단일 모형을 일관되게 능가함을 확인.
  - Ref: [Machine learning powered financial credit scoring: a systematic literature review](https://link.springer.com/article/10.1007/s10462-025-11416-2) (Artificial Intelligence Review, 2025)

- **Hybrid Boosted Attention-based LightGBM (2025)**: LightGBM에 어텐션 메커니즘을 결합한 HBA-LGBM 프레임워크 제안. 다단계 변수 선택, 어텐션 기반 변수 강화, 불균형 데이터 처리 전략을 통합하여 디지털 금융 신용평가 성능을 개선.
  - Ref: [Hybrid boosted attention-based LightGBM framework](https://www.nature.com/articles/s41599-025-05230-y) (Nature Humanities and Social Sciences Communications, 2025)

- **부스팅 알고리즘 비교 분석 (2025)**: LightGBM이 AdaBoost, XGBoost, CatBoost를 능가하여 개인 채무불이행 확률 예측에서 최고 성능을 달성.
  - Ref: [Comparative analysis of boosting algorithms for predicting personal default](https://www.tandfonline.com/doi/full/10.1080/23322039.2025.2465971) (Cogent Economics & Finance, 2025)

- **XGBoost 기반 신용카드 연체 예측 (2024)**: XGBoost가 99.4% 정확도로 다른 ML 모형을 압도. 변수 중요도 분석에서 연체 이력 관련 변수가 최상위를 차지.
  - Ref: [Credit Risk Prediction Using ML and Deep Learning](https://www.mdpi.com/2227-9091/12/11/174) (MDPI Risks, 2024)

- **온라인 대출 신용리스크 (2024)**: 하이브리드 ML 알고리즘이 전통 통계 모형 대비 유의미한 성능 개선을 보임. 특히 P2P 대출 환경에서 효과적.
  - Ref: [Performance Evaluation of Hybrid ML for Online Lending Credit Risk](https://www.tandfonline.com/doi/full/10.1080/08839514.2024.2358661) (Applied AI, 2024)

### 2.3 LLM과 신용평가 (2023~현재)

대규모 언어모형(LLM)의 등장으로 비정형 텍스트 데이터를 신용평가에 활용하는 연구가 급증하고 있다.

**LLM 활용 유형**:

```
유형 1: 비정형 데이터 → 신용 신호 추출
  뉴스, 재무보고서, SNS 텍스트 → LLM → 감성/위험 점수 → ML 모형 입력

유형 2: 모형 결과 → 자연어 해석
  ML 예측 결과 + SHAP 값 → LLM → 한국어 신용평가 보고서 (본 프로젝트)

유형 3: Zero-shot / Few-shot 신용 분류
  고객 프로필 → LLM 프롬프트 → 직접 연체 가능성 판단
```

**최근 연구 동향 (2024~2025)**:

- **LLM 기반 신용리스크 체계적 리뷰 (2025)**: 2020~2025년 60편의 논문을 PRISMA 방법론으로 분석. GPT, FinBERT 등 Transformer 모형이 재무 텍스트에서 장거리 의존성을 포착하여 비정형 데이터 기반 신용리스크 평가에 유망함을 확인. 재현성, 공정성, 환각(hallucination) 문제가 향후 과제로 지적됨.
  - Ref: [Interpretable LLMs for Credit Risk: A Systematic Review and Taxonomy](https://arxiv.org/html/2506.04290v2) (arXiv / Expert Systems with Applications, 2025)

- **CreditLLM (2025)**: 신용 심사를 위한 금융 AI 어시스턴트 구축. 대출 설명서 텍스트를 LLM으로 분석하여 리스크 지표를 추출하는 파이프라인 제안.
  - Ref: [CreditLLM: Constructing Financial AI Assistant for Credit](https://aclanthology.org/2025.finnlp-1.13.pdf) (FinNLP Workshop, 2025)

- **AT-FinGPT (2025)**: 오디오+텍스트 멀티모달 LLM으로 금융 리스크 예측. 실적발표 음성과 텍스트를 동시 분석하여 기업 신용리스크 예측 성능 향상.
  - Ref: [AT-FinGPT: Financial risk prediction via audio-text LLM](https://www.sciencedirect.com/science/article/abs/pii/S1544612325002314) (Finance Research Letters, 2025)

- **P2P 대출 텍스트 기반 리스크 지표 (2024)**: 대출 설명 텍스트를 LLM으로 분석하여 전통적 재무 변수에 추가적인 리스크 신호를 추출. 텍스트 기반 변수가 예측력을 유의미하게 개선.
  - Ref: [Credit Risk Meets LLMs: Building a Risk Indicator from Loan Descriptions](https://papers.ssrn.com/sol3/Delivery.cfm/8e3fd583-32ae-49ef-bf13-ca5fc9aa12ba-MECA.pdf?abstractid=4979155&mirid=1) (SSRN, 2024)

- **프롬프트 기반 LLM 신용 분류 (2024)**: 별도 학습 없이 프롬프트 엔지니어링만으로 LLM이 신용 등급을 분류하는 실험. Few-shot 학습이 Zero-shot 대비 유의미한 성능 향상을 보임.
  - Ref: [Explore the Use of Prompt-Based LLM for Credit Risk Classification](https://www.scirp.org/journal/paperinformation?paperid=143252) (SCIRP, 2024)

### 2.4 설명 가능한 AI (XAI)와 공정성

규제 당국(EU AI Act, 미국 CFPB)은 AI 신용평가의 **설명 가능성**과 **공정성**을 점점 더 강하게 요구하고 있다.

**핵심 기법**:

| 기법 | 유형 | 설명 |
|------|------|------|
| **SHAP** | Post-hoc, 모형 비의존 | 각 변수가 예측에 기여한 정도를 Shapley 값으로 계산 |
| **LIME** | Post-hoc, 모형 비의존 | 개별 예측 주변에서 해석 가능한 대리 모형을 학습 |
| **Feature Importance** | 모형 내장 | 트리 모형의 변수 분할 기여도 (본 프로젝트에서 사용) |
| **PDP/ICE** | Post-hoc | 변수 변화에 따른 예측값 변화를 시각화 |

**최근 연구 동향 (2024~2025)**:

- **AI 신용평가의 성능, 공정성, 설명가능성 체계적 리뷰 (2025)**: 세 가지 목표(성능/공정성/설명가능성) 간의 트레이드오프를 분석. 공정성 제약을 추가하면 성능이 하락할 수 있으나, SHAP/LIME 기반 해석이 이를 보완할 수 있음을 제시.
  - Ref: [Performance, Fairness, and Explainability in AI-Based Credit Scoring](https://www.mdpi.com/1911-8074/19/2/104) (MDPI JRFM, 2025)

- **하이브리드 ML + XAI 신용 결정 (2024)**: Neural Network + XGBoost 하이브리드 모형에 SHAP/LIME을 적용하여 96% 정확도 달성. 전통 해석 가능 모형 대비 4%p 성능 향상.
  - Ref: [Enhancing transparency and fairness in automated credit decisions](https://www.nature.com/articles/s41598-024-75026-8) (Scientific Reports, 2024)

- **SHAP 안정성 문제 (2025)**: 클래스 불균형이 심한 신용 데이터에서 SHAP 설명의 일관성이 크게 저하됨을 발견. 불균형 비율이 높을수록 동일 모형의 SHAP 값이 변동하여, 실무 적용 시 주의가 필요.
  - Ref: [SHAP Stability in Credit Risk Management](https://www.mdpi.com/2227-9091/13/12/238) (MDPI Risks, 2025)

- **신용평가 성별 편향 (2025)**: 성별을 변수에서 제거해도 다른 변수(직업, 소비 패턴)를 통해 성별이 간접적으로 반영되는 "대리 변수 누출(Proxy Leakage)" 현상을 구조적으로 분석. 여성 차용자가 동일 리스크 프로필에서도 낮은 신용점수를 받는 현상을 실증.
  - Ref: [Structural Gender Bias in Credit Scoring: Proxy Leakage](https://arxiv.org/html/2601.18342v1) (arXiv, 2025)

- **금융 XAI 종합 리뷰 (2024)**: 금융 분야 설명 가능한 AI 기법을 체계적으로 정리. SHAP과 LIME이 가장 보편적이나, 모형 내재적(intrinsic) 설명과 수요자 맞춤형(demand-based) 설명에 대한 관심이 증가.
  - Ref: [A comprehensive review on financial explainable AI](https://link.springer.com/article/10.1007/s10462-024-11077-7) (Artificial Intelligence Review, 2024)

### 2.5 본 프로젝트의 위치

```
전통 통계 ──── ML/Boosting ──── Deep Learning ──── LLM
(Logistic)    (LightGBM)       (Neural Net)      (GPT/Claude)
                  ▲                                    ▲
                  │                                    │
              [본 프로젝트: 모형 학습]          [본 프로젝트: 결과 해석]
                  │                                    │
                  └──────── SHAP/Feature Importance ────┘
                              (설명 가능성 연결)
```

본 프로젝트는 다음을 결합한다:
1. **LightGBM** - 최근 연구에서 신용평가 정형 데이터 최고 성능 모형으로 확인 (2025 비교 연구)
2. **Feature Importance** - 변수 중요도를 통한 1차 해석
3. **LLM 해석** - Claude/ChatGPT를 활용한 자연어 기반 2차 해석
4. (심화) **SHAP** - 개별 예측 수준의 변수별 기여도 분석

이 조합은 최근 문헌에서 제시하는 "고성능 ML + 설명 가능성 + LLM 해석"의 흐름과 정확히 일치한다.

---

## 3. 프로젝트 구조

```
credit-scoring/
├── START.md              # 이 파일 (가이드)
├── TODO.md               # 체크리스트
├── setup.sh              # 환경 자동 설정 스크립트
├── requirements.txt      # Python 패키지 목록
├── data/
│   └── credit_data.csv   # 신용평가 데이터 (setup.sh 실행 시 생성)
├── notebooks/
│   └── 01_credit_scoring.ipynb   # 메인 노트북 (단계별 실습)
├── src/
│   └── generate_data.py  # 합성 데이터 생성기
└── outputs/
    ├── model.pkl         # 학습된 모형 (실행 후 생성)
    ├── feature_importance.png
    ├── confusion_matrix.png
    └── roc_curve.png
```

---

## 4. 시작하기 (처음부터 끝까지)

### Step 0: 사전 준비
- Python 3.9 이상 설치 확인
- 터미널(명령 프롬프트) 사용법 기본 이해

```bash
python3 --version   # 3.9 이상이어야 함
```

### Step 1: 환경 설정
```bash
cd credit-scoring/
bash setup.sh
```
이 한 줄이면 가상환경 생성, 패키지 설치, 샘플 데이터 생성이 모두 완료된다.

### Step 2: 노트북 실행
```bash
source venv/bin/activate
jupyter notebook notebooks/
```
브라우저에서 `01_credit_scoring.ipynb`을 열고 셀을 하나씩 실행한다.

### Step 3: 결과 확인
`outputs/` 폴더에 모형 파일과 시각화 결과가 저장된다.

---

## 5. 핵심 개념 설명

### 5.1 신용평가란?
- **정의**: 대출 신청자가 미래에 채무를 이행하지 않을(default) 가능성을 수치로 평가
- **한국**: NICE, KCB 등의 신용평가사가 1~1000점 사이의 신용점수 산출
- **미국**: FICO Score (300~850점)
- **핵심 질문**: "이 사람에게 돈을 빌려줘도 괜찮은가?"

### 5.2 사용하는 데이터 (변수 설명)

| 변수명 | 의미 | 왜 중요한가 |
|-------|------|-----------|
| `default` | 연체 여부 (0=정상, 1=연체) | **예측 대상 (Target)** |
| `revolving_utilization` | 신용카드 사용 비율 | 카드 한도의 몇 %를 쓰는지. 높으면 위험 |
| `age` | 나이 | 연령대별 연체 패턴이 다름 |
| `times_30_59_days_late` | 30~59일 연체 횟수 | 경미한 연체 이력 |
| `times_60_89_days_late` | 60~89일 연체 횟수 | 중간 연체 이력 |
| `times_90_plus_days_late` | 90일 이상 연체 횟수 | **가장 심각한 연체. 핵심 변수** |
| `debt_ratio` | 부채 비율 (부채/소득) | 월 소득 대비 부채 상환 부담 |
| `monthly_income` | 월 소득 | 상환 능력의 직접 지표 |
| `num_open_credit_lines` | 개설된 신용 계좌 수 | 신용 이력의 폭 |
| `num_real_estate_loans` | 부동산 담보 대출 수 | 자산 보유 및 부채 구조 |
| `num_dependents` | 부양가족 수 | 지출 부담 |

### 5.3 머신러닝 모형: LightGBM

**왜 LightGBM인가?**
- 정형 데이터에서 **가장 강력한 모형** 중 하나 (Kaggle 대회 우승 다수)
- XGBoost보다 빠르고, 작은 데이터에서도 잘 작동
- 변수 중요도를 자동으로 계산 → 해석 가능
- 실무에서 신용평가에 실제 사용됨

**작동 원리 (간략)**:
```
1. "연체 횟수 > 2인가?" → 예/아니오 분기
2. "부채비율 > 1인가?"  → 예/아니오 분기
3. ...수백 개의 결정 트리를 순차적으로 학습
4. 각 트리가 이전 트리의 실수를 보완 (Gradient Boosting)
5. 최종: 모든 트리의 예측을 합산 → 연체 확률
```

### 5.4 평가 지표

| 지표 | 의미 | 좋은 값 |
|------|------|--------|
| **AUC-ROC** | 모형의 전체적인 구분 능력 | 0.7 이상이면 양호, 0.8 이상이면 우수 |
| **Precision** | "연체 예측" 중 실제 연체 비율 | 높을수록 좋음 (오탐 감소) |
| **Recall** | 실제 연체 중 잡아낸 비율 | 높을수록 좋음 (미탐 감소) |
| **F1 Score** | Precision과 Recall의 조화 평균 | 균형 잡힌 성능 지표 |

**은행 관점**: Recall이 중요 (연체자를 놓치면 손실이 크기 때문)

### 5.5 LLM 활용 (Claude/ChatGPT)

전통 ML 모형은 "확률 0.73"이라고만 알려준다.
LLM을 결합하면:

```
[모형 출력]
연체 확률: 73%

[LLM 해석]
"이 고객은 고위험군으로 분류됩니다.
 주요 위험 요인:
 1. 90일 이상 연체 이력이 3회로 매우 높습니다.
 2. 부채비율이 1.8로 월소득의 거의 2배에 달합니다.
 권고: 대출 거절 또는 금리 상향 조정을 권장합니다."
```

이것이 **설명 가능한 AI (Explainable AI)**의 핵심이다.

---

## 6. 단계별 학습 로드맵

### Phase 1: 기초 (1~2주)
- [ ] Python 기본 문법 복습 (변수, 반복문, 함수)
- [ ] pandas로 CSV 파일 읽기/조작
- [ ] matplotlib으로 기본 차트 그리기
- **목표**: 데이터를 불러와서 기초 통계 확인 가능

### Phase 2: 모형 구축 (2~3주)
- [ ] scikit-learn 기본 사용법 (train_test_split, fit, predict)
- [ ] LightGBM 설치 및 학습
- [ ] 혼동행렬, AUC-ROC 이해 및 시각화
- **목표**: 모형을 학습시키고 성능을 평가할 수 있음

### Phase 3: 해석 및 발표 (1~2주)
- [ ] 변수 중요도 분석 및 해석
- [ ] LLM으로 결과 해석 자동화 (선택)
- [ ] 발표 자료 준비
- **목표**: "왜 이 사람이 고위험인가?"를 설명할 수 있음

### Phase 4: 심화 (선택)
- [ ] 한국 금융 데이터로 교체 (KRX, 금감원 공시)
- [ ] RAG로 금융 규제 문서 검색 연동
- [ ] SHAP 값으로 개별 예측 해석
- [ ] 웹 대시보드 구축 (Streamlit/Gradio)

---

## 7. 주요 라이브러리 요약

| 라이브러리 | 용도 | 설치 |
|-----------|------|------|
| `pandas` | 데이터 조작 (표 형태) | requirements.txt에 포함 |
| `numpy` | 수치 연산 | requirements.txt에 포함 |
| `scikit-learn` | ML 유틸리티 (분할, 평가) | requirements.txt에 포함 |
| `lightgbm` | 그래디언트 부스팅 모형 | requirements.txt에 포함 |
| `matplotlib` | 기본 시각화 | requirements.txt에 포함 |
| `seaborn` | 통계 시각화 | requirements.txt에 포함 |
| `shap` | 모형 해석 (심화) | `pip install shap` |
| `anthropic` | Claude API (심화) | `pip install anthropic` |

---

## 8. 자주 묻는 질문

### Q: 데이터를 직접 구해야 하나요?
A: 아닙니다. `setup.sh`를 실행하면 현실적인 합성 데이터(10,000건)가 자동 생성됩니다.
실제 데이터를 쓰고 싶다면 Kaggle의 "Give Me Some Credit" 데이터셋을 추천합니다.

### Q: GPU가 필요한가요?
A: 아닙니다. LightGBM은 CPU에서도 매우 빠릅니다. 10,000건 기준 수 초면 학습 완료.

### Q: 코딩 경험이 적어도 되나요?
A: 노트북의 셀을 순서대로 실행하면 결과가 나옵니다.
각 셀에 한국어 설명이 포함되어 있어 따라가기 쉽습니다.

### Q: LLM 없이도 프로젝트가 완성되나요?
A: 네. LLM 해석은 선택사항입니다. ML 모형만으로도 완전한 프로젝트입니다.
LLM 부분은 Claude/ChatGPT 웹에서 프롬프트를 복사-붙여넣기하는 것만으로 시연 가능합니다.

### Q: 발표에서 무엇을 보여주면 되나요?
A: 아래 순서를 추천합니다:
1. 문제 정의 (신용평가가 무엇이고 왜 중요한지)
2. 데이터 탐색 (어떤 변수가 있고, 분포는 어떤지)
3. 모형 학습 결과 (AUC 점수, 혼동행렬)
4. 변수 중요도 (어떤 변수가 예측에 가장 중요한지)
5. LLM 해석 시연 (선택: AI가 결과를 설명하는 데모)

---

## 9. 참고 자료

### 온라인 강좌
- [Kaggle: Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning)
- [Kaggle: Give Me Some Credit Competition](https://www.kaggle.com/c/GiveMeSomeCredit)

### 논문 및 보고서
- Basel Committee: "Credit Scoring and Credit Control" (신용평가 규제 프레임워크)
- 한국은행: "빅데이터를 활용한 개인 신용평가 모형" 보고서

### 한국 금융 데이터 소스 (심화)
- 금융감독원 전자공시 (DART): https://dart.fss.or.kr
- 한국은행 경제통계시스템 (ECOS): https://ecos.bok.or.kr
- KRX 정보데이터시스템: http://data.krx.co.kr

---

## 10. 문제 해결

### setup.sh 실행 오류
```bash
# Permission denied
chmod +x setup.sh && bash setup.sh

# Python not found
python3 --version  # 설치 확인
```

### Jupyter 실행 안됨
```bash
source venv/bin/activate  # 가상환경 활성화 먼저!
jupyter notebook notebooks/
```

### LightGBM 설치 오류 (Mac M1/M2)
```bash
brew install libomp
pip install lightgbm
```

### 그래도 안 될 때
```bash
# 전체 초기화 후 재설치
rm -rf venv/
bash setup.sh
```
