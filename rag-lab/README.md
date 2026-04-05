# RAG Lab: 세 가지 RAG 아키텍처 비교 실습

Standard RAG · LightRAG · Karpathy LLM-Wiki를 직접 구현하고 비교하는 실습 패키지입니다.

## 아키텍처 비교

```
┌─────────────────────────────────────────────────────────────────┐
│ Method 1: Standard RAG                                         │
│   문서 → 청킹 → 임베딩 → FAISS → 쿼리 시 검색 → LLM 생성       │
│   ✅ 빠른 구축  ✅ 대규모 확장  ❌ 청크가 문맥 잃음              │
├─────────────────────────────────────────────────────────────────┤
│ Method 2: LightRAG                                             │
│   문서 → 엔티티 추출 → 지식 그래프 + 벡터 검색 → LLM 생성       │
│   ✅ 관계 활용  ✅ 크로스 문서  ❌ 그래프 품질 의존               │
├─────────────────────────────────────────────────────────────────┤
│ Method 3: Karpathy LLM-Wiki                                    │
│   문서 → LLM이 위키로 컴파일 → index.md → 위키 탐색 → 답변      │
│   ✅ 완전 투명  ✅ 지식 축적  ❌ 높은 초기 비용                   │
└─────────────────────────────────────────────────────────────────┘
```

## 빠른 시작

```bash
# 1. 클론
git clone <repo-url>
cd rag-lab

# 2. 환경 설정
pip install -r requirements.txt

# 3. API 키 설정
cp .env.example .env
# .env 파일에 OPENAI_API_KEY 입력

# 4. 논문 다운로���
python scripts/download_papers.py

# 5. 노트북 실행
jupyter notebook notebooks/
```

## 디렉토리 구조

```
rag-lab/
├── rag_lab/                    # 핵심 라이브러리
│   ├── __init__.py             # 패키지 진입점
│   ├── config.py               # 설정값
│   ├── utils.py                # 유틸리티 (청킹, 로딩)
│   ├── llm.py                  # OpenAI API 래퍼 + 캐싱
│   ├── embeddings.py           # 임베딩 엔진 + FAISS
│   ├── standard_rag.py         # Method 1: Standard RAG
│   ├── lightrag.py             # Method 2: LightRAG
│   ├── karpathy_wiki.py        # Method 3: Karpathy Wiki
│   └── evaluate.py             # LLM-as-Judge 평가기
│
├── notebooks/                  # 실습 노트북 (순서대로)
│   ├── 01_standard_rag.ipynb   # Lab 1: Standard RAG
│   ├── 02_lightrag.ipynb       # Lab 2: LightRAG
│   ├── 03_karpathy_wiki.ipynb  # Lab 3: Karpathy Wiki
│   └── 04_benchmark.ipynb      # Lab 4: 종합 벤치마크
│
├── scripts/
│   ├── download_papers.py      # 논문 다운로드
│   └── run_benchmark.py        # CLI 벤치마크 실행
│
├── papers/                     # NBER 논문 (텍스트)
├── wiki/                       # Karpathy Wiki 출력
├── .cache/                     # LLM 응답 캐시
├── requirements.txt
├── .env.example
└── README.md
```

## 실습 커리큘럼

| Lab | 주제 | 소요 시간 | 핵심 개념 |
|-----|------|----------|----------|
| 01 | Standard RAG | 30-40분 | 청킹, 임베딩, FAISS, 벡터 검색 |
| 02 | LightRAG | 30-40분 | 엔티티 추출, 지식 그래프, 그래프 탐색 |
| 03 | Karpathy Wiki | 40-50분 | 위키 컴파일, index 탐색, lint |
| 04 | 종합 벤치마크 | 20-30분 | LLM-as-Judge, 질문 유형별 분석 |

## 벤치마크 결과 (참고)

5편 NBER 논문, 5개 질문 기준:

| 지표 | Standard RAG | LightRAG | Karpathy Wiki |
|------|:---:|:---:|:---:|
| 정확성 | 8.0 | 8.2 | **9.0** |
| 완전성 | 6.8 | 7.0 | **8.0** |
| 구체성 | 6.2 | 6.4 | **7.4** |
| 통합성 | 5.4 | 5.6 | **6.6** |
| **전체** | **6.6** | **6.8** | **7.8** |
| 구축 시간 | **3초** | 56초 | 72초 |
| 쿼리 시간 | **5초** | 6초 | 13초 |

## API 비용 안내

gpt-4o-mini 기준 예상 비용:
- 전체 실습 1회: ~$0.50-1.00
- 벤치마크 1회: ~$0.30
- 캐시 사용 시 재실행 비용: $0

## 참고 자료

- [Karpathy's LLM Wiki Gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)
- [FAISS Documentation](https://faiss.ai/)
- [Sentence Transformers](https://www.sbert.net/)
- [LightRAG Paper](https://arxiv.org/abs/2410.05779)

## 라이선스

교육용 (MIT License)
