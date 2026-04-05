#!/usr/bin/env python3
"""
NBER 논문 다운로드 스크립트

5편의 NBER Working Paper를 다운로드하고 텍스트로 변환합니다.
(학교 네트워크에서 실행 권장)
"""

import subprocess
import sys
import urllib.request
from pathlib import Path

PAPERS_DIR = Path(__file__).parent.parent / "papers"
PAPERS_DIR.mkdir(exist_ok=True)

# 벤치마크용 NBER 논문 5편
PAPERS = {
    "nber_25000": {
        "title": "The Intergenerational Transmission of Human Capital",
        "authors": "Card, Domnisoru, Taylor",
        "url": "https://www.nber.org/system/files/working_papers/w25000/w25000.pdf",
    },
    "nber_25001": {
        "title": "On the Effects of Linking Voluntary Cap-and-Trade Systems",
        "authors": "Weitzman, Holtsmark",
        "url": "https://www.nber.org/system/files/working_papers/w25001/w25001.pdf",
    },
    "nber_25002": {
        "title": "Dynamics and Efficiency in Decentralized Online Auction Markets",
        "authors": "Hendricks, Sorensen",
        "url": "https://www.nber.org/system/files/working_papers/w25002/w25002.pdf",
    },
    "nber_25003": {
        "title": "Transitions from Career Employment Among Public/Private Workers",
        "authors": "Quinn, Cahill, Giandrea",
        "url": "https://www.nber.org/system/files/working_papers/w25003/w25003.pdf",
    },
    "nber_25004": {
        "title": "Why Have Negative Interest Rates Had Small Effect on Banks?",
        "authors": "Lopez, Rose, Spiegel",
        "url": "https://www.nber.org/system/files/working_papers/w25004/w25004.pdf",
    },
}


def download_and_convert():
    print("=" * 60)
    print("  NBER 논문 다운로드 및 텍스트 변환")
    print("=" * 60)

    # pdftotext 확인
    try:
        subprocess.run(["pdftotext", "-v"], capture_output=True, check=True)
    except FileNotFoundError:
        print(
            "\n⚠ pdftotext가 설치되어 있지 않습니다.\n"
            "설치 방법:\n"
            "  Ubuntu/Debian: sudo apt install poppler-utils\n"
            "  macOS:         brew install poppler\n"
            "  Windows:       conda install -c conda-forge poppler\n"
        )
        sys.exit(1)

    for name, info in PAPERS.items():
        txt_path = PAPERS_DIR / f"{name}.txt"
        if txt_path.exists():
            print(f"  ✓ {name} — 이미 존재")
            continue

        print(f"\n  📥 {name}: {info['title']}")
        print(f"     저자: {info['authors']}")

        # PDF 다운로드
        pdf_path = PAPERS_DIR / f"{name}.pdf"
        try:
            print(f"     다운로드 중...", end=" ", flush=True)
            urllib.request.urlretrieve(info["url"], pdf_path)
            print("완료")
        except Exception as e:
            print(f"\n     ⚠ 다운로드 실패: {e}")
            print(f"     대안: 수동으로 PDF를 papers/ 폴더에 넣으세요")
            continue

        # PDF → 텍스트 변환
        try:
            print(f"     텍스트 변환 중...", end=" ", flush=True)
            subprocess.run(
                ["pdftotext", str(pdf_path), str(txt_path)],
                check=True, capture_output=True,
            )
            # PDF 삭제 (용량 절약)
            pdf_path.unlink()
            size = txt_path.stat().st_size
            print(f"완료 ({size:,} bytes)")
        except subprocess.CalledProcessError as e:
            print(f"\n     ⚠ 변환 실패: {e}")

    # 결과 확인
    files = list(PAPERS_DIR.glob("*.txt"))
    print(f"\n{'=' * 60}")
    print(f"  완료: {len(files)}/{len(PAPERS)} 논문 준비됨")
    print(f"  경로: {PAPERS_DIR}")
    print(f"{'=' * 60}")

    if len(files) == 0:
        print(
            "\n💡 NBER 다운로드가 차단된 경우:\n"
            "   1. 학교 VPN 연결 후 재시도\n"
            "   2. 또는 수동으로 papers/ 폴더에 .txt 파일 추가\n"
        )


if __name__ == "__main__":
    download_and_convert()
