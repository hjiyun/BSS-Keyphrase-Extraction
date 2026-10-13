"""Amazon Fine Food Reviews 전처리 — 기존 amazon_pre.txt 와 동일 파이프라인.

검증: 원문 Id=42(McCann 오트밀, ProductId B001EO5QW8, Summary "Oatmeal For Oatmeal Lovers")를
아래 preprocess() 로 돌리면 기존 data_JOC/amazon_pre.txt 와 바이트 단위로 일치함.

파이프라인: 소문자 → <br/> 제거 → 소유격 's 제거 → 영숫자 외 공백치환(하이픈/구두점 분리·숫자보존)
           → NLTK 영어 불용어 제거 → Porter 스테밍.

사용:
  python3 preprocess_amazon.py <ProductId> [outprefix]
    → <outprefix>_docs.txt  (리뷰 1건 = 1줄, 전처리본; K-문서 결합 실험용)
      <outprefix>_merged.txt (전 리뷰 전처리본을 이어붙인 단일 텍스트; amazon_pre.txt 스타일)
  기본 ProductId=B001EO5QW8(오트밀), outprefix=amazon_oatmeal
"""
import os, re, sys, csv, html
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

csv.field_size_limit(10 ** 7)
HERE = os.path.dirname(os.path.abspath(__file__))
REVIEWS = os.path.join(HERE, "amazon", "Reviews.csv")
SW = set(stopwords.words('english'))
ps = PorterStemmer()


def preprocess(text):
    t = html.unescape(text)                   # HTML 엔티티 디코딩 (&#34;->" 등; 숫자 잔여물 방지)
    t = t.lower()
    t = re.sub(r"\b\d+\s*['′’][\s\d]*[\"”]?", ' ', t)       # 키(피트'인치): 5'4, 5'9, 5'4"
    t = re.sub(r'\b\d+\s*(ft|feet|foot|inch(?:es)?|lbs?|oz|cm|mm|kg)\b', ' ', t)  # 신체 측정단위
    t = re.sub(r'<br\s*/?>', ' ', t)          # HTML 줄바꿈 제거
    t = re.sub(r"'s\b", '', t)                # 소유격 's 제거 (mccann's -> mccann)
    t = re.sub(r"[^a-z0-9]+", ' ', t)         # 영숫자 외 → 공백 (하이픈/구두점 분리, 숫자 보존)
    return ' '.join(ps.stem(w) for w in t.split() if w not in SW)


def load_product(product_id):
    rows = []
    with open(REVIEWS, newline='', encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            if row['ProductId'] == product_id:
                rows.append(row)
    rows.sort(key=lambda r: int(r['Id']))
    return rows


def main():
    pid = sys.argv[1] if len(sys.argv) > 1 else "B001EO5QW8"
    prefix = sys.argv[2] if len(sys.argv) > 2 else "amazon_oatmeal"
    rows = load_product(pid)
    docs = [preprocess(r['Text']) for r in rows]
    docs = [d for d in docs if d.strip()]

    docs_path = os.path.join(HERE, f"{prefix}_docs.txt")
    merged_path = os.path.join(HERE, f"{prefix}_merged.txt")
    with open(docs_path, "w") as f:
        f.write("\n".join(docs) + "\n")
    with open(merged_path, "w") as f:
        f.write(" ".join(docs) + "\n")

    all_toks = " ".join(docs).split()
    print(f"ProductId={pid}  리뷰 {len(rows)}건 (전처리 후 {len(docs)}건)")
    print(f"  merged 총 토큰(중복포함)={len(all_toks)}  고유 토큰(그래프 노드 n)={len(set(all_toks))}")
    print(f"  리뷰별 토큰수: {[len(d.split()) for d in docs]}")
    print(f"저장: {os.path.basename(docs_path)} (1리뷰=1줄), {os.path.basename(merged_path)} (합본)")


if __name__ == "__main__":
    main()
