#!/usr/bin/env python3
"""
논문 1.5.1.1 전처리를 최대한 충실히 재현한다 (POS 필터 추가 버전).

논문 서술:
  - 토큰화(tokenization)
  - POS 태깅 후 정보량 적은 단어(접속사/전치사/동사 등) 제거
    -> 명사/형용사만 후보 단어(그래프 정점)로 남김
  - 윈도우 2 공출현 무방향 그래프
  - 키워드 10개 미만 문서 제외 -> 216개 목표

원본 short_articles/*.abstr (제목+본문 raw) 에서 시작한다.
원본은 건드리지 않고 data_JOC/repro_pos/ 에 결과를 저장한다.

POS 후보: 명사(NN*) + 형용사(JJ*). 논문이 명시적 stemmer를 안 밝혔으므로,
키워드-사전 매칭 일관성을 위해 양쪽 모두 Porter stemming 적용한 버전도 함께 계산한다.
"""
import glob, os, re, csv
from collections import Counter
import nltk
from nltk.stem import PorterStemmer

ROOT = os.path.dirname(os.path.abspath(__file__)) + "/short_articles"
OUT = os.path.dirname(os.path.abspath(__file__)) + "/repro_pos"
os.makedirs(OUT, exist_ok=True)
os.makedirs(OUT + "/pre_process", exist_ok=True)

ps = PorterStemmer()
KEEP_POS = lambda tag: tag.startswith("NN") or tag.startswith("JJ")  # 명사 + 형용사

def candidate_words(text, stem=False):
    """토큰화 -> POS 태깅 -> 명사/형용사만 -> (옵션)stemming. 등장 순서 보존."""
    text = text.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    out = []
    for w, tag in tagged:
        wl = w.lower()
        if not re.search(r"[a-z]", wl):      # 순수 기호/숫자 제거
            continue
        if KEEP_POS(tag):
            out.append(ps.stem(wl) if stem else wl)
    return out

def key_words(text, stem=False):
    """uncontr 키프레이즈를 단어로 분해 (POS 필터 없음 - 정답은 그대로 단어화)."""
    text = re.sub(r"[\t\r;]", " ", text).replace("\n", " ")
    out = []
    for w in text.split():
        wl = w.lower().strip()
        if not re.search(r"[a-z]", wl):
            continue
        out.append(ps.stem(wl) if stem else wl)
    return out

ids = sorted([os.path.basename(f).rsplit(".", 1)[0] for f in glob.glob(f"{ROOT}/*.abstr")], key=int)

def run(stem):
    rows = []
    for i in ids:
        with open(f"{ROOT}/{i}.abstr") as f:
            cand = candidate_words(f.read(), stem=stem)   # 후보 단어 (그래프 정점, 순서)
        vocab = set(cand)
        with open(f"{ROOT}/{i}.uncontr") as f:
            kws = key_words(f.read(), stem=stem)
        truth = set(w for w in kws if w in vocab)          # 그래프에 존재하는 정답 키워드
        rows.append((i, len(truth), len(vocab)))
        if not stem:   # 후보단어 텍스트 저장 (stemming 안 한 버전만 파일로)
            with open(f"{OUT}/pre_process/{i}.abstr", "w") as g:
                g.write(" ".join(cand))
    return rows

def report(rows, label):
    tc = [r[1] for r in rows]
    n = len(rows); sv = sorted(tc)
    ge10 = sum(1 for x in tc if x >= 10)
    print(f"\n=== {label} ===")
    print(f"min={sv[0]} 1Q={sv[n//4]} median={sv[n//2]} mean={sum(sv)/n:.2f} 3Q={sv[3*n//4]} max={sv[-1]}")
    print(f"키워드 >= 10 문서: {ge10}개   (논문 216)")
    sub = sorted(x for x in tc if x >= 10)
    if sub:
        m = len(sub)
        print(f"  └ ≥10 부분집합 통계: min={sub[0]} median={sub[m//2]} mean={sum(sub)/m:.2f} max={sub[-1]}  (논문 Table1.1: min11 med19 mean19.83 max42)")
    return ge10

print(f"문서 {len(ids)}개 처리 중...")
rows_plain = run(stem=False)
rows_stem  = run(stem=True)
report(rows_plain, "POS(명사+형용사) 필터, stemming 미적용")
report(rows_stem,  "POS(명사+형용사) 필터 + Porter stemming")

# CSV 저장
with open(OUT + "/keyword_counts_pos.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "kw_pos_nostem", "vocab_nostem", "kw_pos_stem", "vocab_stem"])
    for a, b in zip(rows_plain, rows_stem):
        w.writerow([a[0], a[1], a[2], b[1], b[2]])
print(f"\n저장: repro_pos/keyword_counts_pos.csv,  후보단어 텍스트: repro_pos/pre_process/*.abstr")
