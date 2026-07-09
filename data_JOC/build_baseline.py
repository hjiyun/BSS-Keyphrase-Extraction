#!/usr/bin/env python3
"""
원논문(Wang et al. 2023) 전처리를 최대한 충실히 재현하여 baseline 데이터셋을 생성한다.

파이프라인 (논문 1.5.1.1 + 원본 create_fcm_words 기준):
  raw abstract -> word_tokenize -> POS 태깅 -> 명사(NN*)+형용사(JJ*)만 정점
              -> 소문자화 -> (옵션) Porter stemming -> 후보 단어 시퀀스(순서/중복 보존)
  키워드(uncontr) -> 동일 정규화 -> 단어 분해 -> 그래프 정점에 존재하는 것만 truth

입력  : short_articles/{id}.abstr (raw), short_articles/{id}.uncontr (gold)
출력  : data_JOC/baseline_preprocessed/   (원본 미변경)
  pre_process/{id}.abstr   : POS 필터된 후보 단어 (공백 구분, fcm 입력용) — 500개 전체
  truth/{id}.uncontr       : 정규화된 gold 키워드 단어 (공백 구분) — 500개 전체
  doc_stats.csv            : id, n_candidate_words, n_keywords_in_graph, group
  selected_ids.txt         : 키워드 >= 10 인 문서 ID (baseline / dense, 391개)
  sparse_ids.txt           : 키워드 <  10 인 문서 ID (sparse, 109개)
"""
import glob, os, re, csv, sys
import nltk
from nltk.stem import PorterStemmer

STEMMING = "--stem" in sys.argv          # 기본 미적용 (논문 Table1.1과 더 일치)
MIN_KEYWORDS = 10                        # 논문: 10개 미만 제외

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = BASE + "/short_articles"
OUT = BASE + "/baseline_preprocessed"
os.makedirs(OUT + "/pre_process", exist_ok=True)
os.makedirs(OUT + "/truth", exist_ok=True)

ps = PorterStemmer()
keep_pos = lambda tag: tag.startswith("NN") or tag.startswith("JJ")
norm = lambda w: ps.stem(w.lower()) if STEMMING else w.lower()
has_alpha = lambda w: re.search(r"[a-z]", w.lower()) is not None

def candidate_words(text):
    """POS 필터된 후보 단어 시퀀스 (그래프 정점, 순서·중복 보존)."""
    text = text.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    out = []
    for w, tag in nltk.pos_tag(nltk.word_tokenize(text)):
        if keep_pos(tag) and has_alpha(w):
            out.append(norm(w))
    return out

def keyword_words(text):
    """uncontr 키프레이즈를 정규화된 단어로 분해."""
    text = re.sub(r"[\t\r;]", " ", text).replace("\n", " ")
    return [norm(w) for w in text.split() if has_alpha(w)]

ids = sorted([os.path.basename(f).rsplit(".", 1)[0] for f in glob.glob(f"{SRC}/*.abstr")], key=int)

stats, dense, sparse = [], [], []
for i in ids:
    with open(f"{SRC}/{i}.abstr") as f:
        cand = candidate_words(f.read())
    vocab = set(cand)
    with open(f"{SRC}/{i}.uncontr") as f:
        kws = keyword_words(f.read())
    truth = [w for w in dict.fromkeys(kws) if w in vocab]   # 그래프에 존재하는 고유 키워드(순서보존)

    # 전처리는 dense/sparse 구분 없이 500개 전체에 동일 적용
    with open(f"{OUT}/pre_process/{i}.abstr", "w") as g:
        g.write(" ".join(cand))
    with open(f"{OUT}/truth/{i}.uncontr", "w") as g:
        g.write(" ".join(truth))

    grp = "dense" if len(truth) >= MIN_KEYWORDS else "sparse"
    stats.append((i, len(vocab), len(truth), grp))
    (dense if grp == "dense" else sparse).append(i)

with open(f"{OUT}/doc_stats.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "n_candidate_words", "n_keywords_in_graph", "group"])
    w.writerows(stats)
with open(f"{OUT}/selected_ids.txt", "w") as f:   # dense / baseline (>=10)
    f.write("\n".join(dense) + "\n")
with open(f"{OUT}/sparse_ids.txt", "w") as f:     # sparse (<10)
    f.write("\n".join(sparse) + "\n")

def summ(group_ids):
    kc = sorted(dict((s[0], s[2]) for s in stats)[i] for i in group_ids)
    n = len(kc)
    return f"{n}개 | min={kc[0]} median={kc[n//2]} mean={sum(kc)/n:.2f} max={kc[-1]}"

print(f"전처리 사양: POS=명사+형용사, stemming={'ON' if STEMMING else 'OFF'}, 윈도우2 (500개 전체 동일 적용)")
print(f"전체     : {summ(ids)}")
print(f"dense  (>= {MIN_KEYWORDS}) : {summ(dense)}   [baseline, selected_ids.txt]")
print(f"sparse (<  {MIN_KEYWORDS}) : {summ(sparse)}   [sparse_ids.txt]")
print(f"출력: baseline_preprocessed/{{pre_process,truth}}/(500), doc_stats.csv, selected_ids.txt, sparse_ids.txt")
