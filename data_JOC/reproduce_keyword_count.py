#!/usr/bin/env python3
"""
graph.generate (Keyphrase_functions.R:32-51) / create_fcm_words 로직을 그대로 재현하여
Hulth short_articles 500개 문서의 '키워드 수'를 논문과 동일한 방식으로 카운트한다.

핵심 로직 (원본과 동일):
  1. 본문: pre_process/{id}.abstr 을 소문자화 + [\t\r;] 제거 + \n->공백 + 공백분리
     -> 고유 토큰 집합 = 그래프 정점(후보 단어) 사전
  2. 키워드: {id}.uncontr 을 동일하게 처리하여 단어 리스트로 분해
  3. truth = 사전(본문 토큰)에 실제로 존재하는 uncontr 단어들의 고유 집합
     -> 이 truth 개수가 논문이 말한 'number of keywords'
"""
import glob, os, re
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__)) + "/short_articles"

def normalize(text):
    text = text.lower()
    text = re.sub(r'[\t\r;]', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()

def doc_keyword_count(doc_id):
    # 1. 본문 사전 (그래프 정점)
    with open(f"{ROOT}/pre_process/{doc_id}.abstr") as f:
        vocab = set(normalize(f.read()))
    # 2. uncontr 키워드 단어
    with open(f"{ROOT}/{doc_id}.uncontr") as f:
        key_words = normalize(f.read())
    # 3. 사전에 존재하는 키워드 단어만 (truth)
    truth = set(w for w in key_words if w in vocab)
    return len(truth), len(vocab), len(set(key_words))

ids = sorted([os.path.basename(f).rsplit('.', 1)[0] for f in glob.glob(f"{ROOT}/*.uncontr")], key=int)

rows = []
for i in ids:
    tcount, vcount, rawk = doc_keyword_count(i)
    rows.append((i, tcount, vcount, rawk))

truth_counts = [r[1] for r in rows]
n = len(rows)
sv = sorted(truth_counts)

print(f"=== 재현 결과: {n}개 문서, '본문에 존재하는 uncontr 키워드 단어' 개수 ===")
print(f"min={sv[0]}  1Q={sv[n//4]}  median={sv[n//2]}  mean={sum(sv)/n:.2f}  3Q={sv[3*n//4]}  max={sv[-1]}")
print()

# 히스토그램
c = Counter(truth_counts)
print(" k  : 문서수")
for k in range(0, 16):
    print(f" {k:2d} : {c.get(k,0)}")
print(" 16+:", sum(v for k,v in c.items() if k >= 16))
print()

# 논문 필터: >=10 (216 목표)
for thr in [10]:
    ge = [r for r in rows if r[1] >= thr]
    print(f">> 키워드 >= {thr} 인 문서: {len(ge)}개   (논문: 216)")

# sparse 꼬리
print()
for thr in [2,3,4,5]:
    le = [r for r in rows if r[1] <= thr]
    print(f"sparse: 키워드 <= {thr} : {len(le)}개")

# CSV 저장
import csv
with open(os.path.dirname(os.path.abspath(__file__)) + "/keyword_counts.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "keyword_count_in_graph", "vocab_size", "raw_uncontr_words"])
    w.writerows(rows)
print("\n저장: data_JOC/keyword_counts.csv")
