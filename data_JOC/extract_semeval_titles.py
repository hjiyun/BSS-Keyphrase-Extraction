"""SemEval-2010(midas/semeval2010 raw)에서 제목을 복원.

구조: [제목] [저자명 소속 이메일]... ABSTRACT ...
제목 경계 찾기: 첫 이메일의 local-part에서 저자명 조각을 얻어, 헤더에서 그 이름이 처음
등장하는 위치를 찾고 그 앞까지를 제목으로 자른다.
(소속명 'High Performance Computing Center' 등이 정답 키워드와 겹치므로 반드시 제외해야 함)

검증: 논문 §1.5.2는 C-42 제목에서 '8개 관측 키워드'를 얻었다 → 재현되는지 확인.
출력: data_JOC/semeval_titles.json  {paper_id: {"title": str, "title_stems": [...]}}
"""
import json, re, os, sys
from huggingface_hub import hf_hub_download
from nltk.stem import PorterStemmer

HERE = os.path.dirname(os.path.abspath(__file__))
ps = PorterStemmer()
STOP = set("""a an the of for on in to and or with by from at as is are was were be been being
this that these those we our us it its their his her he she they them you your i
how does do did what which when where why not no if then than so such very can could
using use used toward towards via into over under between among through during
new novel a.k.a based""".split())

AFFIL = re.compile(r'^(univ|universit|department|dept|institut|laborator|lab|research|center|centre|'
                   r'school|college|corp|inc|ltd|gmbh|academy|faculty|division|microsoft|yahoo|google|'
                   r'ibm|intel|at&t|hp|nokia|siemens)', re.I)

def norm(t):
    return re.sub(r'[^a-z0-9]', '', t.lower())

def find_title(tokens):
    """제목 토큰 구간 [0:cut) 을 반환."""
    # 1) ABSTRACT 위치로 헤더 한정
    abs_i = None
    for i, t in enumerate(tokens[:400]):
        if re.fullmatch(r'abstract', t.strip(' .:'), re.I):
            abs_i = i; break
    header_end = abs_i if abs_i else min(len(tokens), 200)
    header = tokens[:header_end]

    # 2) 첫 이메일에서 저자명 조각 추출
    name_frags = set()
    email_i = None
    for i, t in enumerate(header):
        if '@' in t:
            email_i = i
            local = t.split('@')[0]
            for frag in re.split(r'[._\-\d]+', local):
                if len(frag) >= 3: name_frags.add(frag.lower())
            break

    # 3) 저자명이 헤더에서 처음 등장하는 위치 → 그 앞이 제목
    cut = None
    if name_frags:
        for i, t in enumerate(header):
            tn = norm(t)
            if len(tn) < 3: continue
            if any(tn == f or (len(tn) >= 4 and (tn in f or f in tn)) for f in name_frags):
                cut = i; break
        if cut is not None:
            # 이름 앞의 given name 도 이메일에 있으면 함께 뒤로 물림
            while cut > 1:
                tn = norm(header[cut-1])
                if len(tn) >= 3 and any(tn == f or (len(tn) >= 4 and (tn in f or f in tn)) for f in name_frags):
                    cut -= 1
                else:
                    break

    # 4) fallback: 첫 소속 키워드 앞 (저자명 2토큰 여유를 두고 자름)
    if cut is None or cut < 2:
        for i, t in enumerate(header):
            if i >= 3 and AFFIL.match(t.strip(' ,.')):
                cut = max(2, i - 2); break
    if cut is None:
        cut = min(len(header), 20) if not email_i else max(2, email_i - 3)
    return tokens[:cut]

def stems_of(title_tokens):
    out = []
    for t in title_tokens:
        w = re.sub(r'[^A-Za-z0-9\-]', '', t).lower()
        if not w or w in STOP or len(w) < 2: continue
        # 하이픈 단어는 통째 + 분해 둘 다 (grid-enabled → grid-enabl, grid, enabl)
        parts = [w] + (w.split('-') if '-' in w else [])
        for p in parts:
            p = p.strip('-')
            if p and p not in STOP and len(p) >= 2:
                out.append(ps.stem(p))
    return sorted(set(out))

def main():
    p = hf_hub_download('midas/semeval2010', 'train.jsonl', repo_type='dataset')
    rows = [json.loads(l) for l in open(p)]
    res = {}
    for r in rows:
        toks = r['document']
        tt = find_title(toks)
        res[r['paper_id']] = {'title': ' '.join(tt), 'title_stems': stems_of(tt)}
    out = os.path.join(HERE, 'semeval_titles.json')
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1)
    print(f"제목 복원: {len(res)}개 → {out}\n")

    # ---- 검증: C-42 (논문 §1.5.2 = 제목에서 8개 관측) ----
    tdir = os.path.join(HERE, 'pre_process_reader_truth')
    for pid in ['C-42']:
        print(f"=== {pid} ===")
        print("제목:", res[pid]['title'])
        print("제목 stem:", res[pid]['title_stems'])
        kwf = os.path.join(tdir, pid)
        if os.path.exists(kwf):
            kw_words = set()
            for phrase in open(kwf).read().strip().split(','):
                kw_words.update(phrase.strip().split())
            obs = sorted(set(res[pid]['title_stems']) & kw_words)
            print("정답 키워드 단어들:", sorted(kw_words))
            print(f"\n>>> 관측(제목 ∩ 정답) = {len(obs)}개: {obs}")
            print(f">>> 논문 §1.5.2 = 8개  →  {'✅ 일치' if len(obs)==8 else '⚠️ 불일치'}")

if __name__ == '__main__':
    main()
