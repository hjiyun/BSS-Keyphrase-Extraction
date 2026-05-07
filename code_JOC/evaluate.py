"""
키프레이즈 추출 평가 지표 (Evaluation Metrics)
- author_truth와 reader_truth를 사용한 평가
- 스템된(stemmed) 키프레이즈 비교 지원

평가 지표:
1. Precision (정밀도): 추출한 것 중 정답인 비율
2. Recall (재현율): 정답 중 추출한 비율  
3. F1-Score: Precision과 Recall의 조화평균
4. Partial Match: 부분 일치 허용
5. Top-K Metrics: 상위 K개만 평가
"""

import os
import re
import json
import numpy as np
from collections import Counter
from datetime import datetime


# ========== 설정 ==========
# 정답 파일 경로 (사용자 환경에 맞게 수정)
DATA_DIR = "/home/jiyoon/BSS-Keyphrase-Extraction/data_JOC"
AUTHOR_TRUTH_DIR = os.path.join(DATA_DIR, "pre_process_author_truth")
READER_TRUTH_DIR = os.path.join(DATA_DIR, "pre_process_reader_truth")


# ========== 전처리 함수들 ==========
def normalize_phrase(phrase):
    """구(phrase) 정규화: 소문자, 특수문자 제거, 공백 정리"""
    phrase = phrase.lower().strip()
    phrase = re.sub(r'[^\w\s-]', ' ', phrase)  # 하이픈은 유지
    phrase = re.sub(r'\s+', ' ', phrase).strip()
    return phrase


def load_ground_truth(filepath):
    """
    정답 파일 로드 (쉼표로 구분된 형식)
    
    예: "ensembl kalman filter,data assimil methodolog,..."
    """
    if not os.path.exists(filepath):
        print(f"  [경고] 파일 없음: {filepath}")
        return []
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read().strip()
    
    # 쉼표로 분리
    keyphrases = []
    for part in content.split(','):
        part = normalize_phrase(part)
        if part and len(part) > 1:
            keyphrases.append(part)
    
    return keyphrases


def load_ground_truth_for_document(doc_name, truth_type='author'):
    """
    특정 문서의 정답 로드
    
    Parameters:
    - doc_name: 문서 이름 (예: "C-42.txt" 또는 "C-42")
    - truth_type: 'author' 또는 'reader'
    """
    # 확장자 제거
    doc_id = os.path.splitext(doc_name)[0]
    
    if truth_type == 'author':
        filepath = os.path.join(AUTHOR_TRUTH_DIR, doc_id)
    else:
        filepath = os.path.join(READER_TRUTH_DIR, doc_id)
    
    return load_ground_truth(filepath)


def load_predictions_from_results(results_dict, method='BSS', top_k=None):
    """
    예측 결과에서 키프레이즈 추출
    
    Parameters:
    - results_dict: print_and_save_results에서 반환된 딕셔너리
    - method: 'BSS', 'TextRank', 'Semi-supervised' 중 선택
    - top_k: 상위 K개만 사용 (None이면 전체)
    """
    if method not in results_dict['methods']:
        print(f"  [경고] 방법 '{method}'가 결과에 없습니다.")
        return []
    
    keyphrases = results_dict['methods'][method]
    
    if top_k:
        keyphrases = keyphrases[:top_k]
    
    return [normalize_phrase(item['phrase']) for item in keyphrases]


# ========== 매칭 함수들 ==========
def exact_match(pred, truth):
    """완전 일치 확인"""
    return normalize_phrase(pred) == normalize_phrase(truth)


def partial_match(pred, truth, threshold=0.5):
    """
    부분 일치 확인
    
    두 구가 공통 단어를 threshold 비율 이상 공유하면 일치로 간주
    예: "neural network" vs "deep neural network" → 일치 (2/3 > 0.5)
    """
    pred_words = set(normalize_phrase(pred).split())
    truth_words = set(normalize_phrase(truth).split())
    
    if not pred_words or not truth_words:
        return False
    
    intersection = pred_words & truth_words
    
    # 짧은 쪽 기준으로 비율 계산
    min_len = min(len(pred_words), len(truth_words))
    ratio = len(intersection) / min_len
    
    return ratio >= threshold


def contains_match(pred, truth):
    """
    포함 관계 확인
    
    한 구가 다른 구에 완전히 포함되면 일치로 간주
    예: "network" in "neural network" → 일치
    """
    pred_norm = normalize_phrase(pred)
    truth_norm = normalize_phrase(truth)
    
    # 단어 단위로 포함 관계 확인
    pred_words = set(pred_norm.split())
    truth_words = set(truth_norm.split())
    
    return pred_words.issubset(truth_words) or truth_words.issubset(pred_words)


def stem_match(pred, truth):
    """
    스템 기반 매칭 (접두사 매칭)
    
    단어의 앞 5글자가 같으면 일치로 간주
    예: "computing" vs "comput" → 일치
    """
    pred_words = normalize_phrase(pred).split()
    truth_words = normalize_phrase(truth).split()
    
    if len(pred_words) != len(truth_words):
        return False
    
    for pw, tw in zip(pred_words, truth_words):
        # 앞 5글자 비교 (또는 짧은 쪽 길이)
        min_len = min(len(pw), len(tw), 5)
        if pw[:min_len] != tw[:min_len]:
            return False
    
    return True


# ========== 평가 지표 계산 ==========
def calculate_metrics(predictions, ground_truth, match_type='exact'):
    """
    평가 지표 계산
    
    Parameters:
    - predictions: 예측된 키프레이즈 리스트
    - ground_truth: 정답 키프레이즈 리스트
    - match_type: 'exact', 'partial', 'contains', 'stem' 중 선택
    
    Returns:
    - dict: precision, recall, f1, matched_pairs 등
    """
    if not predictions or not ground_truth:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'true_positives': 0,
            'false_positives': len(predictions) if predictions else 0,
            'false_negatives': len(ground_truth) if ground_truth else 0,
            'total_predictions': len(predictions) if predictions else 0,
            'total_ground_truth': len(ground_truth) if ground_truth else 0,
            'matched_pairs': [],
            'unmatched_predictions': list(predictions) if predictions else [],
            'unmatched_truths': list(ground_truth) if ground_truth else []
        }
    
    # 매칭 함수 선택
    match_functions = {
        'exact': exact_match,
        'partial': partial_match,
        'contains': contains_match,
        'stem': stem_match
    }
    match_fn = match_functions.get(match_type, exact_match)
    
    # 매칭 수행
    matched_preds = set()
    matched_truths = set()
    matched_pairs = []
    
    for i, pred in enumerate(predictions):
        for j, truth in enumerate(ground_truth):
            if j in matched_truths:
                continue
            
            if match_fn(pred, truth):
                matched_preds.add(i)
                matched_truths.add(j)
                matched_pairs.append((pred, truth))
                break
    
    # 지표 계산
    true_positives = len(matched_pairs)
    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truth) - true_positives
    
    precision = true_positives / len(predictions) if predictions else 0
    recall = true_positives / len(ground_truth) if ground_truth else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 매칭되지 않은 항목
    unmatched_predictions = [pred for i, pred in enumerate(predictions) if i not in matched_preds]
    unmatched_truths = [truth for j, truth in enumerate(ground_truth) if j not in matched_truths]
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'total_predictions': len(predictions),
        'total_ground_truth': len(ground_truth),
        'matched_pairs': matched_pairs,
        'unmatched_predictions': unmatched_predictions,
        'unmatched_truths': unmatched_truths
    }


def calculate_top_k_metrics(predictions, ground_truth, k_values=[5, 10, 15, 20], match_type='exact'):
    """
    다양한 Top-K에 대한 평가 지표 계산
    """
    results = {}
    
    for k in k_values:
        top_k_preds = predictions[:k] if len(predictions) >= k else predictions
        metrics = calculate_metrics(top_k_preds, ground_truth, match_type)
        results[f'top_{k}'] = metrics
    
    return results


def evaluate_all_methods(results_dict, ground_truth, top_k=None, match_type='exact'):
    """
    모든 방법에 대해 평가
    """
    evaluation_results = {}
    
    for method in results_dict['methods'].keys():
        predictions = load_predictions_from_results(results_dict, method, top_k)
        metrics = calculate_metrics(predictions, ground_truth, match_type)
        evaluation_results[method] = metrics
    
    return evaluation_results


# ========== 결과 출력 함수들 ==========
def print_evaluation_summary(evaluation_results, title="평가 결과"):
    """평가 결과 요약 출력"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    
    # 헤더
    print(f"\n{'Method':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 80)
    
    for method, metrics in evaluation_results.items():
        print(f"{method:<20} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
              f"{metrics['f1']:>10.4f} {metrics['true_positives']:>6} "
              f"{metrics['false_positives']:>6} {metrics['false_negatives']:>6}")
    
    print("-" * 80)


def print_detailed_results(metrics, method_name="", show_all=False):
    """상세 평가 결과 출력"""
    print(f"\n{'='*70}")
    print(f"상세 평가 결과: {method_name}")
    print(f"{'='*70}")
    
    print(f"\n[기본 지표]")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['true_positives']}/{metrics['total_predictions']})")
    print(f"  Recall:    {metrics['recall']:.4f} ({metrics['true_positives']}/{metrics['total_ground_truth']})")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    
    max_show = 100 if show_all else 10
    
    print(f"\n[매칭된 키프레이즈] ({len(metrics['matched_pairs'])}개)")
    for pred, truth in metrics['matched_pairs'][:max_show]:
        if pred == truth:
            print(f"  ✓ {pred}")
        else:
            print(f"  ≈ {pred} ↔ {truth}")
    if len(metrics['matched_pairs']) > max_show:
        print(f"  ... 외 {len(metrics['matched_pairs']) - max_show}개")
    
    print(f"\n[누락된 정답] ({len(metrics['unmatched_truths'])}개)")
    for truth in metrics['unmatched_truths'][:max_show]:
        print(f"  ✗ {truth}")
    if len(metrics['unmatched_truths']) > max_show:
        print(f"  ... 외 {len(metrics['unmatched_truths']) - max_show}개")
    
    print(f"\n[잘못된 예측] ({len(metrics['unmatched_predictions'])}개)")
    for pred in metrics['unmatched_predictions'][:max_show]:
        print(f"  ✗ {pred}")
    if len(metrics['unmatched_predictions']) > max_show:
        print(f"  ... 외 {len(metrics['unmatched_predictions']) - max_show}개")


def print_top_k_results(top_k_results, method_name=""):
    """Top-K 평가 결과 출력"""
    print(f"\n[Top-K 평가: {method_name}]")
    print(f"{'K':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>6}")
    print("-" * 50)
    
    for k_name, metrics in sorted(top_k_results.items(), key=lambda x: int(x[0].split('_')[1])):
        k = k_name.replace('top_', '')
        print(f"{k:<10} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
              f"{metrics['f1']:>10.4f} {metrics['true_positives']:>6}")


# ========== 결과 저장 함수들 ==========
def save_evaluation_to_csv(all_results, filepath):
    """평가 결과를 CSV로 저장"""
    import csv
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 헤더
        writer.writerow(['Truth_Type', 'Match_Type', 'Method', 'Precision', 'Recall', 'F1', 
                        'TP', 'FP', 'FN', 'Total_Pred', 'Total_Truth'])
        
        for truth_type, match_results in all_results.items():
            for match_type, data in match_results.items():
                if 'full' in data:
                    for method, metrics in data['full'].items():
                        writer.writerow([
                            truth_type, match_type, method,
                            f"{metrics['precision']:.4f}",
                            f"{metrics['recall']:.4f}",
                            f"{metrics['f1']:.4f}",
                            metrics['true_positives'],
                            metrics['false_positives'],
                            metrics['false_negatives'],
                            metrics['total_predictions'],
                            metrics['total_ground_truth']
                        ])
    
    print(f"  CSV 저장: {filepath}")


def save_evaluation_to_txt(all_results, filepath, ground_truths):
    """평가 결과를 TXT로 저장"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("키프레이즈 추출 평가 결과\n")
        f.write("="*80 + "\n\n")
        
        for truth_type, truth_phrases in ground_truths.items():
            f.write(f"[{truth_type} 정답] ({len(truth_phrases)}개)\n")
            for phrase in truth_phrases:
                f.write(f"  - {phrase}\n")
            f.write("\n")
        
        for truth_type, match_results in all_results.items():
            f.write("\n" + "#"*80 + "\n")
            f.write(f"정답 유형: {truth_type}\n")
            f.write("#"*80 + "\n")
            
            for match_type, data in match_results.items():
                f.write(f"\n--- 매칭 타입: {match_type} ---\n")
                
                if 'full' in data:
                    f.write(f"\n{'Method':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}\n")
                    f.write("-" * 50 + "\n")
                    
                    for method, metrics in data['full'].items():
                        f.write(f"{method:<20} {metrics['precision']:>10.4f} "
                               f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f}\n")
    
    print(f"  TXT 저장: {filepath}")


def save_evaluation_to_json(all_results, filepath):
    """평가 결과를 JSON으로 저장"""
    # numpy 타입 변환
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(convert(all_results), f, ensure_ascii=False, indent=2)
    
    print(f"  JSON 저장: {filepath}")


# ========== 통합 평가 함수 ==========
def evaluate_keyphrase_extraction(
    results_dict,
    document_name,
    author_truth_dir=AUTHOR_TRUTH_DIR,
    reader_truth_dir=READER_TRUTH_DIR,
    match_types=['exact', 'partial', 'contains', 'stem'],
    top_k_values=[5, 10, 15, 20, 30, 50],
    output_dir=None
):
    """
    키프레이즈 추출 종합 평가
    
    Parameters:
    - results_dict: print_and_save_results에서 반환된 딕셔너리
    - document_name: 문서 이름 (예: "C-42.txt")
    - author_truth_dir: 저자 정답 디렉토리
    - reader_truth_dir: 독자 정답 디렉토리
    - match_types: 평가할 매칭 타입들
    - top_k_values: Top-K 평가할 K 값들
    - output_dir: 결과 저장 디렉토리 (None이면 저장 안 함)
    
    Returns:
    - dict: 종합 평가 결과
    """
    doc_id = os.path.splitext(document_name)[0]
    
    print(f"\n{'#'*80}")
    print(f"키프레이즈 추출 평가")
    print(f"문서: {document_name}")
    print(f"{'#'*80}")
    
    # 정답 로드
    ground_truths = {}
    
    author_path = os.path.join(author_truth_dir, doc_id)
    author_truth = load_ground_truth(author_path)
    if author_truth:
        ground_truths['Author'] = author_truth
        print(f"\n[저자 정답] {len(author_truth)}개 키프레이즈")
        for phrase in author_truth:
            print(f"  - {phrase}")
    
    reader_path = os.path.join(reader_truth_dir, doc_id)
    reader_truth = load_ground_truth(reader_path)
    if reader_truth:
        ground_truths['Reader'] = reader_truth
        print(f"\n[독자 정답] {len(reader_truth)}개 키프레이즈")
        for phrase in reader_truth:
            print(f"  - {phrase}")
    
    # Combined (저자 + 독자)
    if 'Author' in ground_truths and 'Reader' in ground_truths:
        combined = list(set(ground_truths['Author'] + ground_truths['Reader']))
        ground_truths['Combined'] = combined
        print(f"\n[통합 정답] {len(combined)}개 키프레이즈 (중복 제거)")
    
    if not ground_truths:
        print("\n[오류] 정답 데이터가 없습니다!")
        print(f"  확인 경로: {author_path}, {reader_path}")
        return None
    
    all_results = {}
    
    # 각 정답 유형에 대해 평가
    for truth_name, ground_truth in ground_truths.items():
        all_results[truth_name] = {}
        
        print(f"\n{'='*80}")
        print(f"평가: {truth_name} 정답 기준 ({len(ground_truth)}개)")
        print(f"{'='*80}")
        
        # 각 매칭 타입에 대해 평가
        for match_type in match_types:
            print(f"\n--- 매칭 타입: {match_type} ---")
            
            # 전체 평가
            eval_results = evaluate_all_methods(results_dict, ground_truth, match_type=match_type)
            print_evaluation_summary(eval_results, f"{truth_name} - {match_type}")
            
            all_results[truth_name][match_type] = {
                'full': eval_results,
                'top_k': {}
            }
            
            # Top-K 평가
            for method in results_dict['methods'].keys():
                predictions = load_predictions_from_results(results_dict, method)
                top_k_results = calculate_top_k_metrics(predictions, ground_truth, top_k_values, match_type)
                all_results[truth_name][match_type]['top_k'][method] = top_k_results
            
            # BSS Top-K 결과 출력
            if 'BSS' in all_results[truth_name][match_type]['top_k']:
                print_top_k_results(all_results[truth_name][match_type]['top_k']['BSS'], 'BSS')
    
    # 최고 성능 요약
    print(f"\n{'#'*80}")
    print("최고 성능 요약")
    print(f"{'#'*80}")
    
    for truth_name in ground_truths.keys():
        print(f"\n[{truth_name} 정답 기준]")
        best_f1 = 0
        best_config = ""
        best_metrics = None
        
        for match_type in match_types:
            for method, metrics in all_results[truth_name][match_type]['full'].items():
                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    best_config = f"{method} + {match_type} 매칭"
                    best_metrics = metrics
        
        if best_metrics:
            print(f"  최고 F1: {best_f1:.4f} ({best_config})")
            print(f"    Precision: {best_metrics['precision']:.4f}, Recall: {best_metrics['recall']:.4f}")
    
    # 결과 저장
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{doc_id}_evaluation_{timestamp}"
        
        print(f"\n결과 저장 중...")
        save_evaluation_to_csv(all_results, os.path.join(output_dir, f"{base_name}.csv"))
        save_evaluation_to_txt(all_results, os.path.join(output_dir, f"{base_name}.txt"), ground_truths)
        save_evaluation_to_json(all_results, os.path.join(output_dir, f"{base_name}.json"))
    
    return all_results


# ========== 간단 평가 함수 ==========
def quick_evaluate(predictions, ground_truth, match_type='stem'):
    """
    빠른 평가 (결과 리스트와 정답 리스트만 입력)
    
    Parameters:
    - predictions: 예측 키프레이즈 리스트 (문자열)
    - ground_truth: 정답 키프레이즈 리스트 (문자열)
    - match_type: 매칭 타입
    """
    # 정규화
    preds = [normalize_phrase(p) for p in predictions]
    truths = [normalize_phrase(t) for t in ground_truth]
    
    metrics = calculate_metrics(preds, truths, match_type)
    
    print(f"\n[평가 결과 - {match_type} 매칭]")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['true_positives']}/{len(preds)})")
    print(f"  Recall:    {metrics['recall']:.4f} ({metrics['true_positives']}/{len(truths)})")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    
    return metrics


# ========== 메인 실행 (테스트) ==========
if __name__ == "__main__":
    print("="*80)
    print("키프레이즈 추출 평가 모듈 테스트")
    print("="*80)
    
    # 테스트: C-42 정답 파일 형식
    test_author_truth = "ensembl kalman filter,data assimil methodolog,hydrocarbon reservoir simul,energi explor,tigr grid comput environ,grid comput,cyberinfrastructur develop project,high perform comput,tigr grid middlewar,strateg applic area,gridwai metaschedul,pool licens,grid-enabl"
    
    # 파싱 테스트
    parsed_truth = [normalize_phrase(p) for p in test_author_truth.split(',')]
    print(f"\n[정답 파싱 테스트]")
    print(f"원본: {test_author_truth[:60]}...")
    print(f"\n파싱 결과 ({len(parsed_truth)}개):")
    for phrase in parsed_truth:
        print(f"  - {phrase}")
    
    # 예시 예측 (스템된 형태)
    example_predictions = [
        "ensembl kalman filter",
        "data assimil methodolog", 
        "grid comput",
        "tigr grid",
        "reservoir simul",
        "strateg applic area",
        "gridwai metaschedul",
        "enkf",
        "job schedul"
    ]
    
    print(f"\n[예측 예시] ({len(example_predictions)}개)")
    for pred in example_predictions:
        print(f"  - {pred}")
    
    # 각 매칭 타입으로 평가
    print(f"\n{'='*60}")
    print("매칭 타입별 평가 결과")
    print(f"{'='*60}")
    
    for match_type in ['exact', 'partial', 'contains', 'stem']:
        metrics = calculate_metrics(example_predictions, parsed_truth, match_type)
        print(f"\n[{match_type} 매칭]")
        print(f"  Precision: {metrics['precision']:.4f} ({metrics['true_positives']}/{metrics['total_predictions']})")
        print(f"  Recall:    {metrics['recall']:.4f} ({metrics['true_positives']}/{metrics['total_ground_truth']})")
        print(f"  F1:        {metrics['f1']:.4f}")
        if metrics['matched_pairs']:
            print(f"  매칭된 쌍:")
            for pred, truth in metrics['matched_pairs'][:5]:
                print(f"    - {pred} ↔ {truth}")    