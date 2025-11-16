import requests
import pandas as pd
import time
import os
import json

API_KEYS = [
]


DELAY_BETWEEN_REQUESTS_SECONDS = 10

SYSTEM_PROMPT = """
You are an expert in analyzing IT-related skills.
Your task is to compare a job requirement from a job description (jd_requirement) with a skill extracted from a resume (cv_skill), and classify their relationship into one of the following four labels:

0 – Disjoint:  
The two items belong to different domains, share no meaningful overlap, and cannot substitute for each other in IT work.

1 – Related: Includes the following cases:
- Partial semantic overlap: the two items share some common ground (same domain, same purpose, or used in similar types of work) but neither fully contains the other.
- jd_requirement is a broad concept or domain, and cv_skill is a narrower subfield, subtype, or specific variant within it. Condition: everyone who has cv_skill is assumed to satisfy jd_requirement, but not vice versa.
- cv_skill is a broad concept or domain, and jd_requirement is a narrower subfield, subtype, or specific variant within it. Condition: everyone who satisfies jd_requirement is assumed to have cv_skill, but not vice versa.

Input: a list of 5 requirement–skill pairs:
- Pair 1: jd_requirement = "{jd_req_1}", cv_skill = "{cv_skill_1}"
- Pair 2: jd_requirement = "{jd_req_2}", cv_skill = "{cv_skill_2}"
- Pair 3: jd_requirement = "{jd_req_3}", cv_skill = "{cv_skill_3}"
- Pair 4: jd_requirement = "{jd_req_4}", cv_skill = "{cv_skill_4}"
- Pair 5: jd_requirement = "{jd_req_5}", cv_skill = "{cv_skill_5}"

Output requirement: return the result in JSON format:
{
  "pair_1": 0 or 1,
  "pair_2": 0 or 1,
  "pair_3": 0 or 1,
  "pair_4": 0 or 1,
  "pair_5": 0 or 1
}
"""


def get_summary_from_ai(api_key, skill_pairs):
    """
    Cấu hình API key và gọi API OpenRouter để so sánh 5 cặp kỹ năng.
    """
    try:
        final_prompt = SYSTEM_PROMPT.format(
            jd_req_1=skill_pairs[0][0], cv_skill_1=skill_pairs[0][1],
            jd_req_2=skill_pairs[1][0], cv_skill_2=skill_pairs[1][1],
            jd_req_3=skill_pairs[2][0], cv_skill_3=skill_pairs[2][1],
            jd_req_4=skill_pairs[3][0], cv_skill_4=skill_pairs[3][1],
            jd_req_5=skill_pairs[4][0], cv_skill_5=skill_pairs[4][1]
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "qwen/qwen-2.5-14b-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": final_prompt
                }
            ],
            "temperature": 0
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            return f"ERROR: HTTP {response.status_code} - {response.text}"

    except Exception as e:
        error_message = f"An error occurred: {e}"
        print(f"ERROR: {error_message}")
        return f"ERROR: {error_message}"

def get_summary_from_ai_with_retry(api_key, skill_pairs, max_retries=3):
    """
    Gọi API để so sánh 5 cặp kỹ năng với retry logic cho lỗi 429.
    """
    for attempt in range(max_retries):
        try:
            result = get_summary_from_ai(api_key, skill_pairs)
            
    
            error_patterns = [
                "429 You exceeded your current quota",
                "429 Resource has been exhausted",
                "Resource has been exhausted",
                "check quota",
                "quota"
            ]
            
            is_quota_error = any(pattern in result for pattern in error_patterns)
            
    
            if not is_quota_error:
                return result
            
    
            if attempt < max_retries - 1:
        
                print(f"Gặp lỗi quota limit 429. Đang chờ {wait_time} giây trước khi thử lại (lần {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                print(f"Đã thử {max_retries} lần nhưng vẫn gặp lỗi quota limit.")
                return result
                
        except Exception as e:
            error_message = f"An error occurred: {e}"
            print(f"ERROR: {error_message}")
            
    
            if "429" in str(e) or "quota" in str(e).lower() or "exhausted" in str(e).lower():
                if attempt < max_retries - 1:
                    wait_time = 60 * (attempt + 1)
                    print(f"Gặp lỗi quota trong exception. Đang chờ {wait_time} giây trước khi thử lại...")
                    time.sleep(wait_time)
                else:
                    return f"ERROR: {error_message}"
            else:
        
                if attempt < max_retries - 1:
                    print(f"Đang thử lại lần {attempt + 2}/{max_retries}...")
                    time.sleep(30)
                else:
                    return f"ERROR: {error_message}"
    
    return "ERROR: Max retries exceeded"

def process_skills_comparison():
    """
    Xử lý file CSV chứa cặp CV skills và JD requirements và áp dụng AI để so sánh theo batch 5 cặp.
    """
    csv_file = ""
    output_file = ""
    
    try:
        df = pd.read_csv(csv_file)
        print(f"Đã đọc {len(df)} cặp kỹ năng từ file {csv_file}")
        print("Các cột trong file:", df.columns.tolist())
    except FileNotFoundError:
        print(f"Không tìm thấy file {csv_file}")
        return
    
    required_columns = ['jd_requirement', 'cv_skill']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Thiếu các cột cần thiết: {missing_columns}")
        return
    
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            if len(existing_df) == len(df) and 'ai_prediction' in existing_df.columns:
        
                existing_predictions = existing_df['ai_prediction'].fillna('').astype('string')
                df['ai_prediction'] = existing_predictions
                processed_count = existing_df['ai_prediction'].notna().sum()
                print(f"Tìm thấy file kết quả hiện có với {processed_count} cặp đã xử lý.")
            else:
                df['ai_prediction'] = pd.Series(dtype='string')
        except Exception as e:
            print(f"Lỗi khi đọc file hiện có: {e}. Bắt đầu từ đầu...")
            df['ai_prediction'] = pd.Series(dtype='string')
    else:

        df['ai_prediction'] = pd.Series(dtype='string')
    
    total_pairs = len(df)
    key_count = len(API_KEYS)
    exhausted_keys = set()
    
    unprocessed_mask = df['ai_prediction'].isna() | (df['ai_prediction'] == "")
    remaining_pairs = unprocessed_mask.sum()
    print(f"Còn lại {remaining_pairs} cặp chưa xử lý...")
    
    nan_count = df['ai_prediction'].isna().sum()
    empty_string_count = (df['ai_prediction'] == "").sum()
    print(f"  - Số dòng có ai_prediction = NaN: {nan_count}")
    print(f"  - Số dòng có ai_prediction = '': {empty_string_count}")
    
    batch_size = 5
    for batch_start in range(0, total_pairs, batch_size):
        batch_end = min(batch_start + batch_size, total_pairs)
        batch_indices = list(range(batch_start, batch_end))
        

        batch_processed = True
        for idx in batch_indices:
            current_prediction = df.loc[idx, 'ai_prediction']
            if pd.isna(current_prediction) or current_prediction == "":
                batch_processed = False
                break
        
        if batch_processed:
            print(f"Batch {batch_start+1}-{batch_end} đã được xử lý, bỏ qua...")
            continue
        

        if len(exhausted_keys) >= key_count:
            print("Tất cả API keys đã hết quota. Không thể tiếp tục xử lý.")
            break
        

        skill_pairs = []
        for idx in batch_indices:
            if idx < total_pairs:
                skill_pairs.append((df.loc[idx, 'jd_requirement'], df.loc[idx, 'cv_skill']))
            else:
        
                skill_pairs.append(("", ""))
        

        key_index = (batch_start // batch_size) % key_count
        attempts = 0
        while key_index in exhausted_keys and attempts < key_count:
            key_index = (key_index + 1) % key_count
            attempts += 1
            
        if attempts >= key_count:
            print("Không còn API key khả dụng.")
            break
            
        current_key = API_KEYS[key_index]
        

        for i, (skill1, skill2) in enumerate(skill_pairs[:len(batch_indices)]):
            print(f"  Cặp {batch_start + i + 1}: '{skill1}' vs '{skill2}'")
        

        result = get_summary_from_ai_with_retry(current_key, skill_pairs, max_retries=3)
        

        error_patterns = [
            "ERROR: An error occurred: 429",
            "Resource has been exhausted",
            "check quota",
            "quota"
        ]
        
        if any(pattern in str(result) for pattern in error_patterns):
            exhausted_keys.add(key_index)
    
            
    
            if len(exhausted_keys) < key_count:
                next_key_index = (key_index + 1) % key_count
                attempts = 0
                while next_key_index in exhausted_keys and attempts < key_count:
                    next_key_index = (next_key_index + 1) % key_count
                    attempts += 1
                    
                if attempts < key_count:
            
                    current_key = API_KEYS[next_key_index]
                    result = get_summary_from_ai_with_retry(current_key, skill_pairs, max_retries=2)
                    
            
                    if any(pattern in str(result) for pattern in error_patterns):
                        exhausted_keys.add(next_key_index)
        

        try:
            import json
            if result.startswith("ERROR:"):
        
                for idx in batch_indices:
                    df.loc[idx, 'ai_prediction'] = str(result)
            else:
        
        
                json_start = result.find('{')
                json_end = result.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = result[json_start:json_end]
                    parsed_result = json.loads(json_str)
                    
            
                    for i, idx in enumerate(batch_indices):
                        pair_key = f"pair_{i+1}"
                        if pair_key in parsed_result:
                            df.loc[idx, 'ai_prediction'] = str(parsed_result[pair_key])
                        else:
                            df.loc[idx, 'ai_prediction'] = "ERROR: Missing result"
                else:
            
                    for idx in batch_indices:
                        df.loc[idx, 'ai_prediction'] = f"ERROR: Invalid JSON format - {result}"
        except json.JSONDecodeError as e:
            print(f"Lỗi parse JSON: {e}")
            for idx in batch_indices:
                df.loc[idx, 'ai_prediction'] = f"ERROR: JSON parse error - {result}"
        except Exception as e:
            print(f"Lỗi xử lý kết quả: {e}")
            for idx in batch_indices:
                df.loc[idx, 'ai_prediction'] = f"ERROR: Processing error - {result}"
        
        print(f"-> Kết quả batch: {result}")
        

        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"-> Đã lưu kết quả vào {output_file}")
        

        if batch_end < total_pairs:
            print(f"Đang chờ {DELAY_BETWEEN_REQUESTS_SECONDS} giây...")
            time.sleep(DELAY_BETWEEN_REQUESTS_SECONDS)
    
    print(f"\nHoàn tất! Kết quả đã được lưu vào file '{output_file}'.")

def main():
    """
    Hàm chính để điều phối toàn bộ quy trình.
    """
    process_skills_comparison()


if __name__ == "__main__":
    main()
