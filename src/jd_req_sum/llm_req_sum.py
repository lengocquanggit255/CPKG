import requests
import pandas as pd
import json
import time
import os
import glob

# --- Cấu hình ---
API_KEYS = [

]

DELAY_BETWEEN_REQUESTS_SECONDS = 2

SYSTEM_PROMPT = """
You are an expert in text analysis. Your task is to read a list of skill-related sentences belonging to the same cluster and summarize them into a single representative sentence, following these requirements:

- Produce an abstractive summary; do not copy any sentence verbatim.
- Select only one core skill that best represents the entire cluster.
- If there are synonymous expressions, keep only the most common phrasing.
- For software tools, always normalize the format to "using [tool]".
- Do not use parentheses, do not add explanations, and do not provide comments.
- Output must be a single concise line.

List of skills:
{skill_list}

Output format: one single concise summary line only.
"""



# ------------------------------------------------------------------------------------------
# 1. FUNCTION: Gọi Qwen-14B qua OpenRouter
# ------------------------------------------------------------------------------------------
def call_qwen_summary(api_key, skills_list, key_index=None):
    """
    Gọi Qwen-14B thông qua OpenRouter API để tóm tắt danh sách kỹ năng.
    """
    try:
        formatted_skills = "\n".join([f'"{s}",' for s in skills_list])
        final_prompt = SYSTEM_PROMPT.format(skill_list=formatted_skills)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://your-app.com",
            "X-Title": "Skill Summarization",
        }

        payload = {
            "model": "qwen/qwen-14b-chat",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": final_prompt}
            ],
            "temperature": 0
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=60
        )

        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            return content.strip()

        # Lỗi quota/throttle từ OpenRouter
        if response.status_code == 429:
            return f"ERROR: 429 Rate limit for API key #{key_index+1}"

        # Lỗi key hết hạn
        if response.status_code in [401, 403]:
            return f"ERROR: API key #{key_index+1} expired or unauthorized."

        return f"ERROR: HTTP {response.status_code}: {response.text}"

    except Exception as e:
        return f"ERROR: Exception calling Qwen: {str(e)}"


# ------------------------------------------------------------------------------------------
# 2. Retry logic đơn giản cho OpenRouter
# ------------------------------------------------------------------------------------------
def get_skill_summary_with_retry(api_key, skills_list, key_index=None, max_retries=3):
    for i in range(max_retries):
        result = call_qwen_summary(api_key, skills_list, key_index)

        if "429" not in result:
            return result

        wait_time = (i + 1) * 15
        print(f"⚠️ API key #{key_index+1} bị rate limit. Chờ {wait_time} giây...")
        time.sleep(wait_time)

    return result


# ------------------------------------------------------------------------------------------
# 3. XỬ LÝ CLUSTERING — ĐÃ BỎ SKIP FILE
# ------------------------------------------------------------------------------------------
def process_clustering_results():

    results_folder = ""
    output_folder = ""

    os.makedirs(output_folder, exist_ok=True)

    # Lấy tất cả file
    result_files = glob.glob(os.path.join(results_folder, "clustering_results_*.json"))

    if not result_files:
        print("❌ Không tìm thấy file clustering_results_*.json")
        return

    print(f"📄 Tìm thấy {len(result_files)} file, sẽ xử lý toàn bộ (không skip):")
    for f in result_files:
        print(" -", os.path.basename(f))

    key_count = len(API_KEYS)
    exhausted_keys = set()

    for file_idx, result_file in enumerate(result_files):

        base_name = os.path.basename(result_file)
        output_name = base_name.replace("clustering_results_", "skill_summary_")
        output_path = os.path.join(output_folder, output_name)

        print("\n" + "=" * 80)
        print(f"🔍 ĐANG XỬ LÝ FILE: {base_name}")
        print("=" * 80)

        with open(result_file, "r", encoding="utf-8") as f:
            clustering_data = json.load(f)

        clusters = clustering_data.get("clusters", [])

        print(f"→ Tìm thấy {len(clusters)} clusters")

        # load progress
        existing = {}
        processed_ids = set()

        if os.path.exists(output_path):
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                for item in existing.get("skill_summaries", []):
                    processed_ids.add(item["cluster_id"])
                print(f"→ File output đã có, {len(processed_ids)} cluster đã xử lý.")
            except:
                existing = {}

        if not existing:
            existing = {
                "metadata": {
                    "source_file": base_name,
                    "timestamp": clustering_data.get("metadata", {}).get("timestamp", ""),
                    "total_clusters": len(clusters),
                    "processing_timestamp": pd.Timestamp.now().isoformat()
                },
                "skill_summaries": []
            }

        remaining_clusters = [
            c for c in clusters if c["cluster_id"] not in processed_ids
        ]

        print(f"→ Còn {len(remaining_clusters)} clusters chưa xử lý.")

        for cluster_idx, cluster in enumerate(remaining_clusters):

            cluster_id = cluster["cluster_id"]
            sentences = cluster["sentences"]

            print(f"\n🔹 Cluster {cluster_id} ({len(sentences)} sentences)")

            if len(exhausted_keys) == key_count:
                print("❌ Tất cả API keys đã exhausted → Dừng.")
                break

            # chọn key
            key_index = (cluster_idx + file_idx * 100) % key_count
            while key_index in exhausted_keys:
                key_index = (key_index + 1) % key_count

            api_key = API_KEYS[key_index]

            # gọi LLM
            summary = get_skill_summary_with_retry(api_key, sentences, key_index)

            # key lỗi
            if "expired" in summary or "unauthorized" in summary:
                exhausted_keys.add(key_index)
                print(f"❌ API key #{key_index+1} hết hạn → skip key.")
                continue

            if "429" in summary:
                exhausted_keys.add(key_index)
                print(f"❌ API key #{key_index+1} rate limit → skip key.")
                continue

            # lưu kết quả
            existing["skill_summaries"].append({
                "cluster_id": cluster_id,
                "original_sentences_count": len(sentences),
                "original_sentences": sentences,
                "skill_summary": summary,
                "processing_key": f"API_KEY_{key_index+1}"
            })

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)

            print(f"✅ Summary: {summary}")

            time.sleep(DELAY_BETWEEN_REQUESTS_SECONDS)

        print(f"🎉 Hoàn thành file: {base_name}")


def main():
    process_clustering_results()


if __name__ == "__main__":
    main()
