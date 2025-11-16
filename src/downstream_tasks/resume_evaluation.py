import io
import os
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("Agg")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification



# ============================================================================
# 1. CVProcessor — phiên bản mới dùng model Matching HuggingFace (Phobert)
# ============================================================================

class CVProcessor:
    """
    Class đánh giá CV theo JD:
    - Dùng model matching fine-tuned trên req–skill (Phobert-large)
    - Không còn dùng TF-IDF/XGBoost
    """

    STRENGTH_THRESHOLD = 0.70
    WEAKNESS_THRESHOLD = 0.40

    RUBRIC = [
        (0.8, 1.01, 0.95, "Mức độ phù hợp rất cao",
         "Bạn đã phù hợp với hầu hết các yêu cầu của vị trí này!"),
        (0.5, 0.8, 0.75, "Mức độ phù hợp cao",
         "Bạn phù hợp tốt với phần lớn yêu cầu của vị trí này."),
        (0.25, 0.5, 0.5, "Mức độ phù hợp khá",
         "Bạn có nền tảng tốt nhưng cần bổ sung thêm kỹ năng."),
        (0.0, 0.25, 0.25, "Mức độ phù hợp trung bình",
         "Bạn mới chỉ bắt đầu làm quen với những yêu cầu của vị trí."),
    ]

    def __init__(self, model_dir: str, device: str = None):
        """
        model_dir: thư mục chứa model fine-tuned hoặc tên repo HF.
        """
        print(f"Đang tải model matching từ: {model_dir}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        print("Tải model thành công — CVProcessor sẵn sàng.")

    # ---------------------------------------------------------------
    # Matching score giữa 1 skill CV và 1 skill JD
    # ---------------------------------------------------------------
    def _skill_match_score(self, skill_cv: str, skill_jd: str) -> float:
        """Trả về xác suất match (0-1)."""

        try:
            tokens = self.tokenizer(
                skill_jd,
                skill_cv,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding="max_length"
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**tokens).logits
                probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
                score = float(probs[1])      # class=1 (match)

            return score

        except Exception as e:
            print("Lỗi khi tính match score:", e)
            return 0.0

    # ---------------------------------------------------------------
    # Evaluate tất cả các skill trong CV theo JD
    # ---------------------------------------------------------------
    def _evaluate_scores(self, cv_skills: List[str], jd_skills: List[Dict]) -> Dict[str, Any]:

        cluster_results = {}
        detailed_scores = {}
        total_score = 0.0

        for cluster in jd_skills:
            cluster_name = cluster["cluster"]
            cluster_weight = cluster["weight"]

            cluster_total = 0.0
            skill_details = []

            for sk in cluster["skills"]:
                jd_name = sk["name"]
                w = sk["weight"]

                # Tính điểm cao nhất giữa skill JD và toàn bộ CV skills
                scores = [self._skill_match_score(cv_s, jd_name) for cv_s in cv_skills]
                best_score = max(scores) if scores else 0.0

                cluster_total += best_score * w
                total_score += best_score * w * cluster_weight

                skill_details.append({
                    "name": jd_name,
                    "score": best_score
                })

            # Tính điểm trung bình (chuẩn hóa theo tổng weight)
            cluster_norm = cluster_total / sum(s["weight"] for s in cluster["skills"])

            cluster_results[cluster_name] = cluster_norm
            detailed_scores[cluster_name] = skill_details

        return {
            "total_score": total_score,
            "cluster_scores": cluster_results,
            "detailed_scores": detailed_scores
        }

    # ---------------------------------------------------------------
    # Generate phản hồi dựa trên rubric
    # ---------------------------------------------------------------
    def _generate_feedback(self, results: Dict[str, Any]) -> Dict[str, Any]:

        total_score = results["total_score"]
        cluster_scores = results["cluster_scores"]
        detailed_scores = results["detailed_scores"]

        final_level = ""
        final_feedback = ""
        display_score = 0.0

        for min_s, max_s, disp, level, feedback in self.RUBRIC:
            if min_s <= total_score < max_s:
                final_level = level
                final_feedback = feedback
                display_score = disp
                break

        # Điểm mạnh/yếu
        strengths = []
        weaknesses = []
        for cluster, skills in detailed_scores.items():
            for s in skills:
                if s["score"] >= self.STRENGTH_THRESHOLD:
                    strengths.append(s["name"])
                elif s["score"] <= self.WEAKNESS_THRESHOLD:
                    weaknesses.append(s["name"])

        # Cluster summary
        strong_clusters = [k for k, v in cluster_scores.items() if v >= self.STRENGTH_THRESHOLD]
        weak_clusters = [k for k, v in cluster_scores.items() if v <= self.WEAKNESS_THRESHOLD]

        cluster_summary = ""
        if strong_clusters:
            cluster_summary += f"Bạn mạnh ở nhóm: {', '.join(strong_clusters)}. "
        if weak_clusters:
            cluster_summary += f"Bạn cần cải thiện ở nhóm: {', '.join(weak_clusters)}."
        if not cluster_summary:
            cluster_summary = "Không có điểm mạnh/yếu rõ rệt."

        return {
            "final_score_raw": total_score,
            "final_score_display": display_score,
            "final_level": final_level,
            "general_feedback": final_feedback,
            "cluster_summary": cluster_summary,
            "strengths": strengths if strengths else ["-"],
            "weaknesses": weaknesses if weaknesses else ["-"],
            "cluster_scores_percent": {k: v * 100 for k, v in cluster_scores.items()},
        }

    # ---------------------------------------------------------------
    # Vẽ radar chart và trả về bytes
    # ---------------------------------------------------------------
    def _create_radar_chart(self, scores: Dict[str, float]) -> bytes:

        labels = list(scores.keys())
        values = list(scores.values())
        num = len(labels)

        angles = np.linspace(0, 2 * np.pi, num, endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        ax.plot(angles, values, color="blue", linewidth=2)
        ax.fill(angles, values, color="blue", alpha=0.3)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(["25%", "50%", "75%", "100%"])
        ax.set_ylim(0, 100)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

        return buf.read()

    # ---------------------------------------------------------------
    # Hàm process chính
    # ---------------------------------------------------------------
    def process(self, cv_skills: List[str], jd_skills: List[Dict]) -> Dict[str, Any]:

        print(f"Đang chấm CV với {len(cv_skills)} skill...")

        results = self._evaluate_scores(cv_skills, jd_skills)
        feedback = self._generate_feedback(results)
        radar_img = self._create_radar_chart(feedback["cluster_scores_percent"])

        return {
            "score_raw": feedback["final_score_raw"],
            "score_display": feedback["final_score_display"],
            "level": feedback["final_level"],
            "feedback_general": feedback["general_feedback"],
            "feedback_cluster": feedback["cluster_summary"],
            "strengths": feedback["strengths"],
            "weaknesses": feedback["weaknesses"],
            "cluster_scores": feedback["cluster_scores_percent"],
            "radar_chart_image": radar_img
        }



# ============================================================================
# 2. DATA MẪU (GIỮ NGUYÊN)
# ============================================================================

cleaned_jd_skills_list = [
    {
        "cluster": "Kỹ năng chuyên môn",
        "weight": 0.4,
        "skills": [
            {"name": "Kiến thức về kinh tế, tài chính, quản trị kinh doanh", "weight": 0.0152},
            {"name": "Tư duy hướng dữ liệu, chủ động và trách nhiệm cao", "weight": 0.0909},
            {"name": "Kiến thức về ngành ứng dụng và game mobile", "weight": 0.8939},
        ],
    },
    {
        "cluster": "Công cụ công nghệ",
        "weight": 0.3,
        "skills": [
            {"name": "Thành thạo Excel hoặc Google Sheets", "weight": 0.1126},
            {"name": "Sử dụng SQL để truy vấn dữ liệu", "weight": 0.1937},
            {"name": "Sử dụng Python cho phân tích dữ liệu", "weight": 0.3518},
            {"name": "Thành thạo các công cụ BI (Tableau, Power BI,...)", "weight": 0.1561},
            {"name": "Kinh nghiệm với AppMagic, SensorTower, App Annie", "weight": 0.1858},
        ],
    },
    {
        "cluster": "Kỹ năng mềm",
        "weight": 0.1,
        "skills": [
            {"name": "Kỹ năng phân tích và tư duy phản biện", "weight": 0.4954},
            {"name": "Khả năng trình bày dữ liệu dễ hiểu", "weight": 0.1446},
            {"name": "Kỹ năng giao tiếp tốt", "weight": 0.2985},
            {"name": "Khả năng tự học, tiếp thu nhanh", "weight": 0.0615},
        ],
    },
    {
        "cluster": "Kinh nghiệm chuyên môn",
        "weight": 0.2,
        "skills": [
            {"name": "Kinh nghiệm phân tích và báo cáo dữ liệu", "weight": 0.9586},
            {"name": "Thành thạo tiếng Anh", "weight": 0.0414},
        ],
    },
]

cv_skills_da = [
    "Sử dụng Python cho phân tích dữ liệu (pandas, numpy)",
    "Thành thạo SQL để truy vấn database",
    "Trực quan hóa dữ liệu bằng Power BI và Tableau",
    "Sử dụng Excel và Google Sheets thành thạo",
    "Có kinh nghiệm phân tích và làm báo cáo",
    "Hiểu biết về ngành game mobile",
    "Kỹ năng giao tiếp tốt",
    "Tiếng Anh đọc hiểu"
]



# ============================================================================
# 3. CHẠY DEMO
# ============================================================================

if __name__ == "__main__":

    MODEL_MATCHING_PATH = "models/req_skill_matching"  

    if not os.path.exists(MODEL_MATCHING_PATH):
        print("Không tìm thấy model:", MODEL_MATCHING_PATH)
        exit()

    processor = CVProcessor(model_dir=MODEL_MATCHING_PATH)

    print("\n================= ĐÁNH GIÁ CV =================")
    results = processor.process(cv_skills_da, cleaned_jd_skills_list)

    print("\n--- KẾT QUẢ ---")
    print("Điểm raw:", results["score_raw"])
    print("Điểm hiển thị:", results["score_display"])
    print("Level:", results["level"])
    print("Nhận xét:", results["feedback_general"])
    print("Nhận xét nhóm:", results["feedback_cluster"])
    print("Điểm mạnh:", results["strengths"])
    print("Điểm yếu:", results["weaknesses"])

    # Lưu biểu đồ
    with open("radar_chart.png", "wb") as f:
        f.write(results["radar_chart_image"])

    print("Đã lưu radar chart → radar_chart.png")
