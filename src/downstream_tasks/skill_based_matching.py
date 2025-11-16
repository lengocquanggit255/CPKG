from typing import List, Dict
from cv_processor import CVProcessor


# -----------------------------------------------------------
# Tính điểm cho 1 job dựa trên CV
# -----------------------------------------------------------
def compute_job_score(
    processor: CVProcessor,
    cv_skills: List[str],
    jd_skills: List[Dict]
) -> float:
    """
    Compute matching score between a resume and a single job.
    
    Use the existing _evaluate_scores() inside CVProcessor.
    """
    results = processor._evaluate_scores(cv_skills, jd_skills)
    return results["total_score"]


# -----------------------------------------------------------
# Ranking job theo điểm match với CV
# -----------------------------------------------------------
def rank_jobs_for_resume(
    processor: CVProcessor,
    cv_skills: List[str],
    job_list: List[Dict]
) -> List[Dict]:
    """
    Rank all job nodes for one resume.

    job_list: List like:
        [
          {
            "job_id": "backend_junior",
            "jd_skills": <same schema as cleaned_jd_skills_list>,
          },
          ...
        ]

    Output: sorted list of:
        [
          {"job_id": "...", "score": ...},
          ...
        ]
    """

    ranking = []

    for job in job_list:
        score = compute_job_score(processor, cv_skills, job["jd_skills"])
        ranking.append({
            "job_id": job["job_id"],
            "score": score
        })

    ranking.sort(key=lambda x: x["score"], reverse=True)
    return ranking


# -----------------------------------------------------------
# Ranking CV cho 1 job (hướng ngược lại)
# -----------------------------------------------------------
def rank_resumes_for_job(
    processor: CVProcessor,
    resumes: List[Dict],
    jd_skills: List[Dict]
) -> List[Dict]:
    """
    Rank multiple candidate resumes for a single job.

    resumes: List like:
        [
            {"candidate_id": "cv01", "skills": [...]},
            {"candidate_id": "cv02", "skills": [...]},
            ...
        ]
    """

    results = []

    for cv in resumes:
        score = compute_job_score(processor, cv["skills"], jd_skills)
        results.append({
            "candidate_id": cv["candidate_id"],
            "score": score
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# -----------------------------------------------------------
# DEMO
# -----------------------------------------------------------
if __name__ == "__main__":

    print("=== DEMO SKILL-BASED MATCHING ===")

    MODEL_MATCHING_PATH = "models/req_skill_matching"

    processor = CVProcessor(model_dir=MODEL_MATCHING_PATH)

    cv_skills_demo = [
        "Python", "SQL", "Power BI", "Tableau", "Excel", 
        "Phân tích và báo cáo dữ liệu", "Tiếng Anh"
    ]

    # Job repository demo
    job_repo = [
        {
            "job_id": "data_analyst_junior",
            "jd_skills": [
                {
                    "cluster": "Technical",
                    "weight": 0.5,
                    "skills": [
                        {"name": "Python", "weight": 0.30},
                        {"name": "SQL", "weight": 0.25}
                    ]
                },
                {
                    "cluster": "Tools",
                    "weight": 0.3,
                    "skills": [
                        {"name": "Power BI", "weight": 0.20},
                        {"name": "Tableau", "weight": 0.20},
                        {"name": "Excel", "weight": 0.30}
                    ]
                },
                {
                    "cluster": "Soft Skills",
                    "weight": 0.2,
                    "skills": [
                        {"name": "Communication", "weight": 0.50},
                        {"name": "Critical Thinking", "weight": 0.50}
                    ]
                }
            ]
        },
        {
            "job_id": "business_analyst_intern",
            "jd_skills": [
                {
                    "cluster": "Business",
                    "weight": 0.6,
                    "skills": [
                        {"name": "BRD Documentation", "weight": 0.50},
                        {"name": "User Story", "weight": 0.50}
                    ]
                },
                {
                    "cluster": "Soft Skills",
                    "weight": 0.4,
                    "skills": [
                        {"name": "Communication", "weight": 0.50},
                        {"name": "Presentation", "weight": 0.50}
                    ]
                }
            ]
        }
    ]

    # Run ranking
    ranking = rank_jobs_for_resume(processor, cv_skills_demo, job_repo)

    print("\n--- Job Ranking ---")
    for idx, item in enumerate(ranking, 1):
        print(f"{idx}. {item['job_id']} — Score = {item['score']:.4f}")
