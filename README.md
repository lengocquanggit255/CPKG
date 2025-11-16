
## Overview
CPKG (Career Path Knowledge Graph) is an AI-powered system designed to match CVs with job descriptions through skill extraction, requirement summarization, and intelligent matching algorithms.

## Project Structure

```
CPKG/
├── src/
│   ├── req_skill_matching/
│   │   ├── train_match_model.py          # Training script for skill matching model
│   │   └── llm_matching_req_skill.py     # LLM-based requirement-skill matching
│   ├── jd_req_sum/
│   │   ├── train_sum_model.py            # Training script for summarization model
│   │   └── llm_req_sum.py                # LLM-based requirement summarization
│   └── downstream_tasks/
│       ├── skill_based_matching.py       # Skill-based CV-JD matching
│       ├── resume_evaluation.py          # Resume scoring and evaluation
│       └── career_guidance.py            # Career path recommendations
├── data/
│   ├── req_sum.jsonl                    # Job requirement summarization dataset
│   └── req_skill_matching.jsonl         # Skill-requirement matching dataset
└── README.md                             # This file
```

## Key Components

### 1. Requirement Skill Matching (`req_skill_matching/`)
Matches individual skills from CVs with job requirements using machine learning and LLM.

**Files:**
- `train_match_model.py`: Trains the skill matching model
- `llm_matching_req_skill.py`: Uses LLM for advanced matching

**Input Format (req_skill_matching.jsonl):**
```json
{"skill": "string", "req": "string", "label": 0/1}
```

### 2. Job Description Requirement Summarization (`jd_req_sum/`)
Summarizes and extracts key requirements from job descriptions.

**Files:**
- `train_sum_model.py`: Trains the summarization model
- `llm_req_sum.py`: Uses LLM for requirement extraction

### 3. Downstream Tasks (`downstream_tasks/`)

#### a) Skill-Based Matching (`skill_based_matching.py`)
Matches CVs to job positions based on extracted skills.

#### b) Resume Evaluation (`resume_evaluation.py`)
Scores and evaluates resumes against job requirements.

#### c) Career Guidance (`career_guidance.py`)
Provides career path recommendations based on current and required skills.

## Data Format

### Requirement-Skill Matching Dataset
```jsonl
{
  "skill": "Java programming experience",
  "req": "Need backend developer proficient in Java",
  "label": 1
}
```

### Job Requirement Summarization Dataset
```jsonl
{
  "job_description": "full text of JD",
  "summary": "extracted key requirements"
}
```

## Usage

### Training Models
```bash
# Train skill matching model
python src/req_skill_matching/train_match_model.py

# Train summarization model
python src/jd_req_sum/train_sum_model.py
```

### Running Matching Tasks
```bash
# Skill-based matching
python src/downstream_tasks/skill_based_matching.py

# Resume evaluation
python src/downstream_tasks/resume_evaluation.py

# Career guidance
python src/downstream_tasks/career_guidance.py
```

## Dataset Statistics

- **Requirement-Skill Matching:** 1000+ pairs
- **Job Descriptions:** 500+ unique positions
- **Languages:** Vietnamese, English

## Features

✅ Skill extraction from CVs
✅ Job requirement summarization
✅ Intelligent skill-to-requirement matching
✅ Resume scoring and evaluation
✅ Career path recommendations
✅ Multi-language support (Vietnamese, English)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License.
