import math
import heapq
import requests
import json


# ============================================================
# 1. NODE ENCODING / DECODING
# ============================================================

def encode_node(title: str, level: str) -> str:
    """Convert a (title, level) pair to a string key."""
    return f"{title}||{level}"

def decode_node(code: str):
    """Convert a string key back to (title, level)."""
    title, level = code.split("||")
    return title, level


# ============================================================
# 2. CAREER TRANSITION GRAPH
# ============================================================

class CareerGraph:
    def __init__(self):
        # graph[u][v] = transition probability from u to v
        self.graph = {}

    def add_edge(self, title_u, level_u, title_v, level_v, probability):
        """
        Add a transition from one role to another.
        Example: (Backend Engineer, Junior) → (Backend Engineer, Senior)
        """
        u = encode_node(title_u, level_u)
        v = encode_node(title_v, level_v)

        if u not in self.graph:
            self.graph[u] = {}
        self.graph[u][v] = probability

    def best_path(self, start_code: str, end_code: str):
        """
        Find the most likely career transition path.
        Instead of multiplying probabilities directly (which leads to underflow),
        we sum the negative log of each probability.

        The smaller the summed negative log, the higher the overall probability.
        This allows us to use a Dijkstra-style shortest-path approach.
        """
        pq = []
        heapq.heappush(pq, (0.0, start_code, [start_code]))  # cost, node, path
        visited = {}

        while pq:
            cost, u, path = heapq.heappop(pq)

            if u == end_code:
                # Convert the accumulated log cost back to probability
                total_probability = math.exp(-cost)
                return path, total_probability

            if u in visited and visited[u] <= cost:
                continue
            visited[u] = cost

            if u not in self.graph:
                continue

            for v, prob in self.graph[u].items():
                if prob <= 0:
                    continue
                new_cost = cost - math.log(prob)
                heapq.heappush(pq, (new_cost, v, path + [v]))

        return None, None


# ============================================================
# 3. SKILL GAP COMPUTATION
# ============================================================

def compute_skill_gap(current_requirements, target_requirements):
    """
    Compute the skill gap:
    These are the requirements present in the target role
    but missing from the current role.
    """
    return list(set(target_requirements) - set(current_requirements))


# ============================================================
# 4. LLM CALL USING QWEN 14B (OPENROUTER)
# ============================================================

OPENROUTER_API_KEY = "YOUR_API_KEY_HERE"

def call_qwen_guidance(skill_gap, best_path):
    """
    Send the computed skill gap and optimal career path to Qwen 14B
    for generating structured career guidance.
    """
    SYSTEM_PROMPT = """
You are an experienced career advisor specializing in IT job transitions. 
Provide clear, structured, and actionable advice based on the skill gap 
and the recommended transition path.
"""

    USER_PROMPT = f"""
Skill gap (missing requirements):
{json.dumps(skill_gap, indent=2, ensure_ascii=False)}

Recommended transition path:
{[decode_node(n) for n in best_path]}

Your tasks:
1. Explain why this transition path is reasonable.
2. Explain the missing skills and why they matter.
3. Recommend a structured learning plan and concrete next steps.
4. Give motivational and practical guidance for reaching the target role.
"""

    payload = {
        "model": "qwen/Qwen2.5-14B-Instruct",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        data=json.dumps(payload)
    )

    return response.json()["choices"][0]["message"]["content"]


# ============================================================
# 5. FULL CAREER GUIDANCE PIPELINE
# ============================================================

def career_guidance_pipeline(
    graph: CareerGraph,
    current_title: str, current_level: str,
    target_title: str, target_level: str,
    current_requirements: list, target_requirements: list
):
    """
    Overall process:
    1. Find best path from current to target role.
    2. Compute the skill gap.
    3. Ask Qwen 14B to generate human-level guidance.
    """
    v_c = encode_node(current_title, current_level)
    v_t = encode_node(target_title, target_level)

    # Step 1: best transition path
    best_path, probability_score = graph.best_path(v_c, v_t)
    if best_path is None:
        raise ValueError("No valid transition path found.")

    # Step 2: skill gap
    skill_gap = compute_skill_gap(current_requirements, target_requirements)

    # Step 3: LLM reasoning
    guidance = call_qwen_guidance(skill_gap, best_path)

    return {
        "best_path": [decode_node(n) for n in best_path],
        "path_probability": probability_score,
        "skill_gap": skill_gap,
        "guidance": guidance
    }


# ============================================================
# 6. EXAMPLE RUN (YOU CAN REMOVE OR EDIT)
# ============================================================

if __name__ == "__main__":
    G = CareerGraph()

    # Add sample transitions
    G.add_edge("Backend Engineer", "Intern", "Backend Engineer", "Junior", 0.83)
    G.add_edge("Backend Engineer", "Junior", "Backend Engineer", "Senior", 0.74)
    G.add_edge("Backend Engineer", "Senior", "Backend Engineer", "Lead", 0.62)
    G.add_edge("Backend Engineer", "Junior", "DevOps Engineer", "Junior", 0.22)

    # Example requirement sets
    current_req = ["Python", "REST API", "Git", "Basic SQL"]
    target_req = ["Python", "REST API", "Git", "Advanced SQL", "System Design", "Leadership"]

    result = career_guidance_pipeline(
        graph=G,
        current_title="Backend Engineer",
        current_level="Junior",
        target_title="Backend Engineer",
        target_level="Lead",
        current_requirements=current_req,
        target_requirements=target_req
    )

    print("\n--- Best Path ---")
    print(result["best_path"])

    print("\n--- Path Probability ---")
    print(result["path_probability"])

    print("\n--- Skill Gap ---")
    print(result["skill_gap"])

    print("\n--- AI Career Guidance ---")
    print(result["guidance"])
