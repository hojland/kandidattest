#!/usr/bin/env python3
"""Process raw candidate data into compact format for the web app.
Includes national + local question stances in profile text for embedding."""
import json

STANCE_MAP = {-2: "Helt uenig", -1: "Uenig", 0: "Neutral", 1: "Enig", 2: "Helt enig"}

# Map storkreds area names to local question key prefixes
AREA_TO_PREFIX = {
    "Bornholms Storkreds": "tv2-fv26-bornholms-",
    "Fyns Storkreds": "tv2-fv26-fyns-",
    "Københavns Storkreds": "tv2-fv26-københavns-",
    "Københavns Omegns Storkreds": "tv2-fv26-københavns-omegns-",
    "Nordjyllands Storkreds": "tv2-fv26-nordjyllands-",
    "Nordsjællands Storkreds": "tv2-fv26-nordsjællands-",
    "Sjællands Storkreds": "tv2-fv26-sjællands-",
    "Sydjyllands Storkreds": "tv2-fv26-sydjyllands-",
    "Vestjyllands Storkreds": "tv2-fv26-vestjyllands-",
    "Østjyllands Storkreds": "tv2-fv26-østjyllands-",
}

with open("data/candidates_raw.json") as f:
    raw = json.load(f)

with open("data/questions.json") as f:
    national_questions = json.load(f)

with open("data/local_questions.json") as f:
    local_questions_by_area = json.load(f)

def get_local_questions(area_name):
    """Get local question mapping for a storkreds."""
    for area, prefix in AREA_TO_PREFIX.items():
        if area == area_name:
            slug = prefix.replace("tv2-fv26-", "").rstrip("-")
            return local_questions_by_area.get(slug, {})
    return {}

candidates = []
storkreds_set = set()

for c in raw:
    answers = {}
    profile_parts = []

    if c.get("pitch"):
        profile_parts.append(c["pitch"])

    # National questions
    for qkey, qtext in national_questions.items():
        ans = (c.get("answers") or {}).get(qkey, {})
        score = ans.get("answer")
        comment = ans.get("comment", "")
        if score is not None:
            answers[qkey] = {"score": score, "comment": comment}
            profile_parts.append(f"{qtext}: {STANCE_MAP.get(score, '')}")
            if comment:
                profile_parts.append(comment)

    # Local questions for this candidate's storkreds
    local_qs = get_local_questions(c.get("area", ""))
    for qkey, qtext in local_qs.items():
        ans = (c.get("answers") or {}).get(qkey, {})
        score = ans.get("answer")
        comment = ans.get("comment", "")
        if score is not None:
            answers[qkey] = {"score": score, "comment": comment}
            profile_parts.append(f"[Lokalt] {qtext}: {STANCE_MAP.get(score, '')}")
            if comment:
                profile_parts.append(comment)

    if not answers:
        continue

    area = c.get("area", "")
    storkreds_set.add(area)

    candidates.append({
        "id": c["id"],
        "name": c["name"],
        "party": c["party"],
        "partyLetter": c["partyLetter"],
        "area": area,
        "age": c.get("age"),
        "occupation": c.get("occupation"),
        "pitch": c.get("pitch", ""),
        "priorities": c.get("priorities", []),
        "answers": answers,
        "profileText": "\n".join(profile_parts),
    })

print(f"Processed {len(candidates)} candidates with answers")

# Full version with profileText (for embedding generation)
with open("data/candidates.json", "w") as f:
    json.dump(candidates, f, ensure_ascii=False)

# Compact version without profileText (for frontend, smaller)
compact = [{k: v for k, v in c.items() if k != "profileText"} for c in candidates]
with open("public/candidates.json", "w") as f:
    json.dump(compact, f, ensure_ascii=False)

# Storkreds metadata for the frontend dropdown
storkredse = sorted(storkreds_set)
storkreds_meta = []
for s in storkredse:
    count = sum(1 for c in candidates if c["area"] == s)
    storkreds_meta.append({"name": s, "candidateCount": count})
with open("public/storkredse.json", "w") as f:
    json.dump(storkreds_meta, f, ensure_ascii=False)

print(f"Saved public/candidates.json ({len(compact)} candidates)")
print(f"Saved public/storkredse.json ({len(storkreds_meta)} storkredse)")
