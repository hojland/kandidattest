#!/usr/bin/env python3
"""Extract question texts from TV2 using positional matching within topic sections."""
import json
from playwright.sync_api import sync_playwright

CANDIDATES_PER_STORKREDS = {
    "bornholms": [123165004],
    "fyns": [124250494],
    "københavns": [124153742],
    "københavns-omegns": [124250593],
    "nordjyllands": [124269137],
    "nordsjællands": [124253085],
    "sjællands": [124290041],
    "sydjyllands": [124284506],
    "vestjyllands": [124253065],
    "østjyllands": [124250599],
}

with open("data/candidates_raw.json") as f:
    raw = json.load(f)
raw_by_id = {c["id"]: c for c in raw}

JS_STRUCTURED = """
() => {
    const sections = document.querySelectorAll('.tc_candidatetest__candidate__rating__item');
    const results = [];
    for (const section of sections) {
        const summary = section.querySelector('summary');
        const topic = summary ? summary.textContent.trim() : '';
        const items = section.querySelectorAll('.tc_candidatetest__candidate__rating__item__body');
        const questions = [];
        for (const item of items) {
            const p = item.querySelector('p');
            questions.push(p ? p.textContent.trim() : '');
        }
        results.push({ topic, questions });
    }
    return results;
}
"""


def scrape_page(page, candidate_id):
    url = f"https://nyheder.tv2.dk/folketingsvalg/kandidat/{candidate_id}"
    page.goto(url, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(3000)
    try:
        page.locator('button:has-text("Acceptér alle")').click(timeout=5000)
        page.wait_for_timeout(1000)
    except Exception:
        pass
    for _ in range(30):
        page.evaluate("window.scrollBy(0, 800)")
        page.wait_for_timeout(200)
    page.wait_for_timeout(3000)
    return page.evaluate(JS_STRUCTURED)


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()

    # National questions are consistent across all candidates
    # Local questions differ per storkreds
    national_questions_ordered = []  # list of texts in page order
    local_questions_by_slug = {}  # slug -> list of texts in page order

    for slug, cand_ids in CANDIDATES_PER_STORKREDS.items():
        candidate_id = cand_ids[0]
        print(f"\nProcessing {slug} ({candidate_id})...")
        sections = scrape_page(page, candidate_id)

        storkreds_display = raw_by_id[candidate_id].get("area", "").replace(" Storkreds", "")
        local_texts = []
        national_texts = []

        for sec in sections:
            topic = sec["topic"]
            is_local = "Storkreds" in topic or storkreds_display in topic
            for q in sec["questions"]:
                if is_local:
                    local_texts.append(q)
                else:
                    national_texts.append(q)

        print(f"  Local: {len(local_texts)}, National: {len(national_texts)}")

        # Local questions: map positionally to keys
        local_prefix = f"tv2-fv26-{slug}-"
        local_keys = sorted(
            [k for k in (raw_by_id[candidate_id].get("answers") or {})
             if k.startswith(local_prefix) and (raw_by_id[candidate_id]["answers"][k].get("answer") is not None)],
            key=lambda x: int(x.split("-")[-1])
        )

        # The storkreds section shows local questions in order
        # But we might get fewer questions than keys if some are in national topics
        # For now, map positionally
        local_qs = {}
        for i, text in enumerate(local_texts):
            if i < len(local_keys):
                local_qs[local_keys[i]] = text
                print(f"  {local_keys[i]}: {text[:80]}")
        local_questions_by_slug[slug] = local_qs

        # National questions should be same across all candidates (24 total)
        if not national_questions_ordered:
            national_questions_ordered = national_texts

    browser.close()

    # Map national questions to keys using the fixed ordering
    # National questions on page are grouped by topic, same order for all candidates
    # Keys are tv2-fv26-danmark-1 through -24
    # We need to figure out which page index maps to which key number

    # The page groups national questions by topic. Each topic has 2 questions.
    # The topics and their question numbers need to be determined.
    # Since all 10 candidates show the same 24 national questions in the same order,
    # we can use any candidate's known answer data to map.

    # Use first candidate's data to map by stance matching with the full national list
    first_cand_id = list(CANDIDATES_PER_STORKREDS.values())[0][0]
    first_cand = raw_by_id[first_cand_id]
    answers = first_cand.get("answers", {})

    # Get national keys sorted by number
    national_keys = sorted(
        [k for k in answers if k.startswith("tv2-fv26-danmark-") and answers[k].get("answer") is not None],
        key=lambda x: int(x.split("-")[-1])
    )

    print(f"\nNational: {len(national_questions_ordered)} texts, {len(national_keys)} keys")

    # The issue: page shows 24 national questions but in topic order, not key order.
    # Key order is danmark-1, -2, ..., -24
    # Page order groups by topic.
    # We need to map page order to key order.

    # Strategy: use MULTIPLE candidates. For each page position, check which key
    # has the matching stance across ALL candidates. Unique match = confirmed.

    # Collect stances per page position across all candidates
    page_position_stances = {}  # page_idx -> {candidate_id: stance_score}
    key_stances = {}  # key -> {candidate_id: stance_score}

    for slug, cand_ids in CANDIDATES_PER_STORKREDS.items():
        candidate_id = cand_ids[0]
        cand = raw_by_id[candidate_id]
        cand_answers = cand.get("answers", {})
        for k in national_keys:
            s = cand_answers.get(k, {}).get("answer")
            if s is not None:
                key_stances.setdefault(k, {})[candidate_id] = s

    # Now re-scrape to get national stances per page position
    # Actually, let me scrape stance data too
    JS_WITH_STANCES = """
    () => {
        const sections = document.querySelectorAll('.tc_candidatetest__candidate__rating__item');
        const results = [];
        for (const section of sections) {
            const summary = section.querySelector('summary');
            const topic = summary ? summary.textContent.trim() : '';
            const items = section.querySelectorAll('.tc_candidatetest__candidate__rating__item__body');
            const questions = [];
            for (const item of items) {
                const p = item.querySelector('p');
                const btn = item.querySelector('button[data-candidateanswer="true"]');
                const stanceText = btn ? btn.textContent.trim() : null;
                const stanceMap = {'Helt uenig': -2, 'Uenig': -1, 'Neutral': 0, 'Enig': 1, 'Helt enig': 2};
                questions.push({
                    text: p ? p.textContent.trim() : '',
                    stance: stanceText ? stanceMap[stanceText] : null
                });
            }
            results.push({ topic, questions });
        }
        return results;
    }
    """

    browser2 = p.chromium.launch(headless=True)
    ctx2 = browser2.new_context()
    page2 = ctx2.new_page()

    # Collect national question stances from multiple candidates for mapping
    position_data = []  # [{text, stances: {cand_id: score}}]
    for i in range(len(national_questions_ordered)):
        position_data.append({"text": national_questions_ordered[i], "stances": {}})

    for slug, cand_ids in CANDIDATES_PER_STORKREDS.items():
        candidate_id = cand_ids[0]
        url = f"https://nyheder.tv2.dk/folketingsvalg/kandidat/{candidate_id}"
        page2.goto(url, wait_until="domcontentloaded", timeout=60000)
        page2.wait_for_timeout(3000)
        try:
            page2.locator('button:has-text("Acceptér alle")').click(timeout=5000)
            page2.wait_for_timeout(1000)
        except Exception:
            pass
        for _ in range(30):
            page2.evaluate("window.scrollBy(0, 800)")
            page2.wait_for_timeout(200)
        page2.wait_for_timeout(3000)

        sections = page2.evaluate(JS_WITH_STANCES)
        storkreds_display = raw_by_id[candidate_id]["area"].replace(" Storkreds", "")
        nat_idx = 0
        for sec in sections:
            is_local = "Storkreds" in sec["topic"] or storkreds_display in sec["topic"]
            if is_local:
                continue
            for q in sec["questions"]:
                if nat_idx < len(position_data):
                    position_data[nat_idx]["stances"][candidate_id] = q["stance"]
                    nat_idx += 1

    browser2.close()

    # Now match: for each page position, find the key where ALL candidate stances match
    national_questions = {}
    used_keys = set()
    for pos in position_data:
        best_key = None
        for k in national_keys:
            if k in used_keys:
                continue
            match = True
            for cand_id, page_stance in pos["stances"].items():
                key_stance = key_stances.get(k, {}).get(cand_id)
                if key_stance != page_stance:
                    match = False
                    break
            if match:
                best_key = k
                break
        if best_key:
            national_questions[best_key] = pos["text"]
            used_keys.add(best_key)
            print(f"  MATCHED {best_key}: {pos['text'][:80]}")
        else:
            print(f"  UNMATCHED: {pos['text'][:80]}")

    print(f"\n=== RESULTS ===")
    print(f"National: {len(national_questions)}/24")
    for k in sorted(national_questions, key=lambda x: int(x.split('-')[-1])):
        print(f"  {k}: {national_questions[k]}")

    print(f"\nLocal questions per storkreds:")
    total_local = 0
    for s in sorted(local_questions_by_slug):
        qs = local_questions_by_slug[s]
        total_local += len(qs)
        print(f"  {s}: {len(qs)}/6")
        for k in sorted(qs, key=lambda x: int(x.split('-')[-1])):
            print(f"    {k}: {qs[k][:80]}")

    # Save
    with open("data/questions.json", "w") as f:
        json.dump(national_questions, f, ensure_ascii=False, indent=2)
    with open("data/local_questions.json", "w") as f:
        json.dump(local_questions_by_slug, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(national_questions)} national, {total_local} local questions")
