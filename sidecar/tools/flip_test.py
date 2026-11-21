#!/usr/bin/env python3
# Copyright 2026 RonanLabs. Licensed under the Apache License 2.0.
"""Teacher-forced top-1 flip test, E3 vs E1 (32 prompts x 128 tokens).

  flip_test.py ref   URL MODEL OUT.json   # on E1 (stock): greedy 128-token continuations + teacher-forced top-1
  flip_test.py score URL MODEL REF.json OUT.json   # on E3: teacher-forced top-1 on the same token sequences

Teacher forcing: the prompt is the token ids of (prompt + E1's continuation), max_tokens 1, prompt_logprobs 1; the
top-1 id at each continuation position is compared. flips = positions where E3's top-1 differs from E1's teacher-forced
top-1, over all continuation positions. E1's own greedy-vs-teacher-forced mismatch is reported as the noise floor.
Requests run one at a time (batch composition changes numerics).
"""
import json
import sys
import urllib.request

PROMPTS = [
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. How much does she make every day at the farmers' market if she sells the remainder at $2 per egg? Let's think step by step.",
    "A train leaves Chicago at 3 pm traveling 60 mph. Another leaves St. Louis, 300 miles away, at 4 pm traveling 75 mph toward Chicago. When do they meet? Work it out carefully.",
    "Write a Python function that returns the longest palindromic substring of a string, with a short explanation of its complexity.",
    "Explain, for a student pilot, the difference between Class B and Class C airspace, including entry requirements.",
    "def merge_sort(xs):\n    \"\"\"Sort a list using merge sort.\"\"\"\n",
    "The French Revolution began in 1789. Summarize its main causes in a few paragraphs.",
    "Translate into German: 'The weather was so bad that the flight to Munich was cancelled, and we spent the night at the airport.'",
    "List the steps of the scientific method and give an example experiment for each step.",
    "What is the derivative of x^3 * sin(x)? Show the product rule step by step.",
    "Write a haiku sequence (three haiku) about a thunderstorm over the desert.",
    "In SQL, how do you find the second highest salary in an Employees table? Give two different queries.",
    "Describe how a four-stroke piston aircraft engine works, including the role of the magnetos.",
    "A rectangle has a perimeter of 54 cm and its length is twice its width. Find its area.",
    "Explain what a hash map is and how collisions are handled with separate chaining and open addressing.",
    "Give a short history of the Silk Road and its importance to trade between China and Europe.",
    "Solve for x: 3x + 7 = 2(x - 4) + 15. Explain each step.",
    "Write a short dialogue between a flight instructor and a student about crosswind landings.",
    "What are the main differences between TCP and UDP? When would you use each?",
    "Compose a formal email requesting a two-week extension on a research report deadline.",
    "Explain photosynthesis to a ten-year-old, then give the chemical equation.",
    "A store sells pencils at 3 for $0.75. How much do 2 dozen pencils cost?",
    "Describe the plot of Shakespeare's Macbeth in about 150 words.",
    "Write a bash one-liner that counts the number of lines in all .py files under the current directory, recursively.",
    "What causes the seasons on Earth? Address the common misconception about distance from the Sun.",
    "Explain the concept of density altitude and why it matters for takeoff performance.",
    "Prove that the square root of 2 is irrational.",
    "Give three arguments for and three against a four-day work week.",
    "Write a JavaScript function that debounces another function, with a usage example.",
    "If a car depreciates by 15% per year, what is its value after 3 years if it cost $24,000 new?",
    "Explain how vaccines train the immune system, mentioning B cells and T cells.",
    "Describe the water cycle and the energy sources that drive it.",
    "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now? Explain.",
]
N_GEN = 128


def post(url, path, body, timeout=600):
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def tokenize(url, model, text):
    return post(url, "/tokenize", {"model": model, "prompt": text})["tokens"]


def teacher_forced_top1(url, model, ids, n_prompt):
    r = post(url, "/v1/completions", {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0,
                                      "prompt_logprobs": 1})
    pl = r["choices"][0]["prompt_logprobs"]
    top = []
    for pos in range(n_prompt, len(ids)):
        d = pl[pos] or {}
        best = [int(k) for k, v in d.items() if v.get("rank") == 1]
        top.append(best[0] if best else -1)
    return top


def greedy(url, model, ids):
    r = post(url, "/v1/completions", {"model": model, "prompt": ids, "max_tokens": N_GEN, "temperature": 0,
                                      "ignore_eos": True, "return_tokens_as_token_ids": True, "logprobs": 0})
    toks = r["choices"][0]["logprobs"]["tokens"]
    return [int(t.split(":", 1)[1]) for t in toks]


def main():
    mode, url, model = sys.argv[1], sys.argv[2].replace("/v1", ""), sys.argv[3]
    if mode == "ref":
        out = {"prompts": []}
        for i, text in enumerate(PROMPTS):
            p = tokenize(url, model, text)
            g = greedy(url, model, p)
            tf = teacher_forced_top1(url, model, p + g, len(p))
            out["prompts"].append({"i": i, "prompt_ids": p, "gen_ids": g, "e1_tf_top1": tf})
            print(f"ref {i}: {len(p)} prompt, {len(g)} gen, self-mismatch {sum(a != b for a, b in zip(g, tf))}",
                  flush=True)
        json.dump(out, open(sys.argv[4], "w"))
        return
    ref = json.load(open(sys.argv[4]))
    flips = tot = noise = 0
    per = []
    for rec in ref["prompts"]:
        p, g, tf1 = rec["prompt_ids"], rec["gen_ids"], rec["e1_tf_top1"]
        tf3 = teacher_forced_top1(url, model, p + g, len(p))
        f = sum(a != b for a, b in zip(tf1, tf3))
        flips += f
        tot += len(tf1)
        noise += sum(a != b for a, b in zip(g, tf1))
        per.append({"i": rec["i"], "flips": f, "n": len(tf1)})
        print(f"prompt {rec['i']}: {f}/{len(tf1)} flips", flush=True)
    res = {"prompts": len(per), "positions": tot, "flips": flips, "flip_rate": flips / max(tot, 1),
           "e1_greedy_vs_tf_noise": noise / max(tot, 1), "per_prompt": per}
    json.dump(res, open(sys.argv[5], "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_prompt"}))


if __name__ == "__main__":
    main()
