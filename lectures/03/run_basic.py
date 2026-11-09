"""第3回 宿題（基礎）: INSTRUCTIONS / docstring を1か所ずつ変えて、ツールの呼ばれ方と回答を比べる。

使い方（リポジトリ直下で）:
    python -m uv run --env-file .env python lectures/03/run_basic.py

条件（それぞれ元の設定から1か所だけ変える）:
    baseline       : app/agent.py の INSTRUCTIONS と app/tools.py の docstring そのまま
    no_search_rule : INSTRUCTIONS から「必ず search_glossary で用語集を確認する」の行を消す
    bulk_query     : search_glossary の docstring「1つだけ入れる」を「まとめて入れる」に変える
変えた文面はモデルに渡る文字列そのもの（ファイルを書き換えたときと同じ）。結果は results.json に保存する。
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.exceptions import ModelHTTPError  # noqa: E402
from pydantic_ai.messages import ToolCallPart  # noqa: E402

from app import tools  # noqa: E402
from app.agent import INSTRUCTIONS, build_model  # noqa: E402
from app.config import MODEL  # noqa: E402

OUT = Path(__file__).resolve().parent / "results.json"
QUESTIONS = ["LoRAとQLoRAの違いは？", "AIE2026の講座の宿題にある「3レベル」とは何ですか？"]
EXERCISE_QUESTIONS = ["第2回の内容から確認問題を1問出して"]
REPEAT = 2
WAIT = 5
RETRY_WAIT = 65
MAX_ATTEMPTS = 3

SEARCH_RULE = "- 用語や講座に固有のことを答える前に、必ず search_glossary で用語集を確認する。\n"
ONE_TERM = "query には調べたい用語を1つだけ入れる（例: \"LoRA\"）。"
BULK_TERM = "query には調べたい用語をまとめて入れる（例: \"LoRA QLoRA\"）。"
assert SEARCH_RULE in INSTRUCTIONS and ONE_TERM in tools.search_glossary.__doc__


def search_glossary(query: str) -> str:
    return tools.search_glossary(query)


search_glossary.__doc__ = tools.search_glossary.__doc__.replace(ONE_TERM, BULK_TERM)


def build(instructions: str, search_tool) -> Agent:
    agent = Agent(instructions=instructions)
    agent.tool_plain(search_tool)
    agent.tool_plain(tools.get_lecture_summary)
    agent.tool_plain(tools.make_quiz)
    return agent


VARIANTS = {
    "baseline": build(INSTRUCTIONS, tools.search_glossary),
    "no_search_rule": build(INSTRUCTIONS.replace(SEARCH_RULE, ""), tools.search_glossary),
    "bulk_query": build(INSTRUCTIONS, search_glossary),
}


def run_once(agent: Agent, question: str, model) -> dict:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            result = agent.run_sync(question, model=model)
            break
        except ModelHTTPError as e:
            print(f"  APIエラー {e.status_code}（{attempt}回目）", flush=True)
            if e.status_code not in (429, 500, 503):
                # キーやモデル名の問題は待っても直らないので、すぐ止める
                raise SystemExit(f"中止: {str(e.body)[:300]}")
            if attempt == MAX_ATTEMPTS:
                return {"error": f"{e.status_code}: {str(e.body)[:200]}", "attempts": attempt}
            time.sleep(RETRY_WAIT)
    usage = result.usage() if callable(result.usage) else result.usage
    calls = [
        {"tool": p.tool_name, "args": p.args}
        for m in result.all_messages() for p in m.parts if isinstance(p, ToolCallPart)
    ]
    return {
        "attempts": attempt,
        "tool_calls": calls,
        "requests": getattr(usage, "requests", None),
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "output": result.output,
    }


def main() -> None:
    model = build_model()
    plan = [("baseline", q) for q in EXERCISE_QUESTIONS] + [
        (v, q) for q in QUESTIONS for v in VARIANTS for _ in range(REPEAT)
    ]
    records = []
    for i, (variant, question) in enumerate(plan, 1):
        print(f"[{i}/{len(plan)}] {variant}: {question}", flush=True)
        rec = {"model": MODEL, "variant": variant, "question": question}
        rec.update(run_once(VARIANTS[variant], question, model))
        for c in rec.get("tool_calls", []):
            print(f"  [tool call] {c['tool']}({c['args']})")
        print(f"  requests={rec.get('requests')} error={rec.get('error')}", flush=True)
        records.append(rec)
        OUT.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(WAIT)
    print(f"\n保存しました: {OUT}")


if __name__ == "__main__":
    main()
