"""第1回 宿題（基礎）: 同じ質問を3回ずつ、モデルを変えて実行し、結果を保存する。

使い方（リポジトリのルートで実行）:
    uv run python lectures/01/run_basic.py

GEMINI_API_KEY が未設定なら、画面に表示せずに入力を求める。
キーは app.ask に環境変数で渡すだけで、保存はしない。
結果は lectures/01/results.json に実行ごとに追記される（途中で止まっても残る）。
"""

import getpass
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "lectures" / "01" / "results.json"

# モデルごとの、実行の間隔（秒）。無料枠の毎分上限に収まるようにする
MODELS = {"gemini-3.5-flash-lite": 5, "gemini-3.8-flash": 15}
QUESTIONS = [
    "LoRAとは何ですか？",
    "AIE2026の講座の宿題にある「3レベル」とは何ですか？",
]
REPEAT = 3
MAX_ATTEMPTS = 3  # 429 や 503 で失敗したときの試行回数
RETRY_WAIT = 65


def run_once(model: str, question: str, api_key: str) -> dict:
    env = {**os.environ, "GEMINI_API_KEY": api_key, "AIE_MODEL": model, "PYTHONUTF8": "1"}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        result = subprocess.run(
            [sys.executable, "-m", "app.ask", question],
            cwd=ROOT,
            env=env,
            capture_output=True,
            encoding="utf-8",
        )
        if result.returncode == 0:
            break
        print(f"  失敗（{attempt}/{MAX_ATTEMPTS} 回目）: {result.stdout.strip()[:200]}")
        if attempt < MAX_ATTEMPTS:
            time.sleep(RETRY_WAIT)
    return {
        "returncode": result.returncode,
        "attempts": attempt,
        "output": result.stdout.strip(),
        "error": result.stderr.strip().splitlines()[-1:],
    }


def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY") or getpass.getpass("GEMINI_API_KEY（入力は表示されません）: ")
    if not api_key:
        raise SystemExit("キーが入力されなかったので終了します。")

    results = []
    for question in QUESTIONS:
        for model, wait in MODELS.items():
            for run in range(1, REPEAT + 1):
                print(f"[{model}] {run}/{REPEAT}: {question}")
                record = {"model": model, "question": question, "run": run}
                record.update(run_once(model, question, api_key))
                results.append(record)
                OUTPUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                time.sleep(wait)

    failed = sum(r["returncode"] != 0 for r in results)
    print(f"完了: {len(results)} 回中 {failed} 回失敗。結果: {OUTPUT}")


if __name__ == "__main__":
    main()
