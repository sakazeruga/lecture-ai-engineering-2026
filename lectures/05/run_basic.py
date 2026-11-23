"""第5回 演習 STEP 1〜4 と宿題（基礎：RAG_TOP_K を 5 → 2）をまとめて実行し、出力を保存する。

使い方（リポジトリ直下で）:
    python -m uv run --env-file .env python lectures/05/run_basic.py

出力は lectures/05/runs/ に 1 実行 1 ファイルで保存する。
宿題の実行の前に app/config.py の RAG_TOP_K を 5 から 2 に書き換え、そのまま残す（宿題で変えた 1 か所）。
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "runs"
CONFIG = ROOT / "app" / "config.py"

Q_MAIN = "第2回の説明では、LoRAは何を固定し何を学習しますか？ QLoRAは何を加えますか？"
Q_MULTI = "第3回の例では、コンパクションとセッションをまたぐメモリはどう違いますか。日々のメモはすべて長期メモリに入りますか"

BEFORE = [
    ("01_full_context", ["-m", "app.agent", "--full-context", "--trace", Q_MAIN]),
    ("02_retrieval_only", ["-m", "app.retrieval", "LoRA QLoRA 違い"]),
    ("03_vector_k5", ["-m", "app.agent", "--trace", Q_MAIN]),
    ("04_tree", ["-m", "app.agent", "--tree", "--trace", Q_MAIN]),
    ("05_graph", ["-m", "app.agent", "--graph", "--trace", Q_MAIN]),
    ("06_vector_k5_multi", ["-m", "app.agent", "--trace", Q_MULTI]),
]
AFTER = [
    ("07_vector_k2", ["-m", "app.agent", "--trace", Q_MAIN]),
    ("08_vector_k2_multi", ["-m", "app.agent", "--trace", Q_MULTI]),
]
WAIT = 10
RETRY_WAIT = 65
MAX_ATTEMPTS = 3
RETRYABLE = ("レート制限", "混み合って", "429", "503")


def run(name: str, args: list[str]) -> None:
    env = {**os.environ, "PYTHONUTF8": "1", "PYDANTIC_AI_NO_BANNER": "1"}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"[{name}] {attempt}回目 ...", flush=True)
        p = subprocess.run([sys.executable, *args], cwd=ROOT, env=env,
                           capture_output=True, encoding="utf-8", errors="replace")
        text = p.stdout + ("\n--- stderr ---\n" + p.stderr if p.stderr.strip() else "")
        (OUT / f"{name}.txt").write_text(text, encoding="utf-8")
        if p.returncode == 0:
            last = [l for l in p.stdout.splitlines() if l.startswith(("[time]", "[usage]", "[cost]"))]
            print("  OK " + " / ".join(last), flush=True)
            return
        print(f"  失敗（終了コード {p.returncode}）: {(p.stdout + p.stderr).strip().splitlines()[-1:]}", flush=True)
        if attempt < MAX_ATTEMPTS and any(k in text for k in RETRYABLE):
            time.sleep(RETRY_WAIT)
        else:
            return


def set_top_k(value: int) -> None:
    s = CONFIG.read_text(encoding="utf-8")
    s, n = re.subn(r"^RAG_TOP_K = \d+", f"RAG_TOP_K = {value}", s, flags=re.M)
    assert n == 1, "app/config.py に RAG_TOP_K が見つかりません"
    CONFIG.write_text(s, encoding="utf-8")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    set_top_k(5)
    for name, args in BEFORE:
        run(name, args)
        time.sleep(WAIT)
    set_top_k(2)  # 宿題（基礎）：変えるのはここ 1 か所
    for name, args in AFTER:
        run(name, args)
        time.sleep(WAIT)
    print(f"\n保存しました: {OUT}")


if __name__ == "__main__":
    main()
