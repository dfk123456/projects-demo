# -*- coding: utf-8 -*-
# ============================================================
# api_selftest.py —— Web 服务端到端自检（需先启动服务）
# 用法：python scripts/api_selftest.py [--file xxx.pdf] [--base http://127.0.0.1:8000]
#
# 覆盖：健康检查 → 模型清单 → 上传 → 启动审查 → 轮询/SSE → 编辑保存 → 导出 Word
# ============================================================
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--file", default=os.path.join(ROOT, "data", "samples", "样例高风险合同_40页.pdf"))
    ap.add_argument("--model", default="")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    print("1) 健康检查：", requests.get(f"{base}/api/health", timeout=20).json())

    models = requests.get(f"{base}/api/models", timeout=20).json()
    print("2) 可选模型：", [(m["id"], m["available"]) for m in models["models"]])
    model_id = args.model or models["default"]
    print("   本次使用：", model_id)

    with open(args.file, "rb") as fh:
        up = requests.post(f"{base}/api/upload",
                           files={"file": (os.path.basename(args.file), fh)}, timeout=60).json()
    print("3) 上传结果：", up)

    start = requests.post(f"{base}/api/review/start", json={
        "file_key": up["file_key"], "filename": up["filename"], "model_id": model_id}, timeout=30).json()
    task_id = start["task_id"]
    print("4) 任务已启动：", task_id)

    # SSE 读取进度
    print("5) 实时进度：")
    done = False
    try:
        r = requests.get(f"{base}/api/review/{task_id}/stream", stream=True, timeout=180)
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            ev = json.loads(line[5:])
            if ev.get("type") == "log":
                print("   ·", ev["message"])
            elif ev.get("type") == "progress":
                print(f"   [{int(ev['progress'] * 100):>3}%] {ev['stage']}")
            elif ev.get("type") == "done":
                done = True
                print("   ✔ 完成")
            elif ev.get("type") == "error":
                print("   ✘", ev["error"]); break
            elif ev.get("type") == "closed":
                break
    except Exception as exc:
        print("   （SSE 中断：", str(exc)[:80], "）")

    if not done:                       # 轮询兜底
        for _ in range(120):
            st = requests.get(f"{base}/api/review/{task_id}", timeout=20).json()
            if st["status"] in ("done", "failed"):
                done = st["status"] == "done"
                break
            time.sleep(2)

    report = requests.get(f"{base}/api/review/{task_id}", timeout=30).json().get("report") or {}
    stat = report.get("statistics", {})
    print(f"6) 报告：{stat.get('total')} 条（高 {stat.get('高')} / 中 {stat.get('中')} / 低 {stat.get('低')}）"
          f"，耗时 {report.get('elapsed_sec')}s")

    # 模拟人工编辑：改结论 + 追加一条
    report["executive_summary"] = "（接口自检已编辑）" + (report.get("executive_summary") or "")
    report.setdefault("risks", []).append({
        "id": "manual-1", "clause": "人工补充", "location": "第1页",
        "quote": "示例摘录", "risk_type": "其他", "description": "接口自检新增", "suggestion": "复核后确认",
        "legal_basis": "", "level": "中", "confidence": 1.0, "page": 1})
    saved = requests.put(f"{base}/api/review/{task_id}/report", json={"report": report}, timeout=30).json()
    print("7) 保存后统计：", saved["report"]["statistics"])

    exp = requests.post(f"{base}/api/review/{task_id}/export", json={"fmt": "docx"}, timeout=60).json()
    print("8) 导出结果：", exp)
    down = requests.get(base + exp["url"], timeout=60)
    out_path = os.path.join(ROOT, "data", "exports", exp["filename"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as fh:
        fh.write(down.content)
    print(f"9) 下载成功：{out_path}（{len(down.content) / 1024:.1f} KB）")


if __name__ == "__main__":
    main()
