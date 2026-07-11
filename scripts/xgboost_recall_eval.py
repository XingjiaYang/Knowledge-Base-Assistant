#!/usr/bin/env python3
"""Evaluate XGBoost code search Recall@5 against 130 labeled queries."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from uuid import UUID

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
USERNAME = sys.argv[2] if len(sys.argv) > 2 else "admin"
PASSWORD = sys.argv[3] if len(sys.argv) > 3 else "123456"
RESULTS_FILE = Path("xgboost_recall_eval_130.json")

QUERIES: list[str] = [
    # 1-26: Training and Booster
    "在 Python 里用底层接口（非 sklearn）训练一个 boosting 模型的函数是哪个？",
    "做带交叉验证的训练、并返回每折评估结果的函数是哪个？",
    "用训练好的 Booster 对一个 DMatrix 做预测的方法是哪个？",
    "不构造 DMatrix，直接对 numpy/数组做就地（in-place）预测的方法是哪个？",
    "把 Booster 模型保存到文件（如 .json/.ubj）的方法是哪个？",
    "从文件加载 Booster 模型的方法是哪个？",
    "把 Booster 模型序列化成内存里的原始字节缓冲的方法是哪个？",
    "把模型结构导出成文本或 JSON 描述的方法是哪个？",
    "获取每个特征重要性得分（weight/gain/cover）的方法是哪个？",
    "查询模型当前已经提升了多少轮（树的层数）的方法是哪个？",
    "给已创建的 Booster 设置训练参数的方法是哪个？",
    "用已配置的目标函数推进一轮 boosting 迭代的方法是哪个？",
    "用自定义梯度和二阶导（grad/hess）手动 boost 一轮的方法是哪个？",
    "在多个评估数据集上评估并返回结果字符串的方法是哪个？",
    "在单个 DMatrix 上评估模型的方法是哪个？",
    "把 Booster 的内部配置导出成 JSON 字符串的方法是哪个？",
    "从 JSON 字符串恢复 Booster 配置的方法是哪个？",
    "给 Booster 设置任意键值字符串属性的方法是哪个？",
    "读取 Booster 上某个字符串属性值的方法是哪个？",
    "一次性取出 Booster 所有属性键值对的方法是哪个？",
    "把模型里所有树转换成一个 pandas DataFrame 的方法是哪个？",
    "统计某个特征在各棵树中分裂取值的直方图的方法是哪个？",
    "取出每棵树的文本/JSON dump 列表的方法是哪个？",
    "查询模型使用的特征数量的方法是哪个？",
    "早停后获取最佳迭代轮次的属性是哪个？",
    "以 F-score 形式（基于出现次数）返回特征重要性的方法是哪个？",
    # 27-39: DMatrix
    "设置 DMatrix 标签（label）的方法是哪个？",
    "设置 DMatrix 每条样本权重的方法是哪个？",
    "设置 DMatrix 的 base margin 的方法是哪个？",
    "为排序任务设置分组信息（group）的方法是哪个？",
    "取出 DMatrix 标签的方法是哪个？",
    "把 DMatrix 保存成二进制缓存文件的方法是哪个？",
    "按行索引切出一个新的 DMatrix 子集的方法是哪个？",
    "获取 DMatrix 分位数分桶切分点的方法是哪个？",
    "查询 DMatrix 行数的方法是哪个？",
    "查询 DMatrix 列数（特征数）的方法是哪个？",
    "查询 DMatrix 中非缺失元素个数的方法是哪个？",
    "把 DMatrix 数据重新取回为 scipy CSR 稀疏矩阵的方法是哪个？",
    "一次性设置 DMatrix 多种元信息（label/weight/group 等）的方法是哪个？",
    # 40-44: QuantileDMatrix / DataIter
    "为 hist 算法准备的、省内存的分位数 DMatrix 类是哪个？",
    "支持外存（external memory）训练的分位数 DMatrix 类是哪个？",
    "自定义流式喂数据时需要继承的数据迭代器基类是哪个？",
    "数据迭代器里返回下一批数据的方法是哪个？",
    "数据迭代器里把游标重置到开头的方法是哪个？",
    # 45-59: sklearn
    "scikit-learn 风格的回归器类是哪个？",
    "scikit-learn 风格的分类器类是哪个？",
    "分类器里输出类别概率的方法是哪个？",
    "用于 learning-to-rank 的 sklearn 估计器类是哪个？",
    "随机森林风格的分类器类是哪个？",
    "随机森林风格的回归器类是哪个？",
    "sklearn 估计器训练拟合的方法是哪个？",
    "从 sklearn 估计器里拿到底层原生 Booster 对象的方法是哪个？",
    "sklearn 估计器上读取特征重要性的属性是哪个？",
    "线性模型下读取系数（coefficients）的属性是哪个？",
    "线性模型下读取截距（intercept）的属性是哪个？",
    "训练后取回各数据集评估历史的方法是哪个？",
    "返回每个样本落入各树叶子节点索引的方法是哪个？",
    "拿到要传给底层 xgboost 的参数字典的方法是哪个？",
    "分类器里列出所有类别标签的属性是哪个？",
    # 60-62: plotting
    "画特征重要性条形图的函数是哪个？",
    "把某棵树可视化绘制出来的函数是哪个？",
    "把一棵树转换成 graphviz Source 对象的函数是哪个？",
    # 63-67: callbacks
    "训练时做早停的回调类是哪个？",
    "训练中按计划调整学习率的回调类是哪个？",
    "周期性保存检查点的回调类是哪个？",
    "训练时打印评估指标的回调类是哪个？",
    "自定义训练回调要继承的抽象基类是哪个？",
    # 68-71: config
    "全局设置 xgboost 配置项的函数是哪个？",
    "读取当前全局配置的函数是哪个？",
    "临时修改全局配置的上下文管理器是哪个？",
    "查询 xgboost 编译构建信息（版本/是否带 CUDA 等）的函数是哪个？",
    # 72-80: collective
    "初始化分布式通信器的函数是哪个？",
    "关闭并清理分布式通信器的函数是哪个？",
    "获取当前 worker 的 rank 的函数是哪个？",
    "获取参与分布式训练的 worker 总数的函数是哪个？",
    "判断当前是否处于分布式环境的函数是哪个？",
    "对一个数组做 allreduce 归约的 Python 函数是哪个？",
    "把数据从 root 广播给所有 worker 的函数是哪个？",
    "进入分布式通信会话的上下文管理器类是哪个？",
    "启动用于协调分布式训练的 tracker 类是哪个？",
    # 81-100: C API
    "C API 中从本地文件创建 DMatrix 的函数是哪个？",
    "C API 中从 CSR 稀疏数据创建 DMatrix 的函数是哪个？",
    "C API 中从稠密浮点矩阵创建 DMatrix 的函数是哪个？",
    "C API 中创建 Booster 句柄的函数是哪个？",
    "C API 中对一个 DMatrix 运行预测并返回结果的函数是哪个？",
    "C API 中把 Booster 模型保存到文件的函数是哪个？",
    "C API 中用内置目标推进一轮训练的函数是哪个？",
    "C API 中用自定义梯度/二阶导 boost 一轮的函数是哪个？",
    "C API 中查询 DMatrix 行数的函数是哪个？",
    "C API 中计算并返回特征重要性得分的函数是哪个？",
    "C API 中把 Booster 完整状态序列化进内存缓冲的函数是哪个？",
    "C API 中设置全局配置的函数是哪个？",
    "C API 中通过数据迭代回调构造 QuantileDMatrix 的函数是哪个？",
    "C API 中按行索引切片 DMatrix 的函数是哪个？",
    "C API 中在一组数据集上做一轮评估的函数是哪个？",
    "C API 中把模型结构 dump 出来的函数是哪个？",
    "C API 中执行分布式 allreduce 的通信函数是哪个？",
    "C API 中查询 Booster 特征数量的函数是哪个？",
    "C API 中从内存缓冲加载模型的函数是哪个？",
    "C API 中查询 DMatrix 列数的函数是哪个？",
    # 101-130: ambiguous intent
    "模型训练到后面基本不涨了，能不能让它自己判断该收手就别再练了？",
    "我的训练数据有上亿行，一次性读进内存直接 OOM，有别的办法吗？",
    "训练完想搞清楚模型到底主要靠哪些字段在做判断。",
    "这模型训练一次老半天，关机之后还想接着用，别每次都从头来。",
    "同事把他训练好的模型文件发我了，我这边怎么弄进来直接跑预测？",
    "我想扒开看看模型内部每棵树到底是怎么一步步分叉的。",
    "线上预测每次都要先把数据包成一个对象，又啰嗦又慢，有没有更直接的法子？",
    "二分类我不要最后那个 0 或 1，我要它判定为正类的把握到底有多大。",
    "搜索场景里同一次查询下有一批候选要排个序，这种从属结构怎么告诉模型？",
    "有些样本我觉得更关键，希望模型多照顾它们，别所有样本一视同仁。",
    "训练要跑很久，过程中我想随时盯着每一轮效果到底好不好。",
    "xgboost 自带的损失满足不了我的业务，我想把自己推的梯度塞进去。",
    "一开始想用大点的步子，后面再慢慢收小让它收敛得更稳。",
    "一个要跑好几个小时的训练任务，万一中途挂了我可不想从头再来。",
    "我想把其中某一棵树画成图，贴到汇报材料里。",
    "它自己提前停下来之后，我怎么知道最终到底采用的是第几轮的模型？",
    "我只想拿前面一部分树来预测，看看树少一点效果会差多少。",
    "我的数据里有不少地方是空的没填，xgboost 能自己处理还是得我先补全？",
    "数据里有\"城市\"\"品类\"这种文字分类列，我不想手动 one-hot，能直接喂进去吗？",
    "现在这个模型效果还行，但我想再多加几十棵树继续提升，不想推倒重练。",
    "我不确定到底该训多少轮才合适，想靠交叉验证自动帮我定下来。",
    "我想把模型导成通用的 JSON 文本，方便别的工具去解析它的结构。",
    "多台机器一起训练，总得有个东西负责把这些节点凑齐并协调起来吧？",
    "我想看某个特征到底是在哪些取值附近被频繁拿去切分的。",
    "预测的时候我还想顺带知道每条样本最后落进了哪个叶子节点。",
    "我的特征已经是 cupy/cudf 放在显卡上了，想省掉来回搬运直接拿来训练。",
    "我想把 xgboost 塞进 sklearn 的 GridSearchCV 里一起做超参搜索。",
    "每次从原始 csv 加载转换都很慢，我想存一份处理好的缓存下次秒开。",
    "训练完我想把当时用的那一整套参数配置原样记下来存档。",
    "我想给模型本身贴个标签，比如训练日期、版本号，让它跟着模型一起走。",
]

# Ground truth: single target for Q1-Q100, answer set for Q101-Q130
GROUND_TRUTH: list[str | list[str]] = [
    # 1-26
    "train",
    "cv",
    "Booster.predict",
    "Booster.inplace_predict",
    "Booster.save_model",
    "Booster.load_model",
    "Booster.save_raw",
    "Booster.dump_model",
    "Booster.get_score",
    "Booster.num_boosted_rounds",
    "Booster.set_param",
    "Booster.update",
    "Booster.boost",
    "Booster.eval_set",
    "Booster.eval",
    "Booster.save_config",
    "Booster.load_config",
    "Booster.set_attr",
    "Booster.attr",
    "Booster.attributes",
    "Booster.trees_to_dataframe",
    "Booster.get_split_value_histogram",
    "Booster.get_dump",
    "Booster.num_features",
    "Booster.best_iteration",
    "Booster.get_fscore",
    # 27-39
    "DMatrix.set_label",
    "DMatrix.set_weight",
    "DMatrix.set_base_margin",
    "DMatrix.set_group",
    "DMatrix.get_label",
    "DMatrix.save_binary",
    "DMatrix.slice",
    "DMatrix.get_quantile_cut",
    "DMatrix.num_row",
    "DMatrix.num_col",
    "DMatrix.num_nonmissing",
    "DMatrix.get_data",
    "DMatrix.set_info",
    # 40-44
    "QuantileDMatrix",
    "ExtMemQuantileDMatrix",
    "DataIter",
    "DataIter.next",
    "DataIter.reset",
    # 45-59
    "XGBRegressor",
    "XGBClassifier",
    "XGBClassifier.predict_proba",
    "XGBRanker",
    "XGBRFClassifier",
    "XGBRFRegressor",
    "XGBModel.fit",
    "XGBModel.get_booster",
    "XGBModel.feature_importances_",
    "XGBModel.coef_",
    "XGBModel.intercept_",
    "XGBModel.evals_result",
    "XGBModel.apply",
    "XGBModel.get_xgb_params",
    "XGBClassifier.classes_",
    # 60-62
    "plot_importance",
    "plot_tree",
    "to_graphviz",
    # 63-67
    "EarlyStopping",
    "LearningRateScheduler",
    "TrainingCheckPoint",
    "EvaluationMonitor",
    "TrainingCallback",
    # 68-71
    "set_config",
    "get_config",
    "config_context",
    "build_info",
    # 72-80
    "init",
    "finalize",
    "get_rank",
    "get_world_size",
    "is_distributed",
    "allreduce",
    "broadcast",
    "CommunicatorContext",
    "RabitTracker",
    # 81-100
    "XGDMatrixCreateFromFile",
    "XGDMatrixCreateFromCSR",
    "XGDMatrixCreateFromMat",
    "XGBoosterCreate",
    "XGBoosterPredictFromDMatrix",
    "XGBoosterSaveModel",
    "XGBoosterUpdateOneIter",
    "XGBoosterBoostOneIter",
    "XGDMatrixNumRow",
    "XGBoosterFeatureScore",
    "XGBoosterSerializeToBuffer",
    "XGBSetGlobalConfig",
    "XGQuantileDMatrixCreateFromCallback",
    "XGDMatrixSliceDMatrix",
    "XGBoosterEvalOneIter",
    "XGBoosterDumpModel",
    "XGCommunicatorAllreduce",
    "XGBoosterGetNumFeature",
    "XGBoosterLoadModelFromBuffer",
    "XGDMatrixNumCol",
    # 101-130
    ["EarlyStopping", "Booster.best_iteration"],
    ["ExtMemQuantileDMatrix", "DataIter", "QuantileDMatrix"],
    ["Booster.get_score", "XGBModel.feature_importances_", "plot_importance"],
    ["Booster.save_model", "Booster.save_raw"],
    ["Booster.load_model"],
    ["Booster.trees_to_dataframe", "Booster.dump_model", "Booster.get_dump"],
    ["Booster.inplace_predict"],
    ["XGBClassifier.predict_proba"],
    ["DMatrix.set_group", "XGBRanker"],
    ["DMatrix.set_weight"],
    ["EvaluationMonitor", "XGBModel.evals_result"],
    ["Booster.boost"],
    ["LearningRateScheduler"],
    ["TrainingCheckPoint"],
    ["plot_tree", "to_graphviz"],
    ["Booster.best_iteration", "Booster.num_boosted_rounds"],
    ["Booster.__getitem__", "Booster.predict"],
    ["DMatrix"],
    ["DMatrix.feature_types", "DMatrix"],
    ["train", "Booster.update"],
    ["cv"],
    ["Booster.dump_model", "Booster.save_model"],
    ["RabitTracker", "init"],
    ["Booster.get_split_value_histogram"],
    ["XGBModel.apply", "Booster.predict"],
    ["QuantileDMatrix"],
    ["XGBRegressor", "XGBClassifier", "XGBModel.fit"],
    ["DMatrix.save_binary"],
    ["Booster.save_config"],
    ["Booster.set_attr"],
]


def _request(method: str, path: str, body: dict | None = None, headers: dict | None = None) -> dict:
    url = f"{BASE_URL}{path}"
    data = json.dumps(body).encode() if body else None
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode()
        try:
            return json.loads(body_text)
        except json.JSONDecodeError:
            return {"error": body_text, "status_code": exc.code}


def login() -> str:
    resp = _request("POST", "/auth/login", {"username": USERNAME, "password": PASSWORD})
    if "token" not in resp:
        raise RuntimeError(f"Login failed: {resp}")
    return str(resp["token"])


def code_search(token: str, query: str) -> dict:
    payload = {
        "query": query,
        "repository_ids": ["xgboost"],
        "final_top_k": 5,
    }
    return _request("POST", "/code/search", payload, {"Authorization": f"Bearer {token}"})


def _matches(qualified_name: str, target: str) -> bool:
    qn = qualified_name.strip()
    t = target.strip()
    if not qn or not t:
        return False
    if t == qn:
        return True
    if qn.endswith(f".{t}") or qn.endswith(f"::{t}"):
        return True
    if t.endswith(".") and qn.endswith(t[:-1]):
        return False
    # Handle "Class.method" target against "Class.method" or "method" qualified_name
    if "." in t:
        parts = t.split(".")
        if qn == t or qn.endswith(t) or qn == parts[-1]:
            return True
    if t in qn and (qn == t or qn.endswith(f".{t}") or qn.endswith(f"::{t}")):
        return True
    return False


def _is_hit(functions: list[dict], target: str | list[str]) -> bool:
    targets = [target] if isinstance(target, str) else target
    top5 = [str(f.get("qualified_name", "")) for f in functions[:5]]
    for t in targets:
        for qn in top5:
            if _matches(qn, t):
                return True
    return False


def main() -> None:
    print(f"Base URL: {BASE_URL}")
    token = login()
    print(f"Logged in as {USERNAME}")

    results: list[dict] = []
    hits = 0
    hits_single = 0
    hits_ambiguous = 0
    total_single = 100
    total_ambiguous = 30

    for index, (query, target) in enumerate(zip(QUERIES, GROUND_TRUTH), start=1):
        print(f"\n[{index}/130] {query[:60]}...")
        start = time.time()
        try:
            resp = code_search(token, query)
        except Exception as exc:
            resp = {"error": str(exc)}
        elapsed = time.time() - start

        functions = resp.get("functions", [])
        hit = _is_hit(functions, target)
        if hit:
            hits += 1
            if index <= 100:
                hits_single += 1
            else:
                hits_ambiguous += 1

        summary = {
            "index": index,
            "query": query,
            "elapsed_seconds": round(elapsed, 2),
            "hit": hit,
            "target": target,
            "retrieval_mode": resp.get("retrieval_mode", "unknown"),
            "code_embedding_degraded": resp.get("code_embedding_degraded", True),
            "functions": [
                {
                    "qualified_name": f.get("qualified_name"),
                    "path": f.get("path"),
                    "score": f.get("score"),
                }
                for f in functions[:5]
            ],
            "error": resp.get("error", ""),
        }
        results.append(summary)
        status = "HIT" if hit else "MISS"
        print(f"  -> {status} target={target} top1={functions[0]['qualified_name'] if functions else 'NONE'}")

    metrics = {
        "total": 130,
        "recall_at_5": round(hits / 130, 4),
        "single_intent": {
            "total": total_single,
            "hits": hits_single,
            "recall_at_5": round(hits_single / total_single, 4),
        },
        "ambiguous_intent": {
            "total": total_ambiguous,
            "hits": hits_ambiguous,
            "recall_at_5": round(hits_ambiguous / total_ambiguous, 4),
        },
    }

    output = {"metrics": metrics, "results": results}
    RESULTS_FILE.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nMetrics: {json.dumps(metrics, indent=2, ensure_ascii=False)}")
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
