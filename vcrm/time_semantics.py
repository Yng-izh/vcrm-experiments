"""统一训练与推理的时间语义（P2）。

问题（LongMemEval 全量结果显示的第五条结构性问题）：

    训练侧（``vcrm/training.py::trajectory_samples``）把目标年份写进**题面**
    （``"In 2019: <question>"``），reader 的 ``Current date`` 槽放**观测截止**
    （``current['time']``）。而评测侧（``probe/run_main_eval.py``）把问题里的历史
    锚折算成 ``window_end`` 之后直接当 ``Current date`` 注入 reader，题面里反而不带
    年份。于是同一份 checkpoint 在训练与推理时看到的「当前时刻」含义不同，
    我们专门训出来的历史版本判别能力被这个口径差异抵消。

本模块给出**唯一**口径，训练与推理共用同一个构造函数：

    * ``observation_time`` —— 系统当时已知的最后时刻（``question_date`` /
      episode 观测窗口末刻）。进 reader 的 ``Current date`` 槽
      （``reader.question`` 用 ``query['time']`` 渲染，见 ``vcrm/reader.py``）。
    * ``target_time`` —— 查询指定的历史时刻。**只进题面**（``In 2019: ...``），
      绝不冒充 ``Current date``；它只用于 ``store.select_version`` 的版本选择上界。
    * 没有历史锚时题面不加前缀，``Current date`` 仍是观测截止。

刻意不设置 ``query['as_of']``：那是 TempLAMA 的金标时间通道，在 Predicted 口径
下等于泄漏。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

#: reader 无法解析的时间占位（与 state_index.UNKNOWN_TIME 同义，这里独立定义以免循环导入）。
_UNKNOWN_VALUES = frozenset({'', 'unknown', 'none', 'null'})


def year_of(value: Any) -> Optional[str]:
    """从 ISO 时间串里取年份；取不到返回 None（绝不猜）。"""
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in _UNKNOWN_VALUES:
        return None
    head = text[:4]
    return head if head.isdigit() and len(head) == 4 else None


def target_is_historical(target_time: Any, observation_time: Any = None) -> bool:
    """目标时刻是否是"观测截止之前"的历史时刻（决定题面要不要带年份）。"""
    target_year = year_of(target_time)
    if target_year is None:
        return False
    observation_year = year_of(observation_time)
    if observation_year is None:
        return True
    return int(target_year) < int(observation_year)


def build_reader_query(question: str, *, target_time: Any = None,
                       observation_time: Any = None,
                       target_in_question: bool = True,
                       time_source: Optional[str] = None) -> Dict[str, Any]:
    """构造 reader 查询：``Current date`` = 观测截止，历史目标只进题面。

    与 ``training.trajectory_samples`` 逐字同构（同一个函数、同一种字符串），
    因此训练与推理的 reader 输入不再有不一致的时间语义。
    """
    text = str(question)
    prompt: Dict[str, Any] = {'question': text}
    if target_in_question and target_is_historical(target_time, observation_time):
        prompt['question'] = f"In {year_of(target_time)}: {text}"
    observation = None if observation_time is None else str(observation_time)
    if observation is not None and observation.strip().lower() not in _UNKNOWN_VALUES:
        # reader.question 把 query['time'] 渲染成 "Current date: ..."。
        prompt['time'] = observation
    if time_source:
        prompt['time_source'] = time_source
    prompt['target_time'] = None if target_time is None else str(target_time)
    return prompt


__all__ = ['build_reader_query', 'target_is_historical', 'year_of']
