from __future__ import annotations
from collections import defaultdict
from typing import Callable, Hashable, Sequence, List

def reciprocal_rank_fusion(ranked_lists: Sequence[Sequence[object]], key_fn: Callable[[object], Hashable], k: int = 60) -> List[object]:
    scores=defaultdict(float); reps={}; ranks=defaultdict(list)
    for results in ranked_lists:
        for rank,item in enumerate(results,start=1):
            key=key_fn(item); reps.setdefault(key,item); scores[key]+=1.0/(k+rank); ranks[key].append(rank)
    fused=sorted(reps.values(),key=lambda x:scores[key_fn(x)],reverse=True)
    for item in fused:
        key=key_fn(item)
        if hasattr(item,"metadata"):
            item.metadata["_rrf_score"]=round(scores[key],8); item.metadata["_rrf_ranks"]=ranks[key]
    return fused

def rrf_score(rank:int,k:int=60)->float:
    if rank<1: raise ValueError("rank must be >= 1")
    return 1.0/(k+rank)
