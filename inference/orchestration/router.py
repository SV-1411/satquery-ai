from __future__ import annotations

import re


def route_query(query: str, file_count: int) -> str:
    q = query.lower()
    if file_count >= 2 and re.search(r"sar|radar|optical|sensor|together|fuse|agreement|distinguish|complement", q) and not re.search(r"between|before|after|date|changed|change", q):
        return "OPTICAL_SAR_FUSION"
    if file_count >= 2 and re.search(r"change|changed|between|before|after|increased|decreased", q):
        return "BI_TEMPORAL_CHANGE_VQA"
    if file_count >= 2 and re.search(r"sar|radar|optical|sensor|together|fuse|agreement|distinguish|complement", q):
        return "OPTICAL_SAR_FUSION"
    if file_count > 1:
        return "MULTI_OBSERVATION_SYNTHESIS"
    if re.search(r"highlight|locate|where|mark|region|outline", q):
        return "TEXT_GUIDED_GROUNDING"
    if re.search(r"describe|caption|scene|land-cover", q):
        return "SINGLE_IMAGE_CAPTIONING"
    return "SINGLE_IMAGE_VQA"
