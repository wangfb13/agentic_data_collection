"""Two-region bottle dimensions. Pure calculations; no robot imports."""
from __future__ import annotations
import math


def bottle_dimensions(dimensions):
    """Merge old neck/cap into an enclosing upper cylinder, preserving height."""
    def read(part):
        result={key:float(dimensions[part][key]) for key in ('diameter_mm','height_mm')}
        if any(not math.isfinite(x) or x<=0 for x in result.values()):
            raise ValueError('瓶子直径和高度必须为有限正数')
        return result
    body=read('body')
    if 'upper' in dimensions:upper=read('upper')
    else:
        neck,cap=read('neck'),read('cap')
        upper={'diameter_mm':max(neck['diameter_mm'],cap['diameter_mm']),
               'height_mm':neck['height_mm']+cap['height_mm']}
    return {'body':body,'upper':upper}


def bottle_sections(dimensions):
    return list(bottle_dimensions(dimensions).items())
