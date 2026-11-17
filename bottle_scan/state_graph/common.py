"""Serialization, identifiers and whole-experiment partitioning."""
import hashlib
import json
from pathlib import Path
import numpy as np


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean(v) for v in value]
    if isinstance(value,np.ndarray): return clean(value.tolist())
    if isinstance(value,np.generic): return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value): return None
    return value


def digest(value):
    return hashlib.sha256(json.dumps(clean(value),sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def group_id(row):
    return str(row.get('group_id') or str(row['session'])+'|'+str(row['trial_id']))


def independent(row):
    return bool(row.get('data_role') == 'independent_validation' or row.get('evaluation_only') or row.get('frozen_independent') or row.get('evaluation_scope')=='post_freeze_independent')


def group_folds(rows,n_splits=5):
    from sklearn.model_selection import GroupKFold
    groups=np.array([group_id(r) for r in rows]);n=len(set(groups))
    return list(GroupKFold(min(n_splits,n)).split(rows,groups=groups)) if n>=2 else []
