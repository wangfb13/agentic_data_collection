"""Controller-specific outcome evaluators. Commands are not outcome evidence."""

def _missing(reason):return {'evaluated':False,'success':None,'level':None,'evidence':[{'reason':reason}]}


def e1e2_squeeze_l2(trial,window,condition):
    row=window.get('_legacy')
    if not row:return _missing('not_a_historical_squeeze_record')
    if row.get('downstream_controller')!='e1e2_squeeze':return _missing('different_consumer')
    if row.get('closure_floor')!=condition['consumer']['params']['closure_floor']:return _missing('different_closure_floor')
    if not window['is_terminal'] or type(row.get('downstream_success')) is not bool:return _missing('no_evaluated_terminal_L2')
    return {'evaluated':True,'success':row['downstream_success'],'level':'L2','evidence':[row['downstream_evidence'],{'source':row['provenance']['meta_path']} ]}


def forcepolicy_l2(trial,window,condition):
    if not window.get('is_terminal'):return _missing('not_handoff_window')
    outcome=window.get('consumer_outcome')
    if not outcome or outcome.get('controller')!=condition['consumer']['controller']:return _missing('no_actual_consumer_outcome')
    if outcome.get('params',{}).get('closure_floor')!=condition['consumer']['params']['closure_floor']:return _missing('different_closure_floor')
    version=condition['consumer'].get('version')
    if version and outcome.get('version')!=version:return _missing('different_consumer_version')
    if not outcome.get('version'):return _missing('unrecorded_consumer_version')
    if outcome.get('L1') is False:
        return {'evaluated':True,'success':False,'level':'L1','evidence':[outcome]}
    if type(outcome.get('L1')) is not bool or type(outcome.get('L2')) is not bool:return _missing('L1_or_L2_not_evaluated')
    return {'evaluated':True,'success':outcome['L1'] and outcome['L2'],'level':'L2','evidence':[outcome]}


def live_lift(trial,window,condition):
    out=forcepolicy_l2(trial,window,condition)
    actual=window.get('consumer_outcome') or {}
    if actual.get('consumer_id')==condition['id'] and actual.get('controller')==condition['consumer']['controller']:
        out.update(failure_cause=actual.get('failure_cause'),label_conflict=actual.get('label_conflict',False),
                   audit=actual.get('audit'),automatic_L2=actual.get('automatic_L2'))
    return out
