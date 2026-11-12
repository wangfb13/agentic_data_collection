"""Translate recorded movement and generic vector actions to the existing adapter."""

def executed_action(trial,window,spec):
    if 'u_executed' in window:return {'u':window.get('u_executed'),'receipt':window.get('u_receipt')}
    row=window.get('_legacy',{})
    delta=row.get('executed_delta_a_mm')
    return {'u':[delta] if isinstance(delta,(int,float)) else None,'receipt':row.get('execution_receipt')}


class RecoveryAdapter:
    """Reuse an already-owned V4 runner; caller supplies the observation adapter."""
    def __init__(self,backend,runner,target,observe,event_writer=None):
        self.backend=backend;self.runner=runner;self.target=target
        self.observer=observe;self.writer=event_writer;self.probe=None;self.status=None
    def observe(self):return self.observer(self)
    def execute(self,u):
        action={'axis':'a','interface':'open_retreat_translate_reapproach_probe','delta_mm':float(u[0])}
        row={'requested_action':action,'receipt':None}
        self.target,self.status,self.probe,row=self.backend._recover_once(self.runner,self.target,action,row)
        return row['receipt']
    def emit(self,event):
        if self.writer:self.writer(event)
