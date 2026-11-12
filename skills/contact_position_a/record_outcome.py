"""Append a reviewed lift/hold or task outcome to an existing episode record."""
import argparse
from .scheduler import append_reviewed_outcome


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--events',required=True);p.add_argument('--l2',choices=['success','failure'],required=True);p.add_argument('--l3',choices=['success','failure']);p.add_argument('--evidence',required=True)
    a=p.parse_args();append_reviewed_outcome(a.events,L2=a.l2=='success',L3=None if a.l3 is None else a.l3=='success',evidence=a.evidence)

if __name__=='__main__':main()
