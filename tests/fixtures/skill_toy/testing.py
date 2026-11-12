from functools import lru_cache
from pathlib import Path
from bottle_scan.state_graph.skill_interface import load_spec
from bottle_scan.state_graph.extract_nodes import extract_nodes
from bottle_scan.state_graph.noise import fit_noise

@lru_cache(None)
def data():
    spec=load_spec(Path(__file__).with_name('spec.json'));rows,_=extract_nodes([],spec);noise=fit_noise(rows,spec)
    return rows,spec,noise

@lru_cache(None)
def trained():
    from bottle_scan.state_graph.dynamics import fit_dynamics
    from bottle_scan.state_graph.graph import build_graph
    from bottle_scan.state_graph.labels import generate_labels
    from bottle_scan.state_graph.clusters import cluster_nodes
    from bottle_scan.state_graph.skill import train_models
    rows,spec,noise=data();f=fit_dynamics(rows,noise,spec);g=build_graph(rows,noise,spec,f);labels=generate_labels(g,rows,spec);clusters=cluster_nodes(rows,g,noise,labels,spec)
    bundle,status=train_models(rows,g,labels,noise,f,spec)
    return rows,spec,noise,f,g,labels,clusters,bundle,status
