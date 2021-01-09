"""Draw graph showing fingerprint process from shell images.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import json
import sys

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pygraphviz as pgv

sns.set_style("white")

DPI = 1200
SIZE = "11,4"


if __name__ == "__main__":
    usage = "python draw_graph.py <mol_figs_dir>"
    try:
        mol_figs_dir = sys.argv[1]
    except IndexError:
        sys.exit(usage)

    json_file = os.path.join(mol_figs_dir, "graph.json")
    plain_mol_file = os.path.join(mol_figs_dir, "mol_plain.png")
    atom_types_mol_file = os.path.join(mol_figs_dir, "mol_atom_types.png")

    tree = {}
    with open(json_file, "r") as f:
        tree = json.load(f)

    # plain graph
    label_text = """<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="{}">
                    <TR><TD width="{}" height="{}" fixedsize="true"><IMG SRC="{}" scale="true"/></TD></TR>
                    </TABLE>>"""
    node_attributes = {"imagescale": "both", "fixedsize": True, "penwidth": 0,
                       "margin": 0, "layer": 2}
    edge_attributes = {"arrowsize": .3, "layer": 1, "penwidth": .5,
                       "color": "#00000080"}
    subgraph_dict = {}
    scale_factor = .5
    G = pgv.AGraph(directed=True, dpi=DPI, bgcolor="transparent",
                   outputorder="nodesfirst", nodesep=.05, ranksep=".35",
                   size=SIZE)
    first_level_id = []
    G.add_node("mol_plain", label=label_text.format(10, 150, 50,
                                                    plain_mol_file),
               shape="box", **node_attributes)
    G.add_node("mol_atom_types", label=label_text.format(10, 150, 50,
                                                         atom_types_mol_file),
               shape="box", **node_attributes)
    G.add_subgraph(["mol_plain", "mol_atom_types"], rank="same", rankdir="LR")
    G.add_edge("mol_plain", "mol_atom_types", minlen=10.0, **edge_attributes)

    for node in sorted(tree['nodes'], key=lambda x: (x["level"], x["order"])):
        subgraph_dict.setdefault(node['level'], set()).add(node["id"])
        if node['level'] == 0:
            padding = 0
            scale = scale_factor * .3
            first_level_id.append(node['id'])
            G.add_edge("mol_atom_types", node["id"], **edge_attributes)
        else:
            padding = 9
            scale = scale_factor * 1.

        try:
            G.add_node(node["id"], label=label_text.format(
                           padding, 100 * scale, 100 * scale, node["image"],
                           node["id"]),
                       shape="circle", **node_attributes)
        except KeyError:
            continue

    for rank, ids in sorted(subgraph_dict.items()):
        print("Rank: {}".format(rank))
        for id in sorted(ids):
            print(" {}\t{}".format(id, id % 1024))
        G.add_subgraph(sorted(ids), rank="same", rankdir="LR")

    for i in range(len(first_level_id)):
        if i == len(first_level_id) - 1:
            break
        G.add_edge(first_level_id[i], first_level_id[i + 1], style="invis",
                   rank="same")

    for link in tree["links"]:
        G.add_edge(link["source"], link["target"], **edge_attributes)

    G.layout(args="-Goverlap=false")
    G.draw(os.path.join(mol_figs_dir, 'graph.png'), prog='dot')

    # fingerprint bitvector as "barcode"
    folded_ids = sorted([y % 1024 for ids in subgraph_dict.values()
                         for y in ids])
    print("\n ".join(map(str, folded_ids)))
    bitvect = np.zeros(1024, dtype=np.int)
    bitvect[folded_ids] = 1
    fig = plt.figure(figsize=(50, 10))
    ax = fig.add_subplot(111)
    ax.matshow(bitvect.reshape(1024, 1), vmax=1, interpolation=None)
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(fig, left=True, bottom=True)
    fig.savefig(os.path.join(mol_figs_dir, "bitvector.svg"), dpi=300)
