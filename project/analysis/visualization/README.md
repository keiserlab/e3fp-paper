# Drawing E3FP Graph Images

The following scripts create a graph demonstrating the flow of
information in E3FP fingerprinting. To run these, you must
have [PyGraphviz](https://pygraphviz.github.io/),
[NetworkX](https://networkx.github.io), and
[PyMOL](https://www.pymol.org) installed.

The SDF files for the three molecules whose graphs appear in
Figure 1 of the paper may be found in `$E3FP_PROJECT/data`:
- `CHEMBL2110918.sdf.bz2`: cypenamine (Figure 1a)
- `CHEMBL270807.sdf.bz2`: (R)-2-(2-(2-methylpyrrolidin-1-yl)ethyl)pyridine (Figure 1b)
- `CHEMBL210990.sdf.bz2`: 2-[2-[methyl-[3-[[7-propyl-3-(trifluoromethyl)-1,2-benzoxazol-6-yl]oxy]propyl]amino]pyrimidin-5-yl]acetic acid (Figure 1c)

To generate the graph for cypenamine, for example, run:

```bash
pymol -r make_shell_figures.py -- $E3FP_PROJECT/data/CHEMBL2110918.sdf.bz2
python draw_graph.py CHEMBL2110918
```

The graph will be saved to `CHEMBL2110918/graph.png`.
