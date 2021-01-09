"""Fingerprint molecule, save shell images, and save fingerprinting graph.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import glob
import os
from time import sleep
import json
import logging
import sys

import numpy as np
from pymol import cmd, stored
from pymol.cgo import *
import PIL
from PIL import Image, ImageChops

import sklearn
import sklearn.decomposition
import networkx as nx
from networkx.readwrite import json_graph
import rdkit
import rdkit.Chem

from python_utilities.io_tools import touch_dir
from python_utilities.scripting import setup_logging
from python_utilities.plotting.color_cycles import COLOR_ALPHABET
from python_utilities.plotting.util import compute_yuv_dist

from e3fp.conformer.util import mol_from_sdf
from e3fp.fingerprint.fprinter import Fingerprinter, signed_to_unsigned_int
from e3fp.fingerprint.array_ops import as_unit, make_rotation_matrix, \
                                       make_transform_matrix, transform_array

setup_logging(verbose=True)

CPK_COLORS = {"H": (255, 255, 255),
              "C": (144, 144, 144),
              "N": (48, 80, 248),
              "O": (255, 13, 13),
              "F": (144, 224, 80),
              "P": (255, 128, 0),
              "S": (255, 255, 48),
              "Cl": (31, 240, 31)}

ROOT_NODE_NAME = "mol"
BITS = 2**32
LEVEL = 5
RADIUS_MULTIPLIER = 1.718


def fingerprint_mol(mol):
    fp = Fingerprinter(bits=BITS, level=LEVEL,
                       radius_multiplier=RADIUS_MULTIPLIER, stereo=True,
                       remove_duplicate_substructs=True)
    fp.run(mol=mol, conf=0)
    return fp


def save_aligned_conf_to_pdb(pdb_file, mol, conf_id=0):
    """Save mol conformer to PDB file."""
    conf = mol.GetConformer(conf_id)
    coords = np.array([conf.GetAtomPosition(i) for i
                       in range(conf.GetNumAtoms())])
    pca = sklearn.decomposition.PCA(2)
    pca.fit(coords)
    mean_coord = np.mean(coords, axis=0)
    x = as_unit(pca.components_[0, :])
    y = as_unit(pca.components_[1, :])
    z = np.cross(x, y)
    trans_mat = make_transform_matrix(mean_coord, y=y, z=z)
    trans_coords = transform_array(trans_mat, coords)
    for i in range(conf.GetNumAtoms()):
        conf.SetAtomPosition(i, trans_coords[i, :])
    writer = rdkit.Chem.rdmolfiles.PDBWriter(pdb_file)
    writer.write(mol, conf_id)
    writer.close()
    # renumber atoms by atom_idx
    lines = []
    with open(pdb_file, "rU") as f:
        lines = f.readlines()
    with open(pdb_file, "w") as f:
        i = 0
        for line in lines:
            if line.startswith("HETATM"):
                elem = mol.GetAtomWithIdx(i).GetSymbol()
                atom_name = (elem + "{:d}  ".format(i + 1))[:4]
                line = line[:13] + atom_name + line[17:]
                i += 1
            f.write(line)
    return np.argsort(trans_coords[:, 0], )


def write_pdb_files(fprinter, bits, out_dir=""):
    """Write all substructs to PDB files."""
    pdbs = []
    current_level = fprinter.current_level
    for i in range(0, current_level + 1):
        substruct_dir = os.path.join(out_dir, "substructs{}".format(i))
        pdbs.extend(fprinter.substructs_to_pdb(level=i, bits=bits,
                                               out_dir=substruct_dir))
    return pdbs


def load_shell_in_pymol(pdb_file, shell):
    """Load shell/substruct PDB file in PyMOL."""
    cmd.load(pdb_file, shell.identifier)


def create_shell_graph(fp, radius_multiplier=RADIUS_MULTIPLIER,
                       only_unique=True):
    """Create shell graph by backtracking from final level to root."""
    current_level = fp.current_level
    shells_to_examine = set(fp.get_shells_at_level(current_level,
                                                   exact=only_unique))
    shell_to_node_map = {}
    identifier_to_shell = {}
    shell_level = {signed_to_unsigned_int(k.identifier): int(
                       round(k.radius / radius_multiplier))
                   for k in shells_to_examine}
    shell_atom = {signed_to_unsigned_int(k.identifier): int(k.center_atom)
                  for k in shells_to_examine}
    shell_to_substruct_map = {}
    G = nx.DiGraph()  # create an empty graph
    while len(shells_to_examine) > 0:
        shell = shells_to_examine.pop()
        if shell.is_duplicate:
            continue
        last_shell = shell.last_shell

        shell.identifier = signed_to_unsigned_int(shell.identifier)

        identifier_to_shell.setdefault(shell.identifier, set()).add(shell)
        shell_to_substruct_map[shell] = shell.substruct
        if shell in shell_to_node_map:
            continue
        if last_shell is None:
            G.add_edge(ROOT_NODE_NAME, shell.identifier)
        else:
            if last_shell.is_duplicate:
                last_shell = last_shell.duplicate
            last_shell.identifier = signed_to_unsigned_int(
                last_shell.identifier)
            shells_to_examine.add(last_shell)
            shell_level[last_shell.identifier] = (
                shell_level[shell.identifier] - 1)
            shell_atom[last_shell.identifier] = int(last_shell.center_atom)
            G.add_edge(last_shell.identifier, shell.identifier)
            print("added edge between {} and {}".format(
                last_shell.identifier, shell.identifier))
        shells_to_examine.update(shell.shells)
        for s in shell.shells:
            if s.is_duplicate:
                s = s.duplicate
            s.identifier = signed_to_unsigned_int(s.identifier)
            shell_level[s.identifier] = shell_level[shell.identifier] - 1
            shell_atom[s.identifier] = int(s.center_atom)
            G.add_edge(s.identifier, shell.identifier)
            print("added edge between {} and {}".format(s.identifier,
                                                        shell.identifier))
        shell_to_node_map[shell] = None
    nx.set_node_attributes(G, name='shell', values=identifier_to_shell)
    nx.set_node_attributes(G, name='level', values=shell_level)
    nx.set_node_attributes(G, name='atom_id', values=shell_atom)
    return G


def get_cgo_sphere_obj(center, radius, color=[0., 0., 0.], alpha=1.):
    """Get CGO parameters to create sphere in PyMOL"""
    logging.debug("Creating CGO sphere with center {} and radius {}".format(
        center, radius))
    obj = []
    if alpha < 1.:
        obj.extend([ALPHA, alpha])
    obj.extend([SPHERE] + list(center) + [radius] + color)
    return obj


def get_cgo_segment_obj(start, stop, linespacing=0., color=[0., 0., 0.],
                        alpha=1.):
    """Get CGO parameters to create line segment in PyMOL"""
    obj = []
    if linespacing > 0.:
        seg_len = np.linalg.norm(stop - start)
        norm = (stop - start) / seg_len
        seg_num = int(seg_len / (2 * linespacing))
        starts = np.outer(np.arange(seg_num) * 2 * seg_len, norm)
        stops = np.outer(np.arange(1, seg_num + 1) * 2 * seg_len, norm)
        for i in range(starts.shape[0]):
            obj.extend([VERTEX] + list(starts[i]) + [VERTEX] + list(stops[i]))
    else:
        obj.extend([VERTEX] + list(start) + [VERTEX] + list(stop))
    return obj


def get_cgo_cylinder_obj(start, stop, rad=.2, color=[0., 0., 0.], alpha=1.):
    """Get CGO parameters to create cylinder in PyMOL"""
    logging.debug(
        "Creating CGO cylinder with start {}, stop {}, and radius".format(
            start, stop, rad))
    obj = [CYLINDER, ]
    obj.extend(list(start) + list(stop) + [rad] + list(color) + list(color))
    return obj


def get_cgo_arc_obj(center, norm, start, radians, linespacing=0.,
                    color=[0., 0., 0.], alpha=1., res=.5 * np.pi / 180):
    """Get CGO parameters to create arc in PyMOL"""
    logging.debug(("Creating CGO arc with start {}, center {}, norm {}, "
                   "covering {} deg").format(start, center, norm,
                                             radians * 180 / np.pi))
    obj = []
    obj = [COLOR] + color
    start_ref = start - center
    rad = np.linalg.norm(start_ref)
    if rad < 1e-7:
        return []
    rot1 = make_rotation_matrix(norm, np.array([0., 0., 1.]))
    start2 = np.dot(rot1, start_ref.reshape(3, 1)).T
    rot2 = make_rotation_matrix(start2, np.array([1., 0., 0.]))
    inv_rot = np.dot(rot2, rot1).T
    angles = np.linspace(0., radians, int(float(radians) / res))
    ref_xyz = np.empty((angles.shape[0], 3), dtype=np.double)
    ref_xyz[:, 0] = rad * np.cos(angles)
    ref_xyz[:, 1] = rad * np.sin(angles)
    ref_xyz[:, 2] = 0
    xyz = np.dot(inv_rot, ref_xyz.T).T + center
    if linespacing > 0.:
        i = 0
        skip_num = max(1, int(linespacing / (rad * res)))
        while i < (xyz.shape[0] - 1 - skip_num):
            for j in range(skip_num):
                obj.extend(get_cgo_segment_obj(xyz[i, :], xyz[i + 1, :]))
                i += 1
            i += skip_num
    else:
        for i in range(xyz.shape[0] - 1):
            obj.extend(get_cgo_segment_obj(xyz[i, :], xyz[i + 1, :]))
    return obj


def get_cgo_circle_obj(center, norm, rad, start=None, linespacing=0.,
                       color=[0., 0., 0.], alpha=1.):
    """Get CGO parameters to create circle in PyMOL."""
    if np.linalg.norm(norm - np.array([1., 0., 0.])) > 1e-7:
        start = rad * as_unit(np.cross(np.array([1., 0., 0.]), norm)) + center
    else:
        start = rad * as_unit(np.cross(np.array([0., 1., 0.]), norm)) + center
    return get_cgo_arc_obj(center, norm, start, 2 * np.pi, linespacing, color,
                           alpha)


def get_cgo_cone_obj(start, stop, rad, color=[0., 0., 0.], alpha=1.):
    """Get CGO parameters to create cone in PyMOL."""
    return ([CONE] + list(start) + list(stop) + [rad, 0.] + list(color) +
            list(color)) + [1.0, 0.0]


def get_cgo_arrow_obj(start=np.array([0., 0., 0.]),
                      stop=np.array([0., 0., 1.]), rad=.1, arrow_rad=.2,
                      arrow_height=.3, linespacing=0., color=[0., 0., 0.],
                      alpha=1.):
    """Get CGO parameters to create arrow in PyMOL."""
    logging.debug("Creating CGO cone with start {} and stop {}".format(
        start, stop))
    obj = []
    v = stop - start
    arrow_len = np.linalg.norm(v)
    u = v / arrow_len
    seg_stop = start + max(arrow_len - arrow_height, 0) * u
    obj.extend(get_cgo_cylinder_obj(start, seg_stop, rad, color=color,
                                    alpha=alpha))
    obj.extend(get_cgo_cone_obj(seg_stop, stop, arrow_rad, color=color,
                                alpha=alpha))
    return obj


def get_disk_norm(vaxis, haxis, offset=0.):
    norm = as_unit(vaxis)
    rot1 = make_rotation_matrix(norm, np.array([0., 0., 1.]))
    haxis2 = np.dot(rot1, haxis.reshape(3, 1)).T
    rot2 = make_rotation_matrix(haxis2, np.array([1., 0., 0.]))
    inv_rot = np.dot(rot2, rot1).T
    ref_norm = as_unit(np.array([np.cos(offset), np.sin(offset), 0.]))
    disk_norm = np.dot(inv_rot, ref_norm.reshape(3, 1)).T
    return disk_norm


def draw_xy_axes(scale=1.0):
    """Draw x and y axes in PyMOL."""
    logging.debug("Drawing axes.")
    obj = []
    obj.extend(get_cgo_arrow_obj(start=np.array([0, 0, 0.]),
                                 stop=scale * np.array([0., 1., 0.])))
    obj.extend(get_cgo_arrow_obj(start=np.array([0, 0, -.1]),
                                 stop=scale * np.array([0., 0., 1.])))
    cmd.load_cgo(obj, cmd.get_unused_name('axes'))


def draw_sphere_marker(radius, coords, marker_rad=.15, thickness=.05,
                       color=[0., 0., 0.], alpha=1.):
    logging.debug("Drawing sphere marker.")
    coords = np.asarray(coords)
    coords = radius * coords / np.linalg.norm(coords)
    obj = []
    start = coords
    stop = coords * (radius + thickness) / radius
    obj.extend(get_cgo_cylinder_obj(start, stop, marker_rad, color=color,
                                    alpha=alpha))
    obj.extend(get_cgo_arrow_obj(start=np.array([0, 0, 0.]), stop=coords,
                                 rad=marker_rad / 2., arrow_rad=marker_rad))
    cmd.load_cgo(obj, cmd.get_unused_name('marker'))


def draw_shell_sphere(radius, polar_cone_angle=np.pi / 36.,
                      axis0=np.array([0., 1., 0.]),
                      axis1=np.array([0., 0., 1.]), linewidth=1,
                      linespacing=0.15, alpha=0.35, name=None):
    """Draw sphere and stereo regions at specified radius in PyMOL."""
    center = np.array([0., 0., 0.], dtype=np.double)
    obj = []
    # draw sphere
    obj.extend(get_cgo_sphere_obj(center, radius, color=[1., 1., 1.],
               alpha=alpha))
    cmd.load_cgo(obj, cmd.get_unused_name('sphere_{:.4f}'.format(radius)))

    # draw polar circles
    obj = []
    obj.extend([BEGIN, LINES, COLOR] + [0., 0., 0.])
    v_to_cone_center = radius * np.cos(polar_cone_angle) * axis0
    cone_radius = radius * np.sin(polar_cone_angle)
    obj.extend(get_cgo_circle_obj(center + v_to_cone_center, axis0,
               cone_radius, linespacing=linespacing))
    obj.extend(get_cgo_circle_obj(center - v_to_cone_center, -axis0,
               cone_radius, linespacing=linespacing))
    # draw equator
    obj.extend(get_cgo_circle_obj(center, axis0,
               radius, linespacing=linespacing))
    # draw quadrant arcs
    for i in (1, 3, 5, 7):
        if i in (5, 7):
            color = [.8, .8, .8]
        else:
            color = [0., 0., 0.]
        disk_norm = get_disk_norm(axis0, axis1, i * np.pi / 4.)
        arc_start = (cone_radius * as_unit(np.cross(disk_norm, axis0)) +
                     center + v_to_cone_center)
        obj.extend(get_cgo_arc_obj(
            center, disk_norm, arc_start, np.pi - 2 * polar_cone_angle,
            linespacing=linespacing, color=color))
    # draw axes
    obj.append(END)
    lname = cmd.get_unused_name('contours_{:.4f}'.format(radius))
    cmd.load_cgo(obj, lname)
    cmd.set("cgo_line_width", linewidth, lname)


def pymol_load_with_defaults(pdb_file=None, stick_radius=.1, sphere_scale=.15):
    """Load PyMOL with visualization defaults."""
    pymol_define_colors()
    cmd.bg_color("white")
    if pdb_file is not None:
        cmd.load(pdb_file)
    cmd.set("ray_opaque_background", 0)
    cmd.set("ray_trace_mode", 1)
    cmd.set("antialias", 0)
    cmd.set('stick_radius', stick_radius)
    cmd.set('valence', 1)
    cmd.set('fog_start', 0.15)
    cmd.hide('everything', "all")
    cmd.show('sticks')
    cmd.show('spheres')
    cmd.set('sphere_scale', sphere_scale)
    cmd.set('ray_shadow', 0)
    cmd.set('ray_shadows', 0)
    sleep(0.5)


def get_atom_types(mol, graph):
    logging.info("Get atom types of nodes.")
    root = next(nx.topological_sort(graph))
    assert(root == ROOT_NODE_NAME)
    atom_nodes = sorted(graph[root].keys())
    atom_types_dict = {}
    for node in atom_nodes:
        atom_types_dict[node] = set([x.center_atom for x
                                    in graph.nodes[node]["shell"]])
    return atom_types_dict


def pymol_color_atoms_by_elem(elem_colors=CPK_COLORS):
    """Color atoms by element in PyMOL"""
    logging.info("Coloring atoms.")
    for elem, rgb in elem_colors.items():
        color_name = elem + "_color"
        cmd.set_color(color_name, list(rgb))
        cmd.color(color_name, "e. {}".format(elem))


def define_colors_by_atom_types(d, mol, cycle=COLOR_ALPHABET,
                                elem_colors=CPK_COLORS):
    """Pick atom type colors to be similar to CPK but distinguishable."""
    logging.info("Getting colors from atom types.")
    colors = cycle.copy()
    atom_colors = {}
    for i, (identifier, atoms) in enumerate(
            sorted(d.items(), key=lambda x: (-len(x[1]), x[0]))):
        elem = mol.GetAtomWithIdx(int(list(atoms)[0])).GetSymbol()
        cpk = elem_colors.get(elem, (0, 0, 0))
        rgb = sorted([(x, int(10 * compute_yuv_dist(cpk, x)) * .1, j)
                      for j, x in enumerate(colors)
                      if colors[x] not in ["pink", "quagmire"]],
                     key=lambda x: (x[1], x[2]))[0][0]
        color_name = colors.pop(rgb)
        for a in atoms:
            atom_colors[a] = (rgb, color_name)
    return atom_colors


def pymol_define_colors(color_cycle=COLOR_ALPHABET):
    """Define color names in PyMOL."""
    for rgb, name in color_cycle.items():
        cmd.set_color(name, list(rgb))


def pymol_color_by_atom_types(atom_colors, mode="id"):
    """Color atoms by type in PyMOL."""
    if mode == "id":
        for atom_id, (rgb, color_name) in atom_colors.items():
            cmd.color(color_name, "id {}".format(atom_id + 1))
    elif mode == "name":
        for atom_name, (rgb, color_name) in atom_colors.items():
            cmd.color(color_name, "name {}".format(atom_name))


def partial_opaque_to_opaque(image_file, crop=True):
    """Render transparent parts of PNG to be opaque."""
    logging.info("Making {} opaque.".format(image_file))
    while True:
        try:  # keep checking for image to exist
            img = PIL.Image.open(image_file)
        except IOError:
            continue
        try:  # keep checking for image to be completely saved
            img = img.convert("RGBA")
            break  # it's ready!
        except Exception:
            img.close()
        sleep(0.5)
    pixels = img.getdata()

    new_data = []
    for item in pixels:
        if item[3] != 0:
            new_data.append(item[:3] + (255,))
        else:
            new_data.append(item)

    if crop:
        # adapted from http://stackoverflow.com/a/10616717
        bg = Image.new(img.mode, img.size, img.getpixel((0, 0)))
        diff = ImageChops.difference(img, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            img = img.crop(bbox)

    img.save(image_file, "PNG")


def save_shell_figures(mol_pdb_file, shell_pdb_files, graph, atom_colors,
                       dpi=3000, overwrite=False, out_dir=""):
    """Save all mol/shell/substructure figures."""
    cmd.reinitialize()
    sleep(0.5)

    mol_color_hetero_img = os.path.join(out_dir, "mol_plain.png")
    mol_color_types_img = os.path.join(out_dir, "mol_atom_types.png")
    pymol_load_with_defaults(mol_pdb_file)
    cmd.zoom('all', complete=1, buffer=1)
    if overwrite or not os.path.isfile(mol_color_hetero_img):
        cmd.util.cbaw()
        pymol_color_atoms_by_elem()
        cmd.ray(dpi)
        cmd.png(mol_color_hetero_img)
        partial_opaque_to_opaque(mol_color_hetero_img)
        sleep(0.5)

    if overwrite or not os.path.isfile(mol_color_types_img):
        pymol_color_by_atom_types(atom_colors)
        cmd.ray(dpi)
        cmd.png(mol_color_types_img)
        partial_opaque_to_opaque(mol_color_types_img)
        sleep(0.5)

    stored.names_ids = []
    cmd.iterate("all", "stored.names_ids.append((ID, name))")
    ids_names_map = dict(stored.names_ids)
    atom_colors = {ids_names_map[atom_id + 1]: color for atom_id, color
                   in atom_colors.items()}

    vector_shell_pdb_file = None
    vector_shell = None
    radii = set()
    for pdb_file in shell_pdb_files:
        identifier = int(os.path.basename(pdb_file).split('.')[0])
        shell_image_file = os.path.join(
            out_dir, "{}_shell.png".format(identifier))
        graph.nodes[identifier]["image"] = shell_image_file
        if not overwrite and os.path.isfile(shell_image_file):
            continue
        cmd.reinitialize()
        pymol_load_with_defaults(pdb_file)
        shell = list(graph.nodes[identifier]["shell"])[0]
        radius = shell.radius
        if radius > 0:
            radii.add(radius)
            draw_shell_sphere(radius, linespacing=radius * .15 / 1.5)
            cmd.center('sphere_*')
            cmd.origin('sphere_*')
            if vector_shell is None and round(radius/RADIUS_MULTIPLIER) == 1.:
                vector_shell_pdb_file = pdb_file
                vector_shell = shell

        pymol_color_by_atom_types(atom_colors, mode="name")
        cmd.turn('y', 90)
        cmd.turn('x', 35)
        cmd.zoom('all', complete=1)
        cmd.ray(dpi)
        cmd.png(shell_image_file)
        partial_opaque_to_opaque(shell_image_file)
        sleep(0.5)

    vector_figure = os.path.join(out_dir, "vector_axes.png")
    if overwrite or not os.path.isfile(vector_figure):
        cmd.reinitialize()
        pymol_load_with_defaults(vector_shell_pdb_file)
        radius = vector_shell.radius
        draw_shell_sphere(radius, linespacing=radius * .15 / 1.5)
        center_atom = vector_shell.center_atom
        for atom in vector_shell.atoms:
            dist = cmd.get_distance(
                "{} and id {}".format(vector_shell.identifier, center_atom),
                "{} and id {}".format(vector_shell.identifier, atom),
                state=1)
            if dist <= radius:
                coords = cmd.get_coords("{} and id {}".format(
                    vector_shell.identifier, atom), state=1)[0]
                draw_sphere_marker(radius, coords)
        cmd.center('sphere_*')
        cmd.origin('sphere_*')

        pymol_color_by_atom_types(atom_colors, mode="name")
        cmd.turn('y', 90)
        cmd.turn('x', 35)
        cmd.delete(str(vector_shell.identifier))
        cmd.zoom('all', complete=1)
        cmd.ray(dpi)
        cmd.png(vector_figure)
        partial_opaque_to_opaque(vector_figure)
        sleep(0.5)

    axes_figure = os.path.join(out_dir, "axes.png")
    if overwrite or not os.path.isfile(axes_figure):
        cmd.reinitialize()
        pymol_load_with_defaults()
        draw_shell_sphere(min(radii), linespacing=min(radii) * .15 / 1.5)
        sleep(.1)
        draw_xy_axes(scale=min(radii))
        sleep(.1)
        cmd.turn('y', 90)
        cmd.turn('x', 35)
        cmd.zoom('all', complete=1)
        cmd.ray(dpi)
        cmd.png(axes_figure)
        partial_opaque_to_opaque(axes_figure)
        sleep(0.5)


if __name__ == "__main__":
    usage = "pymol -r make_shell_figures.py -- <sdf_file>"
    try:
        sdf_file = sys.argv[1]
    except IndexError:
        sys.exit(usage)

    mol_name = os.path.basename(sdf_file).split(".")[0]
    out_dir = mol_name
    touch_dir(out_dir)

    json_out_file = os.path.join(out_dir, "graph.json")
    aligned_mol_pdb_file = os.path.join(out_dir, "mol.pdb")

    mol = mol_from_sdf(sdf_file)
    left_to_right_atom_ids = save_aligned_conf_to_pdb(aligned_mol_pdb_file,
                                                      mol)
    fprinter = fingerprint_mol(mol)
    graph = create_shell_graph(fprinter, radius_multiplier=RADIUS_MULTIPLIER,
                               only_unique=True)
    atom_types_dict = get_atom_types(mol, graph)
    atom_colors_dict = define_colors_by_atom_types(atom_types_dict, mol)
    write_pdb_files(fprinter, BITS, out_dir=out_dir)
    pdb_dirs = sorted(glob.glob(os.path.join(out_dir, "substructs*")))
    pdbs = glob.glob(pdb_dirs[-1] + "/*")
    save_shell_figures(aligned_mol_pdb_file, pdbs, graph, atom_colors_dict,
                       overwrite=False, out_dir=out_dir)
    print("Writing JSON file.")
    out_graph = graph.copy()
    out_graph.remove_node(ROOT_NODE_NAME)
    node_order = {}
    left_to_right_atom_ids = list(left_to_right_atom_ids)
    for node in out_graph:
        del(out_graph.nodes[node]["shell"])
        if out_graph.nodes[node]["level"] == 0:
            node_order[node] = left_to_right_atom_ids.index(
                out_graph.nodes[node]["atom_id"])
        else:
            node_order[node] = None
    nx.set_node_attributes(out_graph, name='order', values=node_order)

    out_json_graph = json_graph.node_link_data(out_graph)

    with open(json_out_file, "w") as f:
        f.write(json.dumps(out_json_graph, sort_keys=False, indent=4,
                           separators=(',', ': ')))
