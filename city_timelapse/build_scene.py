"""Procedural city-growth timelapse in Blender (Cycles).

Run headless:
    python3 build_scene.py --out /path/to/frames [--res 960 540] [--samples 12]
        [--frames 1 400]     render only a frame range
        [--save scene.blend] also save the .blend

Every building, road and ground cell carries a "birth" year. A Value node in
the materials and the "Growth" node group is keyframed with the current year,
which hides
anything not yet built (Geometry Nodes delete those faces), so the whole
growth runs from one static mesh.
"""
import argparse
import math
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import timeline as tl  # noqa: E402

RNG = np.random.default_rng(7)
EXTENT = 48.0          # half-size of the ground
CORE = np.array([0.0, 0.0])


# ---------------------------------------------------------------- terrain ---

def river_y(x):
    return 1.7 * np.sin(x * 0.22 + 0.4) + 0.9 * np.sin(x * 0.55 + 1.3) - 1.5


def river_width(x):
    return 0.55 + 0.012 * np.abs(x)


CORE[1] = river_y(0.0) + 1.2


def smooth_noise(x, y, seed, freq):
    """Cheap smooth noise in [-1, 1] from a sum of random plane waves."""
    r = np.random.default_rng(seed)
    out = np.zeros_like(x)
    for _ in range(8):
        a = r.uniform(0, math.pi)
        f = freq * r.uniform(0.6, 1.6)
        out += np.sin((np.cos(a) * x + np.sin(a) * y) * f + r.uniform(0, 6.3))
    return out / 8 * 2.2


def effective_distance(x, y):
    """Distance from the core, stretched along the river and noisy."""
    dx, dy = x - CORE[0], y - CORE[1]
    d = np.sqrt((dx / 1.35) ** 2 + dy ** 2)
    south = y < river_y(x)
    d = np.where(south, d * 1.3 + 0.4, d)
    d *= 1 + 0.28 * smooth_noise(x, y, 1, 0.35)
    return d


VILLAGES = [(-7, 6, 1480), (8, 5, 1520), (-12, -6, 1560), (5, -8, 1600),
            (14, -2, 1640), (-4, 12, 1620), (-15, 3, 1700), (11, 11, 1720)]


def birth_year(x, y):
    b = tl.birth_from_distance(effective_distance(x, y))
    # outlying villages start on their own before the city swallows them
    for vx, vy, vyear in VILLAGES:
        dv = np.hypot(x - vx, y - vy) * (1 + 0.3 * smooth_noise(x, y, 5, 1.2))
        bv = np.where(dv < 1.2, vyear + dv * 120, 1e5)
        b = np.minimum(b, bv)
    return b


def is_park(x, y):
    return (smooth_noise(x, y, 9, 0.5) > 0.75) & (effective_distance(x, y) > 2.5)


# ---------------------------------------------------------------- helpers ---

def mesh_object(name, verts, faces, face_attrs=None, point_attrs=None):
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", np.asarray(verts, np.float32).ravel())
    faces = np.asarray(faces, np.int32)
    n = faces.shape[1]
    me.loops.add(faces.size)
    me.loops.foreach_set("vertex_index", faces.ravel())
    me.polygons.add(len(faces))
    me.polygons.foreach_set("loop_start", np.arange(0, faces.size, n, dtype=np.int32))
    me.update(calc_edges=True)
    for dom, attrs in (("FACE", face_attrs), ("POINT", point_attrs)):
        for key, val in (attrs or {}).items():
            val = np.asarray(val, np.float32)
            if val.ndim == 2:
                a = me.attributes.new(key, "FLOAT_COLOR", dom)
                a.data.foreach_set("color", val.ravel())
            else:
                a = me.attributes.new(key, "FLOAT", dom)
                a.data.foreach_set("value", val)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def boxes(cx, cy, w, d, h, ang, z0=0.0):
    """Arrays of verts/faces for many oriented boxes (top + 4 sides)."""
    n = len(cx)
    c, s = np.cos(ang), np.sin(ang)
    lx = np.array([-1, 1, 1, -1]) * 0.5
    ly = np.array([-1, -1, 1, 1]) * 0.5
    px = cx[:, None] + (lx[None] * w[:, None]) * c[:, None] - (ly[None] * d[:, None]) * s[:, None]
    py = cy[:, None] + (lx[None] * w[:, None]) * s[:, None] + (ly[None] * d[:, None]) * c[:, None]
    zb = np.full((n, 4), z0)
    zt = zb + h[:, None]
    v = np.stack([np.concatenate([px, px], 1), np.concatenate([py, py], 1),
                  np.concatenate([zb, zt], 1)], -1).reshape(-1, 3)
    base = (np.arange(n) * 8)[:, None, None]
    quad = np.array([[4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])
    f = (base + quad[None]).reshape(-1, 4)
    return v, f, 5


def year_value_node(nt, name="Year"):
    node = nt.nodes.new("ShaderNodeValue")
    node.name = node.label = name
    YEAR_NODES.append(node)
    return node


YEAR_NODES = []


def link(nt, a, b):
    nt.links.new(a, b)


def math_node(nt, op, a=None, b=None, clamp=False):
    m = nt.nodes.new("ShaderNodeMath")
    m.operation = op
    m.use_clamp = clamp
    for i, v in enumerate((a, b)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            m.inputs[i].default_value = v
        else:
            link(nt, v, m.inputs[i])
    return m.outputs[0]


_GROWTH_GROUP = None


def add_growth_modifier(ob):
    """Geometry Nodes: delete every face whose 'birth' is after the current year."""
    global _GROWTH_GROUP
    if _GROWTH_GROUP is None:
        ng = bpy.data.node_groups.new("Growth", "GeometryNodeTree")
        ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        gin = ng.nodes.new("NodeGroupInput")
        gout = ng.nodes.new("NodeGroupOutput")
        attr = ng.nodes.new("GeometryNodeInputNamedAttribute")
        attr.data_type = "FLOAT"
        attr.inputs["Name"].default_value = "birth"
        cmp = ng.nodes.new("FunctionNodeCompare")
        cmp.data_type = "FLOAT"
        cmp.operation = "GREATER_THAN"
        year = year_value_node(ng)
        delete = ng.nodes.new("GeometryNodeDeleteGeometry")
        delete.domain = "FACE"
        ng.links.new(attr.outputs["Attribute"], cmp.inputs["A"])
        ng.links.new(year.outputs[0], cmp.inputs["B"])
        ng.links.new(gin.outputs[0], delete.inputs["Geometry"])
        ng.links.new(cmp.outputs["Result"], delete.inputs["Selection"])
        ng.links.new(delete.outputs[0], gout.inputs[0])
        _GROWTH_GROUP = ng
    mod = ob.modifiers.new("Growth", "NODES")
    mod.node_group = _GROWTH_GROUP


# -------------------------------------------------------------- materials ---

def growth_material(name, roughness=0.8):
    """Shader: colour from attribute 'col', freshly built faces a bit brighter."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = roughness
    year = year_value_node(nt).outputs[0]
    birth = nt.nodes.new("ShaderNodeAttribute")
    birth.attribute_name = "birth"
    col = nt.nodes.new("ShaderNodeAttribute")
    col.attribute_name = "col"

    age = math_node(nt, "SUBTRACT", year, birth.outputs["Fac"])
    # freshly built blocks flash slightly brighter
    fresh = math_node(nt, "SUBTRACT", 1.0, math_node(nt, "DIVIDE", age, 25.0, clamp=True))
    gain = math_node(nt, "MULTIPLY_ADD", fresh, 0.35)
    gain.node.inputs[2].default_value = 1.0
    bright = nt.nodes.new("ShaderNodeMix")
    bright.data_type = "RGBA"
    bright.blend_type = "MULTIPLY"
    bright.inputs["Factor"].default_value = 1.0
    link(nt, col.outputs["Color"], bright.inputs["A"])
    comb = nt.nodes.new("ShaderNodeCombineColor")
    for k in ("Red", "Green", "Blue"):
        link(nt, gain, comb.inputs[k])
    link(nt, comb.outputs[0], bright.inputs["B"])
    link(nt, bright.outputs["Result"], bsdf.inputs["Base Color"])

    link(nt, bsdf.outputs[0], out.inputs["Surface"])
    return mat


def ground_material():
    mat = bpy.data.materials.new("Ground")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.95
    coord = nt.nodes.new("ShaderNodeTexCoord")

    # patchwork of fields
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = 2.4
    vor.inputs["Randomness"].default_value = 0.85
    link(nt, coord.outputs["Object"], vor.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.interpolation = "CONSTANT"
    palette = [(0.0, (0.22, 0.30, 0.10)), (0.2, (0.33, 0.36, 0.14)), (0.4, (0.42, 0.40, 0.20)),
               (0.6, (0.28, 0.33, 0.12)), (0.8, (0.48, 0.44, 0.26))]
    cr.elements[0].position, cr.elements[0].color = palette[0][0], (*palette[0][1], 1)
    cr.elements[1].position, cr.elements[1].color = palette[1][0], (*palette[1][1], 1)
    for p, c in palette[2:]:
        e = cr.elements.new(p)
        e.color = (*c, 1)
    link(nt, vor.outputs["Color"], ramp.inputs["Fac"])

    # forests
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 0.18
    noise.inputs["Detail"].default_value = 6
    link(nt, coord.outputs["Object"], noise.inputs["Vector"])
    fr = nt.nodes.new("ShaderNodeMapRange")
    fr.inputs["From Min"].default_value = 0.56
    fr.inputs["From Max"].default_value = 0.6
    link(nt, noise.outputs["Fac"], fr.inputs["Value"])
    forest = nt.nodes.new("ShaderNodeMix")
    forest.data_type = "RGBA"
    link(nt, fr.outputs[0], forest.inputs["Factor"])
    link(nt, ramp.outputs["Color"], forest.inputs["A"])
    forest.inputs["B"].default_value = (0.06, 0.12, 0.04, 1)

    # urban ground creeps in as the city arrives
    year = year_value_node(nt).outputs[0]
    ub = nt.nodes.new("ShaderNodeAttribute")
    ub.attribute_name = "birth"
    urb = math_node(nt, "DIVIDE", math_node(nt, "SUBTRACT", year, ub.outputs["Fac"]), 40.0, clamp=True)
    urban = nt.nodes.new("ShaderNodeMix")
    urban.data_type = "RGBA"
    link(nt, urb, urban.inputs["Factor"])
    link(nt, forest.outputs["Result"], urban.inputs["A"])
    urban.inputs["B"].default_value = (0.42, 0.36, 0.33, 1)
    link(nt, urban.outputs["Result"], bsdf.inputs["Base Color"])
    link(nt, bsdf.outputs[0], out.inputs["Surface"])
    return mat


def water_material():
    mat = bpy.data.materials.new("Water")
    mat.use_nodes = True
    b = mat.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.05, 0.07, 0.07, 1)
    b.inputs["Roughness"].default_value = 0.12
    return mat


# ----------------------------------------------------------------- scene ---

def build_ground():
    n = 300
    xs = np.linspace(-EXTENT, EXTENT, n)
    gx, gy = np.meshgrid(xs, xs)
    gz = 0.05 * smooth_noise(gx, gy, 11, 0.4)
    # carve the river valley flat
    gz -= 0.06 * np.exp(-((gy - river_y(gx)) / 1.5) ** 2)
    verts = np.stack([gx, gy, gz], -1).reshape(-1, 3)
    idx = np.arange(n * n).reshape(n, n)
    faces = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4)
    birth = birth_year(gx, gy) + 15
    birth = np.where(is_park(gx, gy), 1e5, birth).ravel()
    ob = mesh_object("Ground", verts, faces, point_attrs={"birth": birth})
    for p in ob.data.polygons:
        p.use_smooth = True
    ob.data.materials.append(ground_material())
    return ob


def build_river():
    xs = np.linspace(-EXTENT, EXTENT, 600)
    ys = river_y(xs)
    w = river_width(xs)
    left = np.stack([xs, ys + w, np.full_like(xs, 0.004)], -1)
    right = np.stack([xs, ys - w, np.full_like(xs, 0.004)], -1)
    verts = np.concatenate([left, right])
    n = len(xs)
    i = np.arange(n - 1)
    faces = np.stack([i, i + 1, n + i + 1, n + i], -1)
    ob = mesh_object("River", verts, faces)
    ob.data.materials.append(water_material())
    return ob


def in_river(x, y, margin=0.08):
    return np.abs(y - river_y(x)) < river_width(x) + margin


def build_arterials():
    """Old roads radiating from the core; they exist before the houses."""
    paths = []
    for k in range(10):
        a = k / 10 * 2 * math.pi + RNG.uniform(-0.2, 0.2)
        t = np.linspace(0.3, 34, 220)
        wob = np.cumsum(RNG.normal(0, 0.03, len(t)))
        ang = a + wob * 0.6
        x = CORE[0] + np.cos(ang) * t
        y = CORE[1] + np.sin(ang) * t
        keep = ~in_river(x, y, 0.0) & (np.abs(x) < EXTENT) & (np.abs(y) < EXTENT)
        # cut the road where it first meets the river (the bridge is separate)
        first = np.argmin(keep) if not keep.all() else len(keep)
        if first > 3:
            paths.append(np.stack([x[:first], y[:first]], -1))
    # a road along each bank
    xs = np.linspace(-EXTENT, EXTENT, 400)
    for side in (1, -1):
        paths.append(np.stack([xs, river_y(xs) + side * (river_width(xs) + 0.18)], -1))
    return paths


def strip_mesh(paths, width, z, birth_fn):
    verts, faces, births = [], [], []
    base = 0
    for p in paths:
        d = np.gradient(p, axis=0)
        nrm = np.stack([-d[:, 1], d[:, 0]], -1)
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
        l = p + nrm * width / 2
        r = p - nrm * width / 2
        n = len(p)
        verts.append(np.concatenate([np.c_[l, np.full(n, z)], np.c_[r, np.full(n, z)]]))
        i = np.arange(n - 1)
        faces.append(np.stack([i, i + 1, n + i + 1, n + i], -1) + base)
        mid = (p[:-1] + p[1:]) / 2
        births.append(birth_fn(mid[:, 0], mid[:, 1]))
        base += 2 * n
    return np.concatenate(verts), np.concatenate(faces), np.concatenate(births)


def dist_to_paths(x, y, paths):
    pts = np.concatenate([p[::2] for p in paths])
    out = np.full(len(x), 1e9)
    for i in range(0, len(x), 4000):
        dx = x[i:i + 4000, None] - pts[None, :, 0]
        dy = y[i:i + 4000, None] - pts[None, :, 1]
        out[i:i + 4000] = np.sqrt((dx * dx + dy * dy).min(1))
    return out


def build_city():
    paths = build_arterials()
    # roads appear well before the houses along them
    road_birth = lambda x, y: tl.birth_from_distance(effective_distance(x, y) * 0.55)
    v, f, b = strip_mesh(paths, 0.07, 0.012, road_birth)
    col = np.tile([0.62, 0.55, 0.46, 1.0], (len(f), 1))
    roads = mesh_object("Roads", v, f, face_attrs={"birth": b, "col": col})
    roads.data.materials.append(growth_material("Roads", 0.9))
    add_growth_modifier(roads)

    # candidate building plots on a jittered grid
    step = 0.15
    g = np.arange(-24, 24, step)
    px, py = np.meshgrid(g, g)
    px = px.ravel() + RNG.uniform(-0.03, 0.03, px.size)
    py = py.ravel() + RNG.uniform(-0.03, 0.03, py.size)

    birth = birth_year(px, py)
    keep = (birth < 3000) & ~in_river(px, py)
    px, py, birth = px[keep], py[keep], birth[keep]

    # local street grid with a slowly turning orientation
    theta = 0.5 * smooth_noise(px, py, 3, 0.12)
    u = px * np.cos(theta) + py * np.sin(theta)
    w_ = -px * np.sin(theta) + py * np.cos(theta)
    block = 0.75 + 0.25 * (smooth_noise(px, py, 4, 0.2) > 0)
    street = ((u / block) % 1 < 0.17) | ((w_ / block) % 1 < 0.17)
    near_road = dist_to_paths(px, py, paths) < 0.1
    d = effective_distance(px, py)
    park = is_park(px, py)
    density = np.clip(1.1 - d / 30, 0.55, 1.0)
    keep = ~street & ~near_road & ~park & (RNG.random(px.size) < density)
    px, py, birth, theta, d = px[keep], py[keep], birth[keep], theta[keep], d[keep]
    n = len(px)

    w = RNG.uniform(0.07, 0.12, n)
    dp = RNG.uniform(0.07, 0.12, n)
    h = RNG.uniform(0.03, 0.07, n) * np.where(birth > 1850, 1.5, 1.0)

    # colour by era: terracotta, brick, then grey-pink suburbs
    old = np.array([0.55, 0.26, 0.14])
    brick = np.array([0.42, 0.24, 0.19])
    modern = np.array([0.55, 0.47, 0.45])
    t1 = np.clip((birth - 1600) / 250, 0, 1)[:, None]
    t2 = np.clip((birth - 1880) / 80, 0, 1)[:, None]
    c = old * (1 - t1) + brick * t1
    c = c * (1 - t2) + modern * t2
    c *= RNG.uniform(0.8, 1.15, (n, 1))

    # modern towers in the core
    m = 140
    ta = RNG.uniform(0, 2 * math.pi, m)
    tr = np.abs(RNG.normal(0, 1.3, m))
    tx = CORE[0] + np.cos(ta) * tr * 1.4
    ty = CORE[1] + np.sin(ta) * tr
    ok = ~in_river(tx, ty, 0.15)
    tx, ty = tx[ok], ty[ok]
    m = len(tx)
    px = np.concatenate([px, tx])
    py = np.concatenate([py, ty])
    w = np.concatenate([w, RNG.uniform(0.12, 0.22, m)])
    dp = np.concatenate([dp, RNG.uniform(0.12, 0.22, m)])
    h = np.concatenate([h, RNG.gamma(2.0, 0.25, m) + 0.2])
    theta = np.concatenate([theta, RNG.uniform(0, 0.5, m)])
    birth = np.concatenate([birth, RNG.uniform(1965, 2022, m)])
    c = np.concatenate([c, np.tile([0.30, 0.38, 0.46], (m, 1)) * RNG.uniform(0.8, 1.2, (m, 1))])

    v, f, per = boxes(px, py, w, dp, h, theta)
    shade = np.tile([1.0, 0.7, 0.85, 0.75, 0.9], len(px))[:, None]
    fcol = np.repeat(c, per, axis=0) * shade
    fcol = np.c_[fcol, np.ones(len(fcol))]
    ob = mesh_object("Buildings", v, f, face_attrs={"birth": np.repeat(birth, per), "col": fcol})
    ob.data.materials.append(growth_material("Buildings"))
    add_growth_modifier(ob)
    print(f"buildings: {len(px)}")

    build_landmarks()
    return ob


def build_landmarks():
    parts = []  # (verts, faces, birth, colour)

    # city wall: a ring of segments around the old core
    k = 120
    a = np.linspace(0, 2 * math.pi, k, endpoint=False)
    rx, ry = 1.45, 0.95
    wx = CORE[0] + np.cos(a) * rx
    wy = CORE[1] + 0.25 + np.sin(a) * ry
    ok = ~in_river(wx, wy, 0.05)
    wx, wy, a = wx[ok], wy[ok], a[ok]
    seg = 2 * math.pi * 1.2 / k
    v, f, per = boxes(wx, wy, np.full(len(wx), seg * 1.1), np.full(len(wx), 0.04),
                      np.full(len(wx), 0.07), a + math.pi / 2)
    parts.append((v, f, np.full(len(f), 150.0), (0.62, 0.55, 0.45)))

    # bridge
    bx = np.array([CORE[0] + 0.3])
    ry0 = river_y(bx)
    v, f, per = boxes(bx, ry0, np.array([0.08]), np.array([2 * river_width(bx[0]) + 0.5]),
                      np.array([0.03]), np.array([0.05]))
    parts.append((v, f, np.full(len(f), 120.0), (0.55, 0.40, 0.30)))

    # amphitheatre (ring of blocks) and a forum square
    k = 40
    a = np.linspace(0, 2 * math.pi, k, endpoint=False)
    ax, ay = CORE[0] - 0.9, CORE[1] + 0.6
    v, f, per = boxes(ax + np.cos(a) * 0.18, ay + np.sin(a) * 0.13, np.full(k, 0.05),
                      np.full(k, 0.06), np.full(k, 0.06), a)
    parts.append((v, f, np.full(len(f), 110.0), (0.75, 0.68, 0.58)))
    v, f, per = boxes(np.array([CORE[0] + 0.7]), np.array([CORE[1] + 0.5]), np.array([0.32]),
                      np.array([0.26]), np.array([0.05]), np.array([0.0]))
    parts.append((v, f, np.full(len(f), 140.0), (0.78, 0.72, 0.62)))

    # cathedral
    v, f, per = boxes(np.array([CORE[0] + 0.1]), np.array([CORE[1] + 0.35]), np.array([0.34]),
                      np.array([0.12]), np.array([0.16]), np.array([0.0]))
    parts.append((v, f, np.full(len(f), 1100.0), (0.72, 0.68, 0.60)))

    verts, faces, births, cols, off = [], [], [], [], 0
    for v, f, b, c in parts:
        verts.append(v)
        faces.append(f + off)
        births.append(b)
        cols.append(np.tile([*c, 1.0], (len(f), 1)))
        off += len(v)
    ob = mesh_object("Landmarks", np.concatenate(verts), np.concatenate(faces),
                     face_attrs={"birth": np.concatenate(births), "col": np.concatenate(cols)})
    ob.data.materials.append(growth_material("Landmarks"))
    add_growth_modifier(ob)


def build_fire():
    """Orange glow over the old town during the great fire."""
    mat = bpy.data.materials.new("Fire")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (1.0, 0.35, 0.05, 1)
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    year = year_value_node(nt).outputs[0]
    # 1694 +/- ~6 years, peak in the middle
    dist = math_node(nt, "ABSOLUTE", math_node(nt, "SUBTRACT", year, 1694.0))
    amt = math_node(nt, "SUBTRACT", 1.0, math_node(nt, "DIVIDE", dist, 7.0, clamp=True))
    # radial falloff
    coord = nt.nodes.new("ShaderNodeTexCoord")
    grad = nt.nodes.new("ShaderNodeTexGradient")
    grad.gradient_type = "SPHERICAL"
    link(nt, coord.outputs["Object"], grad.inputs["Vector"])
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 6
    link(nt, coord.outputs["Object"], noise.inputs["Vector"])
    shape = math_node(nt, "MULTIPLY", grad.outputs["Fac"], noise.outputs["Fac"])
    fac = math_node(nt, "MULTIPLY", math_node(nt, "MULTIPLY", shape, amt), 2.2, clamp=True)
    em.inputs["Strength"].default_value = 6.0
    link(nt, fac, mix.inputs["Fac"])
    link(nt, tr.outputs[0], mix.inputs[1])
    link(nt, em.outputs[0], mix.inputs[2])
    link(nt, mix.outputs[0], out.inputs["Surface"])

    bpy.ops.mesh.primitive_plane_add(size=2, location=(CORE[0] + 0.2, CORE[1] + 0.3, 0.09))
    ob = bpy.context.active_object
    ob.name = "Fire"
    ob.scale = (1.5, 1.0, 1.0)
    ob.data.materials.append(mat)
    ob.visible_shadow = False


def setup_world_and_light():
    sc = bpy.context.scene
    world = bpy.data.worlds.new("World")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.62, 0.68, 0.74, 1)
    bg.inputs["Strength"].default_value = 0.55

    sun = bpy.data.lights.new("Sun", "SUN")
    sun.energy = 3.6
    sun.angle = math.radians(3)
    sun.color = (1.0, 0.94, 0.85)
    ob = bpy.data.objects.new("Sun", sun)
    ob.rotation_euler = (math.radians(52), 0, math.radians(-35))
    sc.collection.objects.link(ob)


def setup_camera():
    sc = bpy.context.scene
    cam = bpy.data.cameras.new("Camera")
    cam.lens = 32
    cam.clip_end = 400
    co = bpy.data.objects.new("Camera", cam)
    sc.collection.objects.link(co)
    sc.camera = co

    target = bpy.data.objects.new("Target", None)
    sc.collection.objects.link(target)
    tc = co.constraints.new("TRACK_TO")
    tc.target = target
    tc.track_axis = "TRACK_NEGATIVE_Z"
    tc.up_axis = "UP_Y"

    # pull back as the city grows; framing follows the built-up radius
    frames = list(range(tl.FRAME_START, tl.FRAME_END + 1, 6))
    for fr in frames:
        yr = tl.year_at(fr)
        r = float(np.interp(yr, tl.YEARS, tl.RADII))
        if fr < tl.ANIM_START:  # title: slow glide over the empty valley
            k = (fr - tl.FRAME_START) / tl.TITLE_FRAMES
            r = 1.2 + 0.3 * (1 - k)
        dist = 2.2 + r * 1.45
        height = 1.4 + r * 1.2
        orbit = math.radians(-12 + 18 * (fr - tl.FRAME_START) / (tl.FRAME_END - tl.FRAME_START))
        co.location = (CORE[0] + math.sin(orbit) * dist, CORE[1] - math.cos(orbit) * dist, height)
        co.keyframe_insert("location", frame=fr)
        target.location = (CORE[0], CORE[1] + 0.25 + r * 0.15, 0)
        target.keyframe_insert("location", frame=fr)


def animate_year():
    kf = tl.key_frames()
    for node in YEAR_NODES:
        sock = node.outputs[0]
        sock.default_value = tl.YEARS[0] - 30
        sock.keyframe_insert("default_value", frame=tl.FRAME_START)
        for fr, yr in zip(kf, tl.YEARS):
            sock.default_value = yr
            sock.keyframe_insert("default_value", frame=fr)
    for act in bpy.data.actions:
        for fc in iter_fcurves(act):
            for p in fc.keyframe_points:
                p.interpolation = "LINEAR"


def iter_fcurves(act):
    if hasattr(act, "fcurves"):
        yield from act.fcurves
        return
    for layer in act.layers:  # slotted actions (Blender 4.4+)
        for strip in layer.strips:
            for bag in strip.channelbags:
                yield from bag.fcurves


def setup_render(args):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = args.samples
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 4
    sc.cycles.transparent_max_bounces = 8
    sc.render.resolution_x, sc.render.resolution_y = args.res
    sc.render.fps = tl.FPS
    sc.frame_start, sc.frame_end = args.frames or (tl.FRAME_START, tl.FRAME_END)
    sc.render.image_settings.file_format = "PNG"
    sc.render.filepath = os.path.join(args.out, "f_")
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, nargs=2, default=(960, 540))
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--frames", type=int, nargs=2)
    ap.add_argument("--still", type=int, nargs="*", help="render only these frames")
    ap.add_argument("--save")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    setup_world_and_light()
    build_ground()
    build_river()
    build_city()
    build_fire()
    setup_camera()
    animate_year()
    setup_render(args)
    if args.save:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.save))

    sc = bpy.context.scene
    if args.still is not None:
        for fr in args.still:
            sc.frame_set(fr)
            sc.render.filepath = os.path.join(args.out, f"still_{fr:04d}.png")
            bpy.ops.render.render(write_still=True)
    else:
        bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
