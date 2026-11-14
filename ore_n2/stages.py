"""The 17 explicit N2 operations. All functions are deterministic numerical approximations."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter, sobel, maximum_filter, label
from .config import N2Config

def _unit(v: np.ndarray, axis: int = 0, eps: float = 1e-8) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), eps)

def _gradient(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return sobel(a, axis=1, mode="reflect") / 8.0, sobel(a, axis=0, mode="reflect") / 8.0

def n2_01_retinotopic_coordinates(activity: np.ndarray, cfg: N2Config) -> dict:
    """01. Preserve pixel-to-feature-field coordinates and eccentricity."""
    h, w = activity.shape[-2:]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2, (w - 1) / 2
    radius = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    e0 = max(min(h, w) * 0.05, cfg.epsilon)
    eccentricity = np.log1p(radius / e0)
    return {"x": xx, "y": yy, "eccentricity": eccentricity.astype(np.float32),
            "radial_distance": radius.astype(np.float32)}

def n2_02_multiscale_receptive_fields(activity: np.ndarray, cfg: N2Config) -> np.ndarray:
    """02. Build a scale-space population R_s = G_s * F1."""
    return np.stack([gaussian_filter(activity, sigma=(0, s, s), mode="reflect")
                     for s in cfg.scales]).astype(np.float32)

def n2_03_contour_integration(multiscale: np.ndarray, cfg: N2Config) -> dict:
    """03. Link nearby oriented responses using proximity and orientation agreement."""
    base = multiscale[0]
    orient_count, h, w = base.shape
    strength = np.max(base, axis=0)
    gy, gx = _gradient(strength)
    tangent = np.arctan2(gy, gx) + np.pi/2
    yy, xx = np.mgrid[0:h, 0:w]
    accum = np.zeros((h,w), np.float32)
    for dy in range(-cfg.contour_distance, cfg.contour_distance+1):
        for dx in range(-cfg.contour_distance, cfg.contour_distance+1):
            if dy == 0 and dx == 0: continue
            shifted = np.roll(strength, (dy,dx), axis=(0,1))
            distance = np.hypot(dy,dx)
            direction = np.arctan2(dy,dx)
            alignment = np.abs(np.cos(tangent-direction)) ** cfg.contour_alignment_power
            accum += shifted * alignment / distance
    integrated = strength + gaussian_filter(accum, cfg.contour_sigma, mode="reflect") / max(1, (2*cfg.contour_distance+1)**2)
    return {"strength": strength.astype(np.float32), "tangent": tangent.astype(np.float32),
            "integrated": integrated.astype(np.float32)}

def n2_04_boundary_cues(contours: dict, cfg: N2Config) -> dict:
    """04. Estimate candidate boundaries from contour strength and gradient discontinuity."""
    s = contours["strength"]
    gy, gx = _gradient(s)
    grad = np.hypot(gx,gy)
    boundary = np.maximum(s, grad)
    return {"boundary": boundary.astype(np.float32), "boundary_gradient": grad.astype(np.float32)}

def n2_05_contour_completion(contours: dict, boundary: dict, cfg: N2Config) -> np.ndarray:
    """05. Iteratively bridge weak gaps without changing observed contour evidence."""
    observed = contours["strength"]
    state = observed.copy()
    for _ in range(cfg.completion_iterations):
        support = gaussian_filter(state, cfg.contour_sigma, mode="reflect")
        state = np.maximum(observed, cfg.completion_decay * support * (1.0-boundary["boundary"]/(boundary["boundary"].max()+cfg.epsilon)))
    return state.astype(np.float32)

def n2_06_border_ownership(contours: dict, boundary: dict, cfg: N2Config) -> np.ndarray:
    """06. Softly assign each boundary's ownership to its two sides."""
    s = boundary["boundary"]
    gy,gx = _gradient(s)
    norm = np.hypot(gx,gy)+cfg.epsilon
    signed = (gx*(np.arange(s.shape[1])[None,:]-s.shape[1]/2) +
              gy*(np.arange(s.shape[0])[:,None]-s.shape[0]/2))/norm
    p = 1.0/(1.0+np.exp(-np.clip(signed/cfg.ownership_temperature,-30,30)))
    return np.stack((p,1-p),axis=0).astype(np.float32)

def n2_07_figure_ground(boundary: dict, ownership: np.ndarray, cfg: N2Config) -> dict:
    """07. Approximate figure-ground competition with local smoothness."""
    fg = gaussian_filter(ownership[0], 1.0, mode="reflect")
    bg = gaussian_filter(ownership[1], 1.0, mode="reflect")
    logits = fg-bg-cfg.figure_ground_smoothness*boundary["boundary"]
    prob = 1.0/(1.0+np.exp(-np.clip(logits,-30,30)))
    return {"figure_probability": prob.astype(np.float32),
            "ground_probability": (1-prob).astype(np.float32),
            "figure_ground_energy": (np.abs(prob-0.5)+boundary["boundary"]).astype(np.float32)}

def n2_08_junctions_and_curvature(contours: dict, cfg: N2Config) -> dict:
    """08. Detect multi-orientation junctions and estimate curvature kappa=d(theta)/ds."""
    # Orientation diversity is a junction cue; curvature is the spatial change in tangent.
    responses = contours["orientation_maps"]
    total = np.sum(responses,axis=0)+cfg.epsilon
    probs = responses/total[None,:,:]
    entropy = -np.sum(probs*np.log(probs+cfg.epsilon),axis=0)
    theta = contours["tangent"]
    ty,tx = _gradient(theta)
    curvature = np.hypot(tx,ty)
    return {"junction_strength": entropy.astype(np.float32),
            "curvature": gaussian_filter(curvature,cfg.curvature_sigma,mode="reflect").astype(np.float32)}

def n2_09_shape_fragments(completed: np.ndarray, junctions: dict, cfg: N2Config) -> dict:
    """09. Segment contour fragments using adaptive threshold and connected components."""
    threshold = max(float(np.quantile(completed,0.70)),cfg.fragment_threshold)
    mask = completed >= threshold
    labels,count = label(mask)
    fragments=[]
    for idx in range(1,count+1):
        ys,xs=np.where(labels==idx)
        if ys.size:
            fragments.append({"id":int(idx),"area":int(ys.size),
                              "centroid":[float(xs.mean()),float(ys.mean())],
                              "bbox":[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)],
                              "mean_junction":float(junctions["junction_strength"][ys,xs].mean())})
    return {"labels":labels.astype(np.int32),"fragments":fragments}

def n2_10_structural_shape_graph(fragments: dict, cfg: N2Config) -> dict:
    """10. Construct a spatial graph; report Euler characteristic beta0-beta1."""
    nodes=fragments["fragments"]
    edges=[]
    for i,a in enumerate(nodes):
        ax,ay=a["centroid"]
        for b in nodes[i+1:]:
            bx,by=b["centroid"]
            d=float(np.hypot(ax-bx,ay-by))
            if d<=cfg.graph_link_distance:
                edges.append([a["id"],b["id"],d])
    # beta0 is component count in this fragment graph; beta1=E-V+beta0.
    parent={n["id"]:n["id"] for n in nodes}
    def root(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    for a,b,_ in edges:
        ra,rb=root(a),root(b)
        if ra!=rb: parent[ra]=rb
    beta0=len({root(k) for k in parent})
    beta1=max(0,len(edges)-len(nodes)+beta0)
    return {"nodes":nodes,"edges":edges,"beta0":beta0,"beta1":beta1,
            "euler_characteristic":beta0-beta1}

def n2_11_crowding(contours: dict, coordinates: dict, cfg: N2Config) -> dict:
    """11. Model eccentricity-dependent critical spacing d*=bE+w and local crowding."""
    e=coordinates["eccentricity"]
    critical=cfg.crowding_base+cfg.crowding_eccentricity_slope*e
    local=maximum_filter(contours["strength"],size=5,mode="reflect")
    crowd=np.maximum(local-contours["strength"],0.0)/(local+cfg.epsilon)
    adjusted=contours["strength"]+cfg.crowding_strength*crowd
    return {"critical_spacing":critical.astype(np.float32),"crowding":crowd.astype(np.float32),
            "crowding_adjusted_activity":adjusted.astype(np.float32)}

def n2_12_position_scale_tolerance(multiscale: np.ndarray, cfg: N2Config) -> dict:
    """12. Soft max over scale and local position for tolerant feature evidence."""
    t=cfg.invariance_temperature
    scaled=multiscale/t
    m=np.max(scaled,axis=0,keepdims=True)
    lse=(m+np.log(np.mean(np.exp(scaled-m),axis=0,keepdims=True)+cfg.epsilon))*t
    pooled=np.max(lse,axis=(0,1))
    scale_evidence=np.max(multiscale,axis=(0,1))
    return {"invariant_maps":pooled.astype(np.float32),
            "scale_evidence":scale_evidence.astype(np.float32)}

def n2_13_ventral_population(multiscale: np.ndarray, junctions: dict, fragments: dict, cfg: N2Config) -> dict:
    """13. Assemble a shape/form-oriented ventral population without lexical labels."""
    shape=np.max(multiscale,axis=0)
    descriptor=np.stack((shape.mean(axis=0),shape.std(axis=0),
                         junctions["junction_strength"],junctions["curvature"]),axis=0)
    return {"ventral_maps":descriptor.astype(np.float32),
            "ventral_vector":descriptor.mean(axis=(1,2)).astype(np.float32)}

def n2_14_dorsal_population(activity: np.ndarray, cfg: N2Config) -> dict:
    """14. Compute motion-energy channels when temporal input is available."""
    # A single still frame has no temporal derivative; report zero motion honestly.
    gy,gx=_gradient(activity)
    spatial=np.hypot(gx,gy)
    return {"dorsal_spatial_energy":spatial.astype(np.float32),
            "motion_energy":np.zeros_like(activity,dtype=np.float32),
            "motion_available":np.asarray(False)}

def n2_15_recurrent_refinement(feedforward: np.ndarray, feedback: np.ndarray, cfg: N2Config) -> dict:
    """15. Bounded local recurrence A[t+1]=phi(Wff F + Wlat A[t] + Wfb H - b)."""
    target=np.maximum(feedforward+cfg.recurrent_feedback*feedback,0.0)
    state=target.copy()
    deltas=[]
    for _ in range(cfg.recurrent_iterations):
        nxt=np.maximum((1-cfg.recurrent_lateral)*target+
                       cfg.recurrent_lateral*gaussian_filter(state,1.0,mode="reflect"),0.0)
        delta=float(np.linalg.norm(nxt-state)/(np.linalg.norm(state)+cfg.epsilon))
        deltas.append(delta); state=nxt
        if delta<cfg.recurrent_tolerance: break
    return {"refined_activity":state.astype(np.float32),"iterations":len(deltas),
            "relative_deltas":deltas,"converged":bool(deltas and deltas[-1]<cfg.recurrent_tolerance)}

def n2_16_representation_integration(parts: dict[str,np.ndarray]) -> dict:
    """16. Integrate parallel channels while preserving each constituent representation."""
    preserved={k:np.asarray(v).copy() for k,v in parts.items()}
    summaries={k:float(np.mean(v)) for k,v in preserved.items()}
    return {"channels":preserved,"channel_summaries":summaries}

def n2_17_structured_visual_output(integrated: dict, graph: dict, cfg: N2Config) -> dict:
    """17. Emit a structured, traceable visual description—not a grapheme or word decision."""
    channels=integrated["channels"]
    return {"schema":"ore.n2.structured_visual_output.v1",
            "visual_features":{k:{"shape":list(v.shape),"mean":float(np.mean(v)),
                                  "max":float(np.max(v)),"min":float(np.min(v))}
                              for k,v in channels.items()},
            "shape_graph":{"node_count":len(graph["nodes"]),"edge_count":len(graph["edges"]),
                           "beta0":graph["beta0"],"beta1":graph["beta1"],
                           "euler_characteristic":graph["euler_characteristic"]},
            "interpretation_scope":"visual structure only; no grapheme identity or lexical semantics"}
