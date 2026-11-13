"""End-to-end N2 orchestration over preserved N1 spatial feature maps."""
from __future__ import annotations
import time
import numpy as np
from .config import N2Config
from .contracts import N2Result, adapt_n1_result
from . import stages

class N2Processor:
    """Stateless N2 processor. One instance can safely process independent N1 results."""
    def __init__(self, config: N2Config | None = None) -> None:
        self.config=config or N2Config()

    def process(self, n1_result) -> N2Result:
        source=adapt_n1_result(n1_result)
        cfg=self.config
        maps=source.feature_maps
        started=time.perf_counter()
        records={}
        def record(name,value):
            arr=value if isinstance(value,np.ndarray) else None
            if arr is not None:
                if not np.isfinite(arr).all(): raise FloatingPointError(f"{name} produced non-finite values")
                records[name]={"shape":list(arr.shape),"dtype":str(arr.dtype),
                               "min":float(np.min(arr)),"max":float(np.max(arr)),
                               "mean":float(np.mean(arr))}
            return value

        # N2-01: Retinotopic coordinate/eccentricity field.
        coordinates=stages.n2_01_retinotopic_coordinates(maps,cfg)
        # N2-02: Multi-scale receptive-field population.
        multiscale=record("N2-02.multiscale",stages.n2_02_multiscale_receptive_fields(maps,cfg))
        # N2-03: Contour integration.
        contours=stages.n2_03_contour_integration(multiscale,cfg)
        contours["orientation_maps"]=np.max(multiscale,axis=0)
        record("N2-03.contours",contours["integrated"])
        # N2-04: Boundary cues.
        boundary=stages.n2_04_boundary_cues(contours,cfg)
        record("N2-04.boundary",boundary["boundary"])
        # N2-05: Contour completion.
        completed=record("N2-05.completed",stages.n2_05_contour_completion(contours,boundary,cfg))
        # N2-06: Border ownership.
        ownership=record("N2-06.ownership",stages.n2_06_border_ownership(contours,boundary,cfg))
        # N2-07: Figure-ground organization.
        figure_ground=stages.n2_07_figure_ground(boundary,ownership,cfg)
        record("N2-07.figure_probability",figure_ground["figure_probability"])
        # N2-08: Junction and curvature.
        junctions=stages.n2_08_junctions_and_curvature(contours,cfg)
        record("N2-08.curvature",junctions["curvature"])
        # N2-09: Shape fragments.
        fragments=stages.n2_09_shape_fragments(completed,junctions,cfg)
        # N2-10: Structural shape graph.
        graph=stages.n2_10_structural_shape_graph(fragments,cfg)
        # N2-11: Crowding estimate.
        crowding=stages.n2_11_crowding(contours,coordinates,cfg)
        record("N2-11.crowding",crowding["crowding"])
        # N2-12: Position/scale tolerance.
        invariance=stages.n2_12_position_scale_tolerance(multiscale,cfg)
        record("N2-12.invariant_maps",invariance["invariant_maps"])
        # N2-13: Ventral form population.
        ventral=stages.n2_13_ventral_population(multiscale,junctions,fragments,cfg)
        # N2-14: Dorsal spatial/motion population. Temporal motion requires frame sequences.
        dorsal=stages.n2_14_dorsal_population(np.max(maps,axis=0),cfg)
        # N2-15: Bounded recurrent refinement, without semantic/lexical priors.
        recurrent=stages.n2_15_recurrent_refinement(completed,figure_ground["figure_probability"],cfg)
        record("N2-15.refined",recurrent["refined_activity"])
        # N2-16: Preserve and integrate distinct channels.
        parts={"contour":contours["integrated"],"boundary":boundary["boundary"],
               "junction":junctions["junction_strength"],"curvature":junctions["curvature"],
               "shape_fragments":fragments["labels"].astype(np.float32),
               "position_eccentricity":coordinates["eccentricity"],
               "crowding":crowding["crowding"],"invariant":invariance["invariant_maps"],
               "ventral":ventral["ventral_maps"],"dorsal":dorsal["dorsal_spatial_energy"],
               "recurrent":recurrent["refined_activity"],
               "figure_probability":figure_ground["figure_probability"],
               "border_ownership_left":ownership[0],"border_ownership_right":ownership[1]}
        integrated=stages.n2_16_representation_integration(parts)
        # N2-17: Structured output.
        structured=stages.n2_17_structured_visual_output(integrated,graph,cfg)
        representations={**parts,"multiscale":multiscale,"completed_contours":completed,
                         "critical_spacing":crowding["critical_spacing"],
                         "motion_energy":dorsal["motion_energy"]}
        for key,value in representations.items(): record("representation."+key,value)
        diagnostics={"valid":True,"stage_count":17,"records":records,
                     "elapsed_seconds":time.perf_counter()-started,
                     "source_n1_diagnostics":source.source_diagnostics}
        metadata={"pipeline":"ORE N2 Biological Visual Processing","version":"0.1.0",
                  "input_shape":list(maps.shape),"spatial_shape":list(maps.shape[-2:]),
                  "orientation_count":int(maps.shape[0]),"steps":[f"N2-{i:02d}" for i in range(1,18)],
                  "motion_status":"unavailable for single-frame input; motion channel is explicitly zero",
                  "biological_status":"computational approximation; not a validated physiological model"}
        return N2Result(structured,representations,diagnostics,metadata)
