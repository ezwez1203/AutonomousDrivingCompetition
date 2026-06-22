"""Generate an OpenDRIVE (.xodr) string from TrackSpec.

Build one closed-loop road from the reference line, then extend two lanes to the right. Lanes -1 and -2 both drive in the +s direction. CARLA road marks are disabled because visible markings are spawned as runtime mesh actors.
"""
from __future__ import annotations

from .track_spec import TrackSpec


def _f(v: float) -> str:
    """Format OpenDRIVE numbers with enough precision."""
    return repr(float(v))


def _geometry_records(spec: TrackSpec) -> tuple[str, float]:
    """Return planView <geometry> records and total length."""
    rows = []
    total = 0.0
    for geom in spec.geometries():
        attrs = (f'<geometry s="{_f(geom.s)}" x="{_f(geom.x)}" y="{_f(geom.y)}" '
                 f'hdg="{_f(geom.heading)}" length="{_f(geom.length)}">')
        if abs(geom.curvature) < 1e-12:
            inner = "<line/>"
        else:
            inner = f'<arc curvature="{_f(geom.curvature)}"/>'
        rows.append(f"            {attrs}\n                {inner}\n            </geometry>")
        total = geom.s + geom.length
    return "\n".join(rows), total


def _lane_offset(spec: TrackSpec) -> str:
    """Offset for keeping the reference line on the inner road edge.

    With right-side negative lane IDs, the reference line is already the inner edge, so no extra offset is needed.
    """
    return ('<laneOffset s="0.0" a="0.0" b="0.0" c="0.0" d="0.0"/>')


def _width(w: float) -> str:
    return f'<width sOffset="0.0" a="{_f(w)}" b="0.0" c="0.0" d="0.0"/>'


def _roadmark(spec: TrackSpec, mark_type: str) -> str:
    del mark_type
    width = spec.mm(spec.cfg["dimensions"].get("lane_mark_mm", 50))
    return (f'<roadMark sOffset="0.0" type="none" material="standard" '
            f'color="white" width="{_f(width)}" laneChange="both"/>')

def _signals(spec: TrackSpec) -> str:
    elements = spec.cfg.get("elements", {})
    crosswalk = elements.get("crosswalk")
    if not crosswalk:
        return ""
    s_m = spec.mm(float(crosswalk["s"])) % spec.total_length()
    # Reference line is the inner road edge; lanes expand to the right side (negative t).
    # Place the signal just outside the outer edge and keep the pole on the road surface.
    t = -(spec.road_width + 0.6)
    return f"""
        <signals>
            <signal name="Signal_3Light_Post01" id="1001" s="{_f(s_m)}" t="{_f(t)}"
                    zOffset="0.0" hOffset="0.0" roll="0.0" pitch="0.0" orientation="-"
                    dynamic="yes" country="OpenDRIVE" type="1000001" subtype="-1"
                    value="-1.0" text="" height="1.16" width="0.53"/>
        </signals>"""


def _crosswalk_objects(spec: TrackSpec) -> str:
    elements = spec.cfg.get("elements", {})
    crosswalk = elements.get("crosswalk")
    if not crosswalk:
        return ""

    dims = spec.cfg["dimensions"]
    s_center = spec.mm(float(crosswalk["s"])) % spec.total_length()
    crosswalk_length = spec.mm(dims.get("crosswalk_mm", [1000, 100])[1])
    stripe_count = int(spec.cfg.get("lanes", {}).get("crosswalk_stripes", 4))
    stripe_count = max(1, stripe_count)
    stripe_length = min(0.18, crosswalk_length / (stripe_count * 1.5))
    if stripe_count == 1:
        gap = 0.0
    else:
        gap = max(0.05, (crosswalk_length - stripe_count * stripe_length) / (stripe_count - 1))
    width = spec.road_width + 0.6
    t_center = -spec.road_width / 2.0
    start_s = s_center - crosswalk_length / 2.0

    rows = ["        <objects>"]
    for idx in range(stripe_count):
        s_m = (start_s + idx * (stripe_length + gap) + stripe_length / 2.0) % spec.total_length()
        rows.append(
            f'            <object id="{2000 + idx}" name="crosswalk_stripe_{idx + 1}" '
            f's="{_f(s_m)}" t="{_f(t_center)}" zOffset="0.03" '
            f'validLength="0.0" orientation="none" type="crosswalk" subtype="zebra" '
            f'dynamic="no" hdg="0.0" pitch="0.0" roll="0.0" '
            f'height="0.02" width="{_f(width)}" length="{_f(stripe_length)}"/>'
        )
    rows.append("        </objects>")
    return "\n".join(rows)


def generate_xodr(spec: TrackSpec) -> str:
    geoms, length = _geometry_records(spec)
    lw = spec.lane_width
    marks = spec.cfg.get("lanes", {}).get("marks", {})
    inner_mark = marks.get("inner", "solid")
    divider_mark = marks.get("divider", "broken")
    outer_mark = marks.get("outer", "solid")

    # lane: center(0) + right(-1 inner/1lane, -2 outer/2lane)
    # roadMark positions: center=inner solid, lane -1=divider dashed, lane -2=outer solid
    center_lane = f"""                    <center>
                        <lane id="0" type="driving" level="false">
                            {_roadmark(spec, inner_mark)}
                        </lane>
                    </center>"""
    right_lanes = f"""                    <right>
                        <lane id="-1" type="driving" level="false">
                            <link/>
                            {_width(lw)}
                            {_roadmark(spec, divider_mark)}
                        </lane>
                        <lane id="-2" type="driving" level="false">
                            <link/>
                            {_width(lw)}
                            {_roadmark(spec, outer_mark)}
                        </lane>
                    </right>"""

    road = f"""    <road name="{spec.name}" length="{_f(length)}" id="1" junction="-1">
        <link>
            <predecessor elementType="road" elementId="1" contactPoint="end"/>
            <successor elementType="road" elementId="1" contactPoint="start"/>
        </link>
        <type s="0.0" type="rural">
            <speed max="30" unit="mph"/>
        </type>
        <planView>
{geoms}
        </planView>
        <elevationProfile>
            <elevation s="0.0" a="0.0" b="0.0" c="0.0" d="0.0"/>
        </elevationProfile>
        <lateralProfile/>
        <lanes>
            <laneSection s="0.0">
{center_lane}
{right_lanes}
            </laneSection>
        </lanes>
{_crosswalk_objects(spec)}
{_signals(spec)}
    </road>"""

    xodr = f"""<?xml version="1.0" encoding="UTF-8"?>
<OpenDRIVE>
    <header revMajor="1" revMinor="4" name="{spec.name}" version="1.00" date="2026-06-08"
            north="0.0" south="0.0" east="0.0" west="0.0" vendor="carla_autodrive">
        <geoReference><![CDATA[+proj=tmerc +lat_0=0 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs]]></geoReference>
    </header>
{road}
</OpenDRIVE>
"""
    return xodr
