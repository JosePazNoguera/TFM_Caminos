"""Create a corridor-focused zoning proposal from administrative areas and roads."""

from __future__ import annotations

import argparse
import math
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from heapq import heappop, heappush
from pathlib import Path
from typing import Iterable

import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parents[1]
GIS_DIR = ROOT / "Data" / ".gis"
DEFAULT_CORRIDOR = GIS_DIR / "N-211 p.k. 0 - p.k. 10.kmz"
DEFAULT_ROADS = GIS_DIR / "Mapa de Tráfico" / "si_tramotrafimd2025_l.shp"
DEFAULT_DISTRICTS = GIS_DIR / "zonificacion_distritos.shp"
DEFAULT_MUNICIPALITIES = GIS_DIR / "zonificacion_municipios.shp"
MAX_NETWORK_DISTANCE_KM = 1_100.0
ROAD_SNAP_DISTANCE_M = 2_000.0
MAX_ROAD_SEGMENT_M = 1_000.0
DEFAULT_MAX_ZONES = 19
DEFAULT_RADIAL_SECTORS = 8
NON_PENINSULAR_PROVINCES = {"07", "35", "38", "51", "52"}
COMPASS_SECTORS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def _kml_lines(path: Path) -> gpd.GeoDataFrame:
    """Read KML/KMZ LineString features without depending on GDAL KML support."""
    if path.suffix.lower() == ".kmz":
        with zipfile.ZipFile(path) as archive:
            kml_names = [name for name in archive.namelist() if name.lower().endswith(".kml")]
            if not kml_names:
                raise ValueError(f"No KML document was found inside {path}")
            kml_bytes = archive.read(kml_names[0])
    else:
        kml_bytes = path.read_bytes()

    root = ET.fromstring(kml_bytes)
    namespace = {"kml": "http://www.opengis.net/kml/2.2"}
    lines: list[LineString] = []
    for coordinates in root.findall(".//kml:LineString/kml:coordinates", namespace):
        points = []
        for item in (coordinates.text or "").split():
            values = item.split(",")
            if len(values) >= 2:
                points.append((float(values[0]), float(values[1])))
        if len(points) >= 2:
            lines.append(LineString(points))

    if not lines:
        raise ValueError(f"No usable KML LineString was found in {path}")
    return gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")


def _read_corridor(path: Path) -> gpd.GeoDataFrame:
    if path.suffix.lower() in {".kml", ".kmz"}:
        corridor = _kml_lines(path)
    else:
        corridor = gpd.read_file(path)
    if corridor.crs is None:
        raise ValueError(f"The corridor has no CRS: {path}")
    corridor = corridor[corridor.geometry.notna() & ~corridor.geometry.is_empty]
    if corridor.empty:
        raise ValueError(f"The corridor contains no usable geometries: {path}")
    if not corridor.geometry.geom_type.isin(["LineString", "MultiLineString"]).all():
        raise ValueError("The corridor input must contain only line geometries.")
    return corridor


def _as_lines(geometry: object) -> Iterable[LineString]:
    if isinstance(geometry, LineString):
        yield geometry
    elif isinstance(geometry, MultiLineString):
        yield from geometry.geoms


def _graph_node(coordinate: tuple[float, float]) -> tuple[float, float]:
    return round(coordinate[0], 2), round(coordinate[1], 2)


def _build_road_graph(geometries: Iterable[object]) -> dict[tuple[float, float], list[tuple[tuple[float, float], float]]]:
    graph: dict[tuple[float, float], list[tuple[tuple[float, float], float]]] = defaultdict(list)
    for geometry in geometries:
        for line in _as_lines(geometry):
            coordinates = list(line.coords)
            for start, end in zip(coordinates, coordinates[1:]):
                segment_length = math.dist(start[:2], end[:2])
                if segment_length == 0:
                    continue
                pieces = max(1, math.ceil(segment_length / MAX_ROAD_SEGMENT_M))
                points = [
                    _graph_node(
                        (
                            start[0] + (end[0] - start[0]) * index / pieces,
                            start[1] + (end[1] - start[1]) * index / pieces,
                        )
                    )
                    for index in range(pieces + 1)
                ]
                for first, second in zip(points, points[1:]):
                    distance = math.dist(first, second)
                    if distance:
                        graph[first].append((second, distance))
                        graph[second].append((first, distance))
    return graph


def _network_distances(
    graph: dict[tuple[float, float], list[tuple[tuple[float, float], float]]],
    corridor: object,
) -> dict[tuple[float, float], float]:
    """Run Dijkstra from road nodes within the snap distance of the corridor."""
    distances: dict[tuple[float, float], float] = {}
    queue: list[tuple[float, tuple[float, float]]] = []
    for node in graph:
        distance = Point(node).distance(corridor)
        if distance <= ROAD_SNAP_DISTANCE_M:
            distances[node] = distance
            heappush(queue, (distance, node))

    if not queue:
        raise ValueError(
            "No arterial road-network point is within "
            f"{ROAD_SNAP_DISTANCE_M / 1000:.1f} km of the corridor."
        )

    while queue:
        distance, node = heappop(queue)
        if distance != distances[node]:
            continue
        for neighbour, edge_length in graph[node]:
            candidate = distance + edge_length
            if candidate < distances.get(neighbour, math.inf):
                distances[neighbour] = candidate
                heappush(queue, (candidate, neighbour))
    return distances


def _network_distances_from_node(
    graph: dict[tuple[float, float], list[tuple[tuple[float, float], float]]],
    source: tuple[float, float],
) -> dict[tuple[float, float], float]:
    """Run Dijkstra from a single road node at the corridor's central point."""
    if source not in graph:
        raise ValueError(f"The central road-network node {source!r} is not in the graph.")
    distances = {source: 0.0}
    queue = [(0.0, source)]
    while queue:
        distance, node = heappop(queue)
        if distance != distances[node]:
            continue
        for neighbour, edge_length in graph[node]:
            candidate = distance + edge_length
            if candidate < distances.get(neighbour, math.inf):
                distances[neighbour] = candidate
                heappush(queue, (candidate, neighbour))
    return distances


def _radial_sector_index(
    origin: tuple[float, float],
    target: tuple[float, float],
    sector_count: int = DEFAULT_RADIAL_SECTORS,
) -> int:
    """Return the clockwise compass wedge containing a road point from origin."""
    if sector_count < 1:
        raise ValueError("The radial sector count must be positive.")
    dx = target[0] - origin[0]
    dy = target[1] - origin[1]
    bearing = math.degrees(math.atan2(dx, dy)) % 360
    sector_width = 360 / sector_count
    return int(math.floor((bearing + sector_width / 2) / sector_width)) % sector_count


def _radial_sector_name(index: int, sector_count: int) -> str:
    if sector_count == len(COMPASS_SECTORS):
        return COMPASS_SECTORS[index]
    return f"RADIAL_{index + 1:02d}"


def _municipality_code(zone_id: str) -> str:
    return zone_id.split("_", maxsplit=1)[0][:5]


def _is_peninsular_municipality(zone_id: str) -> bool:
    return bool(
        re.fullmatch(r"\d{5}(?:_AM)?", zone_id)
        and zone_id[:2] not in NON_PENINSULAR_PROVINCES
    )


def _zone_frame(
    corridor: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    districts_path: Path | None,
    municipalities_path: Path,
    max_zones: int,
    project_name: str,
) -> tuple[gpd.GeoDataFrame, int, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    municipalities = gpd.read_file(municipalities_path)
    if municipalities.crs is None:
        raise ValueError(f"The municipality polygons have no CRS: {municipalities_path}")
    if "ID" not in municipalities.columns:
        raise ValueError(f"Expected an 'ID' field in {municipalities_path}")
    municipality_ids = municipalities["ID"].astype(str)
    peninsular = municipality_ids.map(_is_peninsular_municipality)
    municipalities = municipalities.loc[peninsular].copy()
    municipalities["ID"] = municipality_ids.loc[peninsular]

    corridor_crs = corridor.estimate_utm_crs()
    if corridor_crs is None:
        raise ValueError("Could not estimate a projected CRS for the corridor.")
    if roads.crs is None:
        raise ValueError("The arterial road layer has no CRS.")
    corridor = corridor.to_crs(corridor_crs)
    roads = roads.to_crs(corridor_crs)
    municipalities = municipalities.to_crs(corridor_crs)
    corridor_geometry = unary_union(corridor.geometry)

    districts: gpd.GeoDataFrame | None = None
    if districts_path is not None:
        districts = gpd.read_file(districts_path)
        if districts.crs is None:
            raise ValueError(f"The district polygons have no CRS: {districts_path}")
        if "ID" not in districts.columns:
            raise ValueError(f"Expected an 'ID' field in {districts_path}")
        districts = districts.to_crs(corridor_crs)

    if roads.geometry.isna().all():
        raise ValueError("The arterial road layer has no usable geometries.")
    roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty]
    if roads.empty:
        raise ValueError("The arterial road layer contains no usable geometries.")

    max_distance_m = MAX_NETWORK_DISTANCE_KM * 1000
    radial_origin = corridor_geometry.centroid
    search_area = radial_origin.buffer(max_distance_m + ROAD_SNAP_DISTANCE_M)
    roads = roads[roads.geometry.intersects(search_area)].copy()
    if roads.empty:
        raise ValueError("No arterial road segments were found near the corridor.")
    roads.geometry = roads.geometry.intersection(search_area)
    graph = _build_road_graph(roads.geometry)
    graph_nodes = list(graph)
    graph_node_frame = gpd.GeoDataFrame(
        {"node": graph_nodes},
        geometry=[Point(node) for node in graph_nodes],
        crs=corridor_crs,
    )
    origin_frame = gpd.GeoDataFrame(
        geometry=[radial_origin],
        crs=corridor_crs,
    )
    snapped_origin = gpd.sjoin_nearest(
        origin_frame,
        graph_node_frame[["node", "geometry"]],
        how="left",
        distance_col="origin_snap_m",
    )
    source_node = snapped_origin.iloc[0]["node"]
    if not isinstance(source_node, tuple):
        raise ValueError("Could not snap the corridor center to the arterial network.")
    origin_snap_m = float(snapped_origin.iloc[0]["origin_snap_m"])
    if origin_snap_m > ROAD_SNAP_DISTANCE_M:
        raise ValueError(
            "The N-211 center is "
            f"{origin_snap_m / 1000:.1f} km from the arterial network; "
            f"the maximum snap distance is {ROAD_SNAP_DISTANCE_M / 1000:.1f} km."
        )
    network_distances = _network_distances_from_node(graph, source_node)
    graph_node_frame["network_distance_m"] = graph_node_frame["node"].map(
        network_distances
    )
    node_frame = graph_node_frame.loc[
        graph_node_frame["network_distance_m"].notna()
        & (graph_node_frame["network_distance_m"] <= max_distance_m)
    ].copy()

    core_records: list[dict[str, object]] = []
    access_candidates: list[dict[str, object]] = []
    core_municipalities: set[str] = set()

    if districts is not None:
        district_ids = districts["ID"].astype(str)
        is_district = district_ids.str.fullmatch(r"\d{7}")
        districts = districts.loc[is_district].copy()
        districts["ID"] = district_ids.loc[is_district]
        touches_corridor = districts.geometry.intersects(corridor_geometry)
        core_districts = districts.loc[touches_corridor]
        for row in core_districts.itertuples():
            core_records.append(
                {
                    "ID": row.ID,
                    "zone_name": row.ID,
                    "layer": "1.Distritos",
                    "source_level": "district",
                    "source_ids": row.ID,
                    "geometry": row.geometry,
                    "network_distance_km": 0.0,
                }
            )
        core_municipalities.update(
            _municipality_code(zone_id) for zone_id in core_districts["ID"]
        )
        surrounding_districts = districts.loc[~touches_corridor]
        for row in surrounding_districts.itertuples():
            if _municipality_code(row.ID) in core_municipalities:
                access_candidates.append(
                    {
                        "source_level": "district",
                        "source_id": row.ID,
                        "geometry": row.geometry,
                    }
                )

    municipality_ids = municipalities["ID"].astype(str)
    touches_corridor = municipalities.geometry.intersects(corridor_geometry)
    for row in municipalities.loc[
        touches_corridor
        & ~municipality_ids.map(_municipality_code).isin(core_municipalities)
    ].itertuples():
        core_records.append(
            {
                "ID": row.ID,
                "zone_name": row.ID,
                "layer": "2.Municipios",
                "source_level": "municipality",
                "source_ids": row.ID,
                "geometry": row.geometry,
                "network_distance_km": 0.0,
            }
        )
        core_municipalities.add(_municipality_code(row.ID))

    core_municipality_mask = municipality_ids.map(_municipality_code).isin(
        core_municipalities
    )
    core_municipality_geometry = unary_union(
        municipalities.loc[core_municipality_mask].geometry
    )
    adjacent_municipalities = municipalities.loc[
        ~core_municipality_mask
        & municipalities.geometry.touches(core_municipality_geometry)
    ]
    adjacent_municipality_codes: set[str] = set()
    for row in adjacent_municipalities.itertuples():
        core_records.append(
            {
                "ID": row.ID,
                "zone_name": row.ID,
                "layer": "2.Municipios",
                "source_level": "municipality",
                "source_ids": row.ID,
                "geometry": row.geometry,
                "network_distance_km": 0.0,
            }
        )
        adjacent_municipality_codes.add(_municipality_code(row.ID))

    municipality_access = municipalities.loc[
        ~municipality_ids.map(_municipality_code).isin(
            core_municipalities | adjacent_municipality_codes
        )
    ]
    for row in municipality_access.itertuples():
        access_candidates.append(
            {
                "source_level": "municipality",
                "source_id": row.ID,
                "geometry": row.geometry,
            }
        )

    if len(core_records) > max_zones:
        raise ValueError(
            f"The corridor touches {len(core_records)} district/municipality zones, "
            f"already above the requested maximum of {max_zones}."
        )

    if access_candidates:
        access = gpd.GeoDataFrame(access_candidates, geometry="geometry", crs=corridor_crs)
        access_points = gpd.GeoDataFrame(
            {"candidate": range(len(access))},
            geometry=access.geometry.representative_point(),
            crs=corridor_crs,
        )
        nearest = gpd.sjoin_nearest(
            access_points,
            node_frame[["node", "network_distance_m", "geometry"]],
            how="left",
            distance_col="last_mile_m",
        )
        nearest = nearest.sort_values("last_mile_m").drop_duplicates("candidate")
        access["network_distance_m"] = math.inf
        access["sector"] = -1
        for row in nearest.itertuples():
            if isinstance(row.node, tuple) and math.isfinite(row.last_mile_m):
                access.loc[row.candidate, "network_distance_m"] = (
                    row.network_distance_m + row.last_mile_m
                )
                access.loc[row.candidate, "sector"] = _radial_sector_index(
                    source_node, row.node, min(
                        DEFAULT_RADIAL_SECTORS, max_zones - len(core_records)
                    )
                )

        reachable = access[access["network_distance_m"] <= max_distance_m].copy()
    else:
        reachable = gpd.GeoDataFrame(
            columns=["source_level", "source_id", "geometry", "network_distance_m"],
            geometry="geometry",
            crs=corridor_crs,
        )

    access_capacity = max_zones - len(core_records)
    if access_candidates and access_capacity < 1:
        raise ValueError(
            "The maximum zone count leaves no zone for the surrounding "
            "Spanish mainland municipalities."
        )
    requested_sectors = min(DEFAULT_RADIAL_SECTORS, access_capacity)
    access_records: list[dict[str, object]] = []
    if not reachable.empty:
        for sector, group in reachable.groupby("sector", sort=True):
            sector_name = _radial_sector_name(int(sector), requested_sectors)
            zone_id = f"{project_name}_PETAL_{sector_name}"
            access_records.append(
                {
                    "ID": zone_id,
                    "zone_name": f"{project_name} arterial petal {sector_name}",
                    "layer": "3.Access",
                    "source_level": "districts/municipalities",
                    "source_ids": ",".join(sorted(group["source_id"].astype(str))),
                    "geometry": unary_union(group.geometry),
                    "network_distance_km": float(
                        group["network_distance_m"].min() / 1000
                    ),
                }
            )
        if len(reachable) != len(access):
            outside = access.loc[
                access["network_distance_m"] > max_distance_m, "source_id"
            ].astype(str)
            raise ValueError(
                f"{len(outside)} Spanish mainland municipalities fall beyond "
                f"the {MAX_NETWORK_DISTANCE_KM:.0f} km road-network catchment: "
                f"{', '.join(outside.head(10))}"
            )

    zones = gpd.GeoDataFrame(
        core_records + access_records,
        geometry="geometry",
        crs=corridor_crs,
    )
    zones = zones.to_crs("EPSG:4326")
    zones.geometry = zones.geometry.make_valid(
        method="structure", keep_collapsed=False
    )
    zones["project"] = project_name
    excluded_count = len(access_candidates) - len(reachable)
    corridor_output = corridor.to_crs("EPSG:4326")[["geometry"]]
    roads_output = roads[["geometry"]].copy()
    roads_output.geometry = roads_output.geometry.simplify(
        100, preserve_topology=True
    )
    roads_output = roads_output.to_crs("EPSG:4326")
    return zones, excluded_count, corridor_output, roads_output


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a corridor zoning proposal using administrative areas and arterial roads."
    )
    parser.add_argument("--corridor", type=Path, default=DEFAULT_CORRIDOR)
    parser.add_argument("--roads", type=Path, default=DEFAULT_ROADS)
    parser.add_argument(
        "--districts",
        type=Path,
        default=DEFAULT_DISTRICTS if DEFAULT_DISTRICTS.exists() else None,
        help="Optional district polygons. If unavailable, touching municipalities form the core.",
    )
    parser.add_argument("--municipalities", type=Path, default=DEFAULT_MUNICIPALITIES)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--max-zones",
        type=int,
        default=DEFAULT_MAX_ZONES,
        help=f"Maximum zones including corridor core (default: {DEFAULT_MAX_ZONES}, maximum: 19).",
    )
    parser.add_argument("--project-name", default="N-211")
    args = parser.parse_args()

    for label, path in (
        ("Corridor", args.corridor),
        ("Arterial roads", args.roads),
        ("Municipalities", args.municipalities),
    ):
        if not path.is_file():
            parser.error(f"{label} input does not exist: {path}")
    if args.districts is not None and not args.districts.is_file():
        parser.error(f"District input does not exist: {args.districts}")
    if not 2 <= args.max_zones < 20:
        parser.error("--max-zones must be between 2 and 19.")

    corridor = _read_corridor(args.corridor)
    roads = gpd.read_file(args.roads)
    output_path = args.output or ROOT / "Data" / f"Zoning_{_slug(args.project_name)}.parquet"
    geojson_path = output_path.with_suffix(".geojson")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    zones, unreachable_count, corridor_output, roads_output = _zone_frame(
        corridor,
        roads,
        args.districts,
        args.municipalities,
        args.max_zones,
        _slug(args.project_name).upper(),
    )
    if zones.empty:
        raise ValueError("No zones were generated from the supplied inputs.")
    zones.to_parquet(output_path, index=False)
    zones.to_file(geojson_path, driver="GeoJSON")
    corridor_path = output_path.with_name(f"{output_path.stem}_corridor.geojson")
    roads_path = output_path.with_name(f"{output_path.stem}_arterial_roads.geojson")
    corridor_output.to_file(corridor_path, driver="GeoJSON")
    roads_output.to_file(roads_path, driver="GeoJSON")

    print(f"Generated {len(zones)} zones (maximum: {args.max_zones}).")
    corridor_touches = zones["layer"].isin(["1.Distritos", "2.Municipios"])
    print(f"  Detailed corridor/adjacent municipality zones: {corridor_touches.sum()}")
    print(f"  Arterial access zones: {(zones['layer'] == '3.Access').sum()}")
    if unreachable_count:
        print(
            f"  Excluded {unreachable_count} surrounding administrative areas "
            f"outside the {MAX_NETWORK_DISTANCE_KM:.0f} km connected road catchment."
        )
    print(f"Parquet: {output_path}")
    print(f"GeoJSON: {geojson_path}")
    print(f"Corridor: {corridor_path}")
    print(f"Arterial roads: {roads_path}")


if __name__ == "__main__":
    main()
