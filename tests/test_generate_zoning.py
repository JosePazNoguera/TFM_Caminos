import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Polygon, box

from Scripts.generate_visual import (
    _build_arc_data,
    _build_html,
    _centroid_coordinates,
    _dissolve_visual_zones,
    generate_zoning_preview,
)
from Scripts.generate_zoning import (
    _build_road_graph,
    _is_peninsular_municipality,
    _kml_lines,
    _network_distances,
    _radial_sector_index,
)


def test_network_distance_follows_road_geometry():
    graph = _build_road_graph([LineString([(0, 0), (3_000, 0)])])
    distances = _network_distances(graph, LineString([(0, -100), (0, 100)]))

    assert distances[(3_000.0, 0.0)] == 3_000


def test_zone_ids_are_limited_to_peninsular_spanish_municipalities():
    assert _is_peninsular_municipality("19086_AM")
    assert _is_peninsular_municipality("01001")
    assert not _is_peninsular_municipality("PT111")
    assert not _is_peninsular_municipality("07001")
    assert not _is_peninsular_municipality("38013_AM")
    assert not _is_peninsular_municipality("51001")


def test_kml_reader_loads_linestring(tmp_path):
    kml = tmp_path / "corridor.kml"
    kml.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
        <kml xmlns="http://www.opengis.net/kml/2.2">
          <Placemark><LineString><coordinates>-2.4,41.0,0 -2.3,41.1,0</coordinates></LineString></Placemark>
        </kml>""",
        encoding="utf-8",
    )

    corridor = _kml_lines(kml)

    assert len(corridor) == 1
    assert corridor.crs.to_epsg() == 4326
    assert corridor.geometry.iloc[0].coords[0] == (-2.4, 41.0)


def test_zoning_preview_needs_no_od_flows(tmp_path):
    zoning_path = tmp_path / "zones.parquet"
    zones = gpd.GeoDataFrame(
        {
            "ID": ["ZONE_01"],
            "layer": ["3.Access"],
            "zone_name": ["Access band"],
            "source_ids": ["10001,10002"],
        },
        geometry=[box(-3, 40, -2, 41)],
        crs="EPSG:4326",
    )
    zones.to_parquet(zoning_path)

    generate_zoning_preview(zoning_path)

    html = zoning_path.with_name("zones_preview.html").read_text(encoding="utf-8")
    assert "Corridor zoning proposal" in html
    assert "ZONE_01" in html
    assert "arterial-roads" in html
    assert "od_flows.csv" not in html


def test_arcs_use_the_zone_geometry_centroids():
    zones = gpd.GeoDataFrame(
        {
            "ID": ["N211_CORE", "MACRO_01"],
            "layer": ["2.Municipios", "3.Access"],
        },
        geometry=[box(-3.2, 40.0, -3.0, 40.2), box(-4.0, 40.0, -3.4, 40.3)],
        crs="EPSG:4326",
    )
    coords = _centroid_coordinates(zones)
    arcs = _build_arc_data(
        pd.DataFrame(
            [{"origin": "N211_CORE", "destination": "MACRO_01", "volume": 12}]
        ),
        coords,
    )

    assert arcs[0]["source"] == [coords["N211_CORE"]["lon"], coords["N211_CORE"]["lat"]]
    assert arcs[0]["target"] == [coords["MACRO_01"]["lon"], coords["MACRO_01"]["lat"]]


def test_map_shows_every_flow_for_selected_zone_without_top_twenty_limit():
    html = _build_html(
        [],
        '{"type":"FeatureCollection","features":[]}',
        initial_zone="N211_CORE",
        arterial_roads_geojson=(
            '{"type":"FeatureCollection","features":['
            '{"type":"Feature","geometry":{"type":"LineString",'
            '"coordinates":[[-3,40],[-2,41]]},"properties":{}}]}'
        ),
    )

    assert "let selectedZone = \"N211_CORE\"" in html
    assert "All destinations from" in html
    assert "All origins to" in html
    assert "Top 20" not in html
    assert ".slice(0, 20)" not in html
    assert "stroked: false" in html
    assert "arterial-roads-layer" in html
    assert "Arterial road network" in html
    assert "[-3,40],[-2,41]" in html
    assert "Arterial-road petals" in html


def test_visual_dissolves_source_polygons_to_matrix_zones():
    source_zones = gpd.GeoDataFrame(
        {
            "ID": ["19001", "19002"],
            "layer": ["4.RestoPeninsula", "4.RestoPeninsula"],
        },
        geometry=[
            box(-4.0, 40.0, -3.5, 40.5),
            Polygon(
                [(-3.5, 40.0), (-3.0, 40.5), (-3.5, 40.5), (-3.0, 40.0)]
            ),
        ],
        crs="EPSG:4326",
    )

    dissolved = _dissolve_visual_zones(source_zones)

    assert dissolved["ID"].iloc[0] == "REST_PENINSULA"
    assert len(dissolved) == 1
    assert dissolved.geometry.iloc[0].geom_type in {"Polygon", "MultiPolygon"}
    assert dissolved.geometry.iloc[0].is_valid
    assert dissolved.geometry.iloc[0].area > 0


def test_radial_sector_indices_follow_clockwise_compass_directions():
    origin = (0.0, 0.0)
    points = [
        (0.0, 10.0),
        (10.0, 10.0),
        (10.0, 0.0),
        (10.0, -10.0),
        (0.0, -10.0),
        (-10.0, -10.0),
        (-10.0, 0.0),
        (-10.0, 10.0),
    ]

    assert [_radial_sector_index(origin, point) for point in points] == list(range(8))


def test_zone_frame_groups_access_polygons_into_road_direction_petals(tmp_path):
    from Scripts.generate_zoning import _zone_frame

    origin_x, origin_y = 500_000.0, 4_400_000.0
    center = (origin_x, origin_y)
    offsets = [
        (0, 30_000),
        (30_000, 30_000),
        (30_000, 0),
        (30_000, -30_000),
        (0, -30_000),
        (-30_000, -30_000),
        (-30_000, 0),
        (-30_000, 30_000),
    ]
    road_lines = [
        LineString([center, (origin_x + dx, origin_y + dy)])
        for dx, dy in offsets
    ]
    roads = gpd.GeoDataFrame(geometry=road_lines, crs="EPSG:32630")
    corridor = gpd.GeoDataFrame(
        geometry=[LineString([(origin_x - 500, origin_y), (origin_x + 500, origin_y)])],
        crs="EPSG:32630",
    )
    municipality_shapes = [box(origin_x - 2_000, origin_y - 2_000,
                               origin_x + 2_000, origin_y + 2_000)]
    municipality_ids = ["19086_AM"]
    for dx, dy in offsets:
        municipality_ids.append(f"{len(municipality_ids) + 1:05d}")
        municipality_shapes.append(
            box(origin_x + dx - 2_000, origin_y + dy - 2_000,
                origin_x + dx + 2_000, origin_y + dy + 2_000)
        )
    municipality_ids.append("01009")
    municipality_shapes.append(
        box(origin_x + 2_000, origin_y - 1_000, origin_x + 4_000, origin_y + 1_000)
    )
    municipalities_path = tmp_path / "municipalities.geojson"
    gpd.GeoDataFrame(
        {"ID": municipality_ids},
        geometry=municipality_shapes,
        crs="EPSG:32630",
    ).to_file(municipalities_path, driver="GeoJSON")

    zones, _, _, _ = _zone_frame(
        corridor,
        roads,
        districts_path=None,
        municipalities_path=municipalities_path,
        max_zones=10,
        project_name="TEST",
    )

    detailed_neighbors = zones.loc[
        zones.layer.eq("2.Municipios"), "ID"
    ].tolist()
    assert set(zones.loc[zones.layer.eq("3.Access"), "ID"]) == {
        f"TEST_PETAL_{direction}"
        for direction in ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    }
    assert set(detailed_neighbors) == {"19086_AM", "01009"}
    assert len(zones) == 10
    assert zones.geometry.is_valid.all()
