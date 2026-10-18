import geopandas as gpd
from shapely.geometry import box

from Scripts.od_matrix_v3 import _cleanup_raw_downloads, _export_spatial_zoning


def test_cleanup_removes_raw_trip_parquets_only_after_od_outputs_exist(tmp_path):
    district_parquet = tmp_path / "Viajes_distritos_2024-03-14_2024-03-14_v2.parquet"
    municipality_parquet = (
        tmp_path / "Viajes_municipios_2024-03-14_2024-03-14_v2.parquet"
    )
    other_parquet = tmp_path / "other.parquet"
    zoning_shapefile = tmp_path / "zoning.shp"
    for path in (district_parquet, municipality_parquet, other_parquet, zoning_shapefile):
        path.write_text("fixture", encoding="utf-8")

    _cleanup_raw_downloads(str(tmp_path))

    assert district_parquet.exists()
    assert municipality_parquet.exists()
    assert other_parquet.exists()

    (tmp_path / "od_matrix.csv").write_text("matrix", encoding="utf-8")
    (tmp_path / "od_flows.csv").write_text("flows", encoding="utf-8")
    _cleanup_raw_downloads(str(tmp_path))

    assert not district_parquet.exists()
    assert not municipality_parquet.exists()
    assert other_parquet.exists()
    assert zoning_shapefile.exists()


def test_spatial_export_ids_match_custom_od_zone_ids(tmp_path):
    zoning_path = tmp_path / "zoning.parquet"
    zoning = gpd.GeoDataFrame(
        {
            "ID": ["19086_AM", "N_211_PETAL_N"],
            "layer": ["2.Municipios", "3.Access"],
            "source_ids": ["19086_AM", "12345,12346"],
        },
        geometry=[box(-3.2, 40.0, -3.0, 40.2), box(-4.0, 40.0, -3.4, 40.3)],
        crs="EPSG:4326",
    )
    zoning.to_parquet(zoning_path)

    _export_spatial_zoning(str(zoning_path), str(tmp_path))

    exported = gpd.read_file(tmp_path / "zoning.shp")

    assert set(exported["zone_id"]) == {"19086_AM", "N_211_PETAL_N"}
