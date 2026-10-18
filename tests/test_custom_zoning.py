import pandas as pd

from Scripts.od_matrix_v3 import (
    _access_zone_mappings,
    _tier_sets_from_zoning,
    assign_output_zone,
    normalise_parquet_id,
)


def test_custom_access_members_map_to_generated_band():
    zoning = pd.DataFrame(
        [
            {
                "ID": "N_211_ACCESS_01",
                "ID_norm": "N_211_ACCESS_01",
                "layer": "3.Access",
                "source_ids": "12345,12346_AM",
            }
        ]
    )
    relations = pd.DataFrame(
        [
            {
                "census_districts": "1234501",
                "districts_mitma": "12345",
                "municipalities": "12345",
                "municipalities_mitma": "12345",
            },
            {
                "census_districts": "1234601",
                "districts_mitma": "12346_AM",
                "municipalities": "12346",
                "municipalities_mitma": "12346_AM",
            },
        ]
    )

    municipality_zones, district_zones = _access_zone_mappings(zoning, relations)

    assert municipality_zones == {
        "12345": "N_211_ACCESS_01",
        "12346_AM": "N_211_ACCESS_01",
    }
    assert district_zones == {
        "12345": "N_211_ACCESS_01",
        "12346_AM": "N_211_ACCESS_01",
    }


def test_access_band_ids_are_preserved_and_ordered():
    access_id = "N_211_ACCESS_01"
    assert normalise_parquet_id(access_id) == access_id
    assert assign_output_zone({"layer": "3.Access", "ID_norm": access_id}) == access_id

    tiers = pd.DataFrame(
        [
            {"ID_norm": "19086_AM", "layer": "2.Municipios"},
            {"ID_norm": access_id, "layer": "3.Access"},
        ]
    )
    _, municipality_zones, _, _ = _tier_sets_from_zoning(tiers)

    assert municipality_zones == {"19086_AM", access_id}
