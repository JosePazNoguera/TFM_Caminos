"""
od_matrix_v3.py
=======================

Build a custom multi-scale Origin-Destination matrix for an infrastructure
project from a ``Zoning.parquet`` file and MITMA mobility data.

The parquet file carries the project-specific zoning as polygons organised
in four layers:

    1.Distritos        → one MITMA district zone per row (project municipality)
    2.Municipios       → one MITMA municipality zone per row
    3.Provincias       → grouped by province prefix → "PROV_XX"
    4.RestoPeninsula   → aggregated into 1 zone: "REST_PENINSULA"

Branching is always on the ``layer`` column. The ``Class`` column has been
removed from the parquet and must not be used.

The public entry point is :func:`build_od_from_parquet`. Two helper functions
:func:`_get_raw_filtered_totals` and :func:`_get_raw_district_origin_totals`
are exposed for the integration test suite.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

# Fix for macOS Python SSL Certificate errors
import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESTO_PENINSULA_LABEL: str = "REST_PENINSULA"

DEFAULT_START_DATE: str = "2024-07-01"
DEFAULT_END_DATE: str = "2024-07-01"

SOCIO_COLS: list[str] = ["income", "age", "gender"]
ACTIVITY_COLS: list[str] = ["activity_origin", "activity_destination"]

FREQUENT_ACTIVITIES: Set[str] = {"home", "other_frequent", "work_or_study"}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _intersect_cols(cols: list[str], df: pd.DataFrame) -> list[str]:
    """Return only those columns from *cols* that exist in *df*."""
    return [c for c in cols if c in df.columns]

# ---------------------------------------------------------------------------
# Parquet ID normalisation and zone assignment
# ---------------------------------------------------------------------------

def normalise_parquet_id(raw_id):
    """Normalise a raw zoning-parquet ID to a valid MITMA zone ID.

    - ``None`` → ``None``
    - 5-char municipality IDs (e.g. ``"14001"``) → unchanged
    - 7-char district IDs (e.g. ``"1402101"``) → unchanged
    - 8-char aggregated-municipality IDs (e.g. ``"14005_AM"``) → unchanged
    - 11-char census section IDs (e.g. ``"34022253049"``) collapse to the
      parent municipality (first 5 characters).
    """
    if raw_id is None:
        return None
    try:
        if pd.isna(raw_id):
            return None
    except (TypeError, ValueError):
        pass
    raw_id = str(raw_id)
    if len(raw_id) > 8 and raw_id.isdigit():
        return raw_id[:5]
    return raw_id

def assign_output_zone(row) -> str | None:
    """Return the output-zone label for a single zoning-parquet row.

    The parquet now uses four tier keywords:

    - ``1.Distritos``       → 7-digit district IDs; output label is the ID itself.
    - ``2.Municipios``      → 5-digit muni IDs (optionally ``_AM``); output label
                              is the ID itself.
    - ``3.Provincias``      → muni/LUA IDs that the user wants *grouped* to
                              province granularity; output label is derived from
                              the 2-digit province prefix → ``PROV_XX``.
    - ``4.RestoPeninsula``  → muni/LUA IDs aggregated to a single
                              ``REST_PENINSULA`` zone.

    ``row`` must expose ``layer`` and ``ID_norm`` (set by
    :func:`normalise_parquet_id`). Supports both dicts and pandas Series.
    """
    layer = row["layer"]
    id_norm = row["ID_norm"]

    if layer == "4.RestoPeninsula":
        return RESTO_PENINSULA_LABEL

    if layer == "3.Access":
        return id_norm

    if layer == "3.Provincias":
        if id_norm is None:
            return None
        province = str(id_norm).split("_")[0][:2]
        return f"PROV_{province}"

    if layer == "2.Municipios":
        return id_norm

    if layer == "1.Distritos":
        return id_norm

    return None

# ---------------------------------------------------------------------------
# Lookup builders from the MITMA relation table
# ---------------------------------------------------------------------------

def _province_label_from_muni(muni_mitma_id: str) -> str:
    """Output label for a muni that belongs to the ``3.Provincias`` tier."""
    return f"PROV_{str(muni_mitma_id).split('_')[0][:2]}"

def build_zone_lookup(
    rel: pd.DataFrame,
    municipios_ids: Set[str],
    provincias_codes: Set[str],
    distritos_ids: Set[str] | None = None,
    custom_municipality_zones: Dict[str, str] | None = None,
    custom_district_zones: Dict[str, str] | None = None,
) -> Dict[str, str]:
    """Map MITMA zone IDs → output-zone labels.

    When *distritos_ids* is provided the lookup keys are ``districts_mitma``
    IDs with the full four-tier priority (Distritos → Municipios → Provincias
    → RestoPeninsula).  When *distritos_ids* is ``None`` the lookup keys are
    ``municipalities_mitma`` IDs with the three-tier priority.
    """
    lookup: Dict[str, str] = {}
    custom_municipality_zones = custom_municipality_zones or {}
    custom_district_zones = custom_district_zones or {}
    if distritos_ids is not None:
        subset = (
            rel[["districts_mitma", "municipalities_mitma"]]
            .dropna(subset=["districts_mitma"])
            .drop_duplicates("districts_mitma")
        )
        for _, row in subset.iterrows():
            d = row["districts_mitma"]
            m = row["municipalities_mitma"]
            if d in distritos_ids:
                lookup[d] = d
            elif d in custom_district_zones:
                lookup[d] = custom_district_zones[d]
            elif m in custom_municipality_zones:
                lookup[d] = custom_municipality_zones[m]
            elif m in municipios_ids:
                lookup[d] = m
            elif str(m)[:2] in provincias_codes:
                lookup[d] = _province_label_from_muni(m)
            else:
                lookup[d] = RESTO_PENINSULA_LABEL
    else:
        subset = (
            rel[["municipalities_mitma"]]
            .dropna(subset=["municipalities_mitma"])
            .drop_duplicates("municipalities_mitma")
        )
        for _, row in subset.iterrows():
            m = row["municipalities_mitma"]
            if m in custom_municipality_zones:
                lookup[m] = custom_municipality_zones[m]
            elif m in municipios_ids:
                lookup[m] = m
            elif str(m)[:2] in provincias_codes:
                lookup[m] = _province_label_from_muni(m)
            else:
                lookup[m] = RESTO_PENINSULA_LABEL
    return lookup

# ---------------------------------------------------------------------------
# Parquet loading and tier-set construction
# ---------------------------------------------------------------------------

def _load_zoning_parquet(parquet_path: str):
    """Load the zoning attributes required for OD zone assignment."""
    zoning = pd.read_parquet(parquet_path)
    columns = ["ID", "layer"]
    if "source_ids" in zoning.columns:
        columns.append("source_ids")
    zoning = zoning[columns].copy()
    zoning["ID_norm"] = zoning["ID"].apply(normalise_parquet_id)
    zoning["output_zone"] = zoning.apply(assign_output_zone, axis=1)
    return zoning

def _tier_sets_from_zoning(
    zoning,
) -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
    """Extract membership sets for each tier plus the province-zone labels.

    Returns a 4-tuple:
        (distritos_ids, municipios_ids, provincias_codes, provincia_zones)

    - ``distritos_ids``       — 7-digit district IDs from ``1.Distritos``.
    - ``municipios_ids``      — muni/LUA IDs from ``2.Municipios``.
    - ``provincias_codes``    — 2-digit province prefixes from ``3.Provincias``.
    - ``provincia_zones``     — set of ``PROV_XX`` labels implied by tier 3
                                 (used for output ordering).
    """
    distritos_ids = set(
        zoning.loc[zoning["layer"] == "1.Distritos", "ID_norm"].dropna().tolist()
    )
    municipios_ids = set(
        zoning.loc[zoning["layer"] == "2.Municipios", "ID_norm"].dropna().tolist()
    )
    municipios_ids.update(
        zoning.loc[zoning["layer"] == "3.Access", "ID_norm"].dropna().tolist()
    )
    provincias_codes = {
        str(pid)[:2] for pid in zoning.loc[zoning["layer"] == "3.Provincias", "ID_norm"].dropna()
    }
    provincia_zones = {f"PROV_{code}" for code in provincias_codes}
    return (
        distritos_ids,
        municipios_ids,
        provincias_codes,
        provincia_zones,
    )


def _access_zone_mappings(
    zoning: pd.DataFrame, rel: pd.DataFrame
) -> tuple[Dict[str, str], Dict[str, str]]:
    """Map MITMA municipality and district IDs to generated access-band IDs."""
    municipality_zones: Dict[str, str] = {}
    district_zones: Dict[str, str] = {}
    municipality_codes = rel["municipalities"].astype(str)
    mitma_municipalities = rel["municipalities_mitma"].astype(str)
    census_districts = rel["census_districts"].astype(str)
    mitma_districts = rel["districts_mitma"].astype(str)

    for row in zoning.loc[zoning["layer"] == "3.Access"].itertuples():
        members = str(row.source_ids).split(",") if hasattr(row, "source_ids") else []
        for member in members:
            member = member.strip()
            if not member:
                continue

            if member.isdigit() and len(member) == 7:
                matches = rel.loc[census_districts == member]
                for district_id in matches["districts_mitma"].dropna().astype(str).unique():
                    previous = district_zones.setdefault(district_id, row.ID_norm)
                    if previous != row.ID_norm:
                        raise ValueError(
                            f"MITMA district {district_id} belongs to multiple access zones."
                        )
                if not matches.empty:
                    continue

            if member.endswith("_AM"):
                matches = rel.loc[mitma_municipalities == member]
            else:
                municipality_code = member[:5]
                matches = rel.loc[municipality_codes == municipality_code]

            if matches.empty:
                raise ValueError(
                    f"Could not match access-zone source ID {member!r} "
                    "to the MITMA zone-relation table."
                )
            for municipality_id in matches["municipalities_mitma"].dropna().astype(str).unique():
                previous = municipality_zones.setdefault(municipality_id, row.ID_norm)
                if previous != row.ID_norm:
                    raise ValueError(
                        f"MITMA municipality {municipality_id} belongs to multiple access zones."
                    )
    for _, row in rel[["districts_mitma", "municipalities_mitma"]].dropna().drop_duplicates().iterrows():
        district_id = str(row["districts_mitma"])
        municipality_id = str(row["municipalities_mitma"])
        if municipality_id in municipality_zones:
            previous = district_zones.setdefault(
                district_id, municipality_zones[municipality_id]
            )
            if previous != municipality_zones[municipality_id]:
                raise ValueError(
                    f"MITMA district {district_id} belongs to multiple access zones."
                )

    return municipality_zones, district_zones

# ---------------------------------------------------------------------------
# MITMA download helpers
# ---------------------------------------------------------------------------

def _load_relation_table(output_dir: str, version: int) -> pd.DataFrame:
    from pyspainmobility import Zones
    zones_obj = Zones(
        zones="municipalities", version=version, output_directory=output_dir
    )
    rel = zones_obj.get_zone_relations()
    rel = rel.copy()
    for col in ("municipalities", "municipalities_mitma", "districts_mitma"):
        if col in rel.columns:
            rel[col] = rel[col].astype("object")
    return rel

def _download_od(
    zones: str,
    start_date: str, end_date: str, output_dir: str, version: int,
    social_agg: bool = False, keep_activity: bool = False,
) -> pd.DataFrame:
    """Download OD data for the given zone granularity ('districts' or 'municipalities')."""
    from pyspainmobility import Mobility
    mob = Mobility(
        version=version,
        zones=zones,
        start_date=start_date,
        end_date=end_date,
        output_directory=output_dir,
    )
    return mob.get_od_data(
        return_df=True, keep_activity=keep_activity, social_agg=social_agg
    )

def _project_municipalities_from_distritos(distritos_ids: Set[str]) -> Set[str]:
    """Derive the project's INE municipality codes from the ``1.Distritos`` tier."""
    return {str(d)[:5] for d in distritos_ids if d is not None}

def _resolve_project_municipality_mitma_ids(
    rel: pd.DataFrame, project_municipality_ines: Set[str]
) -> Set[str]:
    """Resolve the set of MITMA muni IDs covering the project municipalities."""
    ids = set(
        rel.loc[
            rel["municipalities"].isin(project_municipality_ines),
            "municipalities_mitma",
        ]
        .dropna()
        .unique()
        .tolist()
    )
    assert len(ids) > 0, (
        "Could not resolve MITMA municipality IDs for project municipalities "
        f"{sorted(project_municipality_ines)!r} from the relation table. "
        "Is the '1.Distritos' tier empty or malformed?"
    )
    return ids

# ---------------------------------------------------------------------------
# Filtering + mapping pipeline (shared between public entry point and helpers)
# ---------------------------------------------------------------------------

def _cols_for_pipeline(
    time_aggregation: str,
    social_agg: bool = False,
    keep_activity: bool = False,
) -> list[str]:
    """Return the column projection needed downstream for this aggregation."""
    cols = ["id_origin", "id_destination", "n_trips", "trips_total_length_km"]
    if time_aggregation == "daily":
        cols = ["date"] + cols
    elif time_aggregation == "hourly":
        cols = ["hour"] + cols
    if keep_activity:
        cols += ACTIVITY_COLS
    if social_agg:
        cols += SOCIO_COLS
    return cols

def _prepare_pipeline(
    parquet_path: str,
    start_date: str,
    end_date: str,
    output_dir: str,
    version: int,
    social_agg: bool,
    time_aggregation: str = "total",
    ac_provinces: Set[str] | None = None,
    project_municipality_ine: str = "14021",
    keep_activity: bool = False,
):
    """Run steps 1–7 of the blueprint and return all intermediate artefacts."""
    os.makedirs(output_dir, exist_ok=True)
    zoning = _load_zoning_parquet(parquet_path)
    (
        distritos_ids,
        municipios_ids,
        provincias_codes,
        provincia_zones,
    ) = _tier_sets_from_zoning(zoning)

    if ac_provinces:
        provincias_codes.update(ac_provinces)
        provincia_zones.update({f"PROV_{code}" for code in ac_provinces})

    core_municipality_ids = zoning.loc[
        zoning["layer"] == "2.Municipios", "ID_norm"
    ].dropna().astype(str)
    project_municipality_ines = _project_municipalities_from_distritos(
        distritos_ids
    ) | {municipality_id.split("_", maxsplit=1)[0][:5] for municipality_id in core_municipality_ids}
    if not project_municipality_ines:
        project_municipality_ines = {project_municipality_ine}

    rel = _load_relation_table(output_dir, version=version)
    access_municipality_zones, access_district_zones = _access_zone_mappings(
        zoning, rel
    )
    project_municipality_mitma_ids = _resolve_project_municipality_mitma_ids(
        rel, project_municipality_ines
    )

    dist_lookup = build_zone_lookup(
        rel,
        municipios_ids,
        provincias_codes,
        distritos_ids=distritos_ids,
        custom_municipality_zones=access_municipality_zones,
        custom_district_zones=access_district_zones,
    )
    muni_lookup = build_zone_lookup(
        rel,
        municipios_ids,
        provincias_codes,
        custom_municipality_zones=access_municipality_zones,
    )

    cols_needed = _cols_for_pipeline(time_aggregation, social_agg, keep_activity)

    # --- Download A: district-level -------------------------------------------------
    df_dist_raw = _download_od(
        "districts", start_date, end_date, output_dir, version, social_agg, keep_activity
    )
    district_zones_to_fetch = distritos_ids | project_municipality_mitma_ids
    mask_a = (
        df_dist_raw["id_origin"].isin(district_zones_to_fetch)
        | df_dist_raw["id_destination"].isin(district_zones_to_fetch)
    )
    df_dist = df_dist_raw.loc[mask_a, _intersect_cols(cols_needed, df_dist_raw)]
    del df_dist_raw

    # --- Download B: municipality-level --------------------------------------------
    df_muni_raw = _download_od(
        "municipalities", start_date, end_date, output_dir, version, social_agg, keep_activity
    )
    narrow = df_muni_raw[_intersect_cols(cols_needed, df_muni_raw)]
    del df_muni_raw
    excluded_mask = (
        narrow["id_origin"].isin(project_municipality_mitma_ids)
        | narrow["id_destination"].isin(project_municipality_mitma_ids)
    )
    excluded_b_trips = float(narrow.loc[excluded_mask, "n_trips"].sum())
    df_muni = narrow.loc[~excluded_mask]

    return {
        "zoning": zoning,
        "distritos_ids": distritos_ids,
        "municipios_ids": municipios_ids,
        "provincias_codes": provincias_codes,
        "provincia_zones": provincia_zones,
        "rel": rel,
        "project_municipality_ines": project_municipality_ines,
        "project_municipality_mitma_ids": project_municipality_mitma_ids,
        "access_municipality_zones": access_municipality_zones,
        "access_district_zones": access_district_zones,
        "dist_lookup": dist_lookup,
        "muni_lookup": muni_lookup,
        "df_dist": df_dist,
        "df_muni": df_muni,
        "excluded_b_trips": excluded_b_trips,
        "parquet_path": parquet_path,
    }

def _map_and_clean(df: pd.DataFrame, lookup: Dict[str, str], label: str, include_international: bool = False) -> Tuple[pd.DataFrame, int, int]:
    """Apply origin/dest lookup; return (cleaned_df, n_unmapped_orig, n_unmapped_dest)."""
    out = df.copy()
    out["origin_zone"] = out["id_origin"].map(lookup)
    out["dest_zone"] = out["id_destination"].map(lookup)

    if include_international:
        intl_orig_mask = out["origin_zone"].isna() & out["id_origin"].str.match(r"^[A-Za-z]")
        intl_dest_mask = out["dest_zone"].isna() & out["id_destination"].str.match(r"^[A-Za-z]")
        out.loc[intl_orig_mask, "origin_zone"] = "INTERNATIONAL"
        out.loc[intl_dest_mask, "dest_zone"] = "INTERNATIONAL"

    n_unmapped_orig = int(out["origin_zone"].isna().sum())
    n_unmapped_dest = int(out["dest_zone"].isna().sum())
    if n_unmapped_orig or n_unmapped_dest:
        print(
            f"WARNING [{label}]: {n_unmapped_orig} unmapped origins, "
            f"{n_unmapped_dest} unmapped destinations. These rows will be dropped."
        )
    out = out.dropna(subset=["origin_zone", "dest_zone"])
    return out, n_unmapped_orig, n_unmapped_dest

def _aggregate(df_dist_mapped: pd.DataFrame, df_muni_mapped: pd.DataFrame,
               time_aggregation: str) -> pd.DataFrame:
    cols = ["origin_zone", "dest_zone", "n_trips", "trips_total_length_km"]
    extra_cols: list[str] = []
    if time_aggregation == "daily" and "date" in df_dist_mapped.columns:
        extra_cols = ["date"]
    elif time_aggregation == "hourly" and "hour" in df_dist_mapped.columns:
        extra_cols = ["hour"]

    activity_present = _intersect_cols(ACTIVITY_COLS, df_dist_mapped)
    socio_present = _intersect_cols(SOCIO_COLS, df_dist_mapped)

    select_cols = (
        cols
        + _intersect_cols(extra_cols, df_dist_mapped)
        + activity_present
        + socio_present
    )
    a = df_dist_mapped[_intersect_cols(select_cols, df_dist_mapped)].copy()
    b = df_muni_mapped[_intersect_cols(select_cols, df_muni_mapped)].copy()
    df_combined = pd.concat([a, b], ignore_index=True)

    group_cols = (
        ["origin_zone", "dest_zone"]
        + _intersect_cols(extra_cols, df_combined)
        + activity_present
        + socio_present
    )
    od_long = (
        df_combined
        .groupby(group_cols, as_index=False)
        .agg(n_trips=("n_trips", "sum"),
             trip_km=("trips_total_length_km", "sum"))
    )
    return od_long

def _ordered_zones(distr_cercanos_ids: Iterable[str],
                   muni_colind_ids: Iterable[str],
                   resto_comaut_zones: Iterable[str],
                   include_international: bool = False) -> list[str]:
    zones = (
        sorted(distr_cercanos_ids)
        + sorted(muni_colind_ids)
        + sorted(resto_comaut_zones)
        + [RESTO_PENINSULA_LABEL]
    )
    if include_international:
        zones.append("INTERNATIONAL")
    return zones

# ---------------------------------------------------------------------------
# Distance skim (§7.7)
# ---------------------------------------------------------------------------

def _build_distance_skim(od_long_total: pd.DataFrame,
                         zone_order: list[str]) -> pd.DataFrame:
    """Compute a zone × zone average-distance skim from aggregated OD data."""
    skim = od_long_total.copy()

    n = skim["n_trips"].astype(float)
    km = skim["trip_km"].astype(float)
    skim["avg_km"] = km.where(n > 0).div(n.where(n > 0))

    skim_pivot = (
        skim.pivot(index="origin_zone", columns="dest_zone", values="avg_km")
        .reindex(index=zone_order, columns=zone_order)
    )
    skim_pivot.index.name = "origin\\destination"

    if (
        RESTO_PENINSULA_LABEL in skim_pivot.index
        and RESTO_PENINSULA_LABEL in skim_pivot.columns
    ):
        skim_pivot.loc[RESTO_PENINSULA_LABEL, RESTO_PENINSULA_LABEL] = float("nan")

    return skim_pivot

# ---------------------------------------------------------------------------
# Shared matrix + flows writer
# ---------------------------------------------------------------------------

def _write_matrix_and_flows(
    seg_agg: pd.DataFrame,
    zone_order: list[str],
    pivot_path: Path,
    flows_path: Path,
    header_comment: str,
) -> None:
    """Write a wide-pivot OD matrix CSV and a long-format flows CSV.

    This is the single implementation of the pivot → TOTAL row/col → CSV
    and flows → filter REST×REST → CSV logic used by every output writer.
    """
    # --- Wide pivot ---
    seg_pivot = (
        seg_agg
        .pivot(index="origin_zone", columns="dest_zone", values="n_trips")
        .reindex(index=zone_order, columns=zone_order, fill_value=0.0)
        .fillna(0.0)
    )
    seg_pivot.index.name = "origin\\destination"

    if (
        RESTO_PENINSULA_LABEL in seg_pivot.index
        and RESTO_PENINSULA_LABEL in seg_pivot.columns
    ):
        seg_pivot.loc[RESTO_PENINSULA_LABEL, RESTO_PENINSULA_LABEL] = 0.0

    seg_pivot_out = seg_pivot.round(1).copy()
    seg_pivot_out["TOTAL"] = seg_pivot_out.sum(axis=1)
    total_row = seg_pivot_out.sum(axis=0)
    total_row.name = "TOTAL"
    seg_pivot_out = pd.concat([seg_pivot_out, total_row.to_frame().T])
    seg_pivot_out.index.name = seg_pivot.index.name

    with open(pivot_path, "w", encoding="utf-8") as f:
        f.write(f"# {header_comment}\n")
        seg_pivot_out.to_csv(f, float_format="%.1f")

    # --- Long-format flows ---
    flows = seg_agg.copy()
    flows["n_trips"] = flows["n_trips"].round(1)
    flows["trip_km"] = flows["trip_km"].round(1)
    flows = flows[flows["n_trips"] != 0]
    flows = flows[
        ~(
            (flows["origin_zone"] == RESTO_PENINSULA_LABEL)
            & (flows["dest_zone"] == RESTO_PENINSULA_LABEL)
        )
    ]
    flows = flows.sort_values(["origin_zone", "dest_zone"]).reset_index(drop=True)
    flows.to_csv(flows_path, index=False)

# ---------------------------------------------------------------------------
# Activity-split output (keep_activity=True)
# ---------------------------------------------------------------------------

def _write_activity_outputs(
    od_long: pd.DataFrame,
    output_dir: str,
    start_date: str,
    end_date: str,
    zone_order_full: list[str],
) -> None:
    """Write separate OD matrices for frequent and non_frequent trips.

    **frequent**: activity_origin AND activity_destination in
    {home, other_frequent, work_or_study}.

    **non_frequent**: activity_origin OR activity_destination equals
    other_non_frequent.

    Files are written to ``<output_dir>/activity/``:
      - ``od_matrix_{tag}_frequent.csv``      — wide pivot
      - ``od_flows_{tag}_frequent.csv``       — long-format non-zero pairs
      - ``od_matrix_{tag}_non_frequent.csv``
      - ``od_flows_{tag}_non_frequent.csv``
    """
    if "activity_origin" not in od_long.columns or "activity_destination" not in od_long.columns:
        return

    activity_dir = Path(output_dir) / "activity"
    activity_dir.mkdir(parents=True, exist_ok=True)

    tag = start_date if start_date == end_date else f"{start_date}_{end_date}"

    # Normalize activity cols to plain Python objects to avoid ArrowDtype mismatches.
    od_work = od_long.copy()
    for col in ACTIVITY_COLS:
        if col in od_work.columns:
            od_work[col] = od_work[col].astype(object)

    frequent_mask = (
        od_work["activity_origin"].isin(FREQUENT_ACTIVITIES)
        & od_work["activity_destination"].isin(FREQUENT_ACTIVITIES)
    )
    non_frequent_mask = (
        (od_work["activity_origin"] == "other_non_frequent")
        | (od_work["activity_destination"] == "other_non_frequent")
    )

    print("\n  Writing activity-split matrices …")

    for label, mask in [("frequent", frequent_mask), ("non_frequent", non_frequent_mask)]:
        seg = od_work[mask]
        seg_agg = (
            seg.groupby(["origin_zone", "dest_zone"], as_index=False)
            .agg(n_trips=("n_trips", "sum"), trip_km=("trip_km", "sum"))
        )

        _write_matrix_and_flows(
            seg_agg, zone_order_full,
            pivot_path=activity_dir / f"od_matrix_{label}.csv",
            flows_path=activity_dir / f"od_flows_{label}.csv",
            header_comment=(
                f"Activity type: {label}  |  "
                f"REST_PENINSULA x REST_PENINSULA not extracted"
            ),
        )

        print(f"    ↳ {label}: {seg_agg['n_trips'].sum():.0f} trips")

    print(f"\n  Activity split files saved to: {activity_dir}\n")

# ---------------------------------------------------------------------------
# Per-time-slice output (time_aggregation="daily" or "hourly")
# ---------------------------------------------------------------------------

def _write_time_slice_outputs(
    od_long: pd.DataFrame,
    output_dir: str,
    time_aggregation: str,
    zone_order_full: list[str],
) -> None:
    """Write one OD matrix per date (daily) or per hour (hourly).

    Files are written to ``<output_dir>/daily/`` or ``<output_dir>/hourly/``:
      - ``od_matrix_{slice}.csv``  — wide pivot
      - ``od_flows_{slice}.csv``   — long-format non-zero pairs

    where *slice* is ``YYYY-MM-DD`` for daily and ``HH`` (zero-padded) for
    hourly. The total-across-all-slices matrix in the parent output directory
    is unchanged.
    """
    time_col = "date" if time_aggregation == "daily" else "hour"
    if time_col not in od_long.columns:
        print(f"  [warn] Column '{time_col}' not found in od_long — skipping {time_aggregation} slice outputs.")
        return

    slice_dir = Path(output_dir) / time_aggregation
    slice_dir.mkdir(parents=True, exist_ok=True)

    # Normalize to plain Python objects to avoid ArrowDtype comparison issues.
    od_work = od_long.copy()
    od_work[time_col] = od_work[time_col].astype(object)

    unique_slices = sorted(od_work[time_col].dropna().unique(), key=str)
    print(f"\n  Writing {len(unique_slices)} {time_aggregation} slice(s) …")

    for slice_val in unique_slices:
        # Build a filesystem-safe tag.
        if time_aggregation == "hourly":
            try:
                tag = f"{int(slice_val):02d}"
            except (ValueError, TypeError):
                tag = str(slice_val).replace(":", "-").strip()
        else:
            tag = str(slice_val).strip()

        seg = od_work[od_work[time_col] == slice_val]
        if seg.empty:
            continue

        seg_agg = (
            seg.groupby(["origin_zone", "dest_zone"], as_index=False)
            .agg(n_trips=("n_trips", "sum"), trip_km=("trip_km", "sum"))
        )
        if seg_agg.empty:
            continue

        _write_matrix_and_flows(
            seg_agg, zone_order_full,
            pivot_path=slice_dir / f"od_matrix_{tag}.csv",
            flows_path=slice_dir / f"od_flows_{tag}.csv",
            header_comment=(
                f"{time_col.capitalize()}: {slice_val}  |  "
                f"REST_PENINSULA x REST_PENINSULA not extracted"
            ),
        )

        print(f"    ↳ {tag}: {seg_agg['n_trips'].sum():.0f} trips")

    print(f"\n  {time_aggregation.capitalize()} slice files saved to: {slice_dir}\n")

# ---------------------------------------------------------------------------
# Spatial export
# ---------------------------------------------------------------------------

# Filename stem shared by every spatial-zoning output (.geojson, .shp, …).
# Used by :func:`_cleanup_raw_downloads` to protect these files from deletion.
SPATIAL_ZONING_STEM: str = "zoning"


def _export_spatial_zoning(parquet_path: str, output_dir: str) -> None:
    """Generate spatial files with one polygon per aggregated OD-matrix zone.

    The zoning parquet is dissolved on ``output_zone`` (the same label
    :func:`assign_output_zone` produces for the OD matrix), so the resulting
    polygons carry exactly the IDs used as rows/columns of the matrix:
    7-digit districts, 5-digit / ``_AM`` municipalities, ``PROV_XX`` provinces
    and the single ``REST_PENINSULA`` zone.

    Two files are written into *output_dir*:

    - ``od_zoning.geojson`` — reprojected to EPSG:4326 (WGS84 lon/lat) as the
      GeoJSON standard (RFC 7946) requires, so it lines up correctly in web
      mapping tools.
    - ``od_zoning.shp`` (+ ``.shx``/``.dbf``/``.prj``/``.cpg``) — kept in the
      parquet's native CRS for desktop GIS.
    """
    try:
        import geopandas as gpd
    except ImportError:
        print("  [warn] geopandas not installed — skipping spatial zoning export.")
        return

    print("\n  Generating spatial zoning files...")
    try:
        gdf = gpd.read_parquet(parquet_path)
    except Exception as e:
        print(f"  [warn] Failed to read spatial parquet {parquet_path}: {e}")
        return

    gdf["ID_norm"] = gdf["ID"].apply(normalise_parquet_id)
    gdf["output_zone"] = gdf.apply(assign_output_zone, axis=1)

    # Drop unmapped or non-spatial rows
    gdf = gdf.dropna(subset=["output_zone"])
    if gdf.empty:
        print("  [warn] No spatial features left after mapping — skipping spatial export.")
        return

    dissolved = gdf.dissolve(by="output_zone").reset_index()
    
    # Clean geometries: fix slivers and remove internal holes from imperfect dissolve
    from shapely.geometry import Polygon, MultiPolygon
    def remove_interior_rings(geom):
        if geom is None or geom.is_empty:
            return geom
        if isinstance(geom, Polygon):
            return Polygon(geom.exterior)
        elif isinstance(geom, MultiPolygon):
            return MultiPolygon([Polygon(p.exterior) for p in geom.geoms])
        return geom

    # buffer(0) resolves self-intersections and tiny slivers; then remove holes
    dissolved["geometry"] = dissolved["geometry"].apply(lambda g: g.buffer(0) if g is not None else g)
    dissolved["geometry"] = dissolved["geometry"].apply(remove_interior_rings)

    # Use a <=10-char field name so the ESRI Shapefile driver does not silently
    # truncate it (e.g. "output_zone" -> "output_zon"). Both outputs then share
    # the same attribute name carrying the OD-matrix zone IDs.
    dissolved = dissolved.rename(columns={"output_zone": "zone_id"})
    out_cols = ["zone_id", "geometry"]
    dissolved = dissolved[[c for c in out_cols if c in dissolved.columns]]

    out_dir = Path(output_dir)

    # --- Shapefile in the native (projected) CRS for desktop GIS -----------------
    shp_path = out_dir / f"{SPATIAL_ZONING_STEM}.shp"
    try:
        dissolved.to_file(shp_path, driver="ESRI Shapefile")
        print(f"  ↳ Spatial zoning (shapefile) saved to: {shp_path}")
    except Exception as e:
        print(f"  [warn] Failed to write spatial zoning shapefile: {e}")

    # --- GeoJSON reprojected to WGS84 (EPSG:4326) per the GeoJSON standard -------
    geojson_path = out_dir / f"{SPATIAL_ZONING_STEM}.geojson"
    try:
        geojson_gdf = dissolved
        if dissolved.crs is not None and dissolved.crs.to_epsg() != 4326:
            geojson_gdf = dissolved.to_crs(epsg=4326)
        geojson_gdf.to_file(geojson_path, driver="GeoJSON")
        print(f"  ↳ Spatial zoning (GeoJSON) saved to: {geojson_path}")
    except Exception as e:
        print(f"  [warn] Failed to write spatial zoning GeoJSON: {e}")

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_od_from_parquet(
    parquet_path: str,
    start_date: str,
    end_date: str,
    output_dir: str,
    version: int = 2,
    social_agg: bool = False,
    time_aggregation: str = "total",
    save_csv: bool = True,
    save_skim: bool = True,
    save_spatial: bool = True,
    include_international: bool = False,
    ac_provinces: Set[str] | None = None,
    project_municipality_ine: str = "14021",
    keep_activity: bool = False,
) -> pd.DataFrame:
    """Build a custom multi-scale OD matrix from a ``Zoning.parquet`` file.

    Returns a long-format DataFrame with columns
    ``origin_zone, dest_zone, n_trips, trip_km`` (plus ``activity_origin``,
    ``activity_destination`` when ``keep_activity=True``).

    When ``keep_activity=True`` and ``save_csv=True``, separate matrices are
    written under ``<output_dir>/activity/`` for frequent and non_frequent trips:

    - frequent    — origin AND destination in {home, other_frequent, work_or_study}
    - non_frequent — origin OR destination is other_non_frequent

    Parameters
    ----------
    keep_activity : bool, default False
        When True, pass ``keep_activity=True`` to pyspainmobility so that
        ``activity_origin`` / ``activity_destination`` are retained, then
        generate separate OD matrices for frequent and non_frequent trips.

    social_agg : bool, default False
        When True, pass ``social_agg=True`` to pyspainmobility so that socio-demographic segment columns are retained, 
        then generate separate OD matrices for each segment.
    
    time_aggregation : str, default "total"
        Level of temporal aggregation for the output OD matrix. One of:
        - "total" (default): aggregate all trips across the entire date range.
        - "daily": generate one OD matrix per day.
        - "hourly": generate one OD matrix per hour.
    
    ac_provinces : set[str] | None, default None
        2-digit INE province codes for the autonomous community of the project.
    project_municipality_ine : str, default ``"14021"``
        5-digit INE code of the project municipality (Córdoba by default).
    """
    if time_aggregation not in {"total", "daily", "hourly"}:
        raise ValueError(
            f"time_aggregation must be 'total', 'daily' or 'hourly'; got {time_aggregation!r}"
        )

    os.makedirs(output_dir, exist_ok=True)

    ctx = _prepare_pipeline(
        parquet_path=parquet_path,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        version=version,
        social_agg=social_agg,
        time_aggregation=time_aggregation,
        ac_provinces=ac_provinces,
        project_municipality_ine=project_municipality_ine,
        keep_activity=keep_activity,
    )

    df_dist_mapped, unmapped_a_o, unmapped_a_d = _map_and_clean(
        ctx["df_dist"], ctx["dist_lookup"], "Download A", include_international
    )
    df_muni_mapped, unmapped_b_o, unmapped_b_d = _map_and_clean(
        ctx["df_muni"], ctx["muni_lookup"], "Download B", include_international
    )

    od_long = _aggregate(df_dist_mapped, df_muni_mapped, time_aggregation)

    # Build the canonical wide pivot (always "total" for the CSV output)
    od_long_total = (
        od_long.groupby(["origin_zone", "dest_zone"], as_index=False)
        .agg(n_trips=("n_trips", "sum"), trip_km=("trip_km", "sum"))
    )

    zone_order_full = _ordered_zones(
        ctx["distritos_ids"], ctx["municipios_ids"], ctx["provincia_zones"], include_international
    )

    od_pivot = (
        od_long_total
        .pivot(index="origin_zone", columns="dest_zone", values="n_trips")
        .reindex(index=zone_order_full, columns=zone_order_full, fill_value=0.0)
        .fillna(0.0)
    )
    od_pivot.index.name = "origin\\destination"

    missing_distr = set(ctx["distritos_ids"]) - set(od_pivot.index)
    if missing_distr:
        print(f"WARNING: Missing DistrCercanos zones in output: {sorted(missing_distr)}")

    skim_pivot = None
    if save_skim:
        skim_pivot = _build_distance_skim(od_long_total, zone_order_full)

    if save_csv:
        _write_outputs(
            output_dir=output_dir,
            start_date=start_date,
            end_date=end_date,
            od_long=od_long_total,
            od_pivot=od_pivot,
            zone_order_full=zone_order_full,
            ctx=ctx,
            unmapped_counts=(unmapped_a_o, unmapped_a_d, unmapped_b_o, unmapped_b_d),
            skim_pivot=skim_pivot,
        )

        if save_spatial:
            _export_spatial_zoning(parquet_path, output_dir)

        # Per-day / per-hour matrices when time_aggregation is daily or hourly.
        if time_aggregation in {"daily", "hourly"}:
            time_col = "date" if time_aggregation == "daily" else "hour"
            if time_col in od_long.columns:
                _write_time_slice_outputs(
                    od_long=od_long,
                    output_dir=output_dir,
                    time_aggregation=time_aggregation,
                    zone_order_full=zone_order_full,
                )

        # Activity-split matrices when keep_activity is enabled.
        if keep_activity:
            activity_present = [c for c in ACTIVITY_COLS if c in od_long.columns]
            if activity_present:
                _write_activity_outputs(
                    od_long=od_long,
                    output_dir=output_dir,
                    start_date=start_date,
                    end_date=end_date,
                    zone_order_full=zone_order_full,
                )

        # Per-segment socio-demographic matrices when social_agg is enabled.
        socio_present = [c for c in SOCIO_COLS if c in od_long.columns]
        if social_agg and socio_present:
            _write_social_outputs(
                od_long=od_long,
                output_dir=output_dir,
                start_date=start_date,
                end_date=end_date,
                zone_order_full=zone_order_full,
            )

    _cleanup_raw_downloads(output_dir)

    return od_long

# ---------------------------------------------------------------------------
# Per-segment socio-demographic output (social_agg=True)
# ---------------------------------------------------------------------------

def _sanitise_segment_label(value: str) -> str:
    """Turn a raw MITMA segment value into a filesystem-safe label."""
    s = str(value).strip()
    s = s.replace("<", "-").replace(">", "+")
    s = s.replace(" to ", "-")
    s = "".join(c for c in s if c.isalnum() or c in ("-"))
    return s or "unknown"

def _write_social_outputs(
    od_long: pd.DataFrame,
    output_dir: str,
    start_date: str,
    end_date: str,
    zone_order_full: list[str],
) -> None:
    """Write per-segment OD matrices for each socio-demographic dimension."""
    social_dir = Path(output_dir) / "social"
    social_dir.mkdir(parents=True, exist_ok=True)

    tag = start_date if start_date == end_date else f"{start_date}_{end_date}"

    socio_present = [c for c in SOCIO_COLS if c in od_long.columns]
    if not socio_present:
        print("  [warn] No socio-demographic columns found in od_long — skipping social outputs.")
        return

    # Normalize socio cols to plain Python object dtype so that comparison
    # works correctly regardless of whether pyspainmobility returned Arrow-backed
    # or nullable-string types.
    od_work = od_long.copy()
    for col in socio_present:
        od_work[col] = od_work[col].astype(object)

    for dim in socio_present:
        unique_vals = [v for v in od_work[dim].dropna().unique() if v is not None]
        print(f"\n  Writing {len(unique_vals)} segment(s) for '{dim}' …")
        if not unique_vals:
            print(f"  [warn] No non-null values found for '{dim}', skipping.")
            continue

        for val in sorted(unique_vals, key=str):
            label = _sanitise_segment_label(val)
            seg = od_work[od_work[dim] == val]

            seg_agg = (
                seg.groupby(["origin_zone", "dest_zone"], as_index=False)
                .agg(n_trips=("n_trips", "sum"), trip_km=("trip_km", "sum"))
            )

            _write_matrix_and_flows(
                seg_agg, zone_order_full,
                pivot_path=social_dir / f"od_matrix_{dim}_{label}.csv",
                flows_path=social_dir / f"od_flows_{dim}_{label}.csv",
                header_comment=(
                    f"Segment: {dim} = {val}  |  "
                    f"REST_PENINSULA x REST_PENINSULA not extracted"
                ),
            )

            print(f"    ↳ {dim}={val}  →  {seg_agg['n_trips'].sum():.0f} trips")

    print(f"\n  Social segment files saved to: {social_dir}\n")

# ---------------------------------------------------------------------------
# CSV output writers (§7.4)
# ---------------------------------------------------------------------------

def _write_outputs(
    output_dir: str,
    start_date: str,
    end_date: str,
    od_long: pd.DataFrame,
    od_pivot: pd.DataFrame,
    zone_order_full: list[str],
    ctx: dict,
    unmapped_counts: Tuple[int, int, int, int],
    skim_pivot: pd.DataFrame | None = None,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if start_date == end_date:
        tag = f"{start_date}"
    else:
        tag = f"{start_date}_{end_date}"

    # --- Pivot (wide) + Long-format flows -----------------------------------------
    _write_matrix_and_flows(
        od_long, zone_order_full,
        pivot_path=out_dir / "od_matrix.csv",
        flows_path=out_dir / "od_flows.csv",
        header_comment=(
            "REST_PENINSULA x REST_PENINSULA not extracted "
            "(internal flows outside study area)"
        ),
    )

    # --- Distance skim (§7.7) -------------------------------------------------------
    skim_filename: str | None = None
    if skim_pivot is not None:
        skim_filename = "skim_distance.csv"
        skim_path = out_dir / skim_filename
        with open(skim_path, "w", encoding="utf-8") as f:
            f.write(
                "# Average trip distance (km) per zone pair"
                " — REST_PENINSULA x REST_PENINSULA not extracted\n"
            )
            skim_pivot.round(2).to_csv(f, float_format="%.2f", na_rep="")

    # --- Metadata sidecar -----------------------------------------------------------
    grand_total = float(od_long["n_trips"].sum())
    len_a = len(ctx["df_dist"])
    len_b = len(ctx["df_muni"])
    unmapped_total = sum(unmapped_counts)
    meta_path = out_dir / "od_metadata.txt"
    zones_present = int(len(od_pivot.index))
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("OD Matrix — pySpainMobility extraction\n")
        f.write("=======================================\n")
        f.write(f"Parquet file      : {Path(ctx['parquet_path']).name}\n")
        f.write(f"Date range        : {start_date} to {end_date}\n")
        f.write(f"MITMA version     : 2\n")
        intl_str = " + 1 INTERNATIONAL" if "INTERNATIONAL" in od_pivot.index else ""
        f.write(
            f"Zones             : {zones_present} "
            f"({len(ctx['distritos_ids'])} DistrCercanos + "
            f"{len(ctx['municipios_ids'])} MuniColind + "
            f"{len(ctx['provincia_zones'])} PROV_XX + 1 REST_PENINSULA{intl_str})\n"
        )
        f.write(f"Total trips       : {grand_total:.0f}\n")
        f.write(f"Download A rows   : {len_a:,} (district-level, before mapping)\n")
        f.write(f"Download B rows   : {len_b:,} (municipality-level, before mapping)\n")
        f.write(f"Unmapped rows     : {unmapped_total:,}\n")
        if skim_filename is not None:
            f.write(
                f"Skim distance     : {skim_filename}  "
                f"(avg km, NaN = no observed trips)\n"
            )
        f.write(f"Generated         : {datetime.now().isoformat()}\n")

def _cleanup_raw_downloads(output_dir: str) -> None:
    """Remove raw files downloaded by pyspainmobility to save disk space."""
    out_dir = Path(output_dir)
    if not out_dir.exists():
        return

    print("\nCleaning up raw download files to save disk space...")
    extensions_to_delete = [
        "*.gz", "*.shp", "*.shx", "*.dbf", "*.prj", "*.cpg", "*.qpj"
    ]
    for ext in extensions_to_delete:
        for f in out_dir.glob(ext):
            # Never delete the spatial zoning outputs (od_zoning.shp/.shx/.dbf/.prj/…)
            if f.stem == SPATIAL_ZONING_STEM:
                continue
            try:
                f.unlink(missing_ok=True)
            except Exception as e:
                print(f"  ↳ Could not delete {f.name}: {e}")

# ---------------------------------------------------------------------------
# Helpers used by the integration test suite
# ---------------------------------------------------------------------------

def _get_raw_filtered_totals(
    parquet_path: str,
    start_date: str,
    end_date: str,
    output_dir: str,
    version: int = 2,
    social_agg: bool = False,
    ac_provinces: Set[str] | None = None,
    project_municipality_ine: str = "14021",
    keep_activity: bool = False,
) -> Tuple[pd.DataFrame, float, float, float]:
    """Return ``(df_long, raw_total_a, raw_total_b, excluded_b)``."""
    ctx = _prepare_pipeline(
        parquet_path=parquet_path,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        version=version,
        social_agg=social_agg,
        ac_provinces=ac_provinces,
        project_municipality_ine=project_municipality_ine,
        keep_activity=keep_activity,
    )
    raw_total_a = float(ctx["df_dist"]["n_trips"].sum())
    raw_total_b = float(ctx["df_muni"]["n_trips"].sum())
    excluded_b = float(ctx["excluded_b_trips"])

    df_dist_mapped, *_ = _map_and_clean(ctx["df_dist"], ctx["dist_lookup"], "Download A")
    df_muni_mapped, *_ = _map_and_clean(ctx["df_muni"], ctx["muni_lookup"], "Download B")
    df_long = _aggregate(df_dist_mapped, df_muni_mapped, "total")

    return df_long, raw_total_a, raw_total_b, excluded_b

def _get_raw_district_origin_totals(
    parquet_path: str,
    start_date: str,
    end_date: str,
    output_dir: str,
    version: int = 2,
    social_agg: bool = False,
    ac_provinces: Set[str] | None = None,
    project_municipality_ine: str = "14021",
    keep_activity: bool = False,
) -> Dict[str, float]:
    """Return ``{district_mitma_id: total_n_trips}`` for all DistrCercanos districts."""
    ctx = _prepare_pipeline(
        parquet_path=parquet_path,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        version=version,
        social_agg=social_agg,
        ac_provinces=ac_provinces,
        project_municipality_ine=project_municipality_ine,
        keep_activity=keep_activity,
    )
    df_dist = ctx["df_dist"]
    distr_cercanos_ids = ctx["distritos_ids"]

    origins = df_dist[df_dist["id_origin"].isin(distr_cercanos_ids)]
    totals = origins.groupby("id_origin")["n_trips"].sum().to_dict()
    for d in distr_cercanos_ids:
        totals.setdefault(d, 0.0)
    return {str(k): float(v) for k, v in totals.items()}

# ---------------------------------------------------------------------------
# Command-line helpers
# ---------------------------------------------------------------------------

_ANDALUCIA_PROVINCES_DEFAULT: Set[str] = {
    "04", "11", "14", "18", "21", "23", "29", "41"
}

_V1_START = "2020-02-14"
_V1_END   = "2021-05-09"
_V2_START = "2022-01-01"

def _auto_version(start_date: str) -> int:
    """Pick the MITMA dataset version that covers ``start_date``."""
    if _V1_START <= start_date <= _V1_END:
        return 1
    if start_date >= _V2_START:
        return 2
    raise ValueError(
        f"No MITMA dataset covers {start_date!r}: v1 ends {_V1_END}, "
        f"v2 starts {_V2_START}. Pick a date inside either window."
    )

def _prompt(msg: str, default: str | None = None,
            validator=None, transform=None):
    """Prompt the user until they provide valid input."""
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{msg}{suffix}: ").strip()
        if not raw and default is not None:
            raw = default
        if not raw:
            print("  ↳ value required, please try again.")
            continue
        if validator is not None:
            ok, err = validator(raw)
            if not ok:
                print(f"  ↳ {err}")
                continue
        return transform(raw) if transform is not None else raw

def _validate_parquet(path: str):
    p = Path(path).expanduser()
    if not p.exists():
        return False, f"file not found: {p}"
    if not p.is_file():
        return False, f"not a regular file: {p}"
    if p.suffix.lower() != ".parquet":
        return False, f"expected a .parquet file, got {p.suffix!r}"
    return True, None

def _validate_date(s: str):
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return True, None
    except ValueError:
        return False, f"expected YYYY-MM-DD, got {s!r}"

def _validate_choice(choices: set[str], label: str = ""):
    """Return a validator that checks membership in a set of choices (case-insensitive)."""
    def _check(s: str):
        if s.lower() in choices:
            return True, None
        hint = label or ", ".join(sorted(choices))
        return False, f"choose one of: {hint}; got {s!r}"
    return _check

_validate_yn  = _validate_choice({"y", "yes", "n", "no"}, "y / n")
_validate_agg = _validate_choice({"total", "daily", "hourly"}, "total, daily, hourly")

def _yn_to_bool(s: str) -> bool:
    return s.lower() in {"y", "yes"}

def _validate_muni_ine(s: str):
    if len(s) == 5 and s.isdigit():
        return True, None
    return False, f"5-digit INE code expected (e.g. 14021), got {s!r}"

def _validate_province_list(s: str):
    codes = [c.strip() for c in s.replace(",", " ").split()]
    if not codes:
        return False, "at least one 2-digit province code required"
    for c in codes:
        if len(c) != 2 or not c.isdigit():
            return False, f"expected 2-digit codes, got {c!r}"
    return True, None

def _parse_province_list(s: str) -> Set[str]:
    return {c.strip() for c in s.replace(",", " ").split() if c.strip()}

def _run_interactive() -> dict:
    """Gather all parameters needed by :func:`build_od_from_parquet`."""
    print()
    print("=" * 66)
    print("OD project matrix v3 — interactive setup")
    print("=" * 66)
    print("Press Enter to accept the default shown in brackets.\n")

    parquet_path = _prompt(
        "Zoning parquet file (path)",
        validator=_validate_parquet,
        transform=lambda s: str(Path(s).expanduser().resolve()),
    )

    print("\nMITMA Data Versions by Date:")
    print("  - Version 1: 2020-02-14 to 2021-05-09 (COVID period)")
    print("  - Version 2: 2022-01-01 onwards")

    while True:
        start_date = _prompt("Start date (YYYY-MM-DD)",
                             default=DEFAULT_START_DATE,
                             validator=_validate_date)
        end_date = _prompt("End date   (YYYY-MM-DD)",
                           default=start_date,
                           validator=_validate_date)
        if end_date < start_date:
            print(f"  ↳ end-date {end_date} is before start-date {start_date}; "
                  "swapping them.")
            start_date, end_date = end_date, start_date

        try:
            version = _auto_version(start_date)
            print(f"  ↳ Auto-selected MITMA dataset version {version} for {start_date}.\n")
            break
        except ValueError as exc:
            print(f"  ↳ Error: {exc}")
            print("  ↳ Please select a date range covered by MITMA.\n")

    social_agg = _yn_to_bool(_prompt(
        "Break trips down by income / age / gender? (y/N)",
        default="n",
        validator=_validate_yn,
    ))

    keep_activity = _yn_to_bool(_prompt(
        "Generate separate frequent / non_frequent matrices by activity? (y/N)",
        default="n",
        validator=_validate_yn,
    ))

    include_international = _yn_to_bool(_prompt(
        "Include international flows (e.g. France/Portugal) in an 'INTERNATIONAL' zone? (Y/n)",
        default="y",
        validator=_validate_yn,
    ))

    time_aggregation = _prompt(
        "Time aggregation (total / daily / hourly)",
        default="total",
        validator=_validate_agg,
        transform=str.lower,
    )

    save_skim = _yn_to_bool(_prompt(
        "Also generate the distance skim CSV? (Y/n)",
        default="y",
        validator=_validate_yn,
    ))

    save_spatial = _yn_to_bool(_prompt(
        "Generate aggregated spatial zoning (GeoJSON)? (Y/n)",
        default="y",
        validator=_validate_yn,
    ))

    if start_date == end_date:
        default_out_dir = str(Path(__file__).parent.resolve() / f"Results/{start_date}")
    else:
        default_out_dir = str(Path(__file__).parent.resolve() / f"Results/{start_date}_{end_date}")

    output_dir = _prompt(
        "Output directory",
        default=default_out_dir,
        transform=lambda s: str(Path(s).expanduser()),
    )

    # Summary.
    print()
    print("-" * 66)
    print(f"  Parquet              : {parquet_path}")
    print(f"  Dates                : {start_date} → {end_date}")
    print(f"  MITMA version        : {version}")
    print(f"  Social aggregation   : {'yes' if social_agg else 'no'}")
    print(f"  Activity split       : {'yes' if keep_activity else 'no'}")
    print(f"  Intl. zone           : {'yes' if include_international else 'no'}")
    print(f"  Time aggregation     : {time_aggregation}")
    print(f"  Distance skim        : {'yes' if save_skim else 'no'}")
    print(f"  Spatial zoning       : {'yes' if save_spatial else 'no'}")
    print(f"  Output directory     : {output_dir}")
    print("-" * 66)
    go = _yn_to_bool(_prompt("Proceed? (Y/n)", default="y",
                             validator=_validate_yn))
    if not go:
        raise SystemExit("Aborted by user.")

    return dict(
        parquet_path=parquet_path,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        version=version,
        social_agg=social_agg,
        keep_activity=keep_activity,
        time_aggregation=time_aggregation,
        save_csv=True,
        save_skim=save_skim,
        save_spatial=save_spatial,
        include_international=include_international,
    )

if __name__ == "__main__":  # pragma: no cover
    import argparse
    import sys

    interactive_flags = {"-i", "--interactive"}
    no_cli_args = len(sys.argv) == 1
    explicit_interactive = any(f in sys.argv for f in interactive_flags)

    if no_cli_args or explicit_interactive:
        kwargs = _run_interactive()
        df = build_od_from_parquet(**kwargs)
        print(df.head())
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("-i", "--interactive", action="store_true",
                            help="Run with interactive prompts instead of flags.")
        parser.add_argument("--parquet", default="Zoning.parquet")
        parser.add_argument("--start-date", default=DEFAULT_START_DATE)
        parser.add_argument("--end-date", default=DEFAULT_END_DATE)
        parser.add_argument("--output-dir", default="./output")
        parser.add_argument("--version", type=int, default=2)
        parser.add_argument("--social-agg", action="store_true")
        parser.add_argument("--keep-activity", action="store_true",
                            help="Generate separate frequent/non_frequent OD matrices.")
        parser.add_argument(
            "--time-aggregation",
            choices=["total", "daily", "hourly"],
            default="total",
        )
        parser.add_argument(
            "--no-save", action="store_true", help="Skip saving the output CSV/Parquet."
        )
        parser.add_argument(
            "--no-skim", action="store_true", help="Skip generating the distance skim matrix."
        )
        parser.add_argument(
            "--no-spatial", action="store_true", help="Skip generating the spatial zoning file."
        )
        parser.add_argument(
            "--exclude-international", action="store_true",
            help="Exclude international flows."
        )
        parser.add_argument(
            "--ac-provinces",
            nargs="+",
            default=None,
            metavar="PROV",
            help=(
                "2-digit INE province codes for the project's autonomous community "
                "(e.g. --ac-provinces 04 11 14 18 21 23 29 41)."
            ),
        )
        parser.add_argument(
            "--project-municipality",
            default="14021",
            help="5-digit INE code of the project municipality (default: 14021 — Córdoba).",
        )
        args = parser.parse_args()

        df = build_od_from_parquet(
            parquet_path=args.parquet,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir,
            version=args.version,
            social_agg=args.social_agg,
            keep_activity=args.keep_activity,
            time_aggregation=args.time_aggregation,
            save_csv=not args.no_save,
            save_skim=not args.no_skim,
            save_spatial=not args.no_spatial,
            include_international=not args.exclude_international,
            ac_provinces=set(args.ac_provinces) if args.ac_provinces else None,
            project_municipality_ine=args.project_municipality,
        )
        print(df.head())
