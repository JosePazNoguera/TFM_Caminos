# TFM_Caminos

Project workspace for road-corridor mobility analysis, including MITMA
origin-destination matrices, corridor zoning, and interactive maps.

## Environment

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## N-211 trial workflow

The N-211 corridor, administrative polygons, and 2025 arterial-road network
are stored locally under `Data/.gis/`. Generate a first-pass zoning from the
repository root:

```powershell
python Scripts\generate_zoning.py `
  --corridor "Data/.gis/N-211 p.k. 0 - p.k. 10.kmz" `
  --roads "Data/.gis/Mapa de Tráfico/si_tramotrafimd2025_l.shp" `
  --districts "Data/.gis/zonificacion_distritos.shp" `
  --municipalities "Data/.gis/zonificacion_municipios.shp" `
  --output "Data/Zoning_N211.parquet"
```

The generator retains the most-disaggregated Spanish mainland administrative
zones touched by the corridor (districts where available, otherwise
municipalities). Municipalities sharing a boundary with the corridor's
municipality are also retained individually. The remaining mainland
municipalities are dissolved into up to eight directional petals based on their
nearest arterial-road locations viewed from the corridor midpoint. The default
maximum is 19 zones, allowing the current N-211 trial to contain one core
municipality (`19086_AM`), its nine adjoining municipalities, and eight outer
road petals. Road-network distances connect the outer municipalities to the
N-211 arterial network; the catchment extends 1,100 km and roads are snapped
within 2 km of the central point. Portugal, the Balearic and Canary Islands,
Ceuta, and Melilla are excluded because this is a peninsular Spain analysis.
Network distances are not travel-time estimates. The road layer is a 2025
network, not a date-specific 2024 network.

The generator writes a GeoParquet zoning file, a GeoJSON copy, and companion
`*_corridor.geojson` and `*_arterial_roads.geojson` layers. It produces grouped
access-zone IDs; the OD builder resolves those member IDs against the MITMA
relation table and aggregates them into the generated petals. `Data/` and
`Results/` contain local input/output data and are intentionally Git-ignored.

For the first OD trial, use one day and leave social and activity disaggregation
disabled:

```powershell
python Scripts\od_matrix_v3.py `
  --parquet "Data/Zoning_N211.parquet" `
  --start-date 2024-03-14 `
  --end-date 2024-03-14 `
  --output-dir "$PWD\Results\2024-03-14" `
  --project-municipality 19086 `
  --no-skim `
  --no-spatial
```

Use an absolute output path: `pyspainmobility` resolves relative download-cache
paths under the user's home directory. This command writes only the total OD
matrix, flow list, and metadata; it does not request income, age, gender,
activity splits, daily/hourly slices, a distance skim, or a second spatial
zoning export. MITMA data and zone-relation downloads require internet access.

Generate a zoning-only preview (no OD matrix required) with:

```powershell
python -c "from Scripts.generate_visual import generate_zoning_preview; generate_zoning_preview('Data/Zoning_N211.parquet')"
```

The preview is `Data/Zoning_N211_preview.html`. It shows the macro-zones,
corridor, and arterial roads. Once OD flows have been generated, the
aggregated matrix visual can be generated from the project root with:

```powershell
python -c "from Scripts.generate_visual import generate_aggregated_map; generate_aggregated_map()"
```

When prompted, enter `Data/Zoning_N211.parquet` and `2024-03-14`. The visual
opens centered on the corridor-touching core zone with the N-211 corridor in
red and the arterial road network shown as a subdued line overlay. The outer
petals use directional colors from the N-211 midpoint; the core and its
adjoining municipalities remain individually selectable. Selecting a zone
shows every available destination or origin flow, with arc endpoints placed at
the equal-area centroid of each zone. Before rendering, the visual dissolves
input geometries by their OD matrix zone IDs and disables the cluttering
multipart-edge outlines.

### March 14, 2024 trial

The Spain-wide trial generates 18 spatial zones: core municipality `19086_AM`,
nine adjoining municipalities represented individually, and eight
arterial-direction petals covering the remaining Spanish mainland
municipalities. The adjacent municipality IDs are `19053_AM`, `19103_AM`,
`19190`, `19245_AM`, `19257_AM`, `42025_AM`, `42035_AM`, `50034_AM`, and
`50159_AM`. The matrix builder maps each source unit through the MITMA zone
relation; for this run the metadata reports zero unmapped rows.
Social, activity, time-slice, and skim outputs are not generated.

### Internal trips and matrix diagonals

An OD matrix diagonal cell (origin zone equals destination zone) represents
trips whose origin and destination are both within that same aggregated zone.
This includes trips internal to an individually retained municipality and
trips between municipalities grouped into the same outer petal. The matrix
builder retains these diagonal values for project zones. It deliberately sets
the `REST_PENINSULA`-to-`REST_PENINSULA` cell to zero because that zone is not
extracted as a project area. The interactive map omits all same-zone flows from
its arcs, so internal trips remain counted in the matrix but are not drawn as
self-loop lines.
