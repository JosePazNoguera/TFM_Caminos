import pandas as pd
import geopandas as gpd
import json
import numpy as np
import hashlib
import os
import glob
import math
from pathlib import Path

from Scripts.od_matrix_v3 import assign_output_zone, normalise_parquet_id


def _build_html(
    data,
    geojson_str,
    date="",
    corridor_geojson=None,
    initial_zone=None,
    map_center=(-3.0, 40.0),
    map_zoom=10,
    arterial_roads_geojson=None,
):
    """Build the deck.gl HTML visualization from arc data and basemap GeoJSON."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8" />
        <title>Interactive OD Matrix Visualization – {date}</title>
        <script src="https://unpkg.com/deck.gl@latest/dist.min.js"></script>
        <script src="https://api.tiles.mapbox.com/mapbox-gl-js/v1.13.0/mapbox-gl.js"></script>
        <link href="https://api.tiles.mapbox.com/mapbox-gl-js/v1.13.0/mapbox-gl.css" rel="stylesheet" />
        <style>
          body {{ margin: 0; padding: 0; background-color: #111; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }}
          #map {{ width: 100vw; height: 100vh; position: absolute; }}
          #control-panel {{
            position: absolute; top: 20px; left: 20px; z-index: 1000;
            background: rgba(30, 30, 30, 0.85); color: #fff; padding: 20px;
            border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); max-width: 300px; backdrop-filter: blur(5px);
          }}
          h1 {{ margin-top: 0; font-size: 1.2rem; }}
          p {{ font-size: 0.9rem; color: #ccc; }}
          button {{ width: 100%; margin-top: 5px; border-radius: 4px; font-weight: bold; }}
        </style>
      </head>
      <body>
        <div id="control-panel">
          <h1>Interactive OD Matrix <br> Dates: {date}</h1>
          <p>Click on any zone to see flows.</p>
          <div style="margin: 15px 0;">
            <button id="btn-origin" style="background:#4CAF50; color:white; padding:8px; border:none; cursor:pointer;">Show Destinations</button>
            <button id="btn-dest" style="background:#555; color:white; padding:8px; border:none; cursor:pointer;">Show Origins</button>
          </div>
          <p style="margin-bottom:4px;"><span style="color:#d2dee4;">━</span> Arterial road network</p>
          <p style="margin-bottom:4px;">Arterial-road petals</p>
          <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:4px; font-size:0.75rem;">
            <span style="color:#4d9bd1;">N</span><span style="color:#72ad64;">NE</span>
            <span style="color:#d39c4b;">E</span><span style="color:#c96e62;">SE</span>
            <span style="color:#c65373;">S</span><span style="color:#a25dba;">SW</span>
            <span style="color:#637fc0;">W</span><span style="color:#4f9b9a;">NW</span>
          </div>
          <p id="status-text" style="color:#ffcc00; font-weight:bold;">Select a zone...</p>
        </div>
        <div id="map"></div>
        <div id="tooltip" style="position:fixed;z-index:10000;pointer-events:none;display:none;background:rgba(20,20,20,0.95);color:#fff;padding:8px 14px;border-radius:6px;font-size:13px;line-height:1.4;box-shadow:0 2px 10px rgba(0,0,0,0.6);border:1px solid rgba(255,255,255,0.15);white-space:nowrap;font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;"></div>

        <script>
          const allData = {json.dumps(data)};
          const basemapGeojson = {geojson_str};
          const corridorGeojson = {corridor_geojson or '{"type":"FeatureCollection","features":[]}'};
          const arterialRoadsGeojson = {arterial_roads_geojson or '{"type":"FeatureCollection","features":[]}'}; 
          
          let mode = 'origin'; // 'origin' or 'destination'
          let selectedZone = {json.dumps(initial_zone)};
          const petalColors = {{
            N: [77, 155, 209, 155],
            NE: [114, 173, 100, 155],
            E: [211, 156, 75, 155],
            SE: [201, 110, 98, 155],
            S: [198, 83, 115, 155],
            SW: [162, 93, 186, 155],
            W: [99, 127, 192, 155],
            NW: [79, 155, 154, 155]
          }};

          function getZoneFillColor(feature) {{
            if (feature.properties.ID === selectedZone) return [255, 200, 0, 190];
            const match = feature.properties.ID.match(/_PETAL_(N|NE|E|SE|S|SW|W|NW)$/);
            return match ? petalColors[match[1]] : [60, 160, 180, 150];
          }}
          
          document.getElementById('btn-origin').onclick = () => {{
            mode = 'origin';
            document.getElementById('btn-origin').style.background = '#4CAF50';
            document.getElementById('btn-dest').style.background = '#555';
            render();
          }};
          
          document.getElementById('btn-dest').onclick = () => {{
            mode = 'destination';
            document.getElementById('btn-origin').style.background = '#555';
            document.getElementById('btn-dest').style.background = '#4CAF50';
            render();
          }};
          
          const deckgl = new deck.DeckGL({{
            container: 'map',
            mapStyle: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
            initialViewState: {{
              longitude: {map_center[0]},
              latitude: {map_center[1]},
              zoom: {map_zoom},
              pitch: 0,
              bearing: 0
            }},
            controller: true,
            layers: []
          }});

          // Manual tooltip driven by onHover (more reliable than getTooltip)
          function showArcTooltip(info) {{
            const tooltip = document.getElementById('tooltip');
            if (!info || !info.object) {{
              tooltip.style.display = 'none';
              return;
            }}
            const o = info.object;
            const label = mode === 'origin'
              ? `${{o.origin_id}} → ${{o.dest_id}}`
              : `${{o.dest_id}} → ${{o.origin_id}}`;
            const trips = Math.round(o.volume).toLocaleString('en-US');
            tooltip.innerHTML =
              `<div style="font-weight:bold;">${{label}}</div>` +
              `<div style="color:#ffcc00;">${{trips}} trips</div>`;
            tooltip.style.left = (info.x + 14) + 'px';
            tooltip.style.top  = (info.y + 14) + 'px';
            tooltip.style.display = 'block';
          }}

          function hideTooltip() {{
            document.getElementById('tooltip').style.display = 'none';
          }}

          function render() {{
            let displayData = [];
            let status = 'Select a zone to see flows.';
            
            if (selectedZone) {{
              if (mode === 'origin') {{
                displayData = allData.filter(d => d.origin_id === selectedZone);
                status = `All destinations from ${{selectedZone}} (${{displayData.length}})`;
              }} else {{
                displayData = allData.filter(d => d.dest_id === selectedZone);
                status = `All origins to ${{selectedZone}} (${{displayData.length}})`;
              }}
            }}
            document.getElementById('status-text').innerText = status;
            
            const maxVolume = displayData.length > 0 ? Math.max(...displayData.map(d => d.volume)) : 1;

            const layers = [
              new deck.GeoJsonLayer({{
                id: 'geojson-layer',
                data: basemapGeojson,
                pickable: true,
                stroked: false,
                filled: true,
                extruded: false,
                getFillColor: getZoneFillColor,
                onClick: info => {{
                  if (info.object) {{
                    selectedZone = info.object.properties.ID;
                    render();
                  }}
                }}
              }}),
              new deck.GeoJsonLayer({{
                id: 'arterial-roads-layer',
                data: arterialRoadsGeojson,
                pickable: false,
                stroked: true,
                filled: false,
                lineWidthMinPixels: 0.8,
                getLineColor: [210, 222, 228, 190]
              }}),
              new deck.GeoJsonLayer({{
                id: 'corridor-layer',
                data: corridorGeojson,
                pickable: false,
                stroked: true,
                filled: false,
                lineWidthMinPixels: 5,
                getLineColor: [255, 55, 55, 255]
              }}),
              new deck.ArcLayer({{
                id: 'arc-layer',
                data: displayData,
                getSourcePosition: d => d.source,
                getTargetPosition: d => d.target,
                getSourceColor: d => d.color,
                getTargetColor: d => d.color.map(c => Math.max(0, c - 50)),
                getWidth: d => Math.max(2, (d.volume / maxVolume) * 12),
                pickable: true,
                autoHighlight: true,
                onHover: info => showArcTooltip(info)
              }}),
              new deck.TextLayer({{
                id: 'arrow-layer',
                data: displayData,
                // Place arrow at 80% towards the target so it's clearly indicating direction and close to ground
                getPosition: d => [
                  d.source[0] * 0.2 + d.target[0] * 0.8,
                  d.source[1] * 0.2 + d.target[1] * 0.8
                ],
                getText: d => '➤',
                getSize: 40,
                getColor: [255, 255, 255],
                getAngle: d => {{
                  const dx = d.target[0] - d.source[0];
                  const dy = d.target[1] - d.source[1];
                  const angle = Math.atan2(dy, dx) * 180 / Math.PI;
                  return -angle;
                }},
                getAlignmentBaseline: 'center',
                getTextAnchor: 'middle'
              }})
            ];
            
            deckgl.setProps({{ layers }});
          }}
          
          // Initial render
          render();
        </script>
      </body>
    </html>
    """
    return html_content


def _build_zoning_preview_html(
    zones_geojson, labels, center, zoom, roads_geojson, corridor_geojson
):
    return f"""
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8" />
        <title>Corridor zoning preview</title>
        <script src="https://unpkg.com/deck.gl@latest/dist.min.js"></script>
        <script src="https://api.tiles.mapbox.com/mapbox-gl-js/v1.13.0/mapbox-gl.js"></script>
        <link href="https://api.tiles.mapbox.com/mapbox-gl-js/v1.13.0/mapbox-gl.css" rel="stylesheet" />
        <style>
          body {{ margin: 0; background: #111; font-family: Arial, sans-serif; }}
          #map {{ width: 100vw; height: 100vh; }}
          #panel {{
            position: absolute; z-index: 10; top: 16px; left: 16px;
            max-width: 340px; padding: 14px 18px; border-radius: 8px;
            color: #fff; background: rgba(25, 25, 25, 0.9);
          }}
          #tooltip {{
            display: none; position: fixed; z-index: 20; pointer-events: none;
            padding: 8px 12px; border-radius: 5px; color: #fff;
            background: rgba(20, 20, 20, 0.95); white-space: pre-line;
          }}
        </style>
      </head>
      <body>
        <div id="panel">
          <strong>Corridor zoning proposal</strong>
          <p>Click a zone for its ID and source administrative areas. Drag and zoom to explore.</p>
          <p><span style="color:#f5a52d">■</span> Corridor-touching core
             &nbsp; <span style="color:#4691d2">■</span> Radial arterial petal
             &nbsp; <span style="color:#ff4d4d">━</span> Corridor</p>
          <div id="status">Select a zone.</div>
        </div>
        <div id="tooltip"></div>
        <div id="map"></div>
        <script>
          const zones = {zones_geojson};
          const labels = {labels};
          const arterialRoads = {roads_geojson};
          const corridor = {corridor_geojson};
          const tooltip = document.getElementById('tooltip');
          const deckgl = new deck.DeckGL({{
            container: 'map',
            mapStyle: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
            initialViewState: {{
              longitude: {center[0]},
              latitude: {center[1]},
              zoom: {zoom},
              pitch: 0,
              bearing: 0
            }},
            controller: true,
            layers: [
              new deck.GeoJsonLayer({{
                id: 'zoning',
                data: zones,
                pickable: true,
                stroked: true,
                filled: true,
                lineWidthMinPixels: 1.5,
                getFillColor: f => {{
                  if (f.properties.layer !== '3.Access') return [245, 165, 45, 175];
                  const match = f.properties.ID.match(/_PETAL_(N|NE|E|SE|S|SW|W|NW)$/);
                  const colors = {{
                    N: [77, 155, 209, 130], NE: [114, 173, 100, 130],
                    E: [211, 156, 75, 130], SE: [201, 110, 98, 130],
                    S: [198, 83, 115, 130], SW: [162, 93, 186, 130],
                    W: [99, 127, 192, 130], NW: [79, 155, 154, 130]
                  }};
                  return match ? colors[match[1]] : [70, 145, 210, 95];
                }},
                getLineColor: [235, 240, 250, 230],
                onClick: info => {{
                  const p = info.object && info.object.properties;
                  document.getElementById('status').textContent = p
                    ? `${{p.ID}} — ${{p.zone_name}}`
                    : 'Select a zone.';
                }},
                onHover: info => {{
                  if (!info.object) {{
                    tooltip.style.display = 'none';
                    return;
                  }}
                  const p = info.object.properties;
                  tooltip.textContent = `${{p.ID}}\\n${{p.zone_name}}\\nSource units: ${{p.source_ids}}`;
                  tooltip.style.left = `${{info.x + 12}}px`;
                  tooltip.style.top = `${{info.y + 12}}px`;
                  tooltip.style.display = 'block';
                }}
              }}),
              new deck.GeoJsonLayer({{
                id: 'arterial-roads',
                data: arterialRoads,
                stroked: true,
                filled: false,
                pickable: false,
                lineWidthMinPixels: 0.7,
                getLineColor: [185, 190, 198, 150]
              }}),
              new deck.GeoJsonLayer({{
                id: 'corridor',
                data: corridor,
                stroked: true,
                filled: false,
                pickable: false,
                lineWidthMinPixels: 4,
                getLineColor: [255, 65, 65, 255]
              }}),
              new deck.TextLayer({{
                id: 'zone-labels',
                data: labels,
                getPosition: d => [d.longitude, d.latitude],
                getText: d => d.id,
                getSize: 12,
                getColor: [255, 255, 255, 255],
                getTextAnchor: 'middle',
                getAlignmentBaseline: 'center',
                billboard: true
              }})
            ]
          }});
        </script>
      </body>
    </html>
    """


def generate_zoning_preview(zoning_path, output_path=None):
    """Write an interactive map of a zoning file without requiring OD flows."""
    zoning_path = Path(zoning_path)
    if zoning_path.suffix.lower() == ".parquet":
        gdf = gpd.read_parquet(zoning_path)
    else:
        gdf = gpd.read_file(zoning_path)
    if "ID" not in gdf.columns:
        raise ValueError(f"Zoning file must contain an 'ID' column: {zoning_path}")
    if gdf.crs is None:
        raise ValueError(f"Zoning file has no CRS: {zoning_path}")
    gdf = gdf.to_crs(epsg=4326)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if gdf.empty:
        raise ValueError(f"Zoning file contains no usable geometries: {zoning_path}")

    bounds = gdf.total_bounds
    center = ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)
    span = max(bounds[2] - bounds[0], bounds[3] - bounds[1], 0.01)
    zoom = max(3, min(14, math.log2(360 / span)))
    points = gdf.geometry.representative_point()
    labels = []
    core_number = 0
    for row, point in zip(gdf.itertuples(), points):
        if row.layer == "3.Access":
            label = row.ID.rsplit("_PETAL_", maxsplit=1)[-1]
        else:
            core_number += 1
            label = f"C{core_number:02d}"
        labels.append(
            {"id": label, "longitude": point.x, "latitude": point.y}
        )
    geojson = gdf.to_json().replace("</", "<\\/")
    empty_geojson = '{"type":"FeatureCollection","features":[]}'
    roads_path = zoning_path.with_name(f"{zoning_path.stem}_arterial_roads.geojson")
    corridor_path = zoning_path.with_name(f"{zoning_path.stem}_corridor.geojson")
    roads_geojson = empty_geojson
    corridor_geojson = empty_geojson
    if roads_path.exists():
        roads = gpd.read_file(roads_path)
        if roads.crs is None:
            raise ValueError(f"Arterial road preview layer has no CRS: {roads_path}")
        roads_geojson = roads.to_crs(epsg=4326).to_json().replace("</", "<\\/")
    if corridor_path.exists():
        corridor = gpd.read_file(corridor_path)
        if corridor.crs is None:
            raise ValueError(f"Corridor preview layer has no CRS: {corridor_path}")
        corridor_geojson = corridor.to_crs(epsg=4326).to_json().replace("</", "<\\/")
    html_content = _build_zoning_preview_html(
        geojson, json.dumps(labels), center, zoom, roads_geojson, corridor_geojson
    )
    output_path = Path(output_path) if output_path else zoning_path.with_name(
        f"{zoning_path.stem}_preview.html"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")
    print(f"Successfully created zoning preview '{output_path}'.")


def _read_flows(flows_path):
    """Read a flows CSV (origin_zone, dest_zone, n_trips, trip_km) and return filtered data."""
    print(f"Reading flows from {flows_path}...")
    df = pd.read_csv(flows_path)
    df['origin'] = df['origin_zone'].astype(str)
    df['destination'] = df['dest_zone'].astype(str)
    df['volume'] = df['n_trips']

    # Filter out self-loops and special zones
    df = df[
        (df['origin'] != df['destination']) &
        (~df['origin'].isin(['TOTAL', 'REST_PENINSULA'])) &
        (~df['destination'].isin(['TOTAL', 'REST_PENINSULA']))
    ]
    df = df.dropna(subset=['volume'])
    df = df[df['volume'] > 0]

    valid_ids = set(df['origin']).union(set(df['destination']))
    return df, valid_ids


def _build_arc_data(df_long, coords):
    """Convert melted OD DataFrame + coordinate dict into deck.gl arc data."""
    print("Merging coordinates...")
    data = []

    for _, row in df_long.iterrows():
        o = str(row['origin'])
        d = str(row['destination'])
        if o in coords and d in coords:
            hash_obj = hashlib.md5(f"{o}-{d}".encode())
            color_hex = hash_obj.hexdigest()[:6]
            color = [int(color_hex[i:i+2], 16) for i in (0, 2, 4)]

            data.append({
                'source': [coords[o]['lon'], coords[o]['lat']],
                'target': [coords[d]['lon'], coords[d]['lat']],
                'volume': float(row['volume']),
                'origin_id': o,
                'dest_id': d,
                'color': color
            })

    data = sorted(data, key=lambda x: x['volume'], reverse=True)
    return data


def _centroid_coordinates(gdf, id_column="ID"):
    """Return zone centroids as WGS84 coordinates keyed by zone ID."""
    centroids = gdf.to_crs(epsg=3035).geometry.centroid.to_crs(epsg=4326)
    return {
        str(zone_id): {"lon": point.x, "lat": point.y}
        for zone_id, point in zip(gdf[id_column], centroids)
    }


def _dissolve_visual_zones(gdf):
    """Dissolve source polygons to the exact zone IDs used in the OD matrix."""
    if "ID" not in gdf.columns or "layer" not in gdf.columns:
        raise ValueError("The zoning file must contain 'ID' and 'layer' columns.")
    if gdf.crs is None:
        raise ValueError("The zoning file has no CRS.")

    zones = gdf.copy()
    zones["ID_norm"] = zones["ID"].apply(normalise_parquet_id)
    zones["map_zone_id"] = zones.apply(assign_output_zone, axis=1)
    zones = zones.dropna(subset=["map_zone_id"])
    if zones.empty:
        raise ValueError("The zoning file contains no usable OD-zone polygons.")

    zones = zones.to_crs(epsg=3035)
    zones.geometry = zones.geometry.make_valid(
        method="structure", keep_collapsed=False
    )
    zones = zones[["map_zone_id", "layer", "geometry"]]
    dissolved = zones.dissolve(
        by="map_zone_id",
        as_index=False,
        aggfunc={"layer": "first"},
    )
    dissolved = dissolved.rename(columns={"map_zone_id": "ID"}).to_crs(epsg=4326)
    dissolved.geometry = dissolved.geometry.make_valid(
        method="structure", keep_collapsed=False
    )
    return dissolved


# ---------------------------------------------------------------------------
#  Aggregated visual
# ---------------------------------------------------------------------------
def generate_aggregated_map():
    """Generate an aggregated OD matrix visualization."""
    base_dir = Path(__file__).resolve().parents[1]

    # 1. Inputs
    rel_path = input("Enter the relative path to the base zoning parquet file (e.g. Data/ZoningABC.parquet): ")
    date = input("Enter the date for which the matrix was generated (YYYY-MM-DD): ")

    # 2. Read original zoning
    parquet_path = Path(rel_path)
    if not parquet_path.is_absolute():
        parquet_path = base_dir / parquet_path
    print("Reading original zoning data...")
    gdf = gpd.read_parquet(parquet_path)
    print("Dissolving polygons by OD matrix zone...")
    gdf = _dissolve_visual_zones(gdf)

    print("Calculating zone centroids...")
    coords = _centroid_coordinates(gdf)

    # 3. Project to WGS84 for basemap
    print("Projecting zones to WGS84 for the basemap...")
    gdf_4326 = gdf.to_crs(epsg=4326).copy()

    # 4. Read flows
    flows_path = base_dir / "Results" / date / "od_flows.csv"
    df_long, valid_ids = _read_flows(flows_path)

    # 5. Filter basemap to zones present in the matrix
    print("Filtering zones for the basemap...")
    gdf_map = gdf_4326[['ID', 'geometry']]
    geojson_str = gdf_map.to_json()

    # 6. Build arcs
    data = _build_arc_data(df_long, coords)
    if not data:
        print("No valid data points found after filtering.")
        return

    print(f"Generated {len(data)} arcs.")

    # 7. Write HTML
    print("Generating HTML...")
    corridor_path = parquet_path.with_name(f"{parquet_path.stem}_corridor.geojson")
    roads_path = parquet_path.with_name(
        f"{parquet_path.stem}_arterial_roads.geojson"
    )
    corridor_geojson = None
    arterial_roads_geojson = None
    initial_zone = None
    map_center = (-3.0, 40.0)
    map_zoom = 10
    if corridor_path.exists():
        corridor = gpd.read_file(corridor_path)
        if corridor.crs is None:
            raise ValueError(f"Corridor layer has no CRS: {corridor_path}")
        corridor = corridor.to_crs(gdf.crs)
        corridor_geometry = corridor.geometry.union_all()
        corridor_geojson = corridor.to_crs(epsg=4326).to_json()
        core_zones = gdf.loc[
            gdf.geometry.intersects(corridor_geometry)
            & gdf["layer"].isin(["1.Distritos", "2.Municipios"])
        ]
        if core_zones.empty:
            raise ValueError(
                f"No core zoning polygon intersects the corridor in {parquet_path}."
            )
        initial_zone = str(core_zones.iloc[0]["ID"])
        bounds = gdf_4326.total_bounds
        map_center = (
            float((bounds[0] + bounds[2]) / 2),
            float((bounds[1] + bounds[3]) / 2),
        )
        map_zoom = 5
    if roads_path.exists():
        roads = gpd.read_file(roads_path)
        if roads.crs is None:
            raise ValueError(f"Arterial road layer has no CRS: {roads_path}")
        arterial_roads_geojson = (
            roads.to_crs(epsg=4326).to_json().replace("</", "<\\/")
        )
    html_content = _build_html(
        data,
        geojson_str,
        date,
        corridor_geojson,
        initial_zone,
        map_center,
        map_zoom,
        arterial_roads_geojson,
    )

    output_dir = base_dir / "Results" / date
    output_path = output_dir / "od_matrix_visual.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Successfully created '{output_path}'.")


# ---------------------------------------------------------------------------
#  Disaggregated visual
# ---------------------------------------------------------------------------
def generate_disaggregated_map():
    """Generate a disaggregated OD matrix visualization."""
    base_dir = Path(__file__).resolve().parents[1]

    # 1. Inputs
    rel_path = input("Enter the relative path to the original zoning parquet file (e.g. Data/ZoningABC.parquet): ")
    date = input("Enter the date for which the matrix was generated (YYYY-MM-DD): ")
    flows_rel_path = input("Enter the name of the disaggregated flows CSV file (e.g. od_flows_d_AAA.csv): ")

    # 2. Read original zoning
    parquet_path = Path(rel_path)
    if not parquet_path.is_absolute():
        parquet_path = base_dir / parquet_path
    print("Reading original zoning data...")
    gdf = gpd.read_parquet(parquet_path)

    print("Calculating centroids for original zones...")
    coords = _centroid_coordinates(gdf)

    # 3. Prompt for the disaggregated zoning parquet in the Results/<date>/ folder
    disagg_rel = input("Enter the name of the disaggregated zoning parquet file (e.g. zoning_d_AAA.parquet): ")
    disagg_zoning_path = base_dir / "Results" / date / disagg_rel

    # 4. Read disaggregated zoning
    print("Reading disaggregated zoning data...")
    gdf_sub = gpd.read_parquet(disagg_zoning_path)

    if gdf_sub.crs is None:
        gdf_sub.set_crs(epsg=4326, inplace=True)
    elif gdf_sub.crs.to_epsg() != 4326:
        gdf_sub = gdf_sub.to_crs(epsg=4326)

    coords_sub = _centroid_coordinates(gdf_sub, id_column="sub_zone_id")

    # Merge coordinate dictionaries
    coords.update(coords_sub)

    # 5. Build basemap: original zones (minus parent) + sub-zones
    print("Projecting zones to WGS84 for the basemap...")
    gdf_4326 = gdf.to_crs(epsg=4326).copy()
    parent_zone_id = disagg_rel.split('/')[-1].split('.')[0].replace('zoning_d_', '')
    gdf_4326 = gdf_4326[gdf_4326['ID'] != parent_zone_id]

    gdf_sub_map = gdf_sub.copy()
    gdf_sub_map.rename(columns={'sub_zone_id': 'ID'}, inplace=True)

    # 7. Read disaggregated flows
    flows_path = base_dir / "Results" / date / flows_rel_path
    df_long, valid_ids = _read_flows(flows_path)

    # 8. Filter and combine basemap zones
    print("Combining zones for the basemap...")
    gdf_combined = pd.concat(
        [gdf_4326[['ID', 'geometry']], gdf_sub_map[['ID', 'geometry']]],
        ignore_index=True
    )
    gdf_combined = gdf_combined[gdf_combined['ID'].isin(valid_ids)]
    geojson_str = gdf_combined.to_json()

    # 9. Build arcs
    data = _build_arc_data(df_long, coords)
    if not data:
        print("No valid data points found after filtering.")
        return

    print(f"Generated {len(data)} arcs.")

    # 10. Write HTML
    print("Generating HTML...")

    html_content = _build_html(data, geojson_str, date)

    results_dir = os.path.dirname(flows_path)
    output_path = os.path.join(results_dir, "od_matrix_disagg_visual.html")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Successfully created '{output_path}'.")



# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("--- Matrix and Zoning Visualization Generator ---")
    print("  [1] Aggregated map")
    print("  [2] Disaggregated map")
    print("  [3] Zoning proposal preview (no OD flows required)")
    choice = input("Select an option (1/2/3): ").strip()

    if choice == "1":
        generate_aggregated_map()
    elif choice == "2":
        generate_disaggregated_map()
    elif choice == "3":
        base_dir = Path(__file__).resolve().parent
        zoning_path = input(
            r"Enter the zoning GeoParquet/GeoJSON path [Data\Zoning_N211.parquet]: "
        ).strip()
        zoning_path = Path(zoning_path or r"Data\Zoning_N211.parquet")
        if not zoning_path.is_absolute():
            zoning_path = base_dir.parent / zoning_path
        generate_zoning_preview(zoning_path)
    else:
        print("Invalid option. Please enter 1, 2, or 3.")
