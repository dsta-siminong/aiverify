// export function transformData(data, data2, severities, class_names) {
//   const rows = [];
//   const cmRows = [];
//   for (let s = 0; s < severities.length; s++) {
//     const severity = severities[s];
//     const cr = data[s];
//     const cm = data2[s];
//     // classification_report → wide DF‑like rows
//     for (const cl of Object.keys(cr)) {
//       const r = { severity, class: cl, ...cr[cl] };
//       rows.push(r);
//     }
//     // conf_matrix → rows with TP/FP/FN/TN
//     for (const cl of Object.keys(cm)) {
//       const stats = cm[cl];
//       const row = { severity, class: cl, ...stats };
//       cmRows.push(row);
//     }
//   }
//   // Combine: big_df = rows; cm_df = cmRows
//   const combined = [];
//   for (const r of rows) {
//     const cmRow = cmRows.find(
//       (c) => c.severity === r.severity && c.class === r.class
//     );
//     if (cmRow) {
//       combined.push({
//         ...r,
//         ...cmRow,
//         preds_population: cmRow.TP + cmRow.FP,
//         actual_population: cmRow.TP + cmRow.FN,
//       });
//     }
//   }
//   return combined;
// }

export function transformData(data, data2, severities, class_names) {
  const rows = [];
  const cmRows = [];

  // Detect OD format: cr[0] has a "map_50" key (or class keys contain TP directly)
  const isODFormat = data[0] && "map_50" in data[0];

  for (let s = 0; s < severities.length; s++) {
    const severity = severities[s];
    const cr = data[s];
    const cm = data2[s];

    if (isODFormat) {
      // OD: classification_report already contains TP/FP/FN per class
      for (const key of Object.keys(cr)) {
        if (key === "map_50") continue; // skip the top-level mAP key
        const classStats = cr[key];
        rows.push({
          severity,
          class: key,
          // Normalise f1_score → "f1-score" to match old consumers
          "f1-score": classStats.f1_score,
          ...classStats,
          preds_population: (classStats.TP ?? 0) + (classStats.FP ?? 0),
          actual_population: (classStats.TP ?? 0) + (classStats.FN ?? 0),
        });
      }
      // conf_matrix not needed for per-class stats in OD — skip it
    } else {
      // Original image-classification path — unchanged
      for (const cl of Object.keys(cr)) {
        const r = { severity, class: cl, ...cr[cl] };
        rows.push(r);
      }
      for (const cl of Object.keys(cm)) {
        const stats = cm[cl];
        const row = { severity, class: cl, ...stats };
        cmRows.push(row);
      }
    }
  }

  if (isODFormat) {
    return rows; // already fully combined above
  }

  // Original image-classification merge
  const combined = [];
  for (const r of rows) {
    const cmRow = cmRows.find(
      (c) => c.severity === r.severity && c.class === r.class
    );
    if (cmRow) {
      combined.push({
        ...r,
        ...cmRow,
        preds_population: cmRow.TP + cmRow.FP,
        actual_population: cmRow.TP + cmRow.FN,
      });
    }
  }
  return combined;
}

export function ClassLineChart({ combined, className, metrics }) {
  const sub_df = combined.filter((r) => r.class === className);
  const labels = sub_df.map((r) => r.severity);
  const n = labels.length;
  if (n === 0) return <div>No data for class {className}</div>;

  const margin = { top: 20, right: 40, bottom: 40, left: 50 };
  const width = 500;
  const height = 200;
  const allValues = metrics.flatMap((metric) =>
    sub_df.map((r) => r[metric] ?? 0)
  );
  const min = 0;
  const max = Math.max(...allValues, 1); // avoid divide-by-zero
  const numTicks = 5;
  const tickStep = max / (numTicks - 1);
  const ticks = Array.from({ length: numTicks }, (_, i) => i * tickStep);

  // For each metric, compute scaled y values
  const datasets = metrics.map((metric) => {
    const values = sub_df.map((r) => r[metric] ?? 0);
    const y = values.map((v) => {
      const t = (v - min) / (max - min);
      return height - margin.bottom - t * (height - margin.top - margin.bottom);
    });
    return { metric, values, y };
  });

  const xStep = (width - margin.left - margin.right) / Math.max(n - 1, 1);
  const x = (i) => margin.left + i * xStep;

  // Colors per metric (you can tweak this)
  const getColorForMetric = (metric) => {
    const colors = {
      precision: "#0b304b",
      recall: "#ff7f0e",
      "f1-score": "#2ca02c",
      TP: "#2ca02c",
      FP: "#d62728",
      FN: "#c85d00",
      TN: "#1f77b4",
    };
    return colors[metric] || "#777";
  };

  const yScale = (v) => {
    const t = (v - min) / (max - min);
    return height - margin.bottom - t * (height - margin.top - margin.bottom);
  };

  return (
    <div style={{ width: "100%", overflow: "visible" }}>
      <h4 style={{ margin: "0 0 0.5em 0" }}>
        Metrics for class {className}
      </h4>
      <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet" >
        {/* Axes */}
        <line
          x1={margin.left}
          y1={height - margin.bottom}
          x2={width - margin.right}
          y2={height - margin.bottom}
          stroke="black"
          strokeWidth={1}
        />
        <line
          x1={margin.left}
          y1={margin.top}
          x2={margin.left}
          y2={height - margin.bottom}
          stroke="black"
          strokeWidth={1}
        />

        {/* Y axis ticks */}
        {ticks.map((v) => {
          const y = yScale(v);
          return (
            <g key={v}>
              <line
                x1={margin.left - 5}
                y1={y}
                x2={margin.left}
                y2={y}
                stroke="black"
                strokeWidth={1}
              />
              <text
                x={margin.left - 10}
                y={y + 3}
                textAnchor="end"
                fontSize="10"
                fill="black"
              >
                {v}
              </text>
            </g>
          );
        })}

        {/* X axis labels (severities) */}
        {labels.map((label, i) => (
          <text
            key={i}
            x={x(i)}
            y={height - margin.bottom + 15}
            textAnchor="middle"
            fontSize="8"
            fill="black"
            transform={`rotate(-45, ${x(i)}, ${height - margin.bottom + 15})`}
          >
            {label}
          </text>
        ))}

        {/* Lines for each metric */}
        {datasets.map(({ metric, y }) => (
          <polyline
            key={metric}
            fill="none"
            stroke={getColorForMetric(metric)}
            strokeWidth={2}
            points={labels
              .map((_, i) => `${x(i)},${y[i]}`)
              .join(" ")}
          />
        ))}

        {/* Legend */}
        <g transform={`translate(${width - margin.right}, ${margin.top})`}>
          {datasets.map(({ metric }, i) => (
            <g key={metric} transform={`translate(0, ${i * 15})`}>
              <rect x={-15} y={-8} width={10} height={2} fill={getColorForMetric(metric)} />
              <text x={0} y={0} fontSize="10" fill="black">
                {metric}
              </text>
            </g>
          ))}
        </g>
      </svg>
    </div>
  );
}

export function PopulationChart({ combined, className }) {
  const sub_df = combined.filter((r) => r.class === className);
  const labels = sub_df.map((r) => r.severity);
  const n = labels.length;
  if (n === 0) return <div>No data for class {className}</div>;

  const margin = { top: 20, right: 40, bottom: 40, left: 50 };
  const width = 500;
  const height = 200;

  const preds = sub_df.map((r) => r.preds_population ?? 0);
  const actual = sub_df.map((r) => r.actual_population ?? 0);

  //const allValues = [...preds, ...actual];
  const min = 0;
  const max = Math.max(...preds, ...actual); 
  const y = (v) => {
    const t = (v - min) / (max - min);
    return height - margin.bottom - t * (height - margin.top - margin.bottom);
  };

  const numTicks = 5;
  const tickStep = max / (numTicks - 1);
  const ticks = Array.from({ length: numTicks }, (_, i) => i * tickStep);

  const xStep = (width - margin.left - margin.right) / Math.max(n - 1, 1);
  const x = (i) => margin.left + i * xStep;

  return (
    <div style={{ width: "100%", overflow: "visible" }}>
      <h4 style={{ margin: "0 0 0.5em 0" }}>
        Preds vs actual for class {className}
      </h4>
      <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet" >
        {/* Axes */}
        <line
          x1={margin.left}
          y1={height - margin.bottom}
          x2={width - margin.right}
          y2={height - margin.bottom}
          stroke="black"
          strokeWidth={1}
        />
        <line
          x1={margin.left}
          y1={margin.top}
          x2={margin.left}
          y2={height - margin.bottom}
          stroke="black"
          strokeWidth={1}
        />

        {/* Y axis ticks (0–1) */}
        {ticks.map((v) => {
          const yv = y(v);
          return (
            <g key={v}>
              <line
                x1={margin.left - 5}
                y1={yv}
                x2={margin.left}
                y2={yv}
                stroke="black"
                strokeWidth={1}
              />
              <text
                x={margin.left - 10}
                y={yv + 3}
                textAnchor="end"
                fontSize="10"
                fill="black"
              >
                {v}
              </text>
            </g>
          );
        })}

        {/* X axis labels (severities) */}
        {labels.map((label, i) => (
          <text
            key={i}
            x={x(i)}
            y={height - margin.bottom + 15}
            textAnchor="middle"
            fontSize="8"
            fill="black"
            transform={`rotate(-45, ${x(i)}, ${height - margin.bottom + 15})`}
          >
            {label}
          </text>
        ))}

        {/* preds_population line */}
        <polyline
          fill="none"
          stroke="blue"
          strokeWidth={2}
          points={preds
            .map((v, i) => `${x(i)},${y(v)}`)
            .join(" ")}
        />

        {/* actual_population line */}
        <polyline
          fill="none"
          stroke="red"
          strokeWidth={2}
          points={actual
            .map((v, i) => `${x(i)},${y(v)}`)
            .join(" ")}
        />

        {/* Legend */}
        <g transform={`translate(${width - margin.right}, ${margin.top})`}>
          <g transform="translate(0, 0)">
            <rect x={-15} y={-8} width={10} height={2} fill="blue" />
            <text x={0} y={0} fontSize="10" fill="black">
              preds_population
            </text>
          </g>
          <g transform="translate(0, 15)">
            <rect x={-15} y={-8} width={10} height={2} fill="red" />
            <text x={0} y={0} fontSize="10" fill="black">
              actual_population
            </text>
          </g>
        </g>
      </svg>
    </div>
  );
}


function getColorForClass(index, total) {
  const hues = [
    "#1f77b4", // blue
    "#ff7f0e", // orange
    "#2ca02c", // green
    "#d62728", // red
    "#9467bd", // purple
    "#8c564b", // brown
    "#e377c2", // pink
    "#7f7f7f", // gray
    "#bcbd22", // olive
    "#17becf", // cyan
    "#bc16ab", // magenta
    "#d95b00", // dark orange
    "#19eb19", // brat green
  ];
  return hues[index % hues.length];
}

export function StackedBarChart({ combined, class_names }) {
  // pivot: severity → { class0: raw, class1: raw, ... }
  const pivot = {};
  for (const r of combined) {
    if (!pivot[r.severity]) pivot[r.severity] = {};
    pivot[r.severity][r.class] = r.preds_population;
  }

  // normalize each severity so classes sum to 1
  for (const s of Object.keys(pivot)) {
    const row = pivot[s];
    const total = Object.values(row).reduce((a, v) => a + (v ?? 0), 0);
    if (total > 0) {
      for (const cl of Object.keys(row)) {
        row[cl] = row[cl] / total;
      }
    }
  }

  const labels = Object.keys(pivot);
  const nClasses = Object.values(class_names).length;

  if (labels.length === 0) {
    return <div>No data to display</div>;
  }

  const margin = { top: 20, right: 60, bottom: 40, left: 50 };
  const width = 500;
  const height = 200;

  const barWidth = 30;
  const gap = 15;
  const totalWidth = labels.length * (barWidth + gap);

  return (
    <div style={{ width: "100%", overflow: "visible", margin: "1em 0" }}>
      <h3 style={{ margin: "0 0 0.5em 0", fontSize: "1.2em" }}>
        Proportions of Classes per Severity
      </h3>
      <svg
        width="100%"
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Axes */}
        <line
          x1={margin.left}
          y1={height - margin.bottom}
          x2={margin.left + totalWidth}
          y2={height - margin.bottom}
          stroke="black"
          strokeWidth={1}
        />
        <line
          x1={margin.left}
          y1={margin.top}
          x2={margin.left}
          y2={height - margin.bottom}
          stroke="black"
          strokeWidth={1}
        />

        {/* Y axis ticks (0–1) */}
        {[0, 0.25, 0.5, 0.75, 1].map((v) => {
          const y = height - margin.bottom - v * (height - margin.top - margin.bottom);
          return (
            <g key={v}>
              <line
                x1={margin.left - 5}
                y1={y}
                x2={margin.left}
                y2={y}
                stroke="black"
                strokeWidth={1}
              />
              <text
                x={margin.left - 10}
                y={y + 3}
                textAnchor="end"
                fontSize="10"
                fill="black"
              >
                {v}
              </text>
            </g>
          );
        })}

        {/* X axis labels (severities) */}
        {labels.map((s, i) => {
          const x = margin.left + i * (barWidth + gap);
          return (
            <text
              key={s}
              x={x + barWidth / 2}
              y={height - margin.bottom + 20}
              textAnchor="middle"
              fontSize="8"
              transform={`rotate(-45, ${x + barWidth / 2}, ${height - margin.bottom + 20})`}
              fill="black"
            >
              {s}
            </text>
          );
        })}

        {/* Stacked bars per severity */}
        {labels.map((s, i) => {
          const x = margin.left + i * (barWidth + gap);
          let bottom = height - margin.bottom;

          return (
            <g key={s}>
              {Object.values(class_names).map((className, j) => {
                const frac = pivot[s][className] ?? 0;
                const h = frac * (height - margin.top - margin.bottom);
                const y = bottom - h;
                bottom -= h; // 🔑 THIS is the missing step

                if (h <= 0) return null;

                const color = getColorForClass(j, nClasses);

                return (
                  <rect
                    key={className}
                    x={x}
                    y={y}
                    width={barWidth}
                    height={h}
                    fill={color}
                    stroke="#333"
                    strokeWidth={0.5}
                  />
                );
              })}
            </g>
          );
        })}

        {/* Legend */}
        <g transform={`translate(${width - margin.right + 25}, ${margin.top})`}>
          {Object.values(class_names).map((className, i) => (
            <g key={className} transform={`translate(0, ${i * 15})`}>
              <rect
                x={-15}
                y={-8}
                width={10}
                height={10}
                fill={getColorForClass(i, nClasses)}
              />
              <text x={0} y={0} fontSize="10" fill="black">
                {className}
              </text>
            </g>
          ))}
        </g>
      </svg>
    </div>
  );
}

const MAX_COLUMNS = 5;

export function ClassMetricsTable({ combined, className , metrics }) {

  const classRows = combined.filter((r) => r.class === className);
  const severities = [...new Set(classRows.map((r) => r.severity))];

  if (classRows.length === 0) return null;

  const MAX_COLUMNS = 5; // Safe for PDF
  const severityChunks = [];

  for (let i = 0; i < severities.length; i += MAX_COLUMNS) {
    severityChunks.push(severities.slice(i, i + MAX_COLUMNS));
  }

  return (
    <>
      {severityChunks.map((chunk, chunkIndex) => (
        <div
          style={{
            width: "100%",
            overflowX: "auto",
            marginBottom: "1em",
            breakInside: "avoid",
            pageBreakInside: "avoid",
          }}
        >
          <table
            key={chunkIndex}
            style={{
              width: "max-content",
              borderCollapse: "collapse",
              marginTop: "1em",
              marginBottom: "2em",
              tableLayout: "fixed",
            }}
          >
            <thead>
              <tr>
                <th
                  style={{
                    border: "1px solid #ccc",
                    padding: "6px",
                    width: "150px",
                  }}
                >
                  Metric \ Severity
                </th>

                {chunk.map((sev) => (
                  <th
                    key={sev}
                    style={{
                      border: "1px solid #ccc",
                      padding: "6px",
                      wordWrap: "break-word",
                      whiteSpace: "normal",
                      overflowWrap: "break-word",
                    }}
                  >
                    {sev}
                  </th>
                ))}
              </tr>
            </thead>

            <tbody>
              {metrics.map((metric) => (
                <tr key={metric}>
                  <td
                    style={{
                      border: "1px solid #ccc",
                      padding: "6px",
                      fontWeight: "bold",
                    }}
                  >
                    {metric}
                  </td>

                  {chunk.map((sev) => {
                    const row = classRows.find(
                      (r) => r.severity === sev
                    );

                    let value = row?.[metric];

                    if (typeof value === "number") {
                      // Special formatting for confusion matrix counts
                      if (["TP", "FP", "FN", "TN"].includes(metric)) {
                        const total =
                          (row?.TP ?? 0) +
                          (row?.FP ?? 0) +
                          (row?.FN ?? 0) +
                          (row?.TN ?? 0);

                        const pct = total > 0 ? (value / total) * 100 : 0;

                        value = `${value.toFixed(1)} (${pct.toFixed(1)}%)`;
                      } else {
                        // Default formatting for all other metrics
                        value = value.toFixed(3);
                      }
                    }

                    return (
                      <td
                        key={sev}
                        style={{
                          border: "1px solid #ccc",
                          padding: "6px",
                          textAlign: "center",
                        }}
                      >
                        {value ?? "-"}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </>
  );
}

export function MapTable({ classificationReport, severities }) {
  const MAX_COLUMNS = 5;
  const severityChunks = [];
  for (let i = 0; i < severities.length; i += MAX_COLUMNS) {
    severityChunks.push(severities.slice(i, i + MAX_COLUMNS));
  }

  return (
    <>
      {severityChunks.map((chunk, tableIndex) => (
        <table
          key={tableIndex}
          style={{
            width: "100%",
            borderCollapse: "collapse",
            tableLayout: "fixed",
            marginBottom: "1.5em",
          }}
        >
          <thead>
            <tr>
              <th style={{ border: "1px solid #ccc", padding: "6px" }}>
                Severity
              </th>
              {chunk.map((sev) => (
                <th
                  key={sev}
                  style={{
                    border: "1px solid #ccc",
                    padding: "6px",
                    fontSize: "0.85em",
                  }}
                >
                  {sev === "None" ? "No Aug" : sev}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style={{ border: "1px solid #ccc", padding: "6px", fontWeight: "bold" }}>
                mAP@50
              </td>
              {chunk.map((sev, i) => {
                const severityIndex = severities.indexOf(sev);
                const value = classificationReport[severityIndex]?.map_50;
                return (
                  <td
                    key={sev}
                    style={{
                      border: "1px solid #ccc",
                      padding: "6px",
                      textAlign: "center",
                      fontWeight: "bold",
                    }}
                  >
                    {typeof value === "number" ? value.toFixed(3) : "-"}
                  </td>
                );
              })}
            </tr>
          </tbody>
        </table>
      ))}
    </>
  );
}