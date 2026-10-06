// Shared tooltip body for every Recharts chart.
export default function ChartTooltip({ active, payload, label, title, format = (v) => v, note }) {
  if (!active || !payload?.length) return null
  const datum = payload[0].payload
  const extra = note ? note(datum) : null
  return (
    <div className="tooltip">
      <div className="tt-title">{title ? title(datum, label) : label}</div>
      {payload.map((p) => (
        <div className="tt-row" key={p.dataKey}>
          <i className="swatch" style={{ background: p.payload?._color || p.color || p.fill }} />
          {p.name}
          <b>{format(p.value, datum)}</b>
        </div>
      ))}
      {extra && <div className="tt-note">{extra}</div>}
    </div>
  )
}
