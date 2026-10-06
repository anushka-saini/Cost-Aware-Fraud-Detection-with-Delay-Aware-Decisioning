import { useState } from 'react'

export function Segmented({ options, value, onChange, label }) {
  return (
    <div className="segmented" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o} type="button" aria-pressed={value === o} onClick={() => onChange(o)}>
          {o}
        </button>
      ))}
    </div>
  )
}

export function Legend({ items }) {
  return (
    <div className="legend">
      {items.map((item) => (
        <span key={item.label}>
          <i className="swatch" style={{ background: item.color }} />
          {item.label}
        </span>
      ))}
    </div>
  )
}

export function DataTable({ columns, rows }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key} className={c.num ? 'num' : undefined}>
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className={row._highlight ? 'highlight' : undefined}>
              {columns.map((c) => (
                <td key={c.key} className={c.num ? 'num' : c.wrap ? 'wrap' : undefined}>
                  {c.format ? c.format(row[c.key], row) : row[c.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// A chart with its title, an optional status badge, the notebook cell the
// numbers came from, and a table view of the same data.
export default function ChartCard({ title, subtitle, badge, controls, source, note, table, children }) {
  const [view, setView] = useState('Chart')
  return (
    <section className="card">
      <div className="card-head">
        <div className="titles">
          <h3>
            {title}
            {badge}
          </h3>
          {subtitle && <p>{subtitle}</p>}
        </div>
        <div className="card-controls">
          {controls}
          {table && <Segmented options={['Chart', 'Table']} value={view} onChange={setView} label="View" />}
        </div>
      </div>
      {view === 'Table' && table ? <DataTable {...table} /> : children}
      {note && <p className="note">{note}</p>}
      {source && <p className="source">Source: {source}</p>}
    </section>
  )
}
