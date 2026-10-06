export const fixed = (digits) => (v) => (v == null ? '–' : Number(v).toFixed(digits))
export const pct = (digits = 1) => (v) => (v == null ? '–' : `${(v * 100).toFixed(digits)}%`)
export const amount = (v) =>
  v == null ? '–' : Number(v).toLocaleString('en-US', { maximumFractionDigits: 2 })

// Fraud probabilities span many orders of magnitude, so keep small ones readable.
export function probability(p) {
  if (p == null) return '–'
  const percent = p * 100
  if (percent === 0) return '0%'
  if (percent < 0.01) return '< 0.01%'
  return `${percent.toFixed(2)}%`
}

export const MODEL_NAMES = {
  RandomForestClassifier: 'Random Forest',
  LGBMClassifier: 'LightGBM',
}
