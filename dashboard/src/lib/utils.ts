import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export const cn = (...inputs: ClassValue[]) => twMerge(clsx(inputs))

const number0 = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 })
const number2 = new Intl.NumberFormat('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
const compact = new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 })

export const fmt = {
  int: (v: number | null | undefined) => (v == null ? '–' : number0.format(v)),
  money: (v: number | null | undefined) => (v == null ? '–' : number2.format(v)),
  compact: (v: number | null | undefined) => (v == null ? '–' : compact.format(v)),
  pct: (v: number | null | undefined, digits = 2) => (v == null ? '–' : `${(v * 100).toFixed(digits)}%`),
  score: (v: number | null | undefined, digits = 4) => (v == null ? '–' : v.toFixed(digits)),
  // Calibrated probabilities span many orders of magnitude; show small ones without losing them.
  probability: (v: number | null | undefined) => {
    if (v == null) return '–'
    if (v === 0) return '0'
    if (v > 0.999) return '>99.9%'
    if (v >= 0.01) return `${(v * 100).toFixed(1)}%`
    if (v >= 0.0001) return `${(v * 100).toFixed(3)}%`
    return v.toExponential(1)
  },
  threshold: (v: number) => (v >= 0.01 ? v.toFixed(3) : v >= 0.0001 ? v.toFixed(5) : v.toExponential(2)),
  duration: (seconds: number | null | undefined) => {
    if (seconds == null) return '–'
    if (seconds < 60) return `${Math.round(seconds)}s`
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m`
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`
    return `${Math.floor(seconds / 86400)}d ${Math.floor((seconds % 86400) / 3600)}h`
  },
  time: (iso: string) => new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'medium' }),
  clock: (iso: string) => new Date(iso).toLocaleTimeString(),
  signed: (v: number, digits = 4) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(digits)}`,
  // PaySim's step is an hour index: show it as day and hour.
  step: (step: number | null | undefined) => (step == null ? '–' : `day ${Math.floor((step - 1) / 24)}, ${String(step % 24).padStart(2, '0')}:00`),
}

export const sentence = (text: string) => text.charAt(0).toUpperCase() + text.slice(1)
