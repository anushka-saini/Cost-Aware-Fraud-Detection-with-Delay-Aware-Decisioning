// Cost model for the threshold simulator, computed in the browser.
//
// Same rule as the Python evaluation code: a fraud that lands in ALLOW costs
// its amount; a legitimate transaction that lands in BLOCK costs
// costFalseBlock; every transaction sent to REVIEW costs costReview.
//
// Frauds are held exactly. Legitimate transactions come as a histogram over
// score bins, so a threshold is snapped to the lower edge of its bin.

import type { SimulatorData } from './api'

export interface Outcome {
  totalCost: number; fraudAllowed: number; fraudAllowedAmount: number; fraudReviewed: number; fraudBlocked: number
  legitReviewed: number; legitReviewedPct: number; legitBlocked: number; reviewWorkload: number
  blockPrecision: number; blockRecall: number
}

// The histogram grid written by scripts/build_simulator_data.py: one bin from
// 0 to 1e-9, then 3,999 log-spaced bins up to 1.
const BIN_FLOOR = 1e-9
const BIN_RATIO = 10 ** (9 / 3999)

export class CostModel {
  private fraudProb: number[]
  private fraudAmountBelow: number[]      // prefix sums of fraud amounts, ascending by probability
  private legitEdge: number[]
  private legitAtOrAbove: number[]        // suffix sums of legitimate counts
  readonly nFraud: number
  readonly nLegit: number
  readonly edges: number[]

  constructor(data: SimulatorData) {
    this.fraudProb = data.fraud.map(([p]) => p)
    this.fraudAmountBelow = [0]
    for (const [, amount] of data.fraud) this.fraudAmountBelow.push(this.fraudAmountBelow[this.fraudAmountBelow.length - 1] + amount)
    this.legitEdge = data.legit_histogram.map(([edge]) => edge)
    this.legitAtOrAbove = new Array(this.legitEdge.length + 1).fill(0)
    for (let i = this.legitEdge.length - 1; i >= 0; i--) this.legitAtOrAbove[i] = this.legitAtOrAbove[i + 1] + data.legit_histogram[i][1]
    this.nFraud = data.n_fraud
    this.nLegit = data.n_legit
    this.edges = this.legitEdge
  }

  // First index whose value is >= target (values ascending).
  private static lowerBound(values: number[], target: number) {
    let lo = 0, hi = values.length
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if (values[mid] < target) lo = mid + 1
      else hi = mid
    }
    return lo
  }

  private legitFrom(threshold: number) {
    // Legitimate transactions at or above the threshold. The bin that contains
    // the threshold is counted whole (the threshold snaps to its lower edge).
    let index = CostModel.lowerBound(this.legitEdge, threshold)
    if (index > 0) {
      const previous = this.legitEdge[index - 1]
      const upper = previous === 0 ? BIN_FLOOR : previous * BIN_RATIO
      if (threshold < upper) index -= 1
    }
    return this.legitAtOrAbove[index]
  }

  outcome(tReview: number, tBlock: number, costFalseBlock: number, costReview: number): Outcome {
    const block = Math.max(tBlock, tReview)
    const allowedCount = CostModel.lowerBound(this.fraudProb, tReview)
    const belowBlock = CostModel.lowerBound(this.fraudProb, block)
    const fraudBlocked = this.nFraud - belowBlock
    const fraudReviewed = belowBlock - allowedCount
    const legitFlagged = tReview <= 0 ? this.nLegit : this.legitFrom(tReview)
    const legitBlocked = block > 1 ? 0 : this.legitFrom(block)
    const legitReviewed = Math.max(legitFlagged - legitBlocked, 0)
    const fraudAllowedAmount = this.fraudAmountBelow[allowedCount]
    return {
      totalCost: fraudAllowedAmount + costFalseBlock * legitBlocked + costReview * (legitReviewed + fraudReviewed),
      fraudAllowed: allowedCount, fraudAllowedAmount, fraudReviewed, fraudBlocked,
      legitReviewed, legitReviewedPct: legitReviewed / this.nLegit, legitBlocked,
      reviewWorkload: legitReviewed + fraudReviewed,
      blockPrecision: fraudBlocked + legitBlocked ? fraudBlocked / (fraudBlocked + legitBlocked) : 0,
      blockRecall: this.nFraud ? fraudBlocked / this.nFraud : 0,
    }
  }
}

// The sliders move on a log scale: thresholds of interest run from 1e-7 to 1.
export const LOG_MIN = -7
export const toThreshold = (position: number) => (position >= 0 ? 1 : 10 ** position)
export const toPosition = (threshold: number) => (threshold >= 1 ? 0 : Math.max(LOG_MIN, Math.log10(Math.max(threshold, 1e-12))))
