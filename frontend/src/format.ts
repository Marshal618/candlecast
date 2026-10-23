export const pct = (x: number, digits = 1) => `${x >= 0 ? '+' : ''}${(x * 100).toFixed(digits)}%`
export const rate = (x: number) => `${(x * 100).toFixed(0)}%`
export const money = (x: number) =>
  x.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })
