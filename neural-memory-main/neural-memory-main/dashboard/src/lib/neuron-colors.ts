export const NEURON_TYPE_COLORS: Record<string, string> = {
  concept: "#6366f1",
  entity: "#06b6d4",
  time: "#f59e0b",
  action: "#059669",
  state: "#8b5cf6",
  other: "#a8a29e",
  relation: "#ec4899",
  attribute: "#14b8a6",
}

export const NEURON_DEFAULT_COLOR = NEURON_TYPE_COLORS.other

export function colorForNeuronType(type: string): string {
  return NEURON_TYPE_COLORS[type] ?? NEURON_DEFAULT_COLOR
}
