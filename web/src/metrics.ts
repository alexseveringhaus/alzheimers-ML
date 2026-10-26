// Flattens results/metrics.json into table rows. Same walk as iter_metrics()
// in main.py: results are flat ({"NeuralNetwork": {metrics}}) or nested
// ({"combined": {"LogisticRegression": {metrics}}}); a leaf has test_accuracy.

interface MetricsLeaf {
  feature_set?: string
  cv_auc_mean?: number | null
  cv_auc_std?: number | null
  test_auc?: number | null
  test_accuracy: number
}

export interface MetricsRow {
  model: string
  featureSet: string
  cvAuc: string
  testAuc: string
  testAccuracy: string
  deployed: boolean
  cvAucMean: number | null
}

type Node = { [key: string]: unknown }

const isObject = (value: unknown): value is Node => typeof value === 'object' && value !== null

const MODEL_NAMES: Record<string, string> = {
  LogisticRegression: 'Logistic regression',
  RandomForest: 'Random forest',
  NeuralNetwork: 'Neural network',
  CNN: '2D CNN',
}

const capitalize = (s: string) => s.charAt(0).toUpperCase() + s.slice(1)

function toRow(name: string, group: string, leaf: MetricsLeaf): MetricsRow {
  const longitudinal = group === 'LongitudinalTau'
  return {
    model: longitudinal ? 'Longitudinal LR' : MODEL_NAMES[name] ?? name,
    featureSet: longitudinal ? name : name === 'CNN' ? 'PET image slices' : capitalize(leaf.feature_set ?? ''),
    cvAuc: leaf.cv_auc_mean != null && leaf.cv_auc_std != null
      ? `${leaf.cv_auc_mean.toFixed(3)} ± ${leaf.cv_auc_std.toFixed(3)}`
      : 'N/A',
    testAuc: leaf.test_auc != null ? leaf.test_auc.toFixed(3) : 'N/A',
    testAccuracy: `${(leaf.test_accuracy * 100).toFixed(1)}%`,
    // The API serves the combined-feature logistic regression
    deployed: name === 'LogisticRegression' && leaf.feature_set === 'combined',
    cvAucMean: leaf.cv_auc_mean ?? null,
  }
}

/** `group` is the top-level key a row sits under, so longitudinal rows can be named. */
export function collectMetricRows(node: Node, name = '', group = ''): MetricsRow[] {
  if ('test_accuracy' in node) return [toRow(name, group, node as unknown as MetricsLeaf)]
  return Object.entries(node).flatMap(([childName, child]) =>
    isObject(child) ? collectMetricRows(child, childName, group || childName) : [],
  )
}
