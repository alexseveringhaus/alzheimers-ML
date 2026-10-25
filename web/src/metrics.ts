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
}

type Node = { [key: string]: unknown }

const isObject = (value: unknown): value is Node => typeof value === 'object' && value !== null

export function collectMetricRows(node: Node, name = ''): MetricsRow[] {
  if ('test_accuracy' in node) {
    const leaf = node as unknown as MetricsLeaf
    return [{
      model: name,
      featureSet: leaf.feature_set ?? '',
      cvAuc: leaf.cv_auc_mean != null && leaf.cv_auc_std != null
        ? `${leaf.cv_auc_mean.toFixed(3)} ± ${leaf.cv_auc_std.toFixed(3)}`
        : 'N/A',
      testAuc: leaf.test_auc != null ? leaf.test_auc.toFixed(3) : 'N/A',
      testAccuracy: `${(leaf.test_accuracy * 100).toFixed(1)}%`,
    }]
  }
  return Object.entries(node).flatMap(([childName, child]) =>
    isObject(child) ? collectMetricRows(child, childName) : [],
  )
}
