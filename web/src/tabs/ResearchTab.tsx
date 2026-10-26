import featureImportanceUrl from '../../../results/plots/logreg_feature_importance.png'
import trainingCurveUrl from '../../../results/plots/nn_training_curve.png'
import metrics from '../../../results/metrics.json'
import { collectMetricRows, type MetricsRow } from '../metrics'

const rows = collectMetricRows(metrics)
const find = (model: string, featureSet: string) => rows.find((r) => r.model === model && r.featureSet === featureSet)
const deployed = rows.find((r) => r.deployed)
const auc = (row?: MetricsRow) => row?.cvAucMean?.toFixed(3) ?? 'N/A'
const cnn = rows.find((r) => r.model === '2D CNN')

const MODELS = [
  {
    name: 'Logistic regression and random forest',
    text: 'Baselines on tabular region features, each trained on amyloid only, tau only, and both. The scaler is fit inside every cross-validation fold.',
    result: `Best: combined logistic regression, ${auc(deployed)} CV AUC`,
  },
  {
    name: 'Neural network',
    text: 'A three-layer network on all 26 features with dropout, L2 regularization, and early stopping that restores the best weights.',
    result: `${auc(find('Neural network', 'Combined'))} CV AUC`,
  },
  {
    name: 'Longitudinal model',
    text: 'Uses tau scans from three or more visits. For each region it measures the starting level and how fast tau accumulates over time.',
    result: `${auc(find('Longitudinal LR', 'Baseline + slope'))} CV AUC with baseline and slope`,
  },
  {
    name: '2D CNN on PET slices',
    text: 'Learns straight from about 600 raw tau PET scans, cut into 17,000 slices. Slices from one person never appear in both training and test data.',
    result: `${cnn?.testAccuracy ?? 'N/A'} slice-level accuracy`,
  },
]

export function ResearchTab() {
  return (
    <>
      <div className="research-intro">
        <header className="page-intro">
          <h2>How the model was built</h2>
          <p className="lede">
            The deployed model is a logistic regression on amyloid and tau PET measurements from the{' '}
            <a href="https://adni.loni.usc.edu/" target="_blank" rel="noreferrer">ADNI</a> study. It separates cognitively
            normal people from people with Alzheimer's disease. People with mild cognitive impairment were left out for a
            cleaner signal.
          </p>
        </header>
        <section className="panel">
          <dl className="facts">
            <div><dt>Inputs</dt><dd>26 SUVRs (13 regions × 2 tracers)</dd></div>
            <div><dt>Evaluation</dt><dd>80/20 split, 5-fold cross-validation</dd></div>
            <div><dt>Cross-validated AUC</dt><dd>{deployed?.cvAuc ?? 'N/A'}</dd></div>
            <div><dt>Explanations</dt><dd>SHAP, for each prediction</dd></div>
          </dl>
        </section>
      </div>

      <section className="panel">
        <h2>Model comparison</h2>
        <p className="muted small">
          The held-out test set has 75 subjects (12 AD), so test AUC is noisy. Cross-validated AUC is the more reliable
          estimate.
        </p>
        <div className="table-scroll">
          <table className="metrics-table">
            <thead>
              <tr>
                <th scope="col">Model</th><th scope="col">Features</th>
                <th scope="col" className="num">CV AUC</th><th scope="col" className="num">Test AUC</th>
                <th scope="col" className="num">Test accuracy</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={`${r.model}-${r.featureSet}`} className={r.deployed ? 'deployed' : undefined}>
                  <th scope="row">{r.model}{r.deployed && <span className="tag">Deployed</span>}</th>
                  <td>{r.featureSet}</td>
                  <td className="num">{r.cvAuc}</td>
                  <td className="num">{r.testAuc}</td>
                  <td className="num">{r.testAccuracy}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <div className="model-grid">
        {MODELS.map((m) => (
          <section className="panel model-card" key={m.name}>
            <h3>{m.name}</h3>
            <p>{m.text}</p>
            <p className="model-result">{m.result}</p>
          </section>
        ))}
      </div>

      <div className="figures">
        <figure>
          <a href={featureImportanceUrl} target="_blank" rel="noreferrer" title="Open full size">
            <img src={featureImportanceUrl} alt="Logistic regression coefficients for each brain region, for amyloid, tau, and combined features" />
          </a>
          <figcaption>Logistic regression coefficients. Red bars raise the chance of AD; blue bars lower it. Select the image to open it at full size.</figcaption>
        </figure>
        <figure>
          <a href={trainingCurveUrl} target="_blank" rel="noreferrer" title="Open full size">
            <img src={trainingCurveUrl} alt="Neural network training and validation loss by epoch" />
          </a>
          <figcaption>Neural network loss by epoch. The dashed line marks where early stopping ended training.</figcaption>
        </figure>
      </div>
    </>
  )
}
