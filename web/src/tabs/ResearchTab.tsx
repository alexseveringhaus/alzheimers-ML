import featureImportanceUrl from '../../../results/plots/logreg_feature_importance.png'
import trainingCurveUrl from '../../../results/plots/nn_training_curve.png'
import metrics from '../../../results/metrics.json'
import { collectMetricRows } from '../metrics'

const rows = collectMetricRows(metrics)

export function ResearchTab() {
  return (
    <>
      <section>
        <h2>About the model</h2>
        <p>
          The API serves a <strong>Logistic Regression</strong> trained on combined amyloid + tau PET SUVR features
          from the <a href="https://adni.loni.usc.edu/" target="_blank" rel="noreferrer">ADNI</a> dataset. The binary
          classification task is Cognitively Normal (CN = 0) vs Alzheimer's Disease (AD = 1); MCI subjects are
          excluded for a cleaner signal.
        </p>
        <p><strong>Training approach:</strong></p>
        <ul>
          <li>13 amyloid ROI SUVRs + 13 tau ROI SUVRs = 26 input features</li>
          <li>
            <code>StandardScaler</code> normalization inside an <code>sklearn.Pipeline</code> (fit on training data
            only — no leakage)
          </li>
          <li>Stratified 80/20 train/test split; 5-fold cross-validation on the training portion</li>
          <li>
            Model selection: the combined-feature Logistic Regression achieved Test AUC = 1.00 and CV AUC = 0.892 ±
            0.061, outperforming random forest and matching the neural network while remaining fully interpretable via
            SHAP
          </li>
        </ul>
      </section>

      <section>
        <h2>Model comparison</h2>
        <div className="table-scroll">
          <table>
            <thead>
              <tr><th>Model</th><th>Feature set</th><th>CV AUC</th><th>Test AUC</th><th>Test accuracy</th></tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i}>
                  <td>{r.model}</td><td>{r.featureSet}</td><td>{r.cvAuc}</td><td>{r.testAuc}</td><td>{r.testAccuracy}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted small">
          CV AUC is the mean ± standard deviation over 5 folds of the training data. The held-out test set has 75
          subjects (12 AD), so test AUC is noisy; CV AUC is the more reliable estimate.
        </p>
      </section>

      <section>
        <h2>The four models</h2>
        <h3>Logistic Regression + Random Forest</h3>
        <p>
          Baseline classifiers on tabular ROI features. The same pipeline runs on amyloid-only, tau-only, and combined
          features, with a <code>StandardScaler</code> fit inside each cross-validation fold. Logistic regression
          coefficients serve as a feature importance measure.
        </p>
        <h3>Feedforward Neural Network</h3>
        <p>
          A 3-layer MLP on the combined features: Linear(64) → ReLU → Dropout(0.3) → Linear(32) → ReLU → Linear(2).
          It uses L2 regularization (Adam <code>weight_decay=1e-5</code>) and early stopping with patience 30, which
          restores the best weights.
        </p>
        <h3>Longitudinal Logistic Regression</h3>
        <p>
          Uses multi-visit tau data. For each subject and region, it computes the baseline value and the accumulation
          slope over time, using only subjects with at least 3 timepoints. Baseline-only, slope-only, and combined
          feature sets are evaluated separately.
        </p>
        <h3>2D CNN on PET image slices</h3>
        <p>
          Learns directly from raw 3D tau PET volumes. Axial slices 30–65 of about 600 scans give about 17,000 2D
          samples, each z-score normalized. The train/test split is by subject, so slices from one person never appear
          in both. Three convolution + pooling blocks feed two linear layers; it reports slice-level accuracy.
        </p>
      </section>

      <section>
        <h2>Logistic regression feature importance</h2>
        <p className="muted small">
          Coefficients across amyloid-only, tau-only, and combined feature sets. Red = positive weight (higher SUVR →
          more likely AD); blue = negative weight.
        </p>
        <a href={featureImportanceUrl} target="_blank" rel="noreferrer" title="Open full size">
          <img src={featureImportanceUrl} alt="Bar charts of logistic regression coefficients for each brain region, for amyloid, tau, and combined features" />
        </a>
        <p className="muted small">Select the image to open it at full size.</p>
      </section>

      <section>
        <h2>Neural network training curve</h2>
        <p className="muted small">Train vs validation loss with early stopping (red dashed line).</p>
        <a href={trainingCurveUrl} target="_blank" rel="noreferrer" title="Open full size">
          <img src={trainingCurveUrl} alt="Line chart of neural network training and validation loss by epoch" className="narrow-image" />
        </a>
      </section>
    </>
  )
}
