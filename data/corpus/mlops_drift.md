# Data Drift and Model Monitoring

A machine-learning model is trained on a snapshot of the world, but the world
keeps moving. Data drift is the gradual change in the distribution of inputs a
model sees in production compared with the data it was trained on. When drift is
large enough, a model that was accurate at launch can quietly degrade, because
its learned patterns no longer match reality.

Monitoring systems detect drift by comparing the statistics of recent production
data against a reference distribution. Common signals include changes in feature
means and variances, shifts in categorical frequencies, and distances between
distributions such as population stability index or the Kolmogorov-Smirnov
statistic. Prediction drift and a drop in a live quality metric are also warning
signs.

When drift crosses a threshold, a mature pipeline can trigger an automated
response: raising an alert, collecting fresh labelled data, and retraining the
model. Guarding the retrained model with an evaluation gate ensures that a new
version is only promoted if it does not regress on held-out metrics. This
closed loop of monitor, detect, retrain, and gate is central to reliable MLOps.
