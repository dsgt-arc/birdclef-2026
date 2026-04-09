"""EDA and static baselines — no learned models, diagnostic outputs."""

from birdclef_106.analysis.eda.baseline_zero_shot import ZeroShotBaseline
from birdclef_106.analysis.eda.covariance_logit import LogitCovariance
from birdclef_106.analysis.eda.covariance_embedding import EmbeddingCovariance
from birdclef_106.analysis.eda.domain_shift import DistributionShift
from birdclef_106.analysis.eda.feature_importance import FeatureSelection
