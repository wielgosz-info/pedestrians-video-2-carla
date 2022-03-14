from .zero import ZeroMovements
from .linear import Linear
from .lstm import LSTM
from .linear_ae import LinearAE, LinearAEResidual, LinearAEResidualLeaky, LinearAE2D
from .baseline_3d_pose import Baseline3DPose, Baseline3DPoseRot
from .seq2seq import Seq2Seq, Seq2SeqEmbeddings, Seq2SeqResidualA, Seq2SeqResidualB, Seq2SeqResidualC
from .pose_former import PoseFormer, PoseFormerRot
from .spatial_gnn import GNNLinearAutoencoder

MOVEMENTS_MODELS = {
    m.__name__: m
    for m in [
        # For testing
        ZeroMovements,
        Linear,

        # Universal (support MovementsModelOutputType param)
        LinearAE,
        LSTM,
        Seq2Seq,
        Seq2SeqEmbeddings,
        Seq2SeqResidualA,
        Seq2SeqResidualB,
        Seq2SeqResidualC,

        # For pose lifting
        Baseline3DPose,
        Baseline3DPoseRot,
        LinearAEResidual,
        LinearAEResidualLeaky,
        PoseFormer,
        PoseFormerRot,

        # For 2D pose autoencoding
        LinearAE2D,
        GNNLinearAutoencoder,
    ]
}
