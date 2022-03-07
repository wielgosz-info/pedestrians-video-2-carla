from .zero import ZeroMovements
from .linear import Linear
from .lstm import LSTM
from .linear_ae import LinearAE, LinearAEResidual, LinearAEResidualLeaky, LinearAE2D
from .baseline_3d_pose import Baseline3DPose, Baseline3DPoseRot
from .seq2seq import Seq2Seq, Seq2SeqEmbeddings, Seq2SeqResidualA, Seq2SeqResidualB, Seq2SeqResidualC, Seq2Seq2D
from .pose_former import PoseFormer, PoseFormerRot

MOVEMENTS_MODELS = {
    m.__name__: m
    for m in [
        ZeroMovements,
        Linear,
        Baseline3DPose,
        Baseline3DPoseRot,
        LSTM,
        LinearAE,
        LinearAEResidual,
        LinearAEResidualLeaky,
        Seq2Seq,
        Seq2SeqEmbeddings,
        Seq2SeqResidualA,
        Seq2SeqResidualB,
        Seq2SeqResidualC,
        PoseFormer,
        PoseFormerRot,

        # For 2D pose autoencoding
        LinearAE2D,
        Seq2Seq2D,
    ]
}
