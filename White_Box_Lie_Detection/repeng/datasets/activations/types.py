from pydantic import BaseModel

from White_Box_Lie_Detection.repeng.datasets.elk.types import DatasetId, Split
from White_Box_Lie_Detection.repeng.models.llms import LlmId
from White_Box_Lie_Detection.repeng.utils.pydantic_ndarray import NdArray


class ActivationResultRow(BaseModel, extra="forbid"):
    dataset_id: DatasetId
    group_id: str | None
    answer_type: str | None
    activations: dict[str, NdArray]  # (s, d)
    prompt_logprobs: float
    label: bool
    split: Split
    llm_id: LlmId

    class Config:
        arbitrary_types_allowed = True
