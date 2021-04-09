from fastai.train import Interpretation, DatasetType


class ASRInterpretation(Interpretation):
    def __init__(self, wers, cers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wers = wers
        self.cers = cers

    @classmethod
    def from_learner(cls, learn, ds_type:DatasetType=DatasetType.Valid, activ=None):
        "Gets preds, y_true, losses to construct base class from a learner"
        losses, wers, cers, preds, targets = learn.get_preds(ds_type=ds_type, activ=activ)
        return cls(learn=learn, wers=wers, cers=cers, preds=preds, y_true=targets, losses=losses, ds_type=ds_type)
