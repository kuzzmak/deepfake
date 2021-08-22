import abc

from core.face_detection.algorithms.ModelFactory import ModelFactory

from enums import DEVICE


class BaseModelMeta(abc.ABCMeta):
    ...


class BaseModel(metaclass=BaseModelMeta):
    """Base class which every face detection, landmark detection algorithm
    should inherit. On object construction, model is loaded from factory
    and moved to corresponding device. Every subclass can define abstract
    methods which should be implemented by a particular model.
    """

    def __init__(self, model_factory: ModelFactory, device: DEVICE):
        """Constructor.

        Parameters
        ----------
        model_factory : ModelFactory
            class of the model factory
        device : DEVICE
            computation device
        """
        self.device = device
        self.model = model_factory.build_model(device)
        self.model.eval()
        self.model.to(device.value)
