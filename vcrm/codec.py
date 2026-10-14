"""Reserved for the new QA-trained ICAE history codec, after reader validation."""


class VersionCodec:
    def __init__(self, *args, **kwargs):
        raise RuntimeError('The TransMem difference/MSE codec is retired. Validate independent ICAE entries first; '
                           'the version-conditioned QA codec has not been defined or trained.')
