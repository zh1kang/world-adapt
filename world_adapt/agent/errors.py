"""Errors shared by the model-call layers, kept free of imports so any of them can raise these."""


class InfrastructureError(RuntimeError):
    """The model API failed after every retry. Reported separately from agent outcomes."""
