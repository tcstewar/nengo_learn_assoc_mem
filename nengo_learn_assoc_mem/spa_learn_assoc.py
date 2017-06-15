import nengo
import nengo.spa as spa

from . import learn_assoc

class SPALearningAssocMem(spa.module.Module):
    def __init__(self, n_neurons, dimensions, input_vocab, output_vocab,
                 label=None, seed=None,
                 add_to_container=None, **keys):
        super(spa.module.Module, self).__init__(label=label, seed=seed,
                            add_to_container=add_to_container)

        with self:
            self.mem = learn_assoc.LearningAssocMem(
                            n_neurons=n_neurons,
                            dimensions=dimensions,
                            **keys)

        self.input = self.mem.input
        self.output = self.mem.output
        self.correct = self.mem.correct
        self.inputs = dict(default=(self.input, input_vocab))
        self.outputs = dict(default=(self.output, output_vocab))
