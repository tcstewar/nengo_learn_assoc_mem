import pytry
import nengo
import nengo.spa as spa
import numpy as np
import random

import sys
if '.' not in sys.path:
    sys.path.append('.')
import learn_assoc

class LearningAssocMemTrial(pytry.NengoTrial):
    def params(self):
        self.param('number of neurons', n_neurons=10)
        self.param('number of dimensions', dimensions=32)
        self.param('intercept', intercept=0.0)
        self.param('VOja time constance', voja_tau=0.005)
        self.param('VOja learning rate', voja_rate=1e-3)
        self.param('PES learning rate', pes_rate=1e-3)
        self.param('WTA synapse', inhibit_synapse=0.01)
        self.param('WTA strength', inhibit_strength=0.0005)
        self.param('presentation time', t_present=0.1)
        self.param('number of items', n_items=6)
        self.param('number of presentations per item', n_present=4)
        self.param('lower scale on input values', input_scale=0.5)
        self.param('filename for saving/loading', filename='weights.npz')
        self.param('load weights at beginning', load=False)
        self.param('save weights at end', save=False)

    def model(self, p):
        self.items = list(range(p.n_items))*p.n_present
        random.shuffle(self.items)

        vocab_items = ['X%d' % i for i in self.items]
        vocab = spa.Vocabulary(p.dimensions)

        model = nengo.Network()
        with model:
            def stim_func(t):
                index = int(t / p.t_present)
                scale = 1.0
                if index > len(vocab_items)/2:
                    scale = p.input_scale
                index = index % len(vocab_items)
                return vocab.parse('%g * %s' % (scale, vocab_items[index])).v
            stim = nengo.Node(stim_func)

            def correct_func(t):
                index = int(t / p.t_present)
                scale = 1.0
                index = index % len(vocab_items)
                return vocab.parse('%g * %s' % (scale, vocab_items[index])).v
            correct = nengo.Node(correct_func)

            self.mem = learn_assoc.LearningAssocMem(
                    n_neurons=p.n_neurons,
                    dimensions=p.dimensions,
                    intercepts=nengo.dists.Uniform(p.intercept, p.intercept),
                    voja_tau=p.voja_tau,
                    voja_rate=p.voja_rate,
                    pes_rate=p.pes_rate,
                    inhibit_synapse=p.inhibit_synapse,
                    inhibit_strength=p.inhibit_strength,
                    load_from = None if not p.load else p.filename,
                    )
            if p.save:
                self.mem.create_weight_probes()
            nengo.Connection(stim, self.mem.input, synapse=None)
            nengo.Connection(correct, self.mem.correct, synapse=None)

            self.p_output = nengo.Probe(self.mem.output, synapse=0.01)
            self.p_ideal = nengo.Probe(correct, synapse=0.01)

        return model

    def evaluate(self, p, sim, plt):
        sim.run(p.n_present*p.n_items*p.t_present)
        if p.save:
            self.mem.save(p.filename, sim)

        data = sim.data[self.p_output]
        ideal = sim.data[self.p_ideal]
        accuracy = np.sum(data * ideal, axis=1)

        times = np.arange(p.n_items*p.n_present)*p.t_present
        indices = (times / p.dt).astype(int) - 1

        scores = [[] for i in range(p.n_items)]

        for i, index in enumerate(indices):
            a = accuracy[index]
            item = self.items[i]
            scores[item].append(a)

        scores = np.array(scores)

        if plt:
            plt.plot(scores.T, alpha=0.3)
            plt.plot(np.mean(scores, axis=0))








