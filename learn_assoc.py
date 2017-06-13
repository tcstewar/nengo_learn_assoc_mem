import nengo
import nengo.spa as spa
import numpy as np


class LearningAssocMem(nengo.Network):
    def __init__(self, n_neurons, dimensions,
                 intercepts=nengo.dists.Uniform(0,0),
                 voja_tau=0.005,
                 voja_rate=1e-3,
                 pes_rate=1e-3,
                 label=None,
                 inhibit_synapse=0.01,
                 inhibit_strength=0.0005,
                 seed=None,
                 load_from=None
                 ):
        super(LearningAssocMem, self).__init__(label=label)
        if load_from is not None:
            data = np.load(load_from)
            encoders = data['enc']
            decoders = data['dec']
            if seed is None:
                seed = int(data['seed'])
        else:
            encoders = nengo.Default
            decoders = np.zeros((dimensions, n_neurons), dtype=float)

        self.seed = seed

        with self:
            self.mem = nengo.Ensemble(n_neurons=n_neurons,
                                      dimensions=dimensions,
                                      intercepts=intercepts,
                                      encoders=encoders,
                                      seed=seed,
                                      )
            self.input = nengo.Node(None, size_in=dimensions)
            self.output = nengo.Node(None, size_in=dimensions)
            self.correct = nengo.Node(None, size_in=dimensions)

            if voja_rate > 0:
                learning_rule_type = nengo.Voja(post_tau=voja_tau,
                                                learning_rate=voja_rate)
            else:
                learning_rule_type = None
            nengo.Connection(self.input, self.mem,
                             learning_rule_type=learning_rule_type)


            if pes_rate > 0:
                learning_rule_type = nengo.PES(learning_rate=pes_rate)
            else:
                learning_rule_type = None

            self.conn_out = nengo.Connection(self.mem.neurons, self.output,
                transform=decoders,
                learning_rule_type=learning_rule_type,
                )

            if pes_rate > 0:
                nengo.Connection(self.output, self.conn_out.learning_rule)
                nengo.Connection(self.correct, self.conn_out.learning_rule,
                                 transform=-1)

            nengo.Connection(self.mem.neurons, self.mem.neurons,
                transform=inhibit_strength*(np.eye(self.mem.n_neurons)-1),
                synapse=inhibit_synapse)

    def create_weight_probes(self):
        with self:
            self.probe_encoder = nengo.Probe(self.mem, 'scaled_encoders',
                                             sample_every=0.1)
            self.probe_decoder = nengo.Probe(self.conn_out, 'weights',
                                             sample_every=0.1)

    def save(self, filename, sim):
        enc = sim.data[self.probe_encoder][-1]
        dec = sim.data[self.probe_decoder][-1]
        np.savez(filename, enc=enc, dec=dec, seed=self.seed)

