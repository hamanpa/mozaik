# encoding: utf-8
"""
Mozaik connector interface.
"""
import math
import numpy
import mozaik
import time
from mozaik.core import BaseComponent
from mozaik.sheets.vision import SheetWithMagnificationFactor
from parameters import ParameterSet
from mozaik.tools.misc import sample_from_bin_distribution, normal_function
from collections import Counter
from pyNN import random, space

from builtins import zip

logger = mozaik.getMozaikLogger()

class Connector(BaseComponent):
    """
    An abstract interface class for Connectors in mozaik. Each mozaik connector should derive from this class and implement 
    the _connect function. The usage is: create the instance of MozaikConnector and call connect() to realize the connections.
    """
    required_parameters = ParameterSet({
            'target_synapses' : str,
            'short_term_plasticity': ParameterSet({
                    'U': float, 
                    'tau_rec': float, 
                    'tau_fac': float,
                    'tau_psc': float
            }),
    })
    
    def __init__(self, model, name, source, target, parameters):
        logger.info("Creating %s between %s and %s" % (self.__class__.__name__,
                                                       source.__class__.__name__,
                                                       target.__class__.__name__))
        BaseComponent.__init__(self, model, parameters)
        self.name = name
        self.model.register_connector(self)
        self.sim = self.model.sim
        self.source = source
        self.target = target
        self.input = source
        self.target.input = self
        self.cl = None
        
        self.weight_scaler = 1.0 # This scaler has to be always applied to all weights just before sent to pyNN connect command
                                 # This is because certain pyNN synaptic models interpret weights with different units and the Connector
                                 # function here corrects for these - ie. the Connectors in Mozaik will always assume the weights to be in nano-siemens 
        if self.parameters.short_term_plasticity != None:
           self.weight_scaler = 1000.0

    
    def init_synaptic_mechanisms(self,weight=None,delay=None):
      if not self.parameters.short_term_plasticity != None:
        sm = self.sim.StaticSynapse(weight=weight,delay=delay)                   
      else:
        if weight != None:
            sm = self.sim.native_synapse_type("tsodyks_synapse")(weight=weight,delay=delay,**self.parameters.short_term_plasticity)                   
        else:
            sm = self.sim.native_synapse_type("tsodyks_synapse")(**self.parameters.short_term_plasticity)                   
      return sm
        
    def connect(self):
          t0 = time.time()
          self._connect()
          connect_time = time.time() - t0
          logger.info('Connector %s took %.0fs to compute' % (self.__class__.__name__,connect_time))
            
        
    def _connect(self):
      raise NotImplementedError

    def connection_field_plot_continuous(self, index, afferent=True, density=30):
        weights = numpy.array(self.proj.get('weight', format='list', gather=True))
        if afferent:
            idx = numpy.array(numpy.flatnonzero(weights[:,1].flatten()==index))
            x = self.proj.pre.positions[0][weights[idx,0].astype(int)]
            y = self.proj.pre.positions[1][weights[idx,0].astype(int)]
            w = weights[idx,2]
        else:
            idx = numpy.flatnonzero(weights[:,0]==index)
            x = self.proj.post.positions[0][weights[idx,1].astype(int)]
            y = self.proj.post.positions[1][weights[idx,1].astype(int)]
            w = weights[idx,2]

        xi = numpy.linspace(min(x), max(x), 100)
        yi = numpy.linspace(min(y), max(y), 100)
        zi = griddata(x, y, w, xi, yi)

    def store_connections(self, datastore):
        from mozaik.analysis.data_structures import Connections
        # TODO: write docstring that this method is obsolete and is left for
        # compatibility reasons - should not be accessed directly, only for 
        # certail connections created within PyNN, where the connection list
        # is not available

        data = self.proj.get(['weight', 'delay'], format='list', gather=True)
        data = numpy.array(data)
        weights, delays = data[:,[0,1,2]], data[:,[0,1,3]]

        if (not mozaik.mpi_comm) or (mozaik.mpi_comm.rank == mozaik.MPI_ROOT):
            datastore.add_analysis_result(
                Connections(
                            weights,delays,
                            source_size=(self.source.size_x,self.source.size_y),
                            target_size=(self.target.size_x,self.target.size_y),
                            proj_name=self.name,
                            source_name=self.source.name,
                            target_name=self.target.name,
                            analysis_algorithm='connection storage',
                            datastore_path=datastore.parameters.root_directory))



class SpecificConnector(Connector):
    """
    An abstract class for Connectors with connections specified by a given connection list.

    This class implements _connect function based on the given connection list, therefore every derived SpecificConnector class should implement _obtain_connection_list method, which reutrns a connection list.
     
    Connection list should be in the form of list of tuples (or equivalent two dimensional numpy array). Each tuple represents a single connection and should look like: (pre_idx, post_idx, weight, delay), where 'pre_idx' is the index (i.e. order in the Population, not the ID) of presynaptic neuron, 'post_idx' is the index of postsynaptic neuron, 'weight' is the weight of the connection and 'delay' is the delay of the synaptic transmission.
    """


    def _connect(self):
        conn_list = self._obtain_connection_list()
        method = self.sim.FromListConnector(conn_list)
        if len(conn_list) > 0:
            if self.model.parameters.store_connections:
                self.cl = method.conn_list
            self.proj = self.sim.Projection(
                                self.source.pop,
                                self.target.pop,
                                method,
                                synapse_type=self.init_synaptic_mechanisms(),
                                label=self.name,
                                receptor_type=self.parameters.target_synapses)
        else:
            logger.warning("%s(%s): empty projection - pyNN projection not created." % (self.name,self.__class__.__name__))

    
    def _obtain_connection_list(self):
        raise NotImplementedError


class SpecificArborization(SpecificConnector):
    """
    Generic connector which gets directly list of connections as the list of
    quadruplets as accepted by the pyNN FromListConnector.

    This connector cannot be parametrized directly via the parameter file
    because that does not support list of tuples.
    
    This connector also gets rid of very weak synapses (below one-hundreth of the maximum synapse)
    """

    required_parameters = ParameterSet({
        'weight_factor': float,  # the overall (sum) weight that a single target neuron should receive
    })

    def __init__(self, network, source, target, connection_matrix,delay_matrix, parameters, name):
        Connector.__init__(self, network, name, source,
                                             target, parameters)
        self.connection_matrix = connection_matrix
        self.delay_matrix = delay_matrix

    def _obtain_connection_list(self):
        X = numpy.zeros(self.connection_matrix.shape)
        Y = numpy.zeros(self.connection_matrix.shape)
        
        for x in range(0,X.shape[0]):
            for y in range(0,X.shape[1]):
                X[x][y] = x
                Y[x][y] = y
        
        for i in range(0,self.target.pop.size):
            self.connection_matrix[:,i] = self.connection_matrix[:,i] / numpy.sum(self.connection_matrix[:,i])*self.parameters.weight_factor

        # This is due to native synapses models (which we currently use as the short term synaptic plasticity model) 
        # do not apply the 1000 factor scaler as the pyNN synaptic models
        self.connection_matrix = self.connection_matrix * self.weight_scaler
        self.connection_list = list(zip(numpy.array(X).flatten(),numpy.array(Y).flatten(),self.connection_matrix.flatten(),self.delay_matrix.flatten()))
        # get rid of very weak synapses
        z = numpy.max(self.connection_matrix.flatten())
        self.connection_list = [(int(a),int(b),c,d) for (a,b,c,d) in self.connection_list if c>(z/100.0)]

        return self.connection_list


class SpecificProbabilisticArborization(SpecificConnector):
    """
    Generic connector which gets directly list of connections as the list
    of quadruplets as accepted by the pyNN FromListConnector.

    It interprets the weights as proportional probabilities of connectivity,
    and for each neuron out connections it samples num_samples of
    connections that actually get realized according to these weights.
    Each such sample connections will have weight equal to
    weight_factor/num_samples but note that there can be multiple
    connections between a pair of neurons in this sample (in which case the
    weights are set to the multiple of the base weights times the number of
    occurrences in the sample).

    This connector cannot be parameterized directly via the parameter file
    because that does not support list of tuples.
    """

    required_parameters = ParameterSet({
        'weight_factor': float,  # the overall strength of synapses in this connection per neuron (in µS) (i.e. the sum of the strength of synapses in this connection per target neuron)
        'num_samples': int
    })

    def __init__(self, network, source, target, connection_matrix,delay_matrix, parameters, name):
        Connector.__init__(self, network, name, source,target, parameters)
        self.connection_matrix = connection_matrix
        self.delay_matrix = delay_matrix

    def _obtain_connection_list(self):
        # This is due to native synapses models (which we currently use as the short term synaptic plasticity model) 
        # do not apply the 1000 factor scaler as the pyNN synaptic models
        wf = self.parameters.weight_factor * self.weight_scaler
        seeds = mozaik.get_seeds(self.target.pop.size)
        weights = self.connection_matrix
        delays = self.delay_matrix
        cl = []
        for i in range(0,self.target.pop.size):
            co = Counter(
                sample_from_bin_distribution(
                    weights[:, i].flatten(), int(self.parameters.num_samples), seeds[i]
                )
            )
            cl.extend([(int(k),int(i),wf*co[k]/self.parameters.num_samples,delays[k][i]) for k in co.keys()])
            
        return cl
