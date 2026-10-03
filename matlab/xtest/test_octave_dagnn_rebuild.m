function test_octave_dagnn_rebuild()
% Check graph fan-in/out assignments in both Octave and MATLAB.

net = dagnn_bc.DagNN() ;
net.addLayer('conv', dagnn_bc.Conv(), {'input'}, {'hidden'}, {'weights', 'bias'}) ;
net.addLayer('relu', dagnn_bc.ReLU(), {'hidden'}, {'output'}, {}) ;
net.rebuild() ;

assert(isequal([net.vars.fanin], [0 1 1])) ;
assert(isequal([net.vars.fanout], [1 1 0])) ;
assert(isequal([net.params.fanout], [1 1])) ;
assert(isequal(net.getInputs(), {'input'})) ;
assert(isequal(net.getOutputs(), {'output'})) ;
assert(isequal(net.getLayerExecutionOrder(), [1 2])) ;

% Also exercise a network without parameters.
net = dagnn_bc.DagNN() ;
net.addLayer('relu', dagnn_bc.ReLU(), {'input'}, {'output'}, {}) ;
net.rebuild() ;
assert(isequal([net.vars.fanin], [0 1])) ;
assert(isequal([net.vars.fanout], [1 0])) ;
assert(isempty(net.params)) ;
disp('test_octave_dagnn_rebuild: PASS') ;
end
