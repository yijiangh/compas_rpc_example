import compas
import compas_rhino

from compas.datastructures import Mesh
from compas.utilities import XFunc

# make the function available as a wrapped function with the same call signature and return value as the original.
fd_numpy = XFunc('compas.numerical.fd_numpy')

mesh = Mesh.from_obj(compas.get('faces.obj'))

mesh.update_default_vertex_attributes({'is_fixed': False, 'px': 0.0, 'py': 0.0, 'pz': 0.0})
mesh.update_default_edge_attributes({'q': 1.0})

for key, attr in mesh.vertices(True):
    attr['is_fixed'] = mesh.vertex_degree(key) == 2

key_index = mesh.key_index()
vertices  = mesh.get_vertices_attributes('xyz')
edges     = [(key_index[u], key_index[v]) for u, v in mesh.edges()]
fixed     = [key_index[key] for key in mesh.vertices_where({'is_fixed': True})]
q         = mesh.get_edges_attribute('q', 1.0)
loads     = mesh.get_vertices_attributes(('px', 'py', 'pz'), (0.0, 0.0, 0.0))

xyz, q, f, l, r = fd_numpy(vertices, edges, fixed, q, loads)

for key, attr in mesh.vertices(True):
    attr['x'] = xyz[key][0]
    attr['y'] = xyz[key][1]
    attr['z'] = xyz[key][2]