import compas
import time

from compas.datastructures import Mesh
from compas_rhino.artists import MeshArtist
from compas.rpc import Proxy

numerical = Proxy('compas.numerical')

mesh = Mesh.from_obj(compas.get('faces.obj'))

mesh.update_default_vertex_attributes({'px': 0.0, 'py': 0.0, 'pz': 0.0})
mesh.update_default_edge_attributes({'q': 1.0})

key_index = mesh.key_index()

xyz   = mesh.get_vertices_attributes('xyz')
edges = [(key_index[u], key_index[v]) for u, v in mesh.edges()]
fixed = [key_index[key] for key in mesh.vertices_where({'vertex_degree': 2})]
q     = mesh.get_edges_attribute('q', 1.0)
loads = mesh.get_vertices_attributes(('px', 'py', 'pz'), (0.0, 0.0, 0.0))

xyz, q, f, l, r = numerical.fd_numpy(xyz, edges, fixed, q, loads)

for key, attr in mesh.vertices(True):
    index = key
    attr['x'] = xyz[index][0]
    attr['y'] = xyz[index][1]
    attr['z'] = xyz[index][2]
    attr['rx'] = r[index][0]
    attr['ry'] = r[index][1]
    attr['rz'] = r[index][2]

for index, (u, v, attr) in enumerate(mesh.edges(True)):
    attr['f'] = f[index][0]
    attr['l'] = l[index][0]
#    
#artist = MeshArtist(mesh)
#artist.draw_vertices()
#artist.draw_edges()