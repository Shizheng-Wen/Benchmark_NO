import inspect
import torch 
import torch.nn as nn 
from typing import Optional, Union, Mapping, TypeVar, Tuple, Callable
from .edges import sort_edges, remove_duplicate_edges
from .tri import delaunay_edges
from .domain import shift
from .encode import node_pos_encode, edge_pos_encode
from src.utils.buffer import BufferDict
from src.utils.pair import make_pair, is_pair, force_make_pair

T = TypeVar("T")
def make_dict(data:Union[Mapping[str,T],T], default_key:str = 'x')->Mapping[str,T]:
    if not isinstance(data, Mapping):
        return {default_key:data}
    else:
        return data

def radius_bipartite_graph(
    points_a:torch.Tensor,
    points_b:torch.Tensor,
    radii_b:torch.Tensor,
    periodic:bool = False,
    p:float = 2,
    )->torch.Tensor:
    """
    Compute a biparite graph between points_a and points_b by radius
    Parameters
    ----------
    points_a: torch.Tensor
        2D tensor of shape [n_points_a, n_dimension]
    points_b: torch.Tensor
        2D tensor of shape [n_points_b, n_dimension]
    radii_b: torch.Tensor
        1D tensor of shape [n_points_b,]
    periodic: bool
        whether the graph is periodic
    p: float
        the p-norm to use for the distance

    Returns
    -------
    bipartite_edges: torch.Tensor
        2D tensor of shape [2, n_edges]
    """
    assert points_a.ndim == 2 and points_b.ndim == 2, f"The points_a and points_b are expected to be 2D tensors, but got shapes {points_a.shape} and {points_b.shape}"
    assert points_b.shape[0] == radii_b.shape[0], f"The points_b and radii_b should have the same number of points, but got shapes {points_b.shape} and {radii_b.shape}"
    if periodic:
        residual = points_a[:, None, :] - points_b[None, :, :]
        residual = torch.where(residual >= 1., residual - 2., residual)
        residual = torch.where(residual < -1., residual + 2., residual)
        distances = torch.linalg.norm(residual, axis=-1, ord=p) # [n_points_a, n_points_b]
    else:
        distances = torch.cdist(points_a, points_b, p=p) # [n_points_a, n_points_b]

    bipartite_edges = torch.stack(torch.where(distances < radii_b[None, :]), 0) # [2, n_edges]
    return bipartite_edges

def hierarchical_graph(
    points:torch.Tensor, # [n_points, n_dim]
    level:int,
    sample_factor:float = 2.,
    domain_shifts:Optional[torch.Tensor] = None, # [n_shifts, n_dim]
    return_levels:bool = False
    )->Tuple[torch.Tensor, torch.Tensor]:
    """
    Parameters
    ----------
    points: torch.Tensor
        2D tensor of shape [n_points, n_dim]
    level: int
        number of levels
    periodic: bool
        whether the graph is periodic
    sample_factor: float
        factor of subsampling
    domain_shifts: Optional[torch.Tensor]
        2D tensor of shape [n_shifts, n_dim]

    Returns
    -------
    edges: torch.Tensor
        2D tensor of shape [2, n_edges]
    domains: torch.Tensor
        2D tensor of shape [2, n_edges]
    """
    assert level > 0, "The level should be positive"
    edges   = []
    domains = []
    if return_levels:
        levels = []
    for l in range(level):
        # Sub-sample the rmesh
        num_sampled  = int(points.shape[0] / (sample_factor ** l))
        if num_sampled < 4:
            continue
        level_points = points[:num_sampled]
        # Construct a triangulation
        if domain_shifts is not None:
            # Repeat the rmesh in periodic directions
            level_points = shift(level_points, domain_shifts)
        # Get the relevant edges
        extended_edges = delaunay_edges(level_points) # [2, n_edges]
        level_domains  = extended_edges // num_sampled # [2, n_edges]

        level_edges    = extended_edges % num_sampled # [2, n_edges]
      
        if domain_shifts is  not None: # periodic
            is_relevant = torch.any(level_domains == 0, dim=0)
        else: # not periodic
            is_relevant = torch.all(level_domains == 0, dim=0)
        
        level_edges   = level_edges  [:, is_relevant] # [2, n_edges]
        level_domains = level_domains[:, is_relevant] # [2, n_edges]

        if return_levels:
            levels.extend([l] * level_edges.shape[1])

        edges.append(level_edges)
        domains.append(level_domains)

    edges = torch.cat(edges, 1) # [2, n_edges]
    domains = torch.cat(domains, 1) # [2, n_edges]
    if return_levels:
        levels = torch.tensor(levels)

    edges, sort_idx = sort_edges(edges, return_idx=True)
    domains = domains[:, sort_idx]
    edges, unique_idx = remove_duplicate_edges(edges, return_idx=True)
    domains = domains[:, unique_idx]
    if return_levels:
        levels = levels[sort_idx]
        levels = levels[unique_idx]

    if return_levels:
        return edges, domains, levels
    else:
        return edges, domains

def message(graph:'Graph', fn:Callable, 
                    ndata:Optional[
                        Union[
                            torch.Tensor,
                            Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]
                        ] = None,
                    edata:Optional[torch.Tensor] = None,
                    gdata:Optional[torch.Tensor] = None):
    """
    Parameters
    ----------
    graph: Graph
    fn: Callable
        function to apply to the edge data
        the function should recieve parameters

        src_ndata:torch.Tensor
            ND tensor of shape [..., n_edges, n_src_features]
        dst_ndata:torch.Tensor
            ND tensor of shape [..., n_edges, n_dst_features]
        edata:torch.Tensor
            ND tensor of shape [..., n_edges, n_features]
        
        additionally, you could add gndata:Optional[torch.Tensor] as the last argument 
            ND tensor of shape [..., n_src/dst_nodes,n_features]

        and return a tensor of shape [..., n_edges, n_features]

    ndata: Optional[torch.Tensor or Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]
        ND tensor of shape [..., n_nodes, n_features]
        if graph is bipartie, ndata could be None or Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]
        else:            

    edata: Optional[torch.Tensor]
        ND tensor of shape [..., n_edges, n_features]

    gdata: Optional[torch.Tensor]
        ND tensor of shape [..., n_features]


    Returns
    -------
    message: torch.Tensor
        ND tensor of shape [..., n_edges, n_features]
    """
    if graph.is_bipartite:
        assert ndata is None or is_pair(ndata), f"The ndata is expected to be pair, but got {type(ndata)}"

    sig = inspect.signature(fn)
    n_parameters = len(sig.parameters)

    ndata = make_pair(ndata)
    src_ndata, dst_ndata = ndata 
    if src_ndata is not None:
        assert src_ndata.shape[-2] == make_pair(graph.num_nodes)[0]
        src_ndata = src_ndata[..., graph.edges[0], :]
    if dst_ndata is not None:
        assert dst_ndata.shape[-2] == make_pair(graph.num_nodes)[1]
        dst_ndata = dst_ndata[..., graph.edges[1], :]
    if edata is not None:
        assert edata.shape[-2] == graph.num_edges

    if n_parameters == 2:
        return fn(src_ndata, dst_ndata)
    elif n_parameters == 3:
        return fn(src_ndata, dst_ndata, edata)
    elif n_parameters == 4:
        if gdata is not None:
            expand_dim = [-1] * (gdata.ndim + 1)
            expand_dim[-2] = graph.num_edges  
            gdata = gdata[..., None, :].expand(*expand_dim)
        return fn(src_ndata, dst_ndata, edata, gdata)
    else:
        raise ValueError(f"The function {fn} should have 2, 3 or 4 parameters, but got {n_parameters}")

def aggregate(graph:'Graph', fn:Union[Callable, Tuple[Callable, Callable]], 
                  edata:torch.Tensor,
                  ndata:Optional[torch.Tensor]=None,
                  gdata:Optional[torch.Tensor]=None,
                  dtype:Optional[torch.dtype] =None,
                  reduce:str="mean"):
    """
    Parameters
    ----------
    graph:Graph
    fn: Callable
        function to apply to the edge data
        the function should recieve parameters

        for bipartite graph:

            edata:torch.Tensor
                ND Tensor of shape [..., n_src/dst_nodes, n_edge_features]
            ndata:Optional[torch.Tensor]
                ND Tensor of shape [..., n_src/dst_nodes, n_node_features]

            additionally, you could add gndata:Optional[torch.Tensor] as the last argument 
                ND tensor of shape [..., n_src/dst_nodes,n_features]

        for homogeneous graph:

            src_edata:torch.Tensor
                ND Tensor of shape [..., n_nodes, n_edge_features]
            dst_edata:torch.Tensor
                ND Tensor of shape [..., n_nodes, n_edge_features]
            
            ndata: Optional[torch.Tensor]
                ND tensor of shape [..., n_nodes, n_features]

            additionally, you could add gndata:Optional[torch.Tensor] as the last argument 
                ND tensor of shape [..., n_nodes,n_features]

    edata: torch.Tensor
        ND tensor of shape [...,  n_edges, n_features]

    ndata: Optional[torch.Tensor]
        ND tensor of shape [..., n_nodes, n_features]

    gdata: Optional[torch.Tensor]
        ND tensor of shape [..., n_features]
        
    reduce: str
        reduce operation to apply to the aggregated data
        choose from ["sum", "mean", "prod", "amax", "amin"]
        
    """
    assert reduce in ["sum", "mean", "prod", "amax", "amin"], \
        f'The reduce is expected one of ["sum", "mean", "prod", "amax", "amin"], but got {reduce}'
    dtype = edata.dtype if dtype is None else dtype
    src_shape, dst_shape = force_make_pair(list(edata.shape))
    src_shape[-2] = make_pair(graph.num_nodes)[0]
    dst_shape[-2] = make_pair(graph.num_nodes)[1]

    src_edata = torch.zeros(*src_shape, dtype=dtype, device=edata.device)
    dst_edata = torch.zeros(*dst_shape, dtype=dtype, device=edata.device)      

    src_edata = src_edata.index_reduce_(dim=-2, index=graph.edges[0], source=edata, reduce=reduce)  
    dst_edata = dst_edata.index_reduce_(dim=-2, index=graph.edges[1], source=edata, reduce=reduce)
    
    if is_pair(fn):
        n_parameters = [len(inspect.signature(f).parameters) for f in fn]
        assert n_parameters[0] == n_parameters[1], f"The functions {fn} should have the same number of parameters, but got {n_parameters}"
        n_parameters = n_parameters[0]
    else:
        assert isinstance(fn, Callable), f"The fn is expected to be Callable, but got {type(fn)}"
        n_parameters = len(inspect.signature(fn).parameters)

    if graph.is_bipartite:
        
        src_ndata, dst_ndata = make_pair(ndata)
        if n_parameters == 2:
            fn = make_pair(fn)
            return fn[0](src_edata, src_ndata), fn[1](dst_edata, dst_ndata)
        elif n_parameters == 3:
            if gdata is not None:
                expand_dim = [-1] * (gdata.ndim + 1)
                expand_dim[-2] = graph.num_nodes[0]  
                src_gdata = gdata[..., None, :].expand(*expand_dim)
                expand_dim[-2] = graph.num_nodes[1]  
                dst_gdata = gdata[..., None, :].expand(*expand_dim)
                fn = make_pair(fn)
            return fn[0](src_edata, src_ndata, src_gdata), fn[1](dst_edata, dst_ndata, dst_gdata)
        else:
            raise ValueError(f"The function {fn} should have 2 or 3 parameters, but got {n_parameters}")
    else:
        # NOTE: we only consider the directed graph here
        if n_parameters == 3:
            return fn(src_edata, dst_edata, ndata)
        elif n_parameters == 4:
            if gdata is not None:
                expand_dim = [-1] * (gdata.ndim + 1)
                expand_dim[-2] = graph.num_nodes  
                gdata = gdata[..., None, :].expand(*expand_dim)
            return fn(src_edata,dst_edata, ndata, gdata)
        else:
            raise ValueError(f"The function {fn} should have 2 or 3 parameters, but got {n_parameters}")

class Graph(nn.Module):
    edges:torch.Tensor # [2, edges]
    ndata:Optional[BufferDict] # [..., n_nodes, n_features]
    src_ndata:Optional[BufferDict] # [..., n_src_nodes, n_features]
    dst_ndata:Optional[BufferDict] # [..., n_dst_nodes, n_features]
    edata:Optional[BufferDict] # [..., n_edges, n_features]
    gdata:Optional[BufferDict] # [..., n_features]


    def __init__(self, 
                 edges:torch.Tensor, 
                 ndata:Optional[Union[Mapping[str,torch.Tensor],torch.Tensor]] = None,
                 src_ndata:Optional[Union[Mapping[str,torch.Tensor],torch.Tensor]] = None, 
                 dst_ndata:Optional[Union[Mapping[str,torch.Tensor],torch.Tensor]] = None,
                 edata:Optional[Union[Mapping[str,torch.Tensor],torch.Tensor]] = None,
                 gdata:Optional[Union[Mapping[str,torch.Tensor],torch.Tensor]] = None,
                 batch_identifier:Optional[bool] = None):
        super().__init__()

        self.register_buffer("edges", edges)

        if ndata is not None:
            self.ndata = BufferDict(make_dict(ndata))
            self.src_ndata = None
            self.dst_ndata = None
        else:
            self.ndata = None 
            self.src_ndata = BufferDict(make_dict(src_ndata)) if src_ndata is not None else None 
            self.dst_ndata = BufferDict(make_dict(dst_ndata)) if dst_ndata is not None else None
        
        if edata is not None:
            #assert edata.shape[0] == edges.shape[1]
            self.edata = BufferDict(make_dict(edata))
        else:
            self.edata = None   

        if gdata is not None:
            self.gdata = BufferDict(make_dict(gdata))
        else:
            self.gdata = None

        self.batch_identifier = batch_identifier
    @property 
    def ndim(self)->Union[int, Tuple[int, int]]:
        if self.ndata is not None:
            assert 'x' in self.ndata, f"The node data should have a key 'x', got keys {self.ndata.keys()}"
            return self.ndata['x'].shape[-1]
        else:
            assert 'x' in self.src_ndata, f"The node data should have a key 'x', got keys {self.src_ndata.keys()}"
            assert 'x' in self.dst_ndata, f"The node data should have a key 'x', got keys {self.dst_ndata.keys()}"
            return self.src_ndata['x'].shape[-1], self.dst_ndata['x'].shape[-1]
        
    @property 
    def edim(self)->int:
        assert 'x' in  self.edata, f"The edge data should have a key 'x', got keys {self.edata.keys()}"
        return self.edata['x'].shape[-1]
    
    @property 
    def src_ndim(self)->int:
        assert self.is_bipartite, "The graph is not bipartite"
        assert 'x' in self.src_ndata, f"The node data should have a key 'x', got keys {self.src_ndata.keys()}"
        return self.src_ndata['x'].shape[-1]
    
    @property
    def dst_ndim(self)->int:
        assert self.is_bipartite, "The graph is not bipartite"
        assert 'x' in self.dst_ndata, f"The node data should have a key 'x', got keys {self.dst_ndata.keys()}"
        return self.dst_ndata['x'].shape[-1]

    @property 
    def num_edges(self)->int:
        return self.edges.shape[1]
    
    @property 
    def is_bipartite(self)->bool:
        return self.src_ndata is not None and self.dst_ndata is not None

    @property 
    def num_nodes(self)->Union[int, Tuple[int, int]]:
        if self.ndata is not None:
            assert 'x' in self.ndata, f"The node data should have a key 'x', got keys {self.ndata.keys()}"
            return self.ndata['x'].shape[-2]
        else:
            assert 'x' in self.src_ndata, f"The node data should have a key 'x', got keys {self.src_ndata.keys()}"
            return self.src_ndata['x'].shape[-2], self.dst_ndata['x'].shape[-2]

    @property 
    def num_src_nodes(self):
        assert self.is_bipartite, "The graph is not bipartite"
        assert 'x' in self.src_ndata, f"The node data should have a key 'x', got keys {self.src_ndata.keys()}"
        return self.src_ndata['x'].shape[-2]
    
    @property 
    def num_dst_nodes(self):
        assert self.is_bipartite, "The graph is not bipartite"
        assert 'x' in self.dst_ndata, f"The node data should have a key 'x', got keys {self.dst_ndata.keys()}"
        return self.dst_ndata['x'].shape[-2]

    def message(self, fn:Callable, 
                    ndata:Optional[
                        Union[
                            torch.Tensor,
                            Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]
                        ] = None,
                    edata:Optional[torch.Tensor] = None,
                    gdata:Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        fn: Callable
            function to apply to the edge data
            the function should recieve parameters

            src_ndata:torch.Tensor
                ND tensor of shape [..., n_edges, n_src_features]
            dst_ndata:torch.Tensor
                ND tensor of shape [..., n_edges, n_dst_features]
            edata:torch.Tensor
                ND tensor of shape [..., n_edges, n_features]
            
            additionally, you could add gndata:Optional[torch.Tensor] as the last argument 
                ND tensor of shape [..., n_src/dst_nodes,n_features]

            and return a tensor of shape [..., n_edges, n_features]

        ndata: Optional[torch.Tensor or Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]
            ND tensor of shape [..., n_nodes, n_features]
            if graph is bipartie, ndata could be None or Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]
            else:            

        edata: Optional[torch.Tensor]
            ND tensor of shape [..., n_edges, n_features]

        gdata: Optional[torch.Tensor]
            ND tensor of shape [..., n_features]


        Returns
        -------
        message: torch.Tensor
            ND tensor of shape [..., n_edges, n_features]
        """
        if self.is_bipartite:
            assert ndata is None or is_pair(ndata), f"The ndata is expected to be pair, but got {type(ndata)}"
    
        sig = inspect.signature(fn)
        n_parameters = len(sig.parameters)

        ndata = make_pair(ndata)
        src_ndata, dst_ndata = ndata # [n_src_nodes, n_src_features], [n_dst_nodes, n_dst_features]
        if src_ndata is not None:
            assert src_ndata.shape[-2] == make_pair(self.num_nodes)[0]
            src_ndata = src_ndata[..., self.edges[0], :]
        if dst_ndata is not None:
            assert dst_ndata.shape[-2] == make_pair(self.num_nodes)[1]
            dst_ndata = dst_ndata[..., self.edges[1], :]
        if edata is not None:
            assert edata.shape[-2] == self.num_edges

        
        if n_parameters == 2:
            return fn(src_ndata, dst_ndata)
        elif n_parameters == 3:
            return fn(src_ndata, dst_ndata, edata)
        elif n_parameters == 4:
            if gdata is not None:
                expand_dim = [-1] * (gdata.ndim + 1)
                expand_dim[-2] = self.num_edges  
                gdata = gdata[..., None, :].expand(*expand_dim)
            return fn(src_ndata, dst_ndata, edata, gdata)
        else:
            raise ValueError(f"The function {fn} should have 2, 3 or 4 parameters, but got {n_parameters}")
              
    def aggregate(self, fn:Union[Callable, Tuple[Callable, Callable]], 
                  edata:torch.Tensor,
                  ndata:Optional[torch.Tensor]=None,
                  gdata:Optional[torch.Tensor]=None,
                  dtype:Optional[torch.dtype] =None,
                  reduce:str="mean"):
        """
        Parameters
        ----------
        fn: Callable
            function to apply to the edge data
            the function should recieve parameters

            for bipartite graph:

                edata:torch.Tensor
                    ND Tensor of shape [..., n_src/dst_nodes, n_edge_features]
                ndata:Optional[torch.Tensor]
                    ND Tensor of shape [..., n_src/dst_nodes, n_node_features]

                additionally, you could add gdata:Optional[torch.Tensor] as the last argument 
                    ND tensor of shape [..., n_src/dst_nodes,n_features]

            for homogeneous graph:

                src_edata:torch.Tensor
                    ND Tensor of shape [..., n_nodes, n_edge_features]
                dst_edata:torch.Tensor
                    ND Tensor of shape [..., n_nodes, n_edge_features]
                
                ndata: Optional[torch.Tensor]
                    ND tensor of shape [..., n_nodes, n_features]

                additionally, you could add gndata:Optional[torch.Tensor] as the last argument 
                    ND tensor of shape [..., n_nodes,n_features]

        edata: torch.Tensor
            ND tensor of shape [...,  n_edges, n_features]

        ndata: Optional[torch.Tensor]
            ND tensor of shape [..., n_nodes, n_features]

        gdata: Optional[torch.Tensor]
            ND tensor of shape [..., n_features]
            
        reduce: str
            reduce operation to apply to the aggregated data
            choose from ["sum", "mean", "prod", "amax", "amin"]
            
        """
        assert reduce in ["sum", "mean", "prod", "amax", "amin"], \
            f'The reduce is expected one of ["sum", "mean", "prod", "amax", "amin"], but got {reduce}'
        dtype = edata.dtype if dtype is None else dtype
        src_shape, dst_shape = force_make_pair(list(edata.shape))
        src_shape[-2] = make_pair(self.num_nodes)[0]
        dst_shape[-2] = make_pair(self.num_nodes)[1]

        src_edata = torch.zeros(*src_shape, dtype=dtype, device=edata.device)
        dst_edata = torch.zeros(*dst_shape, dtype=dtype, device=edata.device)      
    
        src_edata = src_edata.index_reduce_(dim=-2, index=self.edges[0], source=edata, reduce=reduce)  
        dst_edata = dst_edata.index_reduce_(dim=-2, index=self.edges[1], source=edata, reduce=reduce)
       
        if is_pair(fn):
            n_parameters = [len(inspect.signature(f).parameters) for f in fn]
            assert n_parameters[0] == n_parameters[1], f"The functions {fn} should have the same number of parameters, but got {n_parameters}"
            n_parameters = n_parameters[0]
        else:
            assert isinstance(fn, Callable), f"The fn is expected to be Callable, but got {type(fn)}"
            n_parameters = len(inspect.signature(fn).parameters)

        if self.is_bipartite:
           
            src_ndata, dst_ndata = make_pair(ndata)
            if n_parameters == 2:
                fn = make_pair(fn)
                return fn[0](src_edata, src_ndata), fn[1](dst_edata, dst_ndata)
            elif n_parameters == 3:
                if gdata is not None:
                    expand_dim = [-1] * (gdata.ndim + 1)
                    expand_dim[-2] = self.num_nodes[0]  
                    src_gdata = gdata[..., None, :].expand(*expand_dim)
                    expand_dim[-2] = self.num_nodes[1]  
                    dst_gdata = gdata[..., None, :].expand(*expand_dim)
                    fn = make_pair(fn)
                return fn[0](src_edata, src_ndata, src_gdata), fn[1](dst_edata, dst_ndata, dst_gdata)
            else:
                raise ValueError(f"The function {fn} should have 2 or 3 parameters, but got {n_parameters}")
        else:
            # NOTE: we only consider the directed graph here
            if n_parameters == 3:
                return fn(src_edata, dst_edata, ndata)
            elif n_parameters == 4:
                if gdata is not None:
                    expand_dim = [-1] * (gdata.ndim + 1)
                    expand_dim[-2] = self.num_nodes  
                    gdata = gdata[..., None, :].expand(*expand_dim)
                return fn(src_edata,dst_edata, ndata, gdata)
            else:
                raise ValueError(f"The function {fn} should have 2 or 3 parameters, but got {n_parameters}")

    def get_ndata(self, key:str = 'x')->Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.ndata is not None:
            return self.ndata[key]
        else:
            assert self.src_ndata is not None, "The source node data is not defined"
            assert self.dst_ndata is not None, "The destination node data is not defined"
            return self.src_ndata[key], self.dst_ndata[key]

    def get_edata(self, key:str = 'x')->torch.Tensor:
        assert self.edata is not None, "The edge data is not defined"
        return self.edata[key]

    def get_gdata(self, key:str = 'x')->torch.Tensor:
        assert self.gdata is not None, "The graph data is not defined"
        return self.gdata[key]

    def drop_edge(self, p:float, seed:Optional[int]=None)->"Graph":
        if p <= 0.0:
            return self
        
        if seed is not None:
            torch.manual_seed(seed)
        
        mask = torch.rand(self.num_edges) >= p
        if self.is_bipartite:
            return Graph(
                edges = self.edges[:, mask],
                ndata = self.ndata,
                src_ndata = self.src_ndata.asdict(),
                dst_ndata = self.dst_ndata.asdict(),
                batch_identifier = self.batch_identifier,
                edata = {k:v[...,mask,:] for k,v in self.edata.asdict().items()}
            )
        else:
            return Graph(
                edges = self.edges[:, mask],
                ndata = self.ndata.asdict(),
                src_ndata = self.src_ndata,
                dst_ndata = self.dst_ndata,
                batch_identifier = self.batch_identifier,
                edata = {k:v[...,mask,:] for k,v in self.edata.asdict().items()}
            )


    def batch_broadcast(self, batch_size, ndata_dim:bool = False, edata_dim:bool = False):
        if self.batch_identifier is None:
            if ndata_dim:
                self.src_ndata["x"] = self.src_ndata["x"].unsqueeze(0).expand(batch_size, -1, -1)
                self.dst_ndata["x"] = self.dst_ndata["x"].unsqueeze(0).expand(batch_size, -1, -1)
            if edata_dim:
                self.edata["x"] = self.edata["x"].unsqueeze(0).expand(batch_size, -1, -1)
            
            self.batch_identifier = True



        
    @classmethod
    def with_pos_encode(cls,
        pos:torch.Tensor, # [n_points, n_dim]
        edges:torch.Tensor, #[2, n_edges],
        ndata:Optional[torch.Tensor] = None,
        edata:Optional[torch.Tensor] = None,
        domain_shifts:Optional[torch.Tensor] = None, # [n_shifts, n_dim]
        domain_edges:Optional[torch.Tensor] = None, # [2, n_domain]
        max_edge_length:float = 2.0,
        node_freq:int = 4,
        periodic:bool = False,
        add_dummy_node:bool = False,    
        with_additional_info:bool = True,
        ):

        ndata = node_pos_encode(pos, ndata=ndata, freq=node_freq, periodic=periodic, add_dummy_node=add_dummy_node)
        edata = edge_pos_encode(u = pos, v = pos, edges=edges, edata=edata, periodic=periodic, max_edge_length=max_edge_length, domain_shifts=domain_shifts, domain_edges=domain_edges)
        
        return cls(edges=edges, 
                   ndata={
                       "x":ndata,
                       "pos":pos
                       } if with_additional_info else ndata, 
                   edata=edata)
    
    @classmethod
    def bipartite_with_pos_encode(
        cls,
        edges:torch.Tensor, # [2, n_edges]
        src_pos:torch.Tensor, # [n_src_nodes, n_dim]
        dst_pos:torch.Tensor, # [n_dst_nodes, n_dim]
        src_ndata:Optional[torch.Tensor] = None, # [n_src_nodes, n_src_features]
        dst_ndata:Optional[torch.Tensor] = None, # [n_dst_nodes, n_dst_features]
        edata:Optional[torch.Tensor] = None, # [n_edges, n_features]
        domain_shifts:Optional[torch.Tensor] = None, # [n_shifts, n_dim]
        domain_edges:Optional[torch.Tensor] = None, # [2, n_domain]
        max_edge_length:float = 2.0,
        node_freq:int = 4, # number of frequencies for node features
        periodic:bool = False, # whether the graph is periodic
        add_dummy_node:bool = False, # whether to add a dummy node
        with_additional_info:bool = True,
        ):
        """
        Build the bipartite graph with positional encoding.  

        Parameters:
        -----------
        edges: torch.Tensor
            2D tensor of shape [2, n_edges]
        """
        src_ndata = node_pos_encode(src_pos, src_ndata, freq=node_freq, periodic=periodic, add_dummy_node=add_dummy_node)
        dst_ndata = node_pos_encode(dst_pos, dst_ndata, freq=node_freq, periodic=periodic, add_dummy_node=add_dummy_node)
        edata = edge_pos_encode(src_pos, dst_pos, edges, periodic=periodic, max_edge_length=max_edge_length, domain_shifts=domain_shifts, domain_edges=domain_edges)
        return cls( edges, 
                    src_ndata = {
                       "x":src_ndata,
                       "pos":src_pos
                    } if with_additional_info else src_ndata, 
                    dst_ndata = {
                        "x":dst_ndata,
                        "pos":dst_pos
                    } if with_additional_info else dst_ndata, 
                    edata = edata)


