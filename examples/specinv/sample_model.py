import numpy as np
from geopvi.nfvi.flows import Linear, Constr2Real, Real2Constr
from geopvi.utils import get_offdiag_mask

def build_variational_model(args, ndim, nx, ny, nz, lower, upper, param=None):
    
    flow_cls = eval(args.flow)

    if args.flow == 'Linear':
        cov_template = np.ones((args.kernel_size, args.kernel_size, args.kernel_size))
        off_diag_mask = get_offdiag_mask(cov_template, ndim, nx=nx, ny=ny, nz=nz)

        flows = [
            flow_cls(dim=ndim, kernel=args.kernel, mask=off_diag_mask, param=param)
            for _ in range(args.nflow)
        ]
    else:
        raise NotImplementedError("Only Linear flow implemented")

    if args.ini_dist == 'Uniform':
        flows.insert(0, Constr2Real(dim=ndim, lower=0, upper=1))

    flows.append(Real2Constr(dim=ndim, lower=lower, upper=upper))

    from geopvi.nfvi.models import FlowsBasedDistribution
    variational = FlowsBasedDistribution(flows, base=args.ini_dist)

    return variational