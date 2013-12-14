require('cutils')
require('xlua')
p = xlua.Profiler()

a = torch.Tensor({3,2,1,5,3,3})
b = torch.Tensor({1,2,3,1,5,4})
print(spearman_rho(a,b))
