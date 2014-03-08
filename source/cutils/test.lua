require('cutils')
require('xlua')
p = xlua.Profiler()
torch.setnumthreads(1)
a = torch.rand(300,25)
b = torch.rand(300,25);
sum = torch.Tensor(300,25);
p:start('x')
multi(sum,a,b)
p:lap('x')

p:start('y')
for i = 1,3000 do
	for j = 1,1 do
		sum = a + b
	end
end
p:lap('y')
p:printAll()
