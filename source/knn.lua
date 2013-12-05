
function knn(entries, rep, k)
	local norm_reps = torch.cdiv(entries.reps, torch.repeatTensor(torch.norm(entries.reps, 2, 1), entries.reps:size(1), 1))
	local dist = -norm_reps:t() * rep

	local nn = {}
	for i = 1,k do
		nn[i] = 1
	end

	for i = 1,dist:numel() do
		local d = dist[{i,1}]
		if d < dist[{nn[k],1}] then
			nn[k] = i
			for j = k-1,1,-1 do
				if d < dist[{nn[j],1}] then
					nn[j+1] = nn[j]
					nn[j] = i
				end
			end
		end
	end

	return nn
end

--[[
entries = {reps = torch.rand(5,20)}
rep = entries.reps[{{},{5}}]
print(knn(entries, rep, 5))
]]
