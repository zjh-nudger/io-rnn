require 'tree'

torch.setnumthreads(1)

if #arg == 1 then
	local treebank_path = arg[1]

	-- load treebank
	print('load treebank...')
	local treebank = {}
	local lines = {}
	for line in io.lines(treebank_path) do
		tree = Tree:create_from_string(line)
		for _,stree in ipairs(tree:all_nodes()) do
			if stree.cover[1] < stree.cover[2] and stree.cover[1] > 1 then
				local slot = Tree:new('PADDING')
				local old_children = stree.children
				stree.children = {slot}
				print(tree:to_string())
				stree.children = old_children

				print(stree:to_string())
			end
		end
		
	end

else 
	print("[treebank]")
end
