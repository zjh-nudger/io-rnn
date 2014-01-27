require 'utils'

function right_branching_parsing(words, start_id)
	local start_id = start_id or 1
	if start_id == #words-1 then
		return '(X (X ' .. words[start_id] .. ') (X ' .. words[start_id + 1] .. '))'
	else
		return '(X (X ' .. words[start_id] .. ') ' ..
					right_branching_parsing(words, start_id+1) .. ')'
	end
end

if #arg == 2 then
	local input = arg[1]
	local output = arg[2]

	local f = io.open(output, 'w')
	local i = 1
	for line in io.lines(input) do
		local words = split_string(line)
		if #words > 1 then
			local tree = right_branching_parsing(words)
			f:write(tree .. '\n')
		end

		if math.mod(i, 10000) == 0 then print(i) end
		i = i + 1
	end
	f:close()
else
	print('[input] [output]')
end
