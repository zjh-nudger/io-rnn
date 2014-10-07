require 'dpiornn_gen'

if #arg == 2 then
	local net = IORNN:load(arg[1])
	net:save(arg[2], false)
else
	print("[input file] [output file]")
end
