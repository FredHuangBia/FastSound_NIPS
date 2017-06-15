

function save_t7(sound_raw_path)
	-- return input tensor for the network
	local audioDim = 44100*3
	local t7filename = paths.concat(sound_raw_path, 'merged.t7') 
	if lfs.attributes(t7filename, 'mode') ~= 'file' or 1==1 then
		local merged = torch.zeros(audioDim)
		local filename = paths.concat(sound_raw_path, 'sound.raw')
		local audioRaw = io.open(filename)

		for k = 1, audioDim do
			merged[k]= merged[k] + tonumber(audioRaw:read('*l'))
		end

		torch.save(t7filename, merged)
	end
end

dataset = '/data/vision/billf/object-properties/sound/sound/primitives/data/primV3b/'
from = tonumber(arg[1])
to = tonumber(arg[2])

for i = from, to do
    local outer_folter = ('%.03d'):format( math.floor((i-0.5)/1000) +1)
    local str = string.format("%06d", i)
    dataPath = path.join(dataset,outer_folter,str)
    save_t7(dataPath)
    print(i)
end