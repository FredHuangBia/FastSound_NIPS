require 'randomkit'
require 'nn'
require 'cudnn'
require 'cunn'

function copy(t) 
	-- shallow-copy a table
    if type(t) ~= "table" then return t end
    local meta = getmetatable(t)
    local target = {}
    for k, v in pairs(t) do target[k] = v end
    setmetatable(target, meta)
    return target
end

function round(num)
	if num >= 0 then return math.floor(num+.5)
	else return math.ceil(num-.5) end
end

------------------------------------------------------------------------------------------------------------------------
function load_model(model)
	return torch.load(model)
end

function get_feature(net_input, model)
	-- raw sound file pass into neural net and get the final featuer vector
	local feature = model:forward(net_input)
	-- print(net_input)
	return feature:clone()
end

function AudioNormalize(input)
	if input:max() > 255 then
		return input:mul(2 ^ -23)
	else
		return input:mul(256)
	end
end

function AmpNormalize(input, coeff) 
	return input:div(input:std() * coeff)
end

function get_net_input(t7path, coeff)
	local audio = torch.load(t7path)
	audio = AudioNormalize(audio)
	audio = AmpNormalize(audio,coeff)
	audio = torch.reshape(audio,1,1,132300,1)
	return audio:cuda()
end

function argmax_1D(v)
   local length = v:size(1)
   assert(length > 0)
   -- examine on average half the entries
   local maxValue = torch.max(v)
   for i = 1, v:size(1) do
      if v[i][1] == maxValue then
         return i
      end
   end
end


model = '/data/vision/billf/object-properties/sound/sound/primitives/models/primV3b_cnnF_soundnet8_pretrainnone_mse1_LR0.001/model_92.t7'
local soundnet = load_model(model)
soundnet:remove()

-- local test_path = '/data/vision/billf/object-properties/sound/sound/primitives/data/primV3b/001/000001/merged.t7'
-- local ipt = get_net_input(test_path, 2)
-- local feature = get_feature(ipt, soundnet)
-- print(soundnet)


-- get the first layer filters
local layer1_filters = soundnet:get(1).weight
for i=1,16 do
	filter_out = io.open('/data/vision/billf/object-properties/sound/sound/primitives/code/inference/visualize/filter'..i..'.txt','w')
	for j = 1,64 do
		filter_out:write(layer1_filters[i][1][j][1]..'\n')
	end
	filter_out:close()
end


-- get the activation pattern of second last layer
local activations = torch.zeros(1000,1024)
activations = activations:cuda()
local test_path = '/data/vision/billf/object-properties/sound/sound/primitives/data/primV3b/001/'
for i = 1,1000 do
	local t7_path = test_path..('%.06d'):format(i)..'/merged.t7'
	local ipt = get_net_input(t7_path, 2)
	local feature = get_feature(ipt, soundnet)
	activations[{{i,i},{1,1024}}] = feature[{{1,1},{1,1024}}]
	if i%100 == 0 then
		print('Finished: '..i)
	end
end 
last_layer_fout = io.open('/data/vision/billf/object-properties/sound/sound/primitives/code/inference/visualize/activation_last_layer.txt','w')
for i = 1,1024 do
	local top5 = torch.zeros(5)
	local activation = torch.zeros(1000,1):cuda()
	local maximum = -100
	activation[{{1,1000},{1,1}}] = activations[{{1,1000},{i,i}}]
	for j = 1,5 do
		local max = argmax_1D(activation)
		top5[j] = max
		if j==1 then
			maximum = activation[max][1]
		end
		activation[max][1] = -100
	end
	last_layer_fout:write('Unite-'..i..'-top5: '..top5[1]..' '..top5[2]..' '..top5[3]..' '..top5[4]..' '..top5[5]..' maximum: '..maximum..'\n')
	print('Unite-'..i..'-top5: '..top5[1]..' '..top5[2]..' '..top5[3]..' '..top5[4]..' '..top5[5]..' maximum: '..maximum)
end
last_layer_fout:close()
