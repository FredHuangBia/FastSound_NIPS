require 'randomkit'
require 'nn'
require 'cudnn'
require 'cunn'
package.path = './util/lua/?.lua;' .. package.path
print(package.path)

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


model = '/data/vision/billf/object-properties/sound/sound/primitives/models/primV3b_cnnF_soundnet8_pretrainnone_mse1_LR0.001/model_151.t7'
local soundnet = load_model(model)

-- print(soundnet)

-- model = nnut.loadSoundNet8()
-- print(model)

-- local max = torch.zeros(1024):cuda()
-- local last_layer = soundnet:get(27).weight

-- local tem_weight = torch.zeros(92,1):cuda()
-- for i = 1,1024 do
-- 	tem_weight[{{1,92},{1,1}}] = last_layer[{{1,92},{i,i}}]
-- 	max[i] = torch.max(tem_weight)
-- end
-- fout = io.open("primV3b_weights_151.txt","w")
-- for i = 1,1024 do
-- 	fout:write(max[i]..'\n')
-- end
-- fout:close()

local height = torch.zeros(1024):cuda()
local alpha = torch.zeros(1024):cuda()
local beta = torch.zeros(1024):cuda()
local restitution = torch.zeros(1024):cuda()


local last_layer = soundnet:get(27).weight

local tem_weight = torch.zeros(92,1):cuda()
for i = 1,1024 do
	height[i] = last_layer[89][i]
	alpha[i] = last_layer[90][i]
	beta[i] = last_layer[91][i]
	restitution[i] = last_layer[92][i]
end
fout = io.open("primV3b_weights_151_seperate.txt","w")
for i = 1,1024 do
	fout:write(height[i]..' '..alpha[i]..' '..beta[i]..' '..restitution[i]..'\n')
end
fout:close()
