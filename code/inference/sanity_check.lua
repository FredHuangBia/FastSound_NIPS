require 'nn'
require 'cudnn'
require 'cunn'

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

local targets_dir = {'000471', '021440', '000280', '064253', '037686', '087250', '137516', '119143', '132269', '176814', '133598', '017725', '119741', '144424', '002457', '085380', '173779', '137054', '046234', '058140', '189910', '197542', '039011', '112333', '198869', '146420', '094531', '148375', '043043', '061524', '179585', '029664', '123807', '101913', '017744', '074381', '035464', '025762', '162821', '115195', '016299', '012335', '033396', '100228', '042797', '037923', '192593', '010638'}
local best_dir = {'29-6', '21-6', '23-7', '26-5', '30-5', '30-7', '30-6', '30-6', '26-6', '30-6', '29-5', '12-6', '27-6', '30-6', '30-7', '30-5', '30-5', '27-6', '30-5', '29-5', '29-6', '25-5', '30-5', '27-5', '22-7', '28-6', '28-7', '24-6', '28-6', '22-6', '18-6', '27-6', '30-6', '30-6', '29-5', '30-6', '29-6', '20-6', '30-6', '30-6', '30-6', '29-5', '29-6', '29-6', '27-6', '27-7', '28-6', '28-6'}

local root = '/data/vision/billf/object-properties/sound/sound/primitives/exp/primV3b/'

local model_path = '/data/vision/billf/object-properties/sound/sound/primitives/models/primV3b_cnnF_soundnet8_pretrainnone_mse1_LR0.001/model_93.t7'

local model = torch.load(model_path)
model:evaluate()

for i = 1,#targets_dir do
	local sanity_log = io.open(root .. targets_dir[i] .. '/' .. 'sanity.log','w')
	local t7_path = root .. targets_dir[i] .. '/' .. best_dir[i] .. '/merged.t7'
	print(("============================= %s ============================="):format(targets_dir[i]))
	print("==> Forward passing!")
	local t7 = get_net_input(t7_path,2)
	local init = model:forward(t7)
	local shape = torch.Tensor(1,14)
	local modulus = torch.Tensor(1,10)
	local rotation = torch.Tensor(1,64)
	shape[1] = init[1][{{1,14}}]:double()
	modulus[1] = init[1][{{15,24}}]:double()
	rotation[1] = init[1][{{25,88}}]:double()
	local init_shape, init_shape_label = torch.max(shape,2)
	local init_modulus, init_modulus_label = torch.max(modulus,2)
	local init_rot, init_rot_label = torch.max(rotation,2)
	local initialization = {}
	initialization[1] = init_shape_label[1][1]
	initialization[2] = init_modulus_label[1][1]
	initialization[3] = init_rot_label[1][1]
	for i = 4, 7 do
		initialization[i] = init[1][85+i]
	end
	sanity_log:write(("%d %d %d %f %f %f %f\n"):format(initialization[1],initialization[2],initialization[3],initialization[4],initialization[5],initialization[6],initialization[7]))
end