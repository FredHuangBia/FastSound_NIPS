require 'randomkit'
require 'nn'
require 'cudnn'
require 'cunn'

scripts = '/data/vision/billf/object-properties/sound/sound/primitives/code/inference/'

rotation = {"[0.0,0.0,0.0,1.0]","[0.0,0.0,0.707,0.707]","[0.0,0.707,0.0,0.707]","[0.5,0.5,0.5,0.5]","[0.707,0.0,0.0,0.707]","[0.5,-0.5,0.5,0.5]","[0.707,0.0,0.707,0.0]"}

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

function label2Val(nn_output)
	-- pass in params estimated by nn all ranges between -1 and 1
	local scnVal0 = (3+4.5)/2 + nn_output[1]*(4.5-3)/2
	local scnVal1 = round(nn_output[2]*3+4)
	local matVal0 = 10^((-4-9)/2 + nn_output[3]*(-9+4)/2)
	local matVal1 = 2^((0+5)/2 + nn_output[4]*(5-0)/2)
	local matVal2 = (0.6+0.85)/2 + nn_output[5]*(0.85-0.6)/2
	return {scnVal0, scnVal1, matVal0, matVal1, matVal2}
end

function gen_sound(nn_output, path)
	-- synthesis new sound based on given params
	local real_values = label2Val(nn_output)
	local outputDir = path .. ("%1.3f-%1.3f-%1.3f-%1.3f-%1.3f/"):format(nn_output[1]+1, nn_output[2]+1, nn_output[3]+1, nn_output[4]+1, nn_output[5]+1)
	-- print(outputDir)
	-- print(real_values)
	os.execute(scripts .. "gen_config.sh " .. outputDir .. ' ' .. real_values[1] .. ' ' .. rotation[real_values[2]] .. ' ' .. real_values[3] .. ' ' .. real_values[4] .. ' ' .. real_values[5])
	os.execute("python " .. scripts .. "gen_sound.py " .. outputDir .. " 0 0 101 10 10 10 0 0 0 0 > " .. outputDir .. 'gen_sound.log')
	return outputDir
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

function log_likelihood(fv1, fv2)
	-- compute closeness of two featuer vectors, input are tensors
	return torch.dist(fv1,fv2,2)
end

function sampling(mu,sigma)
	-- sample value by a normal distribution
	local sample = 2
	while sample > 1 or sample < -1 do
		sample = randomkit.normal(mu,sigma/2)
		print("New sample " .. sample)
	end
	return sample
end

-- function sampling(mu,sigma)
-- 	-- sample value by a normal distribution
-- 		local sample = randomkit.normal(mu,sigma)
-- 		print("New sample " .. sample)
-- 	return sample
-- end

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

function save_t7(sound_raw_path)
	-- return input tensor for the network
	local audioDim = 44100*3
	local t7filename = paths.concat(sound_raw_path, 'merged.t7') 
	local merged = torch.zeros(audioDim)
	local filename = paths.concat(sound_raw_path, 'sound.raw')
	local audioRaw = io.open(filename)

	for k = 1, audioDim do
		merged[k]= merged[k] + tonumber(audioRaw:read('*l'))
	end

	torch.save(t7filename, merged)
	return get_net_input(t7filename,2)
end

------------------------------------------------------------------------------------------------------------------------

function sd_descent(sigma, counter)
	return sigma/math.sqrt(counter+1)
end

function a_b_s(model, real_sound, iterations, verbose)
	-- pass in an initialization table of 5 params
	local path = '/data/vision/billf/object-properties/sound/sound/primitives/exp/single_shape/'
	local soundnet = load_model(model)
	local real_sound_t7 = save_t7(real_sound)
	print("==> Forward passing!")
	local init = soundnet:forward(real_sound_t7)
	local initialization = {}
	for i = 1, init:size()[2] do
		initialization[i] = init[1][i]
	end
	soundnet:remove()
	print("==> Getting target feature vector!")
	local target_feature = get_feature(real_sound_t7, soundnet)
	local sigma = {2/5,1/3,2/5,2/5,2/5}
	local counter = {0,0,0,0,0}
	local params = initialization
	local init_sound = gen_sound(params,path)
	local init_sound_t7 = save_t7(init_sound)
	local init_feature = get_feature(init_sound_t7,soundnet)
	local ll = log_likelihood(init_feature, target_feature)
	print(("    Initial distance is %.4f"):format(ll))
	print("==> Sampling starts!")
	for i = 1, iterations do
		print(("=============================iteration %d============================="):format(i))
		for j = 1, #params do
			print(verbose and ("param %d"):format(j))
			local new_params = copy(params)
			new_params[j] = sampling(params[j],sd_descent(sigma[j],counter[j]))
			local new_sound = gen_sound(new_params, path)
			local new_sound_t7 = save_t7(new_sound)
			local new_feature = get_feature(new_sound_t7, soundnet)
			local new_ll = log_likelihood(new_feature,target_feature)
			if new_ll < ll then
				params = copy(new_params)
				ll = new_ll
				counter[j] = counter[j] + 1
				print(verbose and "ACCEPTED!")
				print("Distance " .. ll .. '\n')
			else
				print(verbose and "Rejected\n")
			end
		end
		print(verbose and ("##### Iteration %d Distance %.4f #####\n"):format(i,ll))
	end
	print("=============================Final Result=============================")
	print((ll < math.huge) and ("Final distance is %.4f"):format(ll))
	print("Initialization:")
	print(initialization)
	print("Estimation:")
	print(params)
	local best_dir = path .. ("%1.3f-%1.3f-%1.3f-%1.3f-%1.3f/"):format(params[1]+1, params[2]+1, params[3]+1, params[4]+1, params[5]+1)
	print("Output directory for the best estimate: " .. best_dir)
end

pretrainedModel = '/data/vision/billf/object-properties/sound/sound/primitives/models/primV2c_cnnA_soundnet8_pretrainnone_mse1_LR0.001/model_best.t7'
example_sound = '/data/vision/billf/object-properties/sound/sound/primitives/exp/single_shape'
num_iteration = 50
a_b_s(pretrainedModel,example_sound,num_iteration,true)

