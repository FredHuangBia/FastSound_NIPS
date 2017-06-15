require 'randomkit'
require 'nn'
require 'cudnn'
require 'cunn'

scripts = '/data/vision/billf/object-properties/sound/sound/primitives/code/inference/'

-- rotation = {"[0.0,0.0,0.0,1.0]","[0.0,0.0,0.707,0.707]","[0.0,0.707,0.0,0.707]","[0.5,0.5,0.5,0.5]","[0.707,0.0,0.0,0.707]","[0.5,-0.5,0.5,0.5]","[0.707,0.0,0.707,0.0]"}

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

function euler2quat(euler)
	-- convert extrinsic euler angle to quaterion
	local a, b, c = (euler[1]-euler[3])/2, (euler[1]+euler[3])/2, euler[2]/2
	local i = math.cos(a)*math.sin(c)
	local j = math.sin(a)*math.sin(c)
	local k = math.sin(b)*math.cos(c)
	local r = math.cos(b)*math.cos(c)
	return {i,j,k,r}
end

function label2Val(nn_output)
	-- pass in params estimated by nn all ranges between -1 and 1
	local pool = {-math.pi/2, -math.pi/4, 0, math.pi/4}
	local phi = math.floor((nn_output[1]-1)/16) + 1
	local theta = math.floor((nn_output[1]-1)%16/4) + 1
	local psi = (nn_output[1]-1)%16%4 + 1
	local euler = {pool[phi],pool[theta], pool[psi]}
	local scnVal0 = euler2quat(euler)				-- 1-64 rotation classification into quat
	local scnVal1 = (3+5)/2 + nn_output[2]*(5-3)/2			-- height, linear, continuous
	local matVal0 = 10^((-5-8)/2 - nn_output[3]*(-8+5)/2)		-- alpha, log10 linear, continuous (-1 -> -8, 1 -> -5)
	local matVal1 = 2^((0+5)/2 + nn_output[4]*(5-0)/2)		-- beta, log2 linear, continuous
	local matVal2 = (0.6+0.9)/2 + nn_output[5]*(0.9-0.6)/2		-- restitution, linear, continuous
	return {scnVal0, scnVal1, matVal0, matVal1, matVal2}
end

function gen_sound(nn_output, path, id, i, j)
	-- synthesis new sound based on given params
	local real_values = label2Val(nn_output)
	local outputDir = path .. ('%06d/'):format(id) .. ("%d-%d/"):format(i,j)
	os.execute(scripts .. "mkdir.sh " .. outputDir)
	-- print(outputDir)
	-- print(real_values)
	os.execute(scripts .. "gen_config.sh " .. outputDir .. ' ' .. real_values[2] .. ' ' .. ('[%.3f,%.3f,%.3f,%.3f]'):format(real_values[1][1],real_values[1][2],real_values[1][3],real_values[1][4]) .. ' ' .. real_values[3] .. ' ' .. real_values[4] .. ' ' .. real_values[5])
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

function sample_rotation(cdf)
	-- sample value according to a given cdf (1d tensor)
	local number = randomkit.random_sample()
	local result = 1
	for i = 1, cdf:size(2) do
		if number < cdf[1][i] then
			result = i
			break
		end
	end
	print("New sample " .. result)
	return result
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

	-- torch.save(t7filename, merged)
	return get_net_input(t7filename,2)
end

------------------------------------------------------------------------------------------------------------------------

function sd_descent(sigma, counter)
	return sigma/math.sqrt(counter+1)
end

function get_distribution(Tensor1d)
	local m = nn.SoftMax()
	local pdf = m:forward(Tensor1d)
	local cdf = torch.cumsum(pdf,2)
	return cdf
end

function a_b_s(model, id, iterations, verbose)
	-- pass in an initialization table of 5 params
	local path = '/data/vision/billf/object-properties/sound/sound/primitives/exp/primV2d/'
	local all_sound = '/data/vision/billf/object-properties/sound/sound/primitives/data/primV2d/'

	local real_sound = paths.concat(all_sound,("%06d"):format(id))
	local soundnet = load_model(model)
	soundnet:evaluate()
	local real_sound_t7 = save_t7(real_sound)
	-- print(real_sound_t7)
	print(("============================= %06d ============================="):format(id))
	print("==> Forward passing!")
	local init = soundnet:forward(real_sound_t7)
	print("soundnet.output")
	print(soundnet.output)
	local rotation = torch.Tensor(1,64)
	rotation[1] = init[1][{{1,64}}]:double()
	local rot_cdf = get_distribution(rotation)
	local init_rot, init_rot_label = torch.max(rotation,2)
	local initialization = {}
	initialization[1] = init_rot_label[1][1]
	for i = 2, 5 do
		initialization[i] = init[1][63+i]
	end
	print("Pridiction:")
	print(initialization)
	soundnet:remove()
	print("Real values:")
	initialization = {43, -0.585, 0.3385, 0.2207, 0.8667}
	print("==> Getting target feature vector!")
	local target_feature = get_feature(real_sound_t7, soundnet)

	local sigma = {1/3,1/3,1/3,1/3,1/3}
	local counter = {0,0,0,0,0}
	local params = initialization
	local init_sound = gen_sound(params, path, id, 100, 100)
	local init_sound_t7 = save_t7(init_sound)
	local init_feature = get_feature(init_sound_t7,soundnet)
	local ll = log_likelihood(init_feature, target_feature)
	print(("    Initial distance is %.4f"):format(ll))

	print("==> Sampling starts!")
	local best_i, best_j = 0, 0
	-- for i = 1, iterations do
	-- 	print(("=============================iteration %d============================="):format(i))
	-- 	for j = 1, #params do
	-- 		print(verbose and ("param %d"):format(j))
	-- 		local new_params = copy(params)
	-- 		if j == 1 then
	-- 			new_params[j] = sample_rotation(rot_cdf)
	-- 		else
	-- 			new_params[j] = sampling(params[j],sd_descent(sigma[j],counter[j]))
	-- 		end
	-- 		local new_sound = gen_sound(new_params, path, id, i, j)
	-- 		local new_sound_t7 = save_t7(new_sound)
	-- 		local new_feature = get_feature(new_sound_t7, soundnet)
	-- 		local new_ll = log_likelihood(new_feature,target_feature)
	-- 		if new_ll < ll then
	-- 			params = copy(new_params)
	-- 			ll = new_ll
	-- 			counter[j] = counter[j] + 1
	-- 			best_i, best_j = i, j
	-- 			print(verbose and "ACCEPTED!")
	-- 			print("Distance " .. ll .. '\n')
	-- 		else
	-- 			print(verbose and "Rejected\n")
	-- 		end
	-- 	end
	-- 	print(verbose and ("##### Iteration %d Distance %.4f #####\n"):format(i,ll))
	-- end
	-- print("=============================Final Result=============================")
	-- print((ll < math.huge) and ("Final distance is %.4f"):format(ll))
	-- print("Initialization:")
	-- print(initialization)
	-- print("Estimation:")
	-- print(params)
	-- local best_dir = path .. ('%06d/'):format(id) .. ("%d-%d/"):format(best_i,best_j)
	-- print("Output directory for the best estimate: " .. best_dir)
	-- local log = io.open(path .. ('%06d/'):format(id) .. "best.txt", "w")
	-- log:write(("%d-%d %d %.2f %.2f %.2f %.2f"):format(best_i,best_j, params[1],params[2],params[3],params[4],params[5]))
end

pretrainedModel = '/data/vision/billf/object-properties/sound/sound/primitives/models/primV2d_cnnB_soundnet8_pretrainnone_mse1_LR0.001/model_192.t7'
id = tonumber(arg[1])
num_iteration = tonumber(arg[2])
a_b_s(pretrainedModel,id,num_iteration,true)

