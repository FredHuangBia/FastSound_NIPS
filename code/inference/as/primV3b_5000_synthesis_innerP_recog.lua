require 'randomkit'
require 'nn'
require 'cudnn'
require 'cunn'
local signal = require 'signal'

scripts = '/data/vision/billf/object-properties/sound/sound/primitives/code/inference/'
sound_scripts = '/data/vision/billf/object-properties/sound/sound/primitives/scripts/multi_shape_v1/'

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
	local prim = {62,187,312,437,553,679,805,931,1057,1183,1309,1435,1593,1713}
	local shape = prim[nn_output[1]]
	local modulus = nn_output[2]-1
	local pool = {-math.pi/2, -math.pi/4, 0, math.pi/4}
	local phi = math.floor((nn_output[3]-1)/16) + 1
	local theta = math.floor((nn_output[3]-1)%16/4) + 1
	local psi = (nn_output[3]-1)%16%4 + 1
	local euler = {pool[phi],pool[theta], pool[psi]}
	local scnVal0 = euler2quat(euler)				-- 1-64 rotation classification into quat
	local scnVal1 = (3+5)/2 + nn_output[4]*(5-3)/2			-- height, linear, continuous
	local matVal0 = 10^((-5-8)/2 - nn_output[5]*(-8+5)/2)		-- alpha, log10 linear, continuous (-1 -> -8, 1 -> -5)
	local matVal1 = 2^((0+5)/2 + nn_output[6]*(5-0)/2)		-- beta, log2 linear, continuous
	local matVal2 = (0.6+0.9)/2 + nn_output[7]*(0.9-0.6)/2		-- restitution, linear, continuous
	return {shape, modulus, scnVal0, scnVal1, matVal0, matVal1, matVal2}
end

function gen_sound(nn_output, path, id, i, j)
	-- synthesis new sound based on given params
	local real_values = label2Val(nn_output)
	local outputDir = path .. ('%06d/'):format(id) .. ("%d-%d/"):format(i,j)
	os.execute(scripts .. "mkdir.sh " .. outputDir)
	-- print(nn_output)
	-- print(real_values)
	os.execute(scripts .. "gen_config.sh " .. outputDir .. ' ' .. real_values[4] .. ' ' .. ('[%.3f,%.3f,%.3f,%.3f]'):format(real_values[3][1],real_values[3][2],real_values[3][3],real_values[3][4]) .. ' ' .. real_values[5] .. ' ' .. real_values[6] .. ' ' .. real_values[7])
	os.execute("python " .. sound_scripts .. "gen_sound.py " .. outputDir .. " 0 0 100000 10 10 10 " .. real_values[1] .. real_values[2] .. " 0 0 0 > " .. outputDir .. 'gen_sound.log')
	return outputDir
end

------------------------------------------------------------------------------------------------------------------------
function load_model(model)
	return torch.load(model)
end

function get_feature(net_input, mean)
	local sound=torch.zeros(132300):float()
	for i = 1,132300 do
		sound[i] = net_input[1][1][i][1]
	end
	sound:csub(mean)
	local energy = torch.sum(torch.cmul(sound,sound))
	sound = torch.mul(sound, 100/torch.sqrt(energy))
	-- print(feature)
	return sound:cuda():clone()
end

function log_likelihood(fv1, fv2)
	-- compute closeness of two featuer vectors, input are tensors
	local innerP = torch.cmul(fv1,fv2)
	return torch.sum(innerP)
end

function abs_loss_factor(sound1,sound2) -- sound2 should be target
	local a1 = torch.mean(torch.abs(sound1))
	local a2 = torch.mean(torch.abs(sound2))
	local dif = torch.abs(a1-a2)
	return (a2-dif)/a2
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

function sample_discrete(cdf)
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

function Audiojitter(input,range)
    local scale = math.random() * range
    local num = torch.rand(input:size()):float():csub(0.5):mul(256 * scale)
    num = num:double()
    return input:add(num)
end

function get_net_input(t7path, coeff)
	local audio = torch.load(t7path)
	audio = AudioNormalize(audio)
	audio = AmpNormalize(audio,coeff)
	-- audio = Audiojitter(audio,0.001)
	-- audio = AmpNormalize(audio,coeff)
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

function get_distribution(Tensor1d)
	local m = nn.SoftMax()
	local pdf = m:forward(Tensor1d)
	local cdf = torch.cumsum(pdf,2)
	return cdf
end

function a_b_s(model, id, iterations, verbose)
	-- pass in an initialization table of 5 params
	local path = '/data/vision/billf/object-properties/sound/sound/primitives/exp/primV3b_5000_innerP_recog/'
	local all_sound = '/data/vision/billf/object-properties/sound/sound/primitives/data/primV3b/'
	os.execute('mkdir ' .. path .. ('%06d/'):format(id))
	os.execute('> ' .. path .. ('%06d/'):format(id) .. 'dist.log')
	os.execute('> ' .. path .. ('%06d/'):format(id) .. 'mse.log')
	local dist_log = io.open(path .. ('%06d/'):format(id) .. "dist.log", "w")
	local mse_log = io.open(path .. ('%06d/'):format(id) .. "mse.log", "w")

	local parent_dir = math.floor((id-1)/1000) + 1
	local real_sound = paths.concat(all_sound,("%03d"):format(parent_dir),("%06d"):format(id))
	local soundnet = load_model(model)
	soundnet:evaluate()
	local real_sound_t7 = save_t7(real_sound)
	local mean=torch.sum(real_sound_t7)/132300
	local modulus_mapping = {1.00,2.25,4.00,6.25,9.00,12.2,16.0,20.2,25.0,30.0}

	print(("============================= %06d ============================="):format(id))
	print("==> Forward passing!")
	local init = soundnet:forward(real_sound_t7)
	local shape = torch.Tensor(1,14)
	local modulus = torch.Tensor(1,10)
	local rotation = torch.Tensor(1,64)
	shape[1] = init[1][{{1,14}}]:double()
	modulus[1] = init[1][{{15,24}}]:double()
	rotation[1] = init[1][{{25,88}}]:double()
	local shape_cdf = get_distribution(shape)
	local modulus_cdf = get_distribution(modulus)
	local rot_cdf = get_distribution(rotation)
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
	print(initialization)
	mse_log:write(("%d %d %d %f %f %f %f\n"):format(initialization[1],initialization[2],initialization[3],initialization[4],initialization[5],initialization[6],initialization[7]))

	print("==> Getting target feature vector!")

	local target_feature = get_feature(real_sound_t7, mean)

	local sigma = {1/3,1/3,1/3,1/3,1/3,1/3,1/3}
	local counter = {0,0,0,0,0,0,0}
	local params = initialization
	local init_sound = gen_sound(params, path, id, 0, 0)
	local init_sound_t7 = save_t7(init_sound)
	local init_feature = get_feature(init_sound_t7, mean)
	local ll = log_likelihood(init_feature, target_feature)
	local l_factor = abs_loss_factor(init_sound_t7,real_sound_t7)
	ll = ll*l_factor
	print(("    Initial distance is %.4f"):format(ll))
	dist_log:write(("0 %f\n"):format(ll))
	local best_real = label2Val(params)

	print("==> Sampling starts!")
	local best_i, best_j = 0, 0
	for i = 1, iterations do
		print(("=============================iteration %d============================="):format(i))
		for j = 1, #params do
			print(verbose and ("param %d"):format(j))
			local new_params = copy(params)
			if j == 1 then
				new_params[j] = sample_discrete(shape_cdf)
			elseif j == 2 then
				new_params[j] = sample_discrete(modulus_cdf)
			elseif j == 3 then
				new_params[j] = sample_discrete(rot_cdf)
			else
				new_params[j] = sampling(params[j],sd_descent(sigma[j],counter[j]))
			end
			local new_sound = gen_sound(new_params, path, id, i, j)
			local new_sound_t7 = save_t7(new_sound)
			local new_feature = get_feature(new_sound_t7, mean)
			local new_ll = log_likelihood(new_feature,target_feature)
			local new_l_factor = abs_loss_factor(new_sound_t7,real_sound_t7)
			new_ll = new_ll*new_l_factor
			if new_ll > ll then
				params = copy(new_params)
				ll = new_ll
				counter[j] = counter[j] + 1
				best_i, best_j = i, j
				print(verbose and "ACCEPTED!")
				print("Distance " .. ll .. '\n')
			else
				print(verbose and "Rejected\n")
			end
		end
		best_real = label2Val(params)
		print(verbose and ("##### Iteration %d Distance %.4f #####\n"):format(i,ll))
		dist_log:write(("%d %.4f %d-%d | %d %d %d %f %f %f %f | %d %d [%f,%f,%f,%f] %f %f %f %f\n"):format(i,ll, best_i,best_j, params[1],params[2],params[3],params[4],params[5],params[6],params[7], best_real[1], modulus_mapping[params[2]], best_real[3][1],best_real[3][2],best_real[3][3],best_real[3][4],best_real[4],best_real[5],best_real[6],best_real[7]))
	end
	print("=============================Final Result=============================")
	print((ll < math.huge) and ("Final distance is %.4f"):format(ll))
	print("Initialization:")
	print(initialization)
	print("Estimation:")
	print(params)
	local best_dir = path .. ('%06d/'):format(id) .. ("%d-%d/"):format(best_i,best_j)
	print("Output directory for the best estimate: " .. best_dir)
	os.execute('> ' .. path .. ('%06d/'):format(id) .. 'best.txt')
	local best = io.open(path .. ('%06d/'):format(id) .. "best.txt", "w")
	best:write(("%d-%d %d %d %d %f %f %f %f\n"):format(best_i,best_j, params[1],params[2],params[3],params[4],params[5],params[6],params[7]))
	best:write(("%d %d [%f,%f,%f,%f] %f %f %f %f\n"):format(best_real[1], modulus_mapping[params[2]], best_real[3][1],best_real[3][2],best_real[3][3],best_real[3][4],best_real[4],best_real[5],best_real[6],best_real[7]))
	mse_log:write(("%d %d %d %f %f %f %f\n"):format(params[1],params[2],params[3],params[4],params[5],params[6],params[7]))
end

pretrainedModel = '/data/vision/billf/object-properties/sound/sound/primitives/models/primV3b_5000_cnnF_soundnet8_pretrainnone_mse1_LR0.001/model_50.t7'
id = tonumber(arg[1])
num_iteration = tonumber(arg[2])
a_b_s(pretrainedModel,id,num_iteration,true)

