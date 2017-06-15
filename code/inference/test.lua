
require 'nn'
require 'cunn'
require 'cudnn'


local modelPath = '/data/vision/billf/object-properties/sound/sound/primitives/models/primV2c_cnnA_soundnet8_pretrainnone_mse1_LR0.001/model_100.t7'
local model
model = torch.load(modelPath)

-- print(model)