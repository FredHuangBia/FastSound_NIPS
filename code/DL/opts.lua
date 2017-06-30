--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--

-- primV2d and primV3a single shape, 64 discrete rotations, height, alpha, beta, restitution
-- primV2e 100 shapes, 10 specific ratios, 64 discrete rotations, height, alpha, beta, restitution
-- cnnES cnnE for shape only
-- primV3b 14 primitive shapes, 10 specific ratios, 64 discrete rotations, height, alpha, beta, restitution


local M = { }

function M.parse(arg)
  local cmd = torch.CmdLine()
   
  cmd:text()
  cmd:text('Torch-7 ResNet Training script')
  cmd:text()
  cmd:text('Options:')
  ------------- General options -------------------
  cmd:option('-debug',            false,          'Debug mode')
  cmd:option('-manualSeed',       0,              'Manually set RNG seed')
  cmd:option('-GPUs',             '4',            'ID of GPUs to use by default, separated by ,')
  cmd:option('-backend',          'cudnn',        'Options: cudnn|cunn(obsolete)')
  cmd:option('-cudnn',            'fastest',      'Options: fastest|default|deterministic')
  ------------- Path options ----------------------
  cmd:option('-data',             '../../data',   'Path to dataset')
  cmd:option('-gen',              '../../gen',    'Path to save generated files')
  cmd:option('-resume',           '../../models', 'Path to checkpoint')
  cmd:option('-www',              '../../www',    'Path to dataset')
  ------------- Data options ----------------------
  cmd:option('-nThreads',         32,              'number of data loading threads')
  cmd:option('-dataset',          'primSelf',      'Options: primV3b, primSelf, primV3b_coarse')
  cmd:option('-maxImgs',          200000,         'Number of images in train+val')
  cmd:option('-trainPctg',        0.99,           'Percentage of training images')
  ------------- Training/testing options ----------
  cmd:option('-nEpochs',          300,            'Number of total epochs to run')
  cmd:option('-epochNumber',      0,             'Manual epoch number: 0=retrain|-1=latest|-2=best')
  cmd:option('-saveEpoch',        10,             'Saving at least every % epochs')
  cmd:option('-batchSize',        16,             'mini-batch size (1 = pure stochastic)')
  cmd:option('-testOnly',         false,          'Run on validation set only')
  cmd:option('-visTrain',         3,              'Visualizing training examples')
  cmd:option('-visTest',          3,              'Visualizing testing examples')
  cmd:option('-visWidth',         -1,             '# images per row for visualization')
  cmd:option('-tenCrop',          false,          'Ten-crop testing')
  ------------- Optimization options --------------
  cmd:option('-LR',               0.001,          'initial learning rate')
  cmd:option('-LRDecay',          'stepwise',     'Options: anneal (sgd)|stepwise|pow|none')
  cmd:option('-LRDParam',         200,            'param for learning rate decay')
  cmd:option('-momentum',         0.9,            'momentum')
  cmd:option('-weightDecay',      1e-4,           'weight decay')
  ------------- Model options ---------------------
  -- cmd:option('-netType',          'cnnI',         'Options: cnnI')
  cmd:option('-netType',          'cnnAll',         'Options: cnnAll(for fully and self-supervised),cnnCor4 (for weakly supervised')
  cmd:option('-netSpec',          'soundnet8',    'Options: custom|res{18|34|50|101|200}|soundnet{5|8}, autoEncoder')
  cmd:option('-pretrain',         'none',         'Options: none|default|soundV2b')
  cmd:option('-absLoss',          0,              'Weight for abs derender criterion')
  cmd:option('-mseLoss',          1,              'Weight for mse derender criterion')
  cmd:option('-gdlLoss',          0,              'Weight for gdl derender criterion')
  cmd:option('-customLoss',       0,              'Weight for custom derender criterion')
  ------------- SVM options ---------------------
  cmd:option('-featLayer',        18,             '# layer for features')
  cmd:option('-svmTrain',         '',             'SVM training options')
  cmd:option('-svmTest',          '',             'SVM testing options')
  ------------- Other model options ---------------
  cmd:option('-nmsThres',         0.5,            'Threshold for non-max suppression')
  cmd:option('-shareGradInput',   false,          'Share gradInput tensors to reduce memory usage')
  cmd:option('-resetClassifier',  false,          'Reset the fully connected layer for fine-tuning')
  cmd:option('-nClasses',         0,              'Number of classes in the dataset')
  cmd:option('-suffix',           '',             'Suffix to hashKey')
  cmd:text()  

  return cmd:parse(arg or {})
end

function M.init(opt)
  
  torch.setdefaulttensortype('torch.FloatTensor')
  torch.setnumthreads(1)
  
  math.randomseed(opt.manualSeed)
  torch.manualSeed(opt.manualSeed)
  cutorch.manualSeedAll(opt.manualSeed)
  
  local cmd = torch.CmdLine()
  
  opt.GPUs = opt.GPUs:split(',')
  opt.nGPU = #opt.GPUs
  if opt.nGPU == 1 then
    cutorch.setDevice(1)
  end
 
 -- dataset-specific configurations

  if opt.dataset == 'primV3b_coarse' then
    opt.audioRate = 44100
    opt.audioLen = 3
    opt.audioDim = opt.audioLen * opt.audioRate
    opt.maxObj = 1
    opt.maxXmlLen = 5  -- 7 labels(shape, specific, rotation, height, alpha, beta, restitution) and 1 sound id
    opt.numObjId = 1
    opt.numEntry = 71469
    opt.numObj = 1    
  elseif opt.dataset == 'primV3b' then
    opt.audioRate = 44100
    opt.audioLen = 3
    opt.audioDim = opt.audioLen * opt.audioRate
    opt.maxObj = 1
    opt.maxXmlLen = 8  -- 7 labels(shape, specific, rotation, height, alpha, beta, restitution) and 1 sound id
    opt.numObjId = 1
    opt.numEntry = 200000
    opt.numObj = 1  
  elseif opt.dataset == 'primSelf' then
    opt.audioRate = 44100
    opt.audioLen = 3
    opt.audioDim = opt.audioLen * opt.audioRate
    opt.maxObj = 1
    opt.maxXmlLen = 11  -- 10 labels(shape, specific, rotation x4, height, alpha, beta, restitution) and 1 sound id
    opt.numObjId = 1
    opt.numEntry = 978
    opt.numObj = 1   
  else
    cmd:error('unknown dataset: ' .. opt.dataset)
  end

  require('datasets/' .. opt.dataset .. '-criterion')

  if opt.netType == 'cnnAll' then
    opt.outputSize = 14+10+4+4
    opt.outputSplitType = torch.ones(opt.maxXmlLen)   -- 1 categorial, 2 normal, 3 bernoulli
    opt.criterionWeights = torch.ones(opt.maxXmlLen)      
  elseif opt.netType == 'cnnCor4' then
    opt.outputSize = 3+1+4
    opt.outputSplitType = torch.ones(opt.maxXmlLen)   -- 1 categorial, 2 normal, 3 bernoulli
    opt.criterionWeights = torch.ones(opt.maxXmlLen)   
  end

  -- SVM
  if opt.netType == 'cnnSVM' then
    opt.nEpochs = 1
  end

  -- visualization-related
  opt.visPerInst = 4
  
  if opt.visWidth == -1 then
    opt.visWidth = math.floor(8 / opt.visPerInst) * opt.visPerInst
  end

  -- debug-specific configuraions
  if opt.debug then
    opt.nEpochs = 1
    opt.nThreads = 1
  end

  if opt.resetClassifier then
    if opt.nClasses == 0 then
      cmd:error('-nClasses required when resetClassifier is set')
    end
  end

  -- path-specific
  opt.hashKey = opt.dataset .. '_' .. opt.netType ..
    '_' .. opt.netSpec ..
    (opt.pretrin ~= 'none' and '_pretrain' .. opt.pretrain or '') ..
    (opt.absLoss ~= 0 and '_abs' .. opt.absLoss or '') ..
    (opt.mseLoss ~= 0 and '_mse' .. opt.mseLoss or '') ..
    (opt.gdlLoss ~= 0 and '_gdl' .. opt.gdlLoss or '') ..
    (opt.customLoss ~= 0 and '_custom' .. opt.customLoss or '') ..
    '_LR' .. opt.LR .. 
    (opt.suffix ~= '' and '_' .. opt.suffix or '')
  
  opt.dataRoot = opt.data
  opt.data = path.join(opt.data, opt.dataset)
  opt.gen = path.join(opt.gen, opt.dataset)
  opt.resume = path.join(opt.resume, opt.hashKey)
  opt.www = path.join(opt.www, opt.hashKey)

  pl.dir.makepath(opt.gen)
  pl.dir.makepath(opt.resume)
  pl.dir.makepath(opt.www)
end

return M
