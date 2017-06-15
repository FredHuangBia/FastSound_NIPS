local handler = {}

function handler.initCriterion(criterion, model)
  if criterion.__typename == 'nn.VRMaskRegressReward' then
    criterion.module = model
  end
  
  if criterion.__typename == 'nn.MultiCriterion' or 
    criterion.__typename == 'nn.ParallelCriterion' then
    for i = 1, #criterion.criterions do
      handler.initCriterion(criterion.criterions[i], model)
    end
  end
end

function handler.createCriterion(opt, model)
  local criterion = nn.ParallelCriterion()

  -- Criterion for derendering
  if opt.deLoss then 
    local deCriterion = nn.MultiCriterion() 
    if opt.absDeLoss ~= 0 then
      deCriterion:add(nn.AbsCriterion(), opt.absDeLoss)
    end
    if opt.mseDeLoss ~= 0 then
      deCriterion:add(nn.MSECriterion(), opt.mseDeLoss)
    end
    if opt.gdlDeLoss ~= 0 then
      deCriterion:add(nn.GDLCriterion(), opt.gdlDeLoss)
    end
    if opt.customDeLoss ~= 0 then
      deCriterion:add(nn.DerenderSingleCriterion(opt.criterionWeights, opt.outputSplitPSum), 
        opt.customDeLoss)
    end
    criterion:add(deCriterion)
  else
    criterion:add(nn.ZeroCriterion())
  end

  -- Criterion for rerendering
  if opt.reLoss then 
    local reCriterion = nn.MultiCriterion() 
    if opt.mseReLoss ~= 0 then
      reCriterion:add(nn.VRMaskRegressReward(model, opt.mseReLoss, opt.rho))
    end
    criterion:add(reCriterion)
  else
    criterion:add(nn.ZeroCriterion())
  end

  criterion:cuda()
  
  return criterion
end

return handler
