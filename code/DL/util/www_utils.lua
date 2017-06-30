require 'lfs'
local ut = require 'utils'

local www_utils = {}


function euler2quat(phi, theta, psi)
  local a = (phi-psi)/2
  local b = (phi+psi)/2
  local c = theta/2
  local i = math.cos(a)*math.sin(c)
  local j = math.sin(a)*math.sin(c)
  local k = math.sin(b)*math.cos(c)
  local r = math.cos(b)*math.cos(c)
  local str = ('[%.3f,%.3f,%.3f,%.3f]'):format(i,j,k,r)
  return str
end

function toBits(num,bits)
    -- returns a table of bits, most significant first.
    bits = bits or math.max(1, select(2, math.frexp(num)))
    local t = {} -- will contain the bits        
    for b = bits, 1, -1 do
        t[b] = math.fmod(num, 2)
        num = math.floor((num - t[b]) / 2)
    end
    return t
end

function getRotation(rotation)
  local bin = toBits(rotation,6)
  local phi = tonumber(table.concat({bin[1],bin[2]}),2)+1
  local theta = tonumber(table.concat({bin[3],bin[4]}),2)+1
  local psi = tonumber(table.concat({bin[5],bin[6]}),2)+1
  local rotations = {-math.pi/2, -math.pi/4, 0, math.pi/4}
  local str = euler2quat(rotations[phi],rotations[theta],rotations[psi])
  return str
end


function round(num)
  if num >= 0 then return math.floor(num+.5)
  else return math.ceil(num-.5) end
end

function label2Val(nn_output , dataset)
  -- pass in params estimated by nn all ranges between -1 and 1
  if dataset == 'primV2c' then
    local rotations = {"[0.0,0.0,0.0,1.0]","[0.0,0.0,0.707,0.707]","[0.0,0.707,0.0,0.707]","[0.5,0.5,0.5,0.5]","[0.707,0.0,0.0,0.707]","[0.5,-0.5,0.5,0.5]","[0.707,0.0,0.707,0.0]"}
    local scnVal0 = (3+4.5)/2 + nn_output[1]*(4.5-3)/2
    local scnVal1 = round(nn_output[2]*3+4)
    local matVal0 = 10^((-4-9)/2 + nn_output[3]*(-9+4)/2)
    if nn_output[3] < -1.00 then
      matVal0 = 10^((-4-9)/2 + (-1.00)*(-9+4)/2)
    end
    local matVal1 = 2^((0+5)/2 + nn_output[4]*(5-0)/2)
    local matVal2 = (0.6+0.85)/2 + nn_output[5]*(0.85-0.6)/2
    local rotation = rotations[scnVal1]
    return {scnVal0, rotation, matVal0, matVal1, matVal2}

  elseif dataset == 'primV2d' or dataset=='primV3a'then
    local rotation = getRotation(nn_output[1]-1)
    local height = (3+5)/2 + nn_output[2]*(5-3)/2
    local alpha = 10^((-5-8)/2 - nn_output[3]*(-8+5)/2)
    local beta = 2^((0+5)/2 + nn_output[4]*(5-0)/2)
    local restitution = (0.6+0.9)/2 + nn_output[5]*(0.9-0.6)/2
    return {rotation, height, alpha, beta, restitution}

  elseif dataset == 'primV2e' then
    local shape100 = nn_output[1]-1               -- minus one because from 1 base to 0 base                    
    local prim = {29, 61, 145, 150, 173, 240, 258, 266, 267, 268, 276, 285, 300, 301, 361, 367, 380, 383, 398, 414, 438, 461, 472, 479, 482, 518, 525, 530, 536, 537, 538, 564, 573, 595, 600, 603, 642, 708, 728, 729, 745, 750, 760, 810, 879, 880, 881, 883, 894, 901, 908, 918, 923, 940, 996, 1005, 1070, 1088, 1097, 1125, 1126, 1135, 1155, 1157, 1182, 1217, 1240, 1246, 1251, 1297, 1307, 1312, 1335, 1341, 1364, 1373, 1386, 1400, 1401, 1403, 1415, 1436, 1492, 1503, 1525, 1532, 1546, 1625, 1626, 1634, 1642, 1650, 1658, 1678, 1692, 1708, 1715, 1732, 1739, 1740}
    local shape = prim[nn_output[1]]
    local specific = nn_output[2]-1               -- minus one because from 1 base to 0 base
    local rotation = getRotation(nn_output[3]-1)  -- minus one because from 1 base to 0 base
    local height = (3+5)/2 + nn_output[4]*(5-3)/2
    local alpha = 10^((-5-8)/2 - nn_output[5]*(-8+5)/2)
    local beta = 2^((0+5)/2 + nn_output[6]*(5-0)/2)
    local restitution = (0.6+0.9)/2 + nn_output[7]*(0.9-0.6)/2
    return {shape, specific, rotation, height, alpha, beta, restitution, shape100}    

  elseif dataset=='primV3b' or dataset=='primV3b_5000' then
    local shape14 = nn_output[1]-1   
    local prim = {62,187,312,437,553,679,805,931,1057,1183,1309,1435,1593,1713}
    local shape = prim[nn_output[1]]
    local specific = nn_output[2]-1               -- minus one because from 1 base to 0 base
    local rotation = getRotation(nn_output[3]-1)  -- minus one because from 1 base to 0 base
    local height = (3+5)/2 + nn_output[4]*(5-3)/2
    local alpha = 10^((-5-8)/2 - nn_output[5]*(-8+5)/2)
    local beta = 2^((0+5)/2 + nn_output[6]*(5-0)/2)
    local restitution = (0.6+0.9)/2 + nn_output[7]*(0.9-0.6)/2
    return {shape, specific, rotation, height, alpha, beta, restitution, shape14}   
  end
end

function gen_sound(nn_output, path, wavName , dataset, target_xml )
  local scripts = '/data/vision/billf/object-properties/sound/sound/primitives/code/inference/'
  local outputDir = '/data/vision/billf/object-properties/sound/sound/primitives/tmp/sim'

  local infer_values = label2Val(nn_output, dataset)
  local target_values = label2Val(target_xml, dataset)

  if dataset == 'primV2c' then -- height rotation alpha beta restitution  
    os.execute(scripts .. "gen_config.sh " .. outputDir .. ' ' .. infer_values[1] .. ' ' .. infer_values[2] .. ' ' .. infer_values[3] .. ' ' .. infer_values[4] .. ' ' .. infer_values[5])
    os.execute("python " .. scripts .. "gen_sound.py " .. outputDir .. " 0 0 101 10 10 10 0 0 0 0 > " .. outputDir .. 'gen_sound.log')

  elseif dataset == 'primV2d' or dataset == 'primV3a'then  -- rotation height alpha beta restitution     
    os.execute(scripts .. "gen_config.sh " .. outputDir .. ' ' .. infer_values[2] .. ' ' .. infer_values[1] .. ' ' .. infer_values[3] .. ' ' .. infer_values[4] .. ' ' .. infer_values[5])
    os.execute("python " .. scripts .. "gen_sound.py " .. outputDir .. " 0 0 101 10 10 10 0 0 0 0 > " .. outputDir .. 'gen_sound.log')

  elseif dataset == 'primV2e' or dataset=='primV3b' or dataset=='primV3b_5000' then
    local mapping = '/data/vision/billf/object-properties/sound/sound/primitives/scripts/single_shape_v3/mapping.py'
    if dataset == 'primV3b' or dataset=='primV3b_5000' then
      mapping = '/data/vision/billf/object-properties/sound/sound/primitives/scripts/multi_shape_v1/mapping.py'
    end

    local file = io.open (paths.concat(path,'../','output_value.txt'), 'a+')
    os.execute('python3'..' '..mapping..' '..(infer_values[8]+1)..' '..(infer_values[2]+1)..' > '..paths.concat(path,'../','tmp_output_value.txt'))
    local tmp_file = io.open (paths.concat(path,'../','tmp_output_value.txt'), 'r')
    local line = tmp_file:read('*l')
    file:write( line .. ' ' .. infer_values[3] .. ' ' .. infer_values[4] .. ' ' .. ('%.02e'):format(infer_values[5]) .. ' ' .. ('%.02f'):format(infer_values[6]) ..'\n')
    tmp_file:close()
    file:close()    

    local file = io.open (paths.concat(path,'../','target_value.txt'), 'a+')
    os.execute('python3'..' '..mapping..' '..(target_values[8]+1)..' '..(target_values[2]+1)..' > '..paths.concat(path,'../','tmp_target_value.txt'))
    local tmp_file = io.open (paths.concat(path,'../','tmp_target_value.txt'), 'r')
    local line = tmp_file:read('*l')
    file:write( line .. ' ' .. target_values[3] .. ' ' .. target_values[4] .. ' ' .. ('%.02e'):format(target_values[5]) .. ' ' .. ('%.02f'):format(target_values[6]) ..'\n')
    tmp_file:close()
    file:close()  

    sound_scripts = '/data/vision/billf/object-properties/sound/sound/primitives/scripts/single_shape_v3/'
    os.execute(scripts .. "gen_config.sh " .. outputDir .. ' ' .. infer_values[4] .. ' ' .. infer_values[3].. ' ' .. infer_values[5] .. ' ' .. infer_values[6] .. ' ' .. infer_values[7])
    os.execute("python " .. sound_scripts .. "gen_sound.py " .. outputDir .. " 0 0 100000 10 10 10 " .. infer_values[1] .. infer_values[2] .. " 0 0 0 > " .. outputDir .. 'gen_sound.log')

  end
  os.execute('mv ' .. paths.concat(outputDir,'sound.wav') .. ' ' .. paths.concat(path,wavName) )
end

----------------------------------------------------------------------------------------------------

-- isNorm is deprecated
function www_utils.saveIms(ims, imPath, isNorm, count)
  count = count or 0
  isNorm = isNorm or false
  local numIms = ut.getLength(ims)

  local filenames = {}
  for i = 1, numIms do
    filenames[i] = string.format("%08d.png", count + i)
    -- Scale the image and save
    local im = ims[i]
    if isNorm then
      im = ims[i]:clone()
      local maxVal = im:max()
      local minVal = im:min()
      im:add(-minVal):mul(1.0 / (maxVal - minVal))
    end
    image.save(paths.concat(imPath, filenames[i]), im)
  end

  return filenames, count + numIms
end

function www_utils.saveAudios(audios, audioPath, count, audioRate)
  count = count or 0
  local numAudios = ut.getLength(audios)

  local filenames = {}
  for i = 1, numAudios do
    filenames[i] = string.format("%08d.wav", count + i)
    -- Scale the image and save
    audio.save(paths.concat(audioPath, filenames[i]), audios[i], audioRate)
  end

  return filenames, count + numAudios
end


-- ims and captions are both tables
function www_utils.renderTables(ims, captions, ncol, width, imPath)
  assert(#captions == 0 or #captions == #ims, "#captions should be either 0 or equal to #ims")

  local nrow = math.ceil(#ims / ncol)
  local htmlStr = "<table>\n"

  local k = 0

  for i = 1, nrow do
    local captionRow = "<td>"..i.."</td>"
    local imRow = "<td>"..i.."</td>"
    for j = 1, ncol do
      k = k + 1

      if #captions > 0 then
        captionRow = captionRow .. string.format("<td>%s</td>", captions[k])
      end
      imRow = imRow .. string.format("<td><img width="..width.." src='%s'></img></td>", imPath .. '/' .. ims[k])
      if k >= #ims then break end
    end

    if #captions > 0 then
      htmlStr = htmlStr .. "<tr>" .. captionRow .. "</tr>\n"
    end
    htmlStr = htmlStr .. "<tr>" .. imRow .. "</tr>\n"

    if k >= #ims then break end
  end

  return htmlStr .. "</table>\n"
end

function www_utils.renderAudioTables(ims, audios, captions, ncol, width, imPath)
  assert(#audios == #ims, "#audios should be equal to #ims")
  assert(#captions == 0 or #captions == #ims, "#captions should be either 0 or equal to #ims")

  local nrow = math.ceil(#ims / ncol)
  local htmlStr = "<table>\n"

  local k = 0

  for i = 1, nrow do
    local imRow = "<td>"..i.."</td>"
    local audioRow = "<td>"..i.."</td>"
    local captionRow = "<td>"..i.."</td>"
    for j = 1, ncol do
      k = k + 1

      imRow = imRow .. string.format("<td><img width=" .. width .. 
        " src='%s'></img></td>", imPath .. '/' .. ims[k])
      audioRow = audioRow .. string.format("<td><audio controls><source src='%s' " ..
        "type='audio/wav'>Your browser does not support the audio tag.</audio>", imPath .. '/' .. audios[k])
      if #captions > 0 then
        captionRow = captionRow .. string.format("<td>%s</td>", captions[k])
      end
      if k >= #ims then break end
    end

    htmlStr = htmlStr .. "<tr>" .. imRow .. "</tr>\n"
    htmlStr = htmlStr .. "<tr>" .. audioRow .. "</tr>\n"
    if #captions > 0 then
      htmlStr = htmlStr .. "<tr>" .. captionRow .. "</tr>\n"
    end

    if k >= #ims then break end
  end

  return htmlStr .. "</table>\n"
end

function www_utils.renderABSTables(audio1s, audio2s, audio3s, captions, ncol, width, audioPath)
  assert(#audio1s == #audio2s, "#audios should be equal")
  assert(#captions == 0 or #captions == #audio1s, "#captions should be either 0 or equal to #audios")

  local nrow = math.ceil(#audio1s / ncol)
  local htmlStr = "<table>\n"

  local k = 0

  for i = 1, nrow do
    local audio1Row = "<td>"..i.."</td>"
    local audio2Row = "<td>"..i.."</td>"
    local audio3Row = "<td>"..i.."</td>"
    local captionRow = "<td>"..i.."</td>"
    for j = 1, ncol do
      k = k + 1

      audio1Row = audio1Row .. string.format("<td><audio controls=\"controls\" preload=\"none\"><source src='%s' " ..
        "type='audio/wav'>Your browser does not support the audio tag.</audio>", audioPath .. '/' .. audio1s[k])
      audio2Row = audio2Row .. string.format("<td><audio controls=\"controls\" preload=\"none\"><source src='%s' " ..
        "type='audio/wav'>Your browser does not support the audio tag.</audio>", audioPath .. '/' .. audio2s[k])
      audio3Row = audio3Row .. string.format("<td><audio controls=\"controls\" preload=\"none\"><source src='%s' " ..
        "type='audio/wav'>Your browser does not support the audio tag.</audio>", audioPath .. '/' .. audio3s[k])
      if #captions > 0 then
        captionRow = captionRow .. string.format("<td nowrap>%s</td>", captions[k])
      end
      if k >= #audio1s then break end
    end

    htmlStr = htmlStr .. "<tr>" .. audio1Row .. "</tr>\n"
    htmlStr = htmlStr .. "<tr>" .. audio2Row .. "</tr>\n"
    htmlStr = htmlStr .. "<tr>" .. audio3Row .. "</tr>\n"
    if #captions > 0 then
      htmlStr = htmlStr .. "<tr>" .. captionRow .. "</tr>\n"
    end

    if k >= #audio1s then break end
  end

  return htmlStr .. "</table>\n"
end


function www_utils.renderKeyValues(t)
  -- Write all key_value pairs
  local htmlStr = "<table>\n<tr><td>Key</td><td>Value</td></tr>\n"

  for k, v in pairs(t) do
    htmlStr = htmlStr .. "<tr><td>" .. k .. "</td><td>" .. v .. "</td><tr>\n"
  end

  return htmlStr .. "</table>\n"
end

function www_utils.renderHeader()
  return [[
<head>
<style>
    * {
        font-size: 24px;
    }
</style>
</head>
]]
end

function www_utils.renderHtml(rootDir, ims, captions, ncol, isNorm, width, htmlFile)
  captions = captions or {}
  ncol = ncol or 8
  isNorm = isNorm or false
  width = width or 256
  htmlFile = htmlFile or 'index.html'

  local imRelativeDir = './im'
  local imDir = paths.concat(rootDir, imRelativeDir)

  lfs.mkdir(rootDir)
  lfs.mkdir(imDir)

  local count = 0

  local f = io.open(paths.concat(rootDir, htmlFile), "w")
  f:write('<html>\n'..www_utils.renderHeader()..'<body>\n')

  local imNames = www_utils.saveIms(ims, imDir, isNorm, count)
  local htmlStr = www_utils.renderTables(imNames, captions, ncol, width, imRelativeDir)

  f:write(htmlStr .. '</body>\n</html>')
  f:close()

  return count
end

function www_utils.renderAudioHtml(rootDir, ims, audios, captions, ncol, isNorm, width, htmlFile, audioRate)
  captions = captions or {}
  ncol = ncol or 8
  isNorm = isNorm or false
  width = width or 256
  htmlFile = htmlFile or 'index.html'
  audioRate = audioRate or 44100

  local imRelativeDir = './im'
  local imDir = paths.concat(rootDir, imRelativeDir)

  lfs.mkdir(rootDir)
  lfs.mkdir(imDir)

  local f = io.open(paths.concat(rootDir, htmlFile), "w")
  f:write('<html>\n'..www_utils.renderHeader()..'<body>\n')

  -- local imNames = www_utils.saveIms(ims, imDir, isNorm, 0)
  -- local audioNames = www_utils.saveAudios(audios, imDir, 0, audioRate)
  local imNames = {}
  local audioNames = {}
  for i = 1, (#audios) do
    imNames[i] = 'NoImNoAu'
  end
  audioNames = imNames
  local htmlStr = www_utils.renderAudioTables(imNames, audioNames, captions, ncol, width, imRelativeDir)

  f:write(htmlStr .. '</body>\n</html>')
  f:close()

  return count
end

function www_utils.renderABSHtml(sound, dataset, rootDir, audio1s, audio2s, audio3s, captions , ncol, isNorm, width, htmlFile, audioRate)
  -- print(captions)
  print('Writing visualize html ...')
  captions = captions or {}
  ncol = ncol or 8
  isNorm = isNorm or false
  width = width or 256
  htmlFile = htmlFile or 'index.html'
  audioRate = audioRate or 44100

  local audioRelativeDir = './audio'
  local audioDir = paths.concat(rootDir, audioRelativeDir)

  lfs.mkdir(rootDir)
  lfs.mkdir(audioDir)

  local f = io.open(paths.concat(rootDir, htmlFile), "w")
  f:write('<html>\n'..www_utils.renderHeader()..'<body>\n')

  -- local imNames = www_utils.saveIms(ims, imDir, isNorm, 0)
  -- local audioNames = www_utils.saveAudios(audios, imDir, 0, audioRate)

  local audio1Names = {}
  local audio2Names = {}
  local audio3Names = {}

  file = io.open (paths.concat(rootDir,'targets.txt'), 'w')
  for i = 1, (#audio1s) do
    audio1Names[i] = 'target' .. i ..'.wav'
    file:write( audio1s[i]..' '..paths.concat(audioDir,'synth'..i..'.wav')..'\n')
  end
  file:close()
  for i = 1, (#audio2s) do
    audio2Names[i] = 'output' .. i ..'.wav'
  end
  for i = 1, (#audio3s) do
    audio3Names[i] = 'synth' .. i ..'.wav'
  end

  if sound then
    moveTargetAudio(audioDir, captions, audio1s, audio1Names , dataset)
    if not dataset == 'primV4a' then
      genInferenceAudio(audioDir, captions, audio2Names , dataset)
    end
  end

  local htmlStr = www_utils.renderABSTables(audio1Names, audio2Names, audio3Names, captions, ncol, width, audioRelativeDir)

  f:write(htmlStr .. '</body>\n</html>')
  f:close()

  return count
end


function moveTargetAudio(audioDir, captions, audio1s, audio1Names, dataset) 
  print('Copying target audios into html folder ...')
  for i = 1, (#captions) do
    local root = '/data/vision/billf/object-properties/sound/sound/primitives/data/' .. dataset
    local caption = captions[i]
    local split = {}
    for token in string.gmatch(caption, "[^%s]+") do
      table.insert(split,token)
    end
    if dataset == 'primV2c' then
      local target_xml = {split[2],split[3],split[4],split[5],split[6]}
      -- print(target_xml)  
      local a = {["-1.00"] = 0,["-0.60"] = 1,["-0.20"] = 2,["0.20"] = 3,["0.60"] = 4,["1.00"] = 5}
      local b = {["-1.00"] = 0,["-0.67"] = 1,["-0.33"] = 2,["0.00"] = 3,["0.33"] = 4,["0.67"] = 5, ["1.00"] = 6}

      root = root .. '/scene-' .. a[target_xml[1]] ..'-' .. b[target_xml[2]]..'/'
      root = root .. '/mat-10-10-10-' .. a[target_xml[3]] .. '-' .. a[target_xml[4]] .. '-' .. a[target_xml[5]] .. '/sound.wav'

    elseif dataset == 'primV2d' or dataset == 'primV2e' or dataset=='primV3a' then
      root = root .. '/' .. audio1s[i] .. '/sound.wav'
    elseif  dataset=='primV3b' or dataset=='primV3b_5000' then
      local id = tonumber(audio1s[i])
      local outer_folter = ('%.03d'):format( math.floor((id-0.5)/1000) +1)
      root = root ..'/'..outer_folter..'/'..audio1s[i] .. '/sound.wav'
    elseif dataset=='primV4a' then
      local id = tonumber(audio1s[i])
      local outer_folter = ('%.03d'):format( math.floor((id-0.5)/1000) +1)
      root = root ..'/'..outer_folter..'/'..audio1s[i] .. '/sound.wav'      
    end

    os.execute("cp " .. root .. " " .. paths.concat(audioDir, audio1Names[i]) )
  end
end

function genInferenceAudio(audioDir, captions, audio2Names , dataset) 
  print('Generating inferenced audios into html folder ...')
  for i = 1, (#captions) do
    -- local root = '/data/vision/billf/object-properties/sound/sound/primitives/data/primV2c/'
    local caption = captions[i]
    local split = {}
    for token in string.gmatch(caption, "[^%s]+") do
      table.insert(split,token)
    end
    local output_xml = {}
    if dataset == 'primV2c' or dataset=='primV2d' or dataset=='primV3a' then
      output_xml = {tonumber(split[8]),tonumber(split[9]),tonumber(split[10]),tonumber(split[11]),tonumber(split[12])}

    elseif dataset == 'primV2e' or dataset=='primV3b' or dataset=='primV3b_5000' then
      target_xml = {tonumber(split[2]),tonumber(split[3]),tonumber(split[4]),tonumber(split[5]),tonumber(split[6]),tonumber(split[7]),tonumber(split[8])}
      output_xml = {tonumber(split[10]),tonumber(split[11]),tonumber(split[12]),tonumber(split[13]),tonumber(split[14]),tonumber(split[15]),tonumber(split[16])}

    end
    gen_sound(output_xml, audioDir, audio2Names[i], dataset, target_xml )
  end
end


return www_utils
