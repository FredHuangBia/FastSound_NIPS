function write_frequencies(inputDir, outputDir)
root = '/data/vision/billf/object-properties/sound/sound/primitives/data/v2.3/';
load([root,inputDir,'/bem_input','/freq.mat']);
fileID = fopen([root,outputDir],'w');
fprintf(fileID,'%f\n',freq);