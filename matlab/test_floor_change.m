files = dir('./alt_data/*.txt')
close all
for f = 1 : length(files)
   fullpath = fullfile('./alt_data/', files(f).name)
   floor_change_detection(fullpath)
end