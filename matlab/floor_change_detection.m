function floor_change_detection(viofile)

zero_count_threshold = 20; % about 2 seconds
hPa_threshold = 0.3;
min_hPa_val = 0.005;
partial_hPa_threshold = 0.1; % 10% of hPA_threshold

[t, ~, ~, ~, alt, ~] = vio_parser(viofile, 1);

tstamp = cell2mat(t{1,1})';
pressure = cell2mat(alt{1,1})'*10; % convert from kPA to hPA

% if too many samples have zero delta, then we should make a decision 
pressure_norm =  (pressure-pressure(1));
t_norm = tstamp-tstamp(1);

figure, plot(t_norm, (pressure_norm)), title(viofile), hold on

i_start = -1;
i_end = -1;
curr_direction = 0;
prev_pressure = nan;

zero_count = 0;

partial_direction = nan;
partial_start_idx = -1;

reset = 0;
seed_value = pressure(1);
plot(t_norm(1), pressure_norm(1))
for i = 2 : length(pressure)
    
    if reset
        seed_value = pressure(i-2);
        reset = 0;
    end
    
    delta = pressure(i) - seed_value;
    
    if abs(delta) > 0.3
        if delta/abs(delta) < 0
            plot(t_norm(i), pressure_norm(i), 'V')
        else
            plot(t_norm(i), pressure_norm(i), '^')
        end
        reset = 1;
    end
end

% for i = 2 : length(pressure)
%     
%     delta_pressure = pressure(i) - pressure(i-1);
%     
%     % update start index
%     if abs(delta_pressure) > min_hPa_val && i_start == -1
%         i_start = i-1;
%     end
%     
%     prev_direction = curr_direction;
%     
%     if delta_pressure > min_hPa_val
%         curr_direction = -1;
%         zero_count = 0;
%     elseif delta_pressure < -min_hPa_val
%         curr_direction = 1;
%         zero_count = 0;
%     else
%         zero_count = zero_count + 1;
%         %curr_direction = 0;
%     end
%     
%     %change of slope or timeout (long zero sequence)
%     if prev_direction * curr_direction < 0 || zero_count > zero_count_threshold
%         i_end = i - 1;
%         zero_count = 0;
%         % this triggers analysis
%         if (i_end > 0 && i_start > 0)
%         deltaPressure = abs(pressure(i_end) - pressure(i_start));
%             if deltaPressure > hPa_threshold
%                 disp('Floor change detected!')
%                 % visualization
%                 hold on, plot(t_norm(i_start), pressure_norm(i_start), 'X', 'MarkerSize', 10)
%                 hold on, plot(t_norm(i_end), pressure_norm(i_end), 'O', 'MarkerSize', 10)
%                 
%                 x_max = xlim;
%                 x_max = x_max(2)-x_max(1);
%                 y_max = ylim;
%                 y_max = y_max(2);
%                 
%                 if prev_direction == 1  
%                     x = (t_norm(i_end) + t_norm(i_start))/2;
%                     y = (pressure_norm(i_end) + pressure_norm(i_start))/2;
%                     [x/x_max x/x_max],[y/y_max y/y_max+0.01]
%                     annotation('arrow',[x/x_max x/x_max],[y/y_max y/y_max+0.01])
%                     text(x,y,num2str(deltaPressure/hPa_threshold));
%                     idx = (tstamp >= tstamp(i_start)) & (tstamp <= tstamp(i_end));
%                     patch([t_norm(idx) fliplr(t_norm(idx))], ...
%                           [pressure_norm(idx) zeros(size(pressure_norm(idx)))], ...
%                           [0. 0. 1.], 'FaceAlpha',0.3, 'EdgeColor','none')
%                 else
%                     x = (t_norm(i_end) + t_norm(i_start))/2 - 10;
%                     y = (pressure_norm(i_end) + pressure_norm(i_start))/2;
%                     text(x,y,num2str(deltaPressure/hPa_threshold));
%                     idx = (tstamp >= tstamp(i_start)) & (tstamp <= tstamp(i_end));
%                     patch([t_norm(idx) fliplr(t_norm(idx))], ...
%                           [pressure_norm(idx) zeros(size(pressure_norm(idx)))], ...
%                           [1. 0. 1.], 'FaceAlpha',0.3, 'EdgeColor','none')
%                 end
%                 
%                 % event found, start new search
%                 i_start = i_end + 1;
%                 i_end = -1;
%             
%             else
%                 if deltaPressure/hPa_threshold > partial_hPa_threshold
%                     disp('Partial')
%                     partial_start_idx = i_start;
%                 else
%                     i_start = i_end + 1;
%                     i_end = -1;
%                 end
%             end
%         end
%     elseif prev_direction * curr_direction == 1
%         i_end = i;
%     end
%     
% end






% for i = 1 : length(pressure)
%     
%     % init condition
%     if isnan(prev_pressure)
%         prev_pressure = pressure(i);
%     else
%         % measure delta
%         dPress = pressure(i) - prev_pressure;
%         tmp_direction = 0;
%         prev_pressure = pressure(i);
%        
%         % initialize i_start only once at the beginning 
%         if dPress ~= 0 && i_start == -1
%             i_start = i-1;
%         end
%         
%         % update direction of \Delta(Pressure)
%         if dPress > min_hPa_val
%             tmp_direction = -1;
%             zero_count = 0;
%         elseif dPress < - min_hPa_val
%             tmp_direction = 1;
%             zero_count = 0;
%         else
%             % if no change in gradient, increase time out counter
%             zero_count = zero_count + 1;
%             tmp_direction = 0;
%         end
%         
%         % make sure we have a range to test
%         if (i_start >0)
%             % change of direction or time out (TODO: consider <= 0)
%             if direction * tmp_direction < 0 || zero_count > zero_count_threshold
%                 % run test
%                 i_end = i-2; %need to check why      
%                 %reset timeout counter
%                 zero_count = 0; 
%                 
%                 %check partial measurements exists and direction is
%                 %consistent
%                 if partial_start_idx > 0 && partial_direction == direction
%                     dPressure = abs((pressure(i_end) - pressure(partial_start_idx)));
%                     i_start = partial_start_idx;
%                 else
%                     dPressure = abs((pressure(i_end) - pressure(i_start)));
%                     partial_start_idx = -1;
%                     partial_direction = 0;
%                     
%                 end
%                 
%                 if dPressure > hPa_threshold
%                     % classify event (up or down TODO: how many stories, elevator/stairs)
%                     
%                     idx = (tstamp >= tstamp(i_start)) & (tstamp <= tstamp(i_end));
%                    
%                     disp('Detected change of elevation')
%                     disp(strcat('\Delta(T): ' , num2str(tstamp(i_end) - tstamp(i_start))))
%                     disp(strcat('\Delta(B): ', num2str(abs(pressure(i_end) - pressure(i_start)))))
%                     
%                     if direction == 1
%                         disp('Going up');
%                         patch([t_norm(idx) fliplr(t_norm(idx))], ...
%                         [pressure_norm(idx) zeros(size(pressure_norm(idx)))], ...
%                         [0. 0. 1.], 'FaceAlpha',0.3, 'EdgeColor','none')        
%                     else
%                         disp('Going down');
%                         patch([t_norm(idx) fliplr(t_norm(idx))], ...
%                         [pressure_norm(idx) zeros(size(pressure_norm(idx)))], ...
%                         [1. 0. 0.], 'FaceAlpha',0.3, 'EdgeColor','none')
%                     end
%                 else
%                     % if deltaP doesn't match threshold, check if it is a partial change of floor
%                     if dPressure/hPa_threshold > partial_hPa_threshold
%                         if partial_start_idx == -1
%                             partial_start_idx = i_start;
%                             d =  -(pressure(i_end) - pressure(partial_start_idx)) ... 
%                                 / abs((pressure(i_end) - pressure(partial_start_idx)));
%                             partial_direction = d;
%                             direction = d;
%                         end
%                         idx = (tstamp >= tstamp(i_start)) & (tstamp <= tstamp(i_end));
%                         disp('Partial ');
%                         patch([t_norm(idx) fliplr(t_norm(idx))], ...
%                         [pressure_norm(idx) zeros(size(pressure_norm(idx)))], ...
%                         [1. 0.5 0.5], 'FaceAlpha',0.3, 'EdgeColor','none')
%                     else 
%                         partial_start_idx = -1;
%                         partial_direction = 0;
%                     end
%                 end
% 
%                 direction = tmp_direction;
%                 i_start = i-1;   
%                 prev_pressure = pressure(i_start);
% 
%             elseif direction == 0 && tmp_direction ~= 0
%                 direction = tmp_direction;
%             end
%         end
%     end
%     
%     
% end
