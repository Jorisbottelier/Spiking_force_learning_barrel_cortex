% Creates a list of 7 values for G between 1 and 5, incremented by 0.5 
% and trains a network for each value.

% Gs = 1:0.5:5;
% parfor i=1:length(Gs)
%     filename = ['simtest' num2str(i)];
%     run_sim(2000, 200, 600, 100, 1, 1, Gs(i), 1,...
%     1, 0.05, 0, true, true, 'spikes', ['.' filesep 'Output' filesep filename]);
% end

% Creates a list of 10 values for G between 1 and 10 and trains the network
% for each value

Gs = 1:10;
parfor i=1:length(Gs)
    filename = ['simtest3_' num2str(i)];
    run_sim(2000, 200, 600, 100, 1, 1, Gs(i), 1,...
    1, 0.05, 0, true, true, 'spikes', ['.' filesep 'Output' filesep filename]);
end