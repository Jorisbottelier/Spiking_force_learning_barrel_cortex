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

% Gs = 1:10;
% parfor i=1:length(Gs)
%     filename = ['simtest3_' num2str(i)];
%     run_sim(2000, 200, 600, 100, 1, 1, Gs(i), 1,...
%     1, 0.05, 0, true, true, 'spikes', ['.' filesep 'Output' filesep filename]);
% end

% Looping over the sparsity:

% sparsities = 0:0.1:1;
% parfor i=1:length(sparsities)
%     filename = ['sparsity_simtest' num2str(i)];
%     sparsity_run_sim(2000, 200, 600, 100, 1, 1, 4, 1,...
%     1, 0.05, 0, sparsities(i), true, true, 'spikes', ['.' filesep 'Output' filesep filename]);
% end

% Training and testing reproducability of input types:

input_types = {'spikes', 'PSTH', 'ConvTrace'};
for j=1:2
    for i=1:length(input_types)
        filename = ['a_' char(input_types(i)) '_simtest' num2str(j)];
        run_sim_nonrandom(2000, 200, 2, 2, 1, 1, 4, 1,...
        1, 0.05, 0, true, true, input_types(i), ['.' filesep 'Output' filesep filename]);
    end
end




