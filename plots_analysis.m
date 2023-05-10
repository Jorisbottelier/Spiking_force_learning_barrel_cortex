%% Creates plots for the initialized weight matrices.

% For 1 < G < 5:

% G_vals = 1:0.5:5;
% for i=1:length(G_vals)
%      filename = ['Win_1G_' num2str(G_vals(i)) 'Q_1Winp_1Pexc_0.mat'];
%      load(filename);
%      plotname = ['Network ' num2str(i) ', test accuracy: ' num2str(training_output.acc)];
%      disp(training_output.acc);
%      subplot(3, 3, i);
%      imagesc(training_output.weights.static(50:100, 50:100));
%      axis square;
%      h.colormap = flag;
%      title(plotname);
% end


% For 1< G < 10:

% for i=1:10
%      filename = ['Win_1G_' num2str(i) 'Q_1Winp_1Pexc_0#3.mat'];
%      load(filename);
%      plotname = ['Network ' num2str(i) ', test accuracy: ' num2str(training_output.acc)];
%      disp(training_output.acc);
%      subplot(2, 5, i);
%      imagesc(training_output.weights.static);
%      axis square;
%      title(plotname);
% end

% For G = 5:

% for i=1:9
%      filename = ['Win_1G_5Q_1Winp_1Pexc_0_' num2str(i) '.mat'];
%      load(filename);
%      plotname = ['Network ' num2str(i) ', test accuracy: ' num2str(training_output.acc)];
%      disp(training_output.acc);
%      subplot(3, 3, i);
%      imagesc(training_output.weights.static);
%      axis square;
%      title(plotname);
% end


% Calculating sum and total positive/negative static weights:

% G_vals = 1:0.5:5;
% for i=1:length(G_vals)
%      filename = ['Win_1G_' num2str(G_vals(i)) 'Q_1Winp_1Pexc_0.mat'];
%      load(filename);
%      fprintf(['Network ' num2str(i) ': \n'])
%      fprintf(['Test accuracy: ' num2str(training_output.acc) '\n'])
%      fprintf('sum: ')
%      fprintf('%f', sum(training_output.weights.static, 'all'));
%      fprintf('\n')
%      v = training_output.weights.static < 0;
%      w = training_output.weights.static > 0;
%      x = size(find(v));
%      y = size(find(w));
%      fprintf('Number of weights less than zero: ')
%      fprintf('%d', x(1));
%      fprintf('\n')
%      fprintf('Number of weights greater than zero: ')
%      fprintf('%d', y(1));
%      fprintf('\n\n')
% end


%% Finds and calculates the number of connections for each neuron:

% G_vals = 1:0.5:5;
% for i=1:length(G_vals)
%      filename = ['Win_1G_' num2str(G_vals(i)) 'Q_1Winp_1Pexc_0.mat'];
%      load(filename);
%      connections = zeros(1, length(training_output.weights.static));
%      for j=1:length(training_output.weights.static)
%          row_connections = size(find(training_output.weights.static(j,:)));
%          connections(j) = row_connections(2);
%      end
%      plotname = ['Network ' num2str(i) 'Most connections: ' num2str(max(connections)) ', test accuracy: ' num2str(training_output.acc)];
%      subplot(3, 3, i);
%      histogram(connections);
%      title(plotname);
% end


%% Finds and plots the eigenvalues of the static weight matrix:

%For G between 1 and 5:
 
% G_vals = 1:0.5:5;
% for i=1:length(G_vals)
%      filename = ['Win_1G_' num2str(G_vals(i)) 'Q_1Winp_1Pexc_0.mat'];
%      load(filename);
%      plotname = ['Network ' num2str(i) ', test accuracy: ' num2str(training_output.acc)];
%      eigenvalues = eig(training_output.weights.static);
%      X = real(eigenvalues);
%      Y = imag(eigenvalues);
%      subplot(3, 3, i);
%      scatter(X,Y, 4, '.', 'black');
%      xlim([1.1*min(X) 1.1*max(X)])
%      ylim([1.1*min(Y) 1.1*max(Y)])
%      grid on;
%      axis square;
%      title(plotname);
% end

%For G between 1 and 10:

% for i=1:10
%      filename = ['Win_1G_' num2str(i) 'Q_1Winp_1Pexc_0#3.mat'];
%      load(filename);
%      plotname = ['Network ' num2str(i) ', test accuracy: ' num2str(training_output.acc)];
%      eigenvalues = eig(training_output.weights.static);
%      X = real(eigenvalues);
%      Y = imag(eigenvalues);
%      subplot(2, 5, i);
%      scatter(X,Y, 4, '.', 'black');
%      xlim([1.1*min(X) 1.1*max(X)])
%      ylim([1.1*min(Y) 1.1*max(Y)])
%      grid on;
%      axis square;
%      title(plotname);
% end

% For G = 5:

% for i=1:9
%      filename = ['Win_1G_5Q_1Winp_1Pexc_0_' num2str(i) '.mat'];
%      load(filename);
%      plotname = ['Network ' num2str(i) ', test accuracy: ' num2str(training_output.acc)];
%      eigenvalues = eig(training_output.weights.static);
%      X = real(eigenvalues);
%      Y = imag(eigenvalues);
%      subplot(3, 3, i);
%      scatter(X,Y, 4, '.', 'black');
%      xlim([1.1*min(X) 1.1*max(X)])
%      ylim([1.1*min(Y) 1.1*max(Y)])
%      grid on;
%      axis square;
%      title(plotname);
% end

%% Creates a scatter plot of the total weight change of each training trial 

%For G between 1 and 10:

% for i=1:10
%      filename = ['Win_1G_' num2str(i) 'Q_1Winp_1Pexc_0#3.mat'];
%      load(filename);
%      plotname = ['Network ' num2str(i) ', test accuracy: ' num2str(training_output.acc)];
%      training_trials = 1:600;
%      subplot(2, 5, i);
%      scatter(training_trials, training_output.weight_change, '.', 'black')
%      title(plotname);
% end

%For the nonrandom test:

% for i=1:6
%      filename = ['Win_1G_5Q_1Winp_1Pexc_0_nonrand' num2str(i) '.mat'];
%      load(filename);
%      plotname = ['Network ' num2str(i) ', test accuracy: ' num2str(training_output.acc)];
%      training_trials = 1:600;
%      subplot(2, 5, i);
%      scatter(training_trials, training_output.weight_change, '.', 'black')
%      title(plotname);
% end

%% Plots the output weight matrix


%For G between 1 and 10:

% for i=1:10
%      filename = ['Win_1G_' num2str(i) 'Q_1Winp_1Pexc_0#3.mat'];
%      load(filename);
%      plotname = ['Network ' num2str(i) ', test accuracy: ' num2str(training_output.acc)];
%      weight_numbers = 1:2000;
%      subplot(2, 5, i);
%      scatter(weight_numbers, training_output.weights.output, '.', 'black');
%      ylim([-3 3])
%      title(plotname);
% end

%For the nonrandom test:

% for i=1:6
%      filename = ['Win_1G_5Q_1Winp_1Pexc_0_nonrand' num2str(i) '.mat'];
%      load(filename);
%      plotname = ['Network ' num2str(i) ', test accuracy: ' num2str(training_output.acc)];
%      weight_numbers = 1:2000;
%      subplot(2, 3, i);
%      scatter(weight_numbers, training_output.weights.output, '.', 'black');
%      ylim([-3 3])
%      title(plotname);
% end

%% Analyzing the reproducibility

%% 1. Checking if the weights are the same

%Checking if the initialized static weights are the same (They are)

% static_weights = zeros(2000, 2000, 6);
% for i=1:6
%     filename = ['Win_1G_5Q_1Winp_1Pexc_0_rep' num2str(i)];
%     load(filename);
%     static_weights(:,:,i) = training_output.weights.static;
% end
% 
% for i=1:5
%     disp(sum(sum(static_weights(:,:,i) == static_weights(:,:,i+1))));
% end

%Example
% A = [1 2 3; 4 5 6; 7 8 9];
% B = [1 2 3; 4 5 6; 7 8 9];
% 
% disp(sum(sum(A==B)));

%Checking if the input weights are the same (They are)

% input_weights = zeros(2000, 200, 6);
% for i=1:6
%     filename = ['Win_1G_5Q_1Winp_1Pexc_0_rep' num2str(i)];
%     load(filename);
%     input_weights(:,:,i) = training_output.weights.input;
% end
% 
% for i=1:5
%     disp(sum(sum(input_weights(:,:,i) == input_weights(:,:,i+1))));
% end

%Checking if the feedback weights are the same (They are)

% feedback_weights = zeros(2000, 1, 6);
% for i=1:6
%     filename = ['Win_1G_5Q_1Winp_1Pexc_0_rep' num2str(i)];
%     load(filename);
%     feedback_weights(:,:,i) = training_output.weights.feedback;
% end
% 
% for i=1:5
%     disp(sum(sum(feedback_weights(:,:,i) == feedback_weights(:,:,i+1))));
% end

%Checking if the output weights are the same (Turns out these are all
%different!!!!)

% output_weights = zeros(2000, 1, 6);
% for i=1:6
%     filename = ['Win_1G_5Q_1Winp_1Pexc_0_rep' num2str(i)];
%     load(filename);
%     output_weights(:,:,i) = training_output.weights.output;
% end
% 
% for i=1:5
%     disp(sum(sum(output_weights(:,:,i) == output_weights(:,:,i+1))));
% end

%For the second nonrandom test they are the same:

% output_weights = zeros(2000, 1, 6);
% for i=1:6
%     filename = ['Win_1G_5Q_1Winp_1Pexc_0_nonrand' num2str(i) '.mat'];
%     load(filename);
%     output_weights(:,:,i) = training_output.weights.output;
% end
% 
% for i=1:5
%     disp(sum(sum(output_weights(:,:,i) == output_weights(:,:,i+1))));
% end


%% 2. Checking if the test_output is the same

%checking if the trials are the same (They are)

% check_trials = {};
% for i=1:6
%     filename = ['Win_1G_5Q_1Winp_1Pexc_0_rep' num2str(i)];
%     load(filename);
%     check_trials{i} = training_output.test_output.trials;
% end
% 
% for i=1:5
%     disp(isequal(check_trials{i}, check_trials{i+1}));
% end

%checking if the error is the same (They are different)

% check_error = {};
% for i=1:6
%     filename = ['Win_1G_5Q_1Winp_1Pexc_0_rep' num2str(i)];
%     load(filename);
%     check_error{i} = training_output.test_output.error;
% end
% 
% for i=1:5
%     disp(isequal(check_error{i}, check_error{i+1}));
% end

%checking if Zx is the same (they are the same)

% check_Zx = {};
% for i=1:6
%     filename = ['Win_1G_5Q_1Winp_1Pexc_0_rep' num2str(i)];
%     load(filename);
%     check_Zx{i} = training_output.test_output.Zx;
% end
% 
% for i=1:5
%     disp(isequal(check_Zx{i}, check_Zx{i+1}));
% end

%checking if Z_out is the same (they are different)

% check_Z_out = {};
% for i=1:6
%     filename = ['Win_1G_5Q_1Winp_1Pexc_0_rep' num2str(i)];
%     load(filename);
%     check_Z_out{i} = training_output.test_output.Z_out;
% end
% 
% for i=1:5
%     disp(isequal(check_Z_out{i}, check_Z_out{i+1}));
% end

%checking if the tspikes are the same (they are different)

% check_tspikes = {};
% for i=1:6
%     filename = ['Win_1G_5Q_1Winp_1Pexc_0_rep' num2str(i)];
%     load(filename);
%     check_tspikes{i} = training_output.test_output.tspikes;
% end
% 
% for i=1:5
%     disp(isequal(check_tspikes{i}, check_tspikes{i+1}));
% end

%checking if the stats are the same (they are different)

% check_stats = {};
% for i=1:6
%     filename = ['Win_1G_5Q_1Winp_1Pexc_0_rep' num2str(i)];
%     load(filename);
%     check_stats{i} = training_output.test_output.stats;
% end
% 
% for i=1:5
%     disp(isequal(check_stats{i}, check_stats{i+1}));
% end

%checking if the first_touches are the same (they are the same)

% check_first_touches = zeros(100, 1, 6);
% for i=1:6
%     filename = ['Win_1G_5Q_1Winp_1Pexc_0_rep' num2str(i)];
%     load(filename);
%     check_first_touches(:,:,i) = training_output.test_output.first_touches;
% end
% 
% for i=1:5
%     disp(sum(sum(check_first_touches(:,:,i) == check_first_touches(:,:,i+1))));
% end

%% 3. Checking if the weight_change is the same 

% (they are different)

% check_weight_change = zeros(600, 1, 6);
% for i=1:6
%     filename = ['Win_1G_5Q_1Winp_1Pexc_0_rep' num2str(i)];
%     load(filename);
%     check_weight_change(:,:,i) = training_output.weight_change;
% end
% 
% for i=1:5
%     disp(sum(sum(check_weight_change(:,:,i) == check_weight_change(:,:,i+1))));
% end

%% 4. Checking if train_trials are the same

% (they are the same)

% check_train_trials = cell(1, 6);
% for i=1:6
%     filename = ['Win_1G_5Q_1Winp_1Pexc_0_rep' num2str(i)];
%     load(filename);
%     check_train_trials{i} = training_output.train_trials;
% end
% 
% for i=1:5
%     disp(isequal(check_train_trials{i}, check_train_trials{i+1}));
% end

%% Plotting the differences

%% 1. Weight_change

% for i=1:6
%      filename = ['Win_1G_5Q_1Winp_1Pexc_0_rep' num2str(i)];
%      load(filename);
%      plotname = ['Network ' num2str(i) ', test accuracy: ' num2str(training_output.acc)];
%      training_trials = 1:600;
%      subplot(2, 3, i);
%      scatter(training_trials, training_output.weight_change, '.', 'black')
%      title(plotname);
% end

%% 2. Zx and Z_out 

% for i=1:6
%     filename = ['Win_1G_5Q_1Winp_1Pexc_0_nonrand' num2str(i) '.mat'];
%     load(filename);
%     
%     full_Zx = [];  % Initialize an empty array to store the concatenated result
%     for j = 1:100
%         full_Zx = cat(2, full_Zx, training_output.test_output.Zx{j});  % Concatenate the i-th double array vertically to the result
%     end
%     
%     full_Z_out = [];  % Initialize an empty array to store the concatenated result
%     for j = 1:100
%         full_Z_out = cat(1, full_Z_out, training_output.test_output.Z_out{j});  % Concatenate the i-th double array vertically to the result
%     end
%     
%     % Create the plot
%     subplot(2, 3, i); % Create a new figure
%     hold on; % Enable hold to overlay plots on the same axes
%     plot(full_Zx, 'b'); % Plot Zx with blue color
%     plot(full_Z_out, 'r'); % Plot Z_out with red color
%     
%     % Add labels and legend
%     ylim([-4,4]);
%     xlabel('X-axis label'); % Replace with your desired x-axis label
%     ylabel('Y-axis label'); % Replace with your desired y-axis label
%     title(['Network ' num2str(i) ', test accuracy: ' num2str(training_output.acc)]); % Set the plot title
%     legend('Zx', 'Z_{out}'); % Add a legend to differentiate between Zx and Z_out
%     
%     hold off; % Disable hold to prevent further overlays on the same axes
% end

%% 3. Error

% for i=1:6
%      filename = ['Win_1G_5Q_1Winp_1Pexc_0_rep' num2str(i)];
%      load(filename);
%      plotname = ['Network ' num2str(i) ', test accuracy: ' num2str(training_output.acc)];
%      
%      test_trials = 1:100;
%      error_double = zeros(1, 100);
%      for j=1:100
%         error_double(j) = training_output.test_output.error{j}; 
%      end
%      
%      subplot(2, 3, i);
%      scatter(test_trials, error_double, '.', 'black');
%      title(plotname);
% end

%% 4.

%% Checking if trail selector is reproducible (it is)

% file = load('trainable_trials');
% trainable_trials = file.trainable_trials;

% get the shuffled train and test trials in the ratio 1:1, prox:dist
% [train_trials, test_trials] = trial_selector(trainable_trials.prox_touch,...
%     trainable_trials.dist_no_touch, N_train, N_test);
% get the fixed train and test trials; one distal and one proximal trial
% [train_trials1, test_trials1] = fixed_trial_selector(trainable_trials.prox_touch,...
%     trainable_trials.dist_no_touch, 600, 100);
% [train_trials2, test_trials2] = fixed_trial_selector(trainable_trials.prox_touch,...
%     trainable_trials.dist_no_touch, 600, 100);

% disp(isequal(train_trials1, train_trials2));
% disp(isequal(test_trials1, test_trials2));

%% Checking if make_trial_spikes is reproducible (it is not) 
%% (Now there is a nonrandom version)

% load the KernelStruct
% filename = ['.' filesep 'Input' filesep 'KernelStruct.mat'];
% 
% if ~exist(filename, 'file')
%     error('KernelStruct.mat is not in the input folder')
% end
% 
% KernelStruct = load(filename);
% KernelStruct = KernelStruct.KernelStruct;
% 
% % load the whiskmat
% filename = ['.' filesep 'Input' filesep 'whiskmat.mat'];
% 
% if ~exist(filename, 'file')
%     error('whiskmat.mat is not in the input folder')
% end
% 
% whiskmat = load(filename);
% whiskmat = whiskmat.filtered_whiskmat;

% create a cell of make_trial_spikes with four iterations using the same
% input values
% SpikeTrainStructs = cell(1, 4);
% for i=1:4
%     SpikeTrainStructs(i) = make_trial_spikes_nonrandom(train_trials1(1).session, train_trials1(1).trial,...
%                 whiskmat, KernelStruct);
% end
%  
% %check if the SpikeTrainStructs generated by make_trial_spikes are equal
% %(they are not equal)
% for i=1:3
%     disp(isequal(SpikeTrainStructs(i), SpikeTrainStructs(i+1)));
% end

%check in the ConvTraces in the SpikeTrainStructs are equal (they are)
% for i=1:3
%     disp(isequal(SpikeTrainStructs{i}.ConvTrace, SpikeTrainStructs{i+1}.ConvTrace));
% end

%check in the PSTH in the SpikeTrainStructs are equal (they are)
% for i=1:3
%     disp(isequal(SpikeTrainStructs{i}.PSTH, SpikeTrainStructs{i+1}.PSTH));
% end

%check in the SpikeTimes in the SpikeTrainStructs are equal (they are not!)
% for i=1:3
%     disp(isequal(SpikeTrainStructs{i}.SpikeTimes, SpikeTrainStructs{i+1}.SpikeTimes));
% end

%check in the SpikeCounts in the SpikeTrainStructs are equal (they are not!)
% for i=1:3
%     disp(isequal(SpikeTrainStructs{i}.SpikeCount, SpikeTrainStructs{i+1}.SpikeCount));
% end

%plot the generated spikes 
% spike_train = make_trial_spikes_nonrandom(train_trials1(1).session, train_trials1(1).trial,...
%                   whiskmat, KernelStruct);
% 
% subplot(3,1,1);
% plot(spike_train{1,1}.ConvTrace{1});
% subplot(3,1,2);
% plot(spike_train{1,1}.PSTH{1});
% subplot(3,1,3);
% spikey = zeros(1, 2005);
% for i=1:length(spike_train{1,1}.SpikeTimes{1,1})
%     spikey(i) = 1;
% end
% disp(spike_train{1,1}.SpikeTimes{1,1});
% plot(spikey);




%% make_trial_spikes uses dynamic_spike_maker, check if this function is deterministic:

% 1. Check if dynamic_spike_maker is deterministic:
% (It is not, however the _nonrandom version is)

% % load the KernelStruct
% filename = ['.' filesep 'Input' filesep 'KernelStruct.mat'];
% 
% if ~exist(filename, 'file')
%     error('KernelStruct.mat is not in the input folder')
% end
% 
% KernelStruct = load(filename);
% KernelStruct = KernelStruct.KernelStruct;
% 
% % load the whiskmat
% filename = ['.' filesep 'Input' filesep 'whiskmat.mat'];
% 
% if ~exist(filename, 'file')
%     error('whiskmat.mat is not in the input folder')
% end
% 
% whiskmat = load(filename);
% whiskmat = whiskmat.filtered_whiskmat;
% 
% results1 = cell(1, 4);
% for i=1:4
%     % select sessions from the whiskingmat
%     session_index = find(strcmp({whiskmat.session}, train_trials1(1).session));
%     session_mat = whiskmat(session_index);
%     
%     % select the trial from the sessions
%     trial_index = find([session_mat.trialId] == train_trials1(1).trial);
%     trial_mat = session_mat(trial_index);
% 
%     results1(i) = dynamic_spike_maker_nonrandom(KernelStruct, trial_mat);
% end
% 
% for i=1:3
%     disp(isequal(results1(i), results1(i+1)));
% end


% 2. dynamic_spike_maker uses make_whisker_trace, check determinism:
% (make_whisker_trace produces the same result when using the same input)

% % load the KernelStruct
% filename = ['.' filesep 'Input' filesep 'KernelStruct.mat'];
% 
% if ~exist(filename, 'file')
%     error('KernelStruct.mat is not in the input folder')
% end
% 
% KernelStruct = load(filename);
% KernelStruct = KernelStruct.KernelStruct;
% 
% % load the whiskmat
% filename = ['.' filesep 'Input' filesep 'whiskmat.mat'];
% 
% if ~exist(filename, 'file')
%     error('whiskmat.mat is not in the input folder')
% end
% 
% whiskmat = load(filename);
% whiskmat = whiskmat.filtered_whiskmat;
% 
% results2 = cell(1, 4);
% for i=1:4
%     % select sessions from the whiskingmat
%     session_index = find(strcmp({whiskmat.session}, train_trials1(1).session));
%     session_mat = whiskmat(session_index);
%     
%     % select the trial from the sessions
%     trial_index = find([session_mat.trialId] == train_trials1(1).trial);
%     trial_mat = session_mat(trial_index);
% 
%     pole = 1;
% 
%     results2{i} = make_whisker_trace(trial_mat, pole);
% end
% 
% for i=1:3
%     disp(isequal(results2(i), results2(i+1)));
% end

% 3. dynamic_spike maker also uses kernel_recording_to_spiketrain, this
% function must not be deterministic










