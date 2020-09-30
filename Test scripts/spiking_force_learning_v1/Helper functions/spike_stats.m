function [ A_t, ISI, Cv ] = spike_stats( tspike, trial_length , N )
% Input an array with which neurons spikes when, and return the ISI, the
% coefficient of variation and the average activity.

% Average activity

% calculate the spikecounts per neuron
[A_t, ~] = histcounts(tspike(:,1), 1:1:N + 1); 
A_t = A_t / (trial_length/1000);

% ISI
ISI = [];
for n=1:N
    index = find(tspike(:,1)==n);
    spike_neuron(n).time = tspike(index,2);
    spike_neuron(n).ISI = diff(spike_neuron(n).time);
    ISI = [ISI diff(spike_neuron(n).time)'];
end

% Coefficient of variation
Cv = zeros(N,1);
for n = 1:N
    Cv(n,1) = std(spike_neuron(n).ISI)/mean(spike_neuron(n).ISI);
end


end

