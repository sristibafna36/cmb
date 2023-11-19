%%  Modelling Competition 
%   PSY-3102-1
%   Model: Reinforcement Learning (Unbiased) + RW Learning Rule
%   Script 

%% Learning Phase

%%  Cleanup 
rng = ('shuffle');  
clc;
clear; 
current_folder = pwd; 

%%  Setting up Environment 

u(1,:) = [0.9 0.6];   % reward probabilities for the rich context
u(2,:) = [0.4 0.1];   % reward probabilities for the poor context
rescale = 0;     % NA to task 
states = [repmat(1,1,50) repmat(2,1,50)];   % setting 2 states for the model: altering between rich and poor contexts
K = 2;  % no of options available to agent to choose between per trial

%%  Setting Simulation 

npart = 1000; % no of participants
T = numel(states);     % no of trials
Q_init = 0.5;  % initial Q value = 0.5 since the model is unbiased

%%  Setting Parameter Values 

for n = 1:npart

    param = [rand rand];    % sampling initial param values from uniform distribution of (0,1)
    param(3) = [param(2)];         % establishing baseline learning rates 
    reduce = rand; % creating random value between 0 and 1 to change alpha values to produce positivity/ negativity bias
    param2 = [ param(1) param(2) param(3)*reduce];  % alpha for positivity bias condition
    param3 = [ param(1) param(2)*reduce param(3)];  % alpha for negativity bias condition
       
    p1(n,:) = param;    
    p2(n,:) = param2;
    p3(n,:) = param3; 

end 

 [choicesRe(n,:),  outcomesRe(n,:), probaRe(n,:),  Q1Re(n,:),  Q2Re(n,:)]  = RL_unbiased_func(param, T, u, Q_init, rescale, states);
 [choicesRe2(n,:), outcomesRe2(n,:),probaRe2(n,:), Q1Re2(n,:), Q2Re2(n,:)] = RL_unbiased_func(param2, T, u, Q_init, rescale, states);
 [choicesRe3(n,:), outcomesRe3(n,:),probaRe3(n,:), Q1Re3(n,:), Q2Re3(n,:)] = RL_unbiased_func(param3, T, u, Q_init, rescale, states);
    

 %% Learning Rates 

colors(2,:)=[223 83 107]./255;
colors(1,:)=[146 208 80]./255;
colors(4,:)=[223 83 107]./255;
colors(3,:)=[146 208 80]./255;
figure
subplot(1,3,1)
violinplot(p1(:,2:3)', colors, ...
    -0, 1, 14, 'Unbiased', '','','');
xticklabels({'\alpha_+','\alpha_-'});
box ON
plot(0:5,repmat(0,6,1),'k','Linewidth',1);
set(gca,'Fontsize',15)
subplot(1,3,2)
violinplot(p2(:,2:3)', colors, ...
    -0, 1, 14, 'Positivity bias', '','','');
xticklabels({'\alpha_+','\alpha_-'});
box ON
plot(0:5,repmat(0,6,1),'k','Linewidth',1);
set(gca,'Fontsize',15)
subplot(1,3,3)
violinplot(p3(:,2:3)', colors, ...
    -0, 1, 14, 'Negativity bias', '','','');
xticklabels({'\alpha_+','\alpha_-'});
box ON
plot(0:5,repmat(0,6,1),'k','Linewidth',1);
set(gca,'Fontsize',15)

%% Differentiating across Rich and Poor contexts and Depressed vs Control groups 
% to translate with our matrices
% 2 is optimistic is controls
% 3 is pessimisti is patients 
% 1:50 rich 
% 51:100 poor

richcont=choicesRe2(:,1:50)'-1; % rich context, control group, positivity bias (optimistic alpha)
poorcont=choicesRe2(:,51:100)'-1;  % poor context, control group, positivity bias (optimistic alpha)
richdepr=choicesRe3(:,1:50)'-1; % rich context, depressed patients, negativity bias (pessimistic alpha)
poordepr=choicesRe3(:,51:100)'-1; % poor context, depressed patients, negativity bias (pessimistic alpha) 
% Continuous plot with function smooth
green(1,:)=[0.1 0.5 0];   % green = control group
green(2,:)=[0.4 0.8 0.5];
orange(1,:)=[0.9 0.4 0];     % orange = depressed patients 
orange(2,:)=[1 0.7 0.3];
   

% Learning curves without legends
figure('Name','5pts Smoothed Correct Choice','NumberTitle','off','Renderer', 'painters');
x=subplot(1,2,1);
Smooth_SurfaceCurvePlot(richcont,green(1,:),green(1,:),2,0.25,0.35,0.85,12,'','','',[0:10:50]);
hold on 
Smooth_SurfaceCurvePlot(poorcont,green(2,:),green(2,:),2,0.25,0.35,0.85,12,'','','',[0:10:50]);
grid on
hold off
axis([1 50 0.35 0.85]);
set(gca,'Fontsize',18);
legend(x(1,:),{'rich','poor'}, 'location','south');
%legend(x(1,1),{'rich','poor','significativity'});
y=subplot(1,2,2);
Smooth_SurfaceCurvePlot(richdepr,orange(1,:),orange(1,:),2,0.25,0.35,0.85,12,'','','',[0:10:50]);
hold on 
Smooth_SurfaceCurvePlot(poordepr,orange(2,:),orange(2,:),2,0.25,0.35,0.85,12,'','','',[0:10:50]);
grid on

axis([1 50 0.35 0.85]);
set(gca,'Fontsize',18);
legend(y(1,:),{'rich','poor'}, 'location','south');




%%  Creating placeholder vectors to record data

%all_choice_probabilities = NaN(npart, T, K);
%all_choices = NaN(npart, T);
%all_rewards = NaN(npart, T); 

%%  Run simulations

%for sub = 1:npart % for each participant, calculate outcomes per trial 

     % [choice_probabilities, rewards, choices]  = RL_unbiased_func(p1, p2, p3, sub, T, K, u, Q_init, states);
      %all_choices(sub,:) = squeeze(choices);
      %all_rewards(sub,:) = squeeze(rewards);
      %all_choice_probabilities(sub,:, :) = squeeze(choice_probabilities);

%end   


%param_values{1} = p1; % beta values
%param_values{2} = p2; % "optimistic" alpha values (positivity bias)
%param_values{3} = p3; % "pessimistic" alpha values (negativity bias) 
%save(fullfile(current_folder, 'RL_unbiased_LearningPhase_results.mat'), 'all_choices', 'all_rewards', 'all_choice_probabilities', 'T', 'K', 'u', 'param_values')





