clc; clear; close all;

% ==== Parameters ====
popSize = 10;
offspringSize = 10;
nGenerations = 1000;
xMin = -10;
xMax = 10;

crossoverRate = 0.8;
mutationRate = 0.01;
tournamentSize = 3;  % 可以調整為 2 或更大

fitnessFunc = @(x) -15 * sin(2 * x).^2 - (x - 2).^2 + 160;

% ==== Initial Population ====
pop = xMin + (xMax - xMin) * rand(popSize, 1);

% ==== For Recording ====
bestFitnessHistory = zeros(1, nGenerations);
bestXHistory = zeros(1, nGenerations);

% ==== Main EA Loop ====
for gen = 1:nGenerations
    % Evaluate fitness
    fitness = fitnessFunc(pop);

    % Record best
    [bestFitness, idx] = max(fitness);
    bestFitnessHistory(gen) = bestFitness;
    bestXHistory(gen) = pop(idx);

    % ==== Tournament Selection for Parents ====
    offspring = zeros(offspringSize, 1);
    for i = 1:2:offspringSize
        % Select parents using tournament selection
        p1 = tournamentSelect(pop, fitness, tournamentSize);
        p2 = tournamentSelect(pop, fitness, tournamentSize);

        % Crossover
        if rand < crossoverRate
            alpha = rand;
            c1 = alpha * p1 + (1 - alpha) * p2;
            c2 = alpha * p2 + (1 - alpha) * p1;
        else
            c1 = p1; c2 = p2;
        end

        % Mutation
        if rand < mutationRate
            c1 = c1 + 0.1 * (xMax - xMin) * randn;
        end
        if rand < mutationRate
            c2 = c2 + 0.1 * (xMax - xMin) * randn;
        end

        % Bound check
        c1 = min(max(c1, xMin), xMax);
        c2 = min(max(c2, xMin), xMax);

        offspring(i) = c1;
        if i + 1 <= offspringSize
            offspring(i+1) = c2;
        end
    end

    % ==== μ + λ Survival Selection ====
    allPop = [pop; offspring];
    allFitness = fitnessFunc(allPop);
    [~, sortedIdx] = sort(allFitness, 'descend');
    pop = allPop(sortedIdx(1:popSize));
end

% ==== Final Result ====
fprintf('Best x = %.5f\n', bestXHistory(end));
fprintf('Best fitness = %.5f\n', bestFitnessHistory(end));

% ==== Plotting ====
figure('Color', 'w', 'Position', [100, 100, 600, 400]);

subplot(2,1,1);
xPlot = linspace(xMin, xMax, 1000);
yPlot = fitnessFunc(xPlot);
plot(xPlot, yPlot, 'b-', 'LineWidth', 2);
xlabel('X'); ylabel('Fitness');
title('Fitness Function f(x)');
grid on;

subplot(2,1,2);
plot(1:nGenerations, bestFitnessHistory, 'r-', 'LineWidth', 2);
xlabel('Generation'); ylabel('Best fitness');
title('Best Fitness Over Generations (EA + Tournament)');
grid on;

% ==== Tournament Selection Function ====
function winner = tournamentSelect(pop, fitness, tSize)
    idx = randperm(length(pop), tSize);
    subPop = pop(idx);
    subFit = fitness(idx);
    [~, bestIdx] = max(subFit);
    winner = subPop(bestIdx);
end
