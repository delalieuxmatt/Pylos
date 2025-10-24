package be.kuleuven.pylos.main;

import be.kuleuven.pylos.battle.Battle;
import be.kuleuven.pylos.battle.BattleMT;
import be.kuleuven.pylos.battle.BattleResult;
import be.kuleuven.pylos.battle.RoundRobin;
import be.kuleuven.pylos.game.PylosBoard;
import be.kuleuven.pylos.game.PylosGame;
import be.kuleuven.pylos.game.PylosGameObserver;
import be.kuleuven.pylos.player.PylosPlayer;
import be.kuleuven.pylos.player.PylosPlayerObserver;
import be.kuleuven.pylos.player.PylosPlayerType;
import be.kuleuven.pylos.player.codes.PlayerFactoryCodes;
import be.kuleuven.pylos.player.codes.PylosPlayerBestFit;
import be.kuleuven.pylos.player.codes.PylosPlayerMiniMax;
import be.kuleuven.pylos.player.student.StudentPlayer;
import be.kuleuven.pylos.player.student.StudentPlayer2;
import be.kuleuven.pylos.player.student.StudentPlayerRandomFit;

import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;

public class PylosMain {

    public static void main(String[] args) {
        /* !!! jvm argument !!! -ea */

        //startSingleGame();
        //startBattle();
        startBattleMultithreaded();
        //startRoundRobinTournament();
    }

    public static void startSingleGame() {

        Random random = new Random(0);

        PylosPlayer playerLight = new PylosPlayerBestFit();
        PylosPlayer playerDark = new PylosPlayerMiniMax(4);

        PylosBoard pylosBoard = new PylosBoard();
        PylosGame pylosGame = new PylosGame(pylosBoard, playerLight, playerDark, random, PylosGameObserver.CONSOLE_GAME_OBSERVER, PylosPlayerObserver.NONE);

        pylosGame.play();
    }

    public static void startBattle() {
        int nRuns = 500;
        PylosPlayerType p1 = new PylosPlayerType("Student") {
            @Override
            public PylosPlayer create() {

                return new StudentPlayer();
                // return new PylosPlayerBestFit();
            }
        };

        PylosPlayerType p2 = new PylosPlayerType("Minimax4") {
            @Override
            public PylosPlayer create() {
                return new PylosPlayerMiniMax(4);
            }
        };

        Battle.play(p1, p2, nRuns);
    }

    public static void startBattleMultithreaded() {
        int nRuns = 100; // Runs per parameter set
        int nThreads = 8;

        // --- NEW: Time constraint ---
        // Max average time per game in milliseconds
        final double MAX_TIME_PER_GAME_MS = 1000.0;

        // --- Hyperparameter Tuning Setup ---
        int[] heightScores =    {0, 50, 100, 150};
        int[] squareScores =    {0, 50, 100, 150};
        int[] availableScores = {0, 50, 100, 150};
        int[] trappedScores =   {0, 50, 100, 150}; // Using non-negative as requested
        int[] reserveScore = {0, 50, 100, 150};

        // --- Variables to track the best result ---
        double bestWinRate = -1.0;
        int[] bestParams = new int[5]; // h, s, a, t, r

        // --- Define the constant opponent ---
        PylosPlayerType p2 = new PylosPlayerType("Minimax4") {
            @Override
            public PylosPlayer create() {
                return new PylosPlayerMiniMax(4);
            }
        };

        int totalCombinations = heightScores.length * squareScores.length * availableScores.length * trappedScores.length * reserveScore.length;
        System.out.println("Starting massive hyperparameter tuning...");
        System.out.println("Testing " + totalCombinations + " combinations (all non-negative) against Minimax4.");
        System.out.println("Constraint: Average time must be < " + (MAX_TIME_PER_GAME_MS / 1000.0) + " sec/game.");
        System.out.println("---------------------------------------------------------");

        int combinationCount = 0;

        // 4. Start the grid search loop
        for (int h : heightScores) {
            for (int s : squareScores) {
                for (int a : availableScores) {
                    for (int t : trappedScores) {
                        for (int r : reserveScore) {
                            final int currentH = h;
                            final int currentS = s;
                            final int currentA = a;
                            final int currentT = t;
                            final int currentR = r;
                            combinationCount++;

                            // 1. Create the P1 player type with current params
                            String playerName = String.format("Student(H:%d,S:%d,A:%d,T:%d, R:%d)", h, s, a, t, r);
                            PylosPlayerType p1 = new PylosPlayerType(playerName) {
                                @Override
                                public PylosPlayer create() {
                                    // Use your StudentPlayer2 constructor
                                    return new StudentPlayer2(currentH, currentS, currentA, currentT, currentR);
                                }
                            };

                            // 2. Run the battle (with print=false)
                            BattleResult result = BattleMT.play(p1, p2, nRuns, nThreads, false);

                            // 3. --- NEW: Check time constraint ---
                            double avgTimePerGameMs = (double) result.runTime / nRuns;

                            // Check if the constraint is met
                            if (avgTimePerGameMs >= MAX_TIME_PER_GAME_MS) {
                                // Constraint FAILED. This set of parameters is invalid.
                                System.out.printf("[%d/%d] Params(H:%d, S:%d, A:%d, T:%d, R:%d) -> FAILED: Time %.2f sec/game. Skipping.%n",
                                        combinationCount, totalCombinations, h, s, a, t, r, avgTimePerGameMs / 1000.0);
                                continue; // Skip to the next set of parameters
                            }

                            // 4. Get the win rate (only if time constraint passed)
                            int p1Wins = result.p1Wins(); // Uses the method from BattleResult
                            int totalGames = nRuns;

                            double winRate = (totalGames == 0) ? 0 : (double) p1Wins / totalGames;

                            // 5. Print progress (now includes time)
                            System.out.printf("[%d/%d] Params(H:%d, S:%d, A:%d, T:%d, R:%d) -> Win Rate: %.2f%% (%d/%d) | Time: %.2f sec/game%n",
                                    combinationCount, totalCombinations, h, s, a, t, r, winRate * 100, p1Wins, totalGames, avgTimePerGameMs / 1000.0);

                            // 6. Track the best result
                            if (winRate > bestWinRate) {
                                bestWinRate = winRate;
                                bestParams[0] = h;
                                bestParams[1] = s;
                                bestParams[2] = a;
                                bestParams[3] = t;
                                bestParams[4] = r;

                                // Print new bests immediately
                                System.out.printf("!!! NEW BEST: %.2f%% with H:%d, S:%d, A:%d, T:%d, R:%d !!!%n",
                                        bestWinRate * 100, h, s, a, t, r);
                            }
                        }
                    }
                }
            }
        }

        // 7. Report the final best result
        System.out.println("---------------------------------------------------------");
        System.out.println("Hyperparameter tuning finished.");

        if (bestWinRate == -1.0) {
            System.out.println("No parameter set met the time constraint of < " + (MAX_TIME_PER_GAME_MS / 1000.0) + " sec/game.");
        } else {
            System.out.println("Best parameters found (meeting time constraints):");
            System.out.printf("  Height Score:    %d%n", bestParams[0]);
            System.out.printf("  Square Score:    %d%n", bestParams[1]);
            System.out.printf("  Available Score: %d%n", bestParams[2]);
            System.out.printf("  Trapped Score:   %d%n", bestParams[3]);
            System.out.printf("  Best Win Rate vs Minimax4: %.2f%%%n", bestWinRate * 100);
        }
        System.out.println("---------------------------------------------------------");
    }

    public static void startRoundRobinTournament() {
        //Same requirements apply as for startBattleMultithreaded()

        //Create your own PlayerFactory containing all PlayerTypes you want to test
        PlayerFactoryCodes pFactory = new PlayerFactoryCodes();
        //PlayerFactoryStudent pFactory = new PlayerFactoryStudent();

        int nRunsPerCombination = 1000;
        int nThreads = 8;

        Set<RoundRobin.Match> matches = RoundRobin.createTournament(pFactory);

        RoundRobin.play(matches, nRunsPerCombination, nThreads);

        List<BattleResult> results = matches.stream().map(c -> c.battleResult).collect(Collectors.toList());

        RoundRobin.printWinsMatrix(results, pFactory.getTypes());
    }
}
