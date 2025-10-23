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

        // This will now run the tuning process
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

    /**
     * This method has been modified to run a tuning process.
     * It will loop through all combinations of weights,
     * battle each combination against Minimax4, and print the results for each battle.
     */
    public static void startBattleMultithreaded() {
        //Please refrain from using Collections.shuffle(List<?> list) in your player,
        //as this is not ideal for use across multiple threads.
        //Use Collections.shuffle(List<?> list, Random random) instead, with the Random object from the player (PylosPlayer.getRandom())

        System.out.println("🚀 Starting weight tuning process...");

        // --- Define the weights and parameters for tuning ---
        int[] heightScores = {0, 30, 60};
        int[] squareScores = {0, 20, 40};
        int[] availableScores = {0, 50, 100};
        int[] trappedScores = {0, 20, 40};

        int nRuns = 100; // Runs per weight combination
        int nThreads = 8; // Number of threads for BattleMT

        long startTime = System.currentTimeMillis();
        int combinations = heightScores.length * squareScores.length * availableScores.length * trappedScores.length;
        int count = 0;

        // --- Loop through all combinations ---
        for (int h : heightScores) {
            for (int s : squareScores) {
                for (int a : availableScores) {
                    for (int t : trappedScores) {
                        count++;
                        String weightsStr = "h=" + h + ", s=" + s + ", a=" + a + ", t=" + t;
                        String playerName = "Student(" + weightsStr + ")";

                        System.out.println("\n=================================================");
                        System.out.println("Testing combination " + count + " of " + combinations + ": " + weightsStr);
                        System.out.println("=================================================");


                        // --- Create player types for this specific battle ---
                        PylosPlayerType p1 = new PylosPlayerType(playerName) {
                            @Override
                            public PylosPlayer create() {
                                // Pass the current loop's weights to the constructor
                                return new StudentPlayer2(h, s, a, t);
                            }
                        };
                        PylosPlayerType p2 = new PylosPlayerType("Minimax4") {
                            @Override
                            public PylosPlayer create() {
                                return new PylosPlayerMiniMax(4);
                            }
                        };

                        // --- Run the battle ---
                        // BattleMT.play() automatically prints the results of this battle to the console.
                        BattleMT.play(p1, p2, nRuns, nThreads);
                    }
                }
            }
        }

        // --- Print final summary ---
        long endTime = System.currentTimeMillis();
        System.out.println("\n=================================================");
        System.out.println("🏆 Weight Tuning Complete 🏆");
        System.out.println("Total time: " + (endTime - startTime) / 1000.0 + " seconds.");
        System.out.println("Total combinations tested: " + combinations);
        System.out.println("=================================================");
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